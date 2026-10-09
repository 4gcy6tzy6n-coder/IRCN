"""Pilot, freeze, and confirmatory CBCC experiment runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cbcc_tasks import dataset, generate, N_TICKS, N_NODES, N_CIRCUITS
from cbcc_model import RecurrentModel, WIDTH, rollout_full
from cbcc_counterfactuals import paired_labels, joint_benefit
from cbcc_scheduler import POLICIES, choose, fit_influence, mac_costs
from cbcc_evaluation import power_n, holm, spearman

TRAIN_SEEDS = tuple(range(100, 110))
VALIDATION_SEEDS = tuple(range(1200, 1208))
PILOT_SEEDS = tuple(range(1300, 1308))
N_PER_SEED = 8
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUT = HERE / "results"
RUN_ID = "preflight"
RUN_PHASE = "smoke"
UPDATE_MAC = 8 * (32 * 8 + 8 * 8 + 8)
BUDGET_MAC_PER_TICK = UPDATE_MAC + 4 * (3 * 8 + 9)
BASELINES = ("random", "fixed_frequency", "activity", "no_influence")
MAX_CONFIRMATORY_SEEDS = 512


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def trained_models():
    train_a = dataset("sparse_key", TRAIN_SEEDS, N_PER_SEED)
    train_b = dataset("feedback_switch", TRAIN_SEEDS, N_PER_SEED)
    alphas = {}
    trained = {}
    for family, data in (("sparse_key", train_a), ("feedback_switch", train_b)):
        fold_scores = {ridge: [] for ridge in RIDGE_GRID}
        zero_fold_scores = []
        seeds = sorted({seed for seed, _ in data})
        for fold in range(5):
            validation_seeds = {seed for i, seed in enumerate(seeds) if i % 5 == fold}
            fit_data = [item for item in data if item[0] not in validation_seeds]
            held_data = [item for item in data if item[0] in validation_seeds]
            zero_fold_scores.append(float(np.mean([np.mean(seq.y ** 2) for _, seq in held_data])))
            for ridge in RIDGE_GRID:
                candidate = RecurrentModel()
                candidate.fit_readout(fit_data, ridge=ridge)
                errors = []
                for _, seq in held_data:
                    _, prediction = rollout_full(candidate, seq)
                    errors.append(float(np.mean((prediction - seq.y) ** 2)))
                fold_scores[ridge].append(float(np.mean(errors)))
        selected_ridge = min(RIDGE_GRID, key=lambda r: np.mean(fold_scores[r]))
        model = RecurrentModel()
        model.fit_readout(data, ridge=selected_ridge)
        model.ridge_cv_scores = {str(r): [float(v) for v in scores] for r, scores in fold_scores.items()}
        model.training_cv_zero_mse = [float(v) for v in zero_fold_scores]
        model.training_cv_selected_mse = [float(v) for v in fold_scores[selected_ridge]]
        trained[family] = model
        alphas[family] = selected_ridge
    # The scheduler is trained only on sparse_key; transfer to feedback_switch is zero-shot.
    xs, ys = [], []
    for _, seq in train_a:
        f, y = paired_labels(trained["sparse_key"], seq)
        xs.append(f); ys.append(y)
    estimator = fit_influence(np.concatenate(xs), np.concatenate(ys))
    return trained, estimator


def model_selection_metadata(models):
    return {family: {"ridge": model.readout_ridge,
                     "training_cv_scores": model.ridge_cv_scores,
                     "training_cv_zero_mse": model.training_cv_zero_mse,
                     "training_cv_selected_mse": model.training_cv_selected_mse}
            for family, model in models.items()}


def validate_freeze(freeze, models=None):
    """Reject stale or edited pilot freezes before confirmatory execution."""
    required = {
        "version": "CBCC-confirmatory-v2",
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "pilot_seeds": list(PILOT_SEEDS),
        "test_seed_start": 3000,
        "budget_mac_per_tick": BUDGET_MAC_PER_TICK,
        "n_per_seed": N_PER_SEED,
    }
    for key, expected in required.items():
        if freeze.get(key) != expected:
            raise ValueError(f"freeze contract mismatch for {key}: expected {expected!r}")
    if not freeze.get("sample_size_feasible"):
        raise ValueError("freeze marks confirmatory sample size infeasible")
    comparisons = freeze.get("comparisons", {})
    test_seeds = freeze.get("test_seeds", {})
    families = {"sparse_key", "feedback_switch"}
    if set(comparisons) != families or set(test_seeds) != families:
        raise ValueError("freeze must define both registered task families")
    for family, comparison in comparisons.items():
        n = comparison.get("n_confirmatory_seeds")
        if not isinstance(n, int) or not 8 <= n <= MAX_CONFIRMATORY_SEEDS:
            raise ValueError(f"invalid confirmatory sample size for {family}")
        if comparison.get("primary_baseline") not in BASELINES:
            raise ValueError(f"invalid primary baseline for {family}")
        if not comparison.get("per_seed_budget_match_within_1pct"):
            raise ValueError(f"primary baseline is not per-seed budget-matched for {family}")
        if test_seeds[family] != list(range(3000, 3000 + n)):
            raise ValueError(f"confirmatory seed list mismatch for {family}")
    validation = freeze.get("task_learnability_validation_check", {})
    if set(validation) != families or not all(item.get("eligible") for item in validation.values()):
        raise ValueError("freeze does not pass the fresh validation learnability gate")
    if models is not None and freeze.get("model_selection") != model_selection_metadata(models):
        raise ValueError("freeze model-selection metadata differs from recomputed training pipeline")


def evaluate_sequence(policy, model, estimator, seq, seed, sequence_index, initial_credit=0):
    t0 = time.perf_counter()
    h = np.zeros((N_NODES, WIDTH), dtype=np.float64)
    states = np.zeros((N_TICKS, N_NODES, WIDTH), dtype=np.float64)
    predictions, events = [], []
    rng = np.random.default_rng(np.random.SeedSequence([seed, sequence_index, 991]))
    spent = updates = 0
    tick_costs = []
    budget_credit = initial_credit
    for tick in range(N_TICKS):
        budget_credit += BUDGET_MAC_PER_TICK
        selected, cost = choose(policy, model, estimator, seq, states, tick,
                                budget_credit, rng)
        budget_credit -= cost
        before = h.copy()
        for c in selected:
            lo, hi = c * 8, (c + 1) * 8
            h[lo:hi] = model.step_circuit(before, seq.x[tick], c)
        pred = model.predict(h)
        predictions.append(pred)
        states[tick] = h
        spent += cost
        tick_costs.append(cost)
        updates += len(selected)
        events.append({"tick": tick, "selected_circuits": selected, "mac_proxy": cost,
                       "state_changed": bool(np.any(h != before))})
    elapsed = time.perf_counter() - t0
    err = np.asarray(predictions) - seq.y
    active = np.flatnonzero(seq.y)
    cue_hit = seq.key_circuit in events[seq.cue_tick]["selected_circuits"]
    switch_hit = seq.switch_tick is None or seq.key_circuit in events[seq.switch_tick]["selected_circuits"]
    missed = not (cue_hit and switch_hit)
    recovery = recovery_cost(err, tick_costs, seq.target_onset) if missed else None
    return {"loss": float(np.mean(err ** 2)), "active_loss": float(np.mean(err[active] ** 2)) if active.size else float("nan"),
            "recovery_cost_mac": recovery, "recovered": recovery is not None,
            "critical_event_recall": float(cue_hit), "switch_event_recall": float(switch_hit), "event_missed": missed,
            "wall_seconds": elapsed, "mac_proxy": spent, "updates": updates,
            "budget_left": budget_credit,
            "critical_recall": float(cue_hit),
            "events": events}


def recovery_cost(errors, tick_costs, start):
    if start >= len(errors):
        return None
    for tick in range(start, len(errors) - 1):
        if np.all(np.abs(errors[tick:tick + 2]) <= 0.1):
            return int(sum(tick_costs[start:tick + 2]))
    return None


def evaluate(seeds, models, estimator, families=("sparse_key", "feedback_switch")):
    rows, raw = [], []
    for family in families:
        model = models[family]
        for seed in seeds:
            per_policy = {p: [] for p in (*POLICIES, "full_update", "zero_output")}
            credits = {p: 0 for p in POLICIES}
            diagnostics = []
            for i in range(N_PER_SEED):
                seq = generate(family, seed, i)
                diagnostic_states, _ = rollout_full(model, seq)
                diagnostic_features, diagnostic_labels = paired_labels(model, seq)
                dimension = diagnostic_features.shape[1]
                mean, scale, coefficients = estimator[:dimension], estimator[dimension:2*dimension], estimator[2*dimension:]
                diagnostic_scores = ((diagnostic_features - mean) / scale) @ coefficients
                joint_gaps = []
                for tick in range(0, N_TICKS - 15 + 1, 4):
                    for left in range(N_CIRCUITS):
                        for right in range(left + 1, N_CIRCUITS):
                            joint = joint_benefit(model, seq, tick, (left, right), states=diagnostic_states)
                            sum_pred = diagnostic_scores[tick * N_CIRCUITS + left] + diagnostic_scores[tick * N_CIRCUITS + right]
                            joint_gaps.append(abs(joint - sum_pred))
                cf_diag = {"influence_spearman": spearman(diagnostic_scores, diagnostic_labels),
                           "joint_additivity_abs_gap": float(np.mean(joint_gaps))}
                diagnostics.append(cf_diag)
                for p in POLICIES:
                    try:
                        result = evaluate_sequence(p, model, estimator, seq, seed, i, credits[p])
                    except Exception as error:
                        (OUT / "failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
                        with (OUT / "failures.jsonl").open("a") as f:
                            f.write(json.dumps({"family": family, "seed": seed, "sequence": i,
                                                "policy": p, "error": repr(error)}, sort_keys=True, default=json_default) + "\n")
                        raise
                    per_policy[p].append(result)
                    credits[p] = result["budget_left"]
                    raw.append({"run_id": RUN_ID, "phase": RUN_PHASE, "family": family, "seed": seed, "sequence": i, "policy": p, **cf_diag, **{k:v for k,v in result.items() if k != "events"}})
                    for event in result["events"]:
                        raw_event = {"run_id": RUN_ID, "phase": RUN_PHASE, "family": family, "seed": seed, "sequence": i, "policy": p, **event}
                        (OUT / "events.jsonl").parent.mkdir(parents=True, exist_ok=True)
                        with (OUT / "events.jsonl").open("a") as f:
                            f.write(json.dumps(raw_event, sort_keys=True, default=json_default) + "\n")
                full_t0 = time.perf_counter()
                states, preds = rollout_full(model, seq)
                full_seconds = time.perf_counter() - full_t0
                full_loss = float(np.mean((preds - seq.y) ** 2))
                full_err = preds - seq.y
                full_recovery = None
                per_policy["full_update"].append({"loss": full_loss, "active_loss": float("nan"), "wall_seconds": full_seconds,
                                                   "mac_proxy": N_TICKS*4*UPDATE_MAC, "updates": N_TICKS*4,
                                                   "critical_recall": 1.0, "critical_event_recall": 1.0,
                                                   "switch_event_recall": 1.0, "event_missed": False,
                                                   "recovery_cost_mac": full_recovery, "recovered": False})
                zero_t0 = time.perf_counter()
                zero_predictions = np.zeros(N_TICKS, dtype=np.float64)
                zero_loss = float(np.mean((zero_predictions - seq.y) ** 2))
                zero_seconds = time.perf_counter() - zero_t0
                per_policy["zero_output"].append({"loss": zero_loss, "active_loss": float("nan"),
                                                   "mac_proxy": 0, "updates": 0, "critical_recall": 0.0,
                                                   "critical_event_recall": 0.0, "switch_event_recall": 0.0, "event_missed": True,
                                                   "recovery_cost_mac": None, "recovered": False,
                                                   "wall_seconds": zero_seconds})
            # Each task seed is one statistical unit; within-seed sequences are averaged.
            for p, values in per_policy.items():
                rows.append({"run_id": RUN_ID, "phase": RUN_PHASE, "family": family, "seed": seed, "policy": p,
                             **{key: finite_mean(v[key] for v in values) for key in ("loss", "active_loss", "wall_seconds", "mac_proxy", "updates", "critical_recall", "critical_event_recall", "switch_event_recall", "recovery_cost_mac")},
                             "total_mac_proxy": float(sum(v["mac_proxy"] for v in values)),
                             "total_mac_budget": float(N_PER_SEED * N_TICKS * BUDGET_MAC_PER_TICK),
                             "mac_budget_utilization": float(sum(v["mac_proxy"] for v in values) / (N_PER_SEED * N_TICKS * BUDGET_MAC_PER_TICK)),
                             "miss_fraction": float(np.mean([v.get("event_missed", False) for v in values])),
                             "recovery_fraction_given_miss": (float(sum(v["recovered"] for v in values) / sum(v.get("event_missed", False) for v in values)) if any(v.get("event_missed", False) for v in values) else float("nan")),
                             "recovery_censor_fraction_given_miss": (float(sum(v.get("event_missed", False) and not v["recovered"] for v in values) / sum(v.get("event_missed", False) for v in values)) if any(v.get("event_missed", False) for v in values) else float("nan")),
                             "influence_spearman": float(np.mean([d["influence_spearman"] for d in diagnostics if np.isfinite(d["influence_spearman"])])) if diagnostics else float("nan"),
                             "joint_additivity_abs_gap": float(np.mean([d["joint_additivity_abs_gap"] for d in diagnostics])) if diagnostics else float("nan")})
    return rows, raw


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def finite_mean(items):
    values = [float(x) for x in items if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else float("nan")


def write_run_failure(reason):
    (OUT / "failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (OUT / "failures.jsonl").open("a") as handle:
        handle.write(json.dumps({"run_id": RUN_ID, "phase": RUN_PHASE,
                                 "error": reason}, sort_keys=True) + "\n")


def main():
    global OUT, RUN_ID, RUN_PHASE
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "confirm"))
    parser.add_argument("--freeze", type=Path, help="required confirmatory freeze file from the pilot")
    args = parser.parse_args()
    freeze = None
    if args.mode == "confirm":
        if args.freeze is None or not args.freeze.is_file():
            raise SystemExit("confirmatory mode requires an existing pilot freeze file")
        freeze = json.loads(args.freeze.read_text())
        validate_freeze(freeze)
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    RUN_PHASE = args.mode
    OUT = HERE / "results" / f"{args.mode}_{RUN_ID}"
    OUT.mkdir(parents=True, exist_ok=False)
    # Preserve each complete run's JSONL; experiment runs are append-only.
    models, estimator = trained_models()
    if args.mode == "pilot":
        RUN_PHASE = "validation"
        validation_rows, validation_raw = evaluate(VALIDATION_SEEDS, models, estimator)
        write_csv(OUT / "validation_seed_metrics.csv", validation_rows)
        write_csv(OUT / "validation_raw_metrics.csv", validation_raw)
        eligibility = {}
        for family in ("sparse_key", "feedback_switch"):
            fam = [r for r in validation_rows if r["family"] == family]
            full = float(np.mean([r["loss"] for r in fam if r["policy"] == "full_update"]))
            zero = float(np.mean([r["loss"] for r in fam if r["policy"] == "zero_output"]))
            eligibility[family] = {"full_update_mse": full, "zero_output_mse": zero, "eligible": full < zero}
        if not all(value["eligible"] for value in eligibility.values()):
            (OUT / "validation_gate.json").write_text(json.dumps(eligibility, indent=2) + "\n")
            write_run_failure("validation learnability gate failed; pilot and confirmatory evaluation prohibited")
            raise SystemExit("validation full-update model did not beat zero-output baseline in both task families")
        RUN_PHASE = "pilot"
        rows, raw = evaluate(PILOT_SEEDS, models, estimator)
        write_csv(OUT / "pilot_seed_metrics.csv", rows)
        write_csv(OUT / "pilot_raw_metrics.csv", raw)
        comparisons = {}
        budget_matches = {}
        model_selection = model_selection_metadata(models)
        for family in ("sparse_key", "feedback_switch"):
            fam = [r for r in rows if r["family"] == family]
            by_policy_seed = {p: {r["seed"]: r for r in fam if r["policy"] == p}
                              for p in ("cbcc", *BASELINES)}
            cbcc = {seed: r["loss"] for seed, r in by_policy_seed["cbcc"].items()}
            budget_matches[family] = {
                p: {str(seed): abs(r["total_mac_proxy"] - by_policy_seed["cbcc"][seed]["total_mac_proxy"])
                    / max(by_policy_seed["cbcc"][seed]["total_mac_proxy"], 1.0)
                    for seed, r in by_policy_seed[p].items()}
                for p in BASELINES
            }
            eligible = [p for p in BASELINES if all(
                budget_matches[family][p][str(seed)] <= 0.01 for seed in cbcc)]
            selected_baseline = min(eligible, key=lambda p: np.mean([r["loss"] for r in fam if r["policy"] == p])) if eligible else None
            if selected_baseline is None:
                relative, n = np.asarray([]), 0
            else:
                baseline_loss = {r["seed"]: r["loss"] for r in fam if r["policy"] == selected_baseline}
                relative = np.asarray([(baseline_loss[s] - cbcc[s]) / max(abs(baseline_loss[s]), 1e-12) for s in cbcc])
                n = power_n(float(relative.std(ddof=1)), margin=0.05, target_effect=0.10, alpha=0.05/8, power=0.8)
            comparisons[family] = {"pilot_relative_improvement_mean": float(relative.mean()) if len(relative) else None,
                                   "pilot_sd": float(relative.std(ddof=1)) if len(relative) > 1 else None, "n_confirmatory_seeds": n,
                                   "primary_baseline": selected_baseline,
                                   "per_seed_budget_match_within_1pct": selected_baseline is not None,
                                   "baseline_policy_losses": {p: float(np.mean([r['loss'] for r in fam if r['policy']==p])) for p in BASELINES}}
        freeze = {"version": "CBCC-confirmatory-v2", "train_seeds": TRAIN_SEEDS,
                  "validation_seeds": VALIDATION_SEEDS, "pilot_seeds": PILOT_SEEDS, "test_seed_start": 3000,
                  "confirmatory_power": "Paired normal approximation; H0 <=5% relative improvement, H1=10%; power=.80; alpha=.05/8 conservative Holm first step; minimum 8 seeds; maximum feasible 512 seeds",
                  "budget_mac_per_tick": BUDGET_MAC_PER_TICK, "n_per_seed": N_PER_SEED,
                  "comparisons": comparisons, "model_selection": model_selection,
                  "multiplicity": "Holm correction across all four scheduler controls in both task families (8 tests)",
                  "budget_matched_baselines": budget_matches,
                  "task_learnability_validation_check": eligibility,
                  "sample_size_feasible": all(8 <= c["n_confirmatory_seeds"] <= MAX_CONFIRMATORY_SEEDS and c["primary_baseline"] is not None for c in comparisons.values()),
                  "test_seeds": {f: list(range(3000, 3000 + c["n_confirmatory_seeds"])) for f,c in comparisons.items()},
                  "interpretation": "confirmatory test seeds frozen after independent pilot; do not change model, policy or threshold after this file is created"}
        (OUT / "confirmatory_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
        if not freeze["sample_size_feasible"]:
            write_run_failure("pilot infeasible: no per-seed budget-matched baseline or required sample exceeds 512 seeds")
            raise SystemExit("pilot found no budget-matched baseline or required sample exceeds the frozen feasibility limit")
        print(json.dumps(freeze, indent=2))
    else:
        assert freeze is not None
        validate_freeze(freeze, models=models)
        if not all(value["eligible"] for value in freeze["task_learnability_validation_check"].values()):
            raise SystemExit("validation learnability gate failed; confirmatory execution is prohibited")
        if not freeze["sample_size_feasible"]:
            raise SystemExit("frozen sample size is infeasible; confirmatory execution is prohibited")
        (OUT / "confirmatory_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
        families = ("sparse_key", "feedback_switch")
        rows, raw = [], []
        for family in families:
            fam_rows, fam_raw = evaluate(freeze["test_seeds"][family], models, estimator, (family,))
            rows.extend(fam_rows)
            raw.extend(fam_raw)
        write_csv(OUT / "confirmatory_seed_metrics.csv", rows)
        write_csv(OUT / "confirmatory_raw_metrics.csv", raw)
        summaries = {}
        all_comparisons = []
        for family in families:
            fam = [r for r in rows if r["family"] == family]
            vals = {p: {r["seed"]: r["loss"] for r in fam if r["policy"] == p} for p in (*POLICIES, "full_update")}
            comparisons = []
            for p in BASELINES:
                d = np.asarray([(vals[p][s] - vals["cbcc"][s]) / max(abs(vals[p][s]), 1e-12) for s in vals["cbcc"]])
                # Test the frozen 5% success margin, not merely positive benefit.
                centered = d - 0.05
                obs = float(centered.mean()); rng = np.random.default_rng(7171)
                draws = np.mean(centered[None, :] * rng.choice([-1.0, 1.0], (50000, len(d))), axis=1)
                pval = float((1 + np.sum(draws >= obs)) / (len(draws) + 1))
                boots = np.mean(d[rng.integers(0, len(d), (10000, len(d)))], axis=1)
                item = {"family": family, "baseline": p, "relative_improvement_mean": float(d.mean()),
                        "margin_adjusted_mean": obs,
                        "ci95": [float(x) for x in np.quantile(boots, [0.025,0.975])], "p_raw": pval}
                comparisons.append(item); all_comparisons.append(item)
        adjusted = holm([c["p_raw"] for c in all_comparisons])
        for c, pa in zip(all_comparisons, adjusted): c["p_holm"] = pa
        for family in families:
            comparisons = [c for c in all_comparisons if c["family"] == family]
            strongest = next(c for c in comparisons if c["baseline"] == freeze["comparisons"][family]["primary_baseline"])
            fam_rows = [r for r in rows if r["family"] == family]
            by_policy_seed = {p: {r["seed"]: r for r in fam_rows if r["policy"] == p}
                              for p in ("cbcc", strongest["baseline"])}
            seed_budget_differences = {
                str(seed): abs(r["total_mac_proxy"] - by_policy_seed["cbcc"][seed]["total_mac_proxy"])
                / max(by_policy_seed["cbcc"][seed]["total_mac_proxy"], 1.0)
                for seed, r in by_policy_seed[strongest["baseline"]].items()
            }
            budget_difference = max(seed_budget_differences.values(), default=float("inf"))
            summaries[family] = {"n_seeds": len(freeze["test_seeds"][family]), "comparisons": comparisons,
                                 "frozen_primary_baseline": strongest["baseline"],
                                 "max_per_seed_actual_budget_relative_difference": budget_difference,
                                 "per_seed_budget_differences": seed_budget_differences,
                                 "primary_pass": bool(budget_difference <= 0.01 and strongest["relative_improvement_mean"] >= .05 and strongest["p_holm"] < .05)}
        result = {"freeze": freeze, "families": summaries,
                  "overall_continue": all(s["primary_pass"] for s in summaries.values()),
                  "claim_ceiling": "synthetic diagnostic tasks only; feedback_switch policy evaluation is zero-shot with respect to influence-estimator training"}
        (OUT / "confirmatory_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if OUT.exists():
            (OUT / "failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
            with (OUT / "failures.jsonl").open("a") as handle:
                handle.write(json.dumps({"run_id": RUN_ID, "phase": RUN_PHASE, "error": repr(error)}, default=json_default) + "\n")
        raise
