"""Seed-level paired inference and frozen E1 gate."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
from scipy.stats import t


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _one_sided_t_p_less(values: np.ndarray, null: float) -> float:
    diff = values - null
    if diff.size < 2:
        return 1.0
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0 if float(np.mean(diff)) < 0.0 else 1.0
    statistic = float(np.mean(diff) / (sd / math.sqrt(diff.size)))
    return float(t.cdf(statistic, df=diff.size - 1))


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"overall_pass": False, "workloads": {}, "decision_reason": "No E1 result rows were produced."}
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload"]), str(row["method"]), int(row["n"]), int(row["seed"]))].append(row)
    workloads: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {
        f"{workload}:{baseline}": 1.0
        for workload in ("sparse_pulses", "smooth_periodic")
        for baseline in ("B1", "B2", "B3")
    }
    log_ratios: dict[str, np.ndarray] = {}
    selected: dict[str, tuple[str, np.ndarray, float]] = {}

    for workload in ("sparse_pulses", "smooth_periodic"):
        baseline_seed_logs: dict[str, list[float]] = {}
        max_error: dict[str, float] = {}
        methods = ("B1", "B2", "B3")
        for method in (*methods, "P"):
            errors, seed_logs = [], defaultdict(lambda: defaultdict(list))
            for (wl, mth, n, seed), reps in grouped.items():
                if wl != workload or mth != method:
                    continue
                errors.extend(float(r["nrmse"]) for r in reps)
                seed_logs[seed][n].extend(float(r["wall_seconds"]) for r in reps)
            max_error[method] = max(errors, default=float("inf"))
            baseline_seed_logs[method] = [
                float(np.mean([math.log(float(np.median(values))) for _, values in sorted(by_size.items())]))
                for _, by_size in sorted(seed_logs.items()) if len(by_size) == 3
            ]
        def complete(method: str) -> bool:
            return all(len(grouped.get((workload, method, n, seed), [])) == 10 for n in (64, 128, 256) for seed in (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009))

        eligible = [m for m in methods if max_error[m] <= 1e-3 and len(baseline_seed_logs[m]) == 10 and complete(m)]
        baseline = min(eligible, key=lambda m: float(np.median(baseline_seed_logs[m]))) if eligible else None
        if baseline is None or max_error["P"] > 1e-3 or not complete("P"):
            workloads[workload] = {"pass": False, "baseline": baseline, "max_nrmse": max_error["P"], "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "holm_p": 1.0, "reason": "no eligible sparse baseline or IRCN exceeded NRMSE limit"}
            continue
        # The independent unit is seed; aggregate paired size cells within seed before inference.
        for base in methods:
            per_seed: dict[int, list[float]] = defaultdict(list)
            for (wl, mth, n, seed), reps in grouped.items():
                if wl != workload or mth != "P":
                    continue
                b_reps = grouped.get((wl, base, n, seed), [])
                if not b_reps:
                    continue
                p_median = float(np.median([float(r["wall_seconds"]) for r in reps]))
                b_median = float(np.median([float(r["wall_seconds"]) for r in b_reps]))
                per_seed[seed].append(math.log(p_median / b_median))
            arr = np.asarray([float(np.mean(per_seed[s])) for s in sorted(per_seed) if len(per_seed[s]) == 3], dtype=float)
            key = f"{workload}:{base}"
            p_values[key] = _one_sided_t_p_less(arr, math.log(0.90)) if arr.size == 10 else 1.0
            log_ratios[key] = arr
        key = f"{workload}:{baseline}"
        arr = log_ratios.get(key, np.asarray([]))
        if arr.size == 10:
            mean = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
            margin = float(t.ppf(0.975, df=arr.size - 1) * sem)
            selected[workload] = (baseline, arr, p_values[key])
            workloads[workload] = {"pass": False, "baseline": baseline, "max_nrmse": max_error["P"], "ratio": math.exp(mean), "ci_low": math.exp(mean - margin), "ci_high": math.exp(mean + margin), "holm_p": 1.0, "seed_log_ratios": arr.tolist()}
        else:
            workloads[workload] = {"pass": False, "baseline": baseline, "max_nrmse": max_error["P"], "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "holm_p": 1.0, "reason": "incomplete paired seed cells"}

    adjusted = holm_adjust(p_values) if p_values else {}
    for workload, (baseline, _arr, _p) in selected.items():
        item = workloads[workload]
        item["holm_p"] = adjusted.get(f"{workload}:{baseline}", 1.0)
        item["pass"] = bool(item["max_nrmse"] <= 1e-3 and item["ratio"] <= 0.90 and item["ci_high"] < 1.0 and item["holm_p"] < 0.05)
    passes = sum(bool(v.get("pass")) for v in workloads.values())
    overall = passes == 2
    if overall:
        reason = "Both co-primary workloads passed the frozen accuracy, >=10% latency, confidence-interval, and Holm-adjusted criteria. E2 is recommended for a separate preregistration only."
    elif passes == 1:
        reason = "Exactly one co-primary workload passed; pivot to that workload. E2 is not opened until a narrowed preregistration is approved."
    else:
        reason = "Neither co-primary workload passed the frozen G1 criteria; stop this acceleration/connectome-extension route. E2 is not recommended."
    return {"overall_pass": overall, "workloads": workloads, "all_contrast_p_values": p_values, "holm_adjusted_p_values": adjusted, "decision_reason": reason}
