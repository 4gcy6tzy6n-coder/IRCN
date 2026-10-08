#!/usr/bin/env python3
"""Run the frozen v0.6.0 synchronous trainability / influence bridge."""
from __future__ import annotations

import csv
import gzip
import io
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "result" / "v0.6.0"
RUN_ID = os.environ.get("IRCN_V06_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", RUN_ID):
    raise ValueError(f"invalid run id: {RUN_ID!r}")
RUN_DIR = RESULT_DIR / "runs" / RUN_ID
sys.path.insert(0, str(MODEL_DIR / "src"))

from ircn_v06 import (  # noqa: E402
    GatedCell,
    N_FEATURES,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    average_ranks,
    balanced_row_weights,
    canonical_array_hash,
    configure_torch,
    counterfactual_labels,
    feature_rows,
    generate_input,
    generate_teacher,
    graph,
    paired_cluster_bootstrap,
    rollout,
    spearman,
    top_precision,
    trajectories,
)


class InfluenceMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(6062)
        self.w1 = torch.nn.Parameter(torch.empty((16, N_FEATURES), dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float64))
        self.w2 = torch.nn.Parameter(torch.empty((1, 16), dtype=torch.float64))
        self.b2 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.w1, generator=gen)
        torch.nn.init.xavier_uniform_(self.w2, generator=gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, self.w1, self.b1)), self.w2, self.b2).squeeze(-1)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array_digest = canonical_array_hash(state[name].detach().cpu().numpy())
        digest.update(name.encode("utf-8") + b"\0" + array_digest.encode("ascii") + b"\n")
    return digest.hexdigest()


def arrays_sha256(**arrays) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array_digest = canonical_array_hash(np.asarray(arrays[name], dtype=np.float64))
        digest.update(name.encode("utf-8") + b"\0" + array_digest.encode("ascii") + b"\n")
    return digest.hexdigest()


class EarlyStoppingTracker:
    """Keep lowest-loss weights; apply min_delta only to patience resets."""

    def __init__(self, min_delta: float, patience: int) -> None:
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.stale_epochs = 0

    def observe(self, loss: float, epoch: int, state: dict[str, torch.Tensor]) -> bool:
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}: {loss}")
        improvement = self.best_loss - loss
        if loss < self.best_loss - 1e-12:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.best_state = {key: value.detach().clone() for key, value in state.items()}
        if improvement > self.min_delta:
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def fit_standardization(x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> dict:
    """Fit weighted scaling from training rows only."""
    weights = np.asarray(weights, dtype=np.float64)
    if x_train.ndim != 2 or y_train.shape != (len(x_train),) or weights.shape != (len(x_train),):
        raise ValueError("training arrays and weights have inconsistent shapes")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all() or not np.isfinite(weights).all():
        raise ValueError("training arrays and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    x_mean = np.average(x_train, axis=0, weights=weights)
    x_var = np.average((x_train - x_mean) ** 2, axis=0, weights=weights)
    x_std = np.sqrt(x_var)
    x_std[x_std == 0.0] = 1.0
    y_mean = float(np.average(y_train, weights=weights))
    y_std = float(np.sqrt(np.average((y_train - y_mean) ** 2, weights=weights))) or 1.0
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def aggregate_seed_metrics(metric_rows: list[dict], replicates: int = 10_000, seed: int = 8686) -> tuple[list[dict], dict]:
    """Aggregate trajectories within seeds, then bootstrap the independent seeds."""
    if not metric_rows:
        raise ValueError("cannot aggregate empty metrics")
    seed_rows = []
    for top_seed in sorted({int(row["top_seed"]) for row in metric_rows}):
        group = [row for row in metric_rows if int(row["top_seed"]) == top_seed]
        keys = [key for key in group[0] if key not in ("top_seed", "trajectory_index", "n_opportunities", "status")]
        entry = {"top_seed": top_seed, "trajectory_count": len(group), "status": "OK"}
        for key in keys:
            vals = np.asarray([row[key] for row in group], dtype=np.float64)
            entry[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        seed_rows.append(entry)
    summary = {}
    keys = [key for key in seed_rows[0] if key not in ("top_seed", "trajectory_count", "status")]
    for key in keys:
        vals = np.asarray([row[key] for row in seed_rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
            "sd": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan"),
            "bootstrap_95_percentile_ci": list(paired_cluster_bootstrap(vals, replicates=replicates, seed=seed)),
        }
    return seed_rows, summary


def write_failure_record(error: BaseException, run_id: str, result_dir: Path = RESULT_DIR) -> Path:
    failure_dir = result_dir / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    path = failure_dir / f"{run_id}.json"
    failure = {"status": "FAILED", "exception_type": type(error).__name__, "message": str(error), "timestamp_unix": time.time()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(failure, stream, indent=2)
        stream.write("\n")
    return path


def freeze_source_provenance() -> dict[str, str]:
    tracked = [
        path for path in (MODEL_DIR / "src").rglob("*") if path.is_file() and path.suffix == ".py"
    ]
    tracked += [MODEL_DIR / "run_experiment.py", MODEL_DIR / "MODEL_CONTRACT_v0.1.md", MODEL_DIR / "pre_registration_v1.yaml", MODEL_DIR / "requirements.lock"]
    tracked += [path for path in (RESULT_DIR / "tests").rglob("*.py") if path.is_file()]
    records = []
    for path in sorted(set(tracked)):
        records.append({"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)})
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_manifest_sha = hashlib.sha256(payload).hexdigest()
    config_paths = [MODEL_DIR / "MODEL_CONTRACT_v0.1.md", MODEL_DIR / "pre_registration_v1.yaml", MODEL_DIR / "requirements.lock"]
    config_payload = "\n".join(f"{p.name}:{file_sha256(p)}" for p in config_paths).encode("utf-8")
    configuration_sha = hashlib.sha256(config_payload).hexdigest()
    write_csv(RUN_DIR / "manifests" / "source_hashes.csv", ["path", "sha256"], records)
    return {"source_manifest_sha256": source_manifest_sha, "configuration_sha256": configuration_sha}


def write_input_provenance(items: list, dirs: dict[str, Path]) -> None:
    rows = []
    for item in items:
        rows.append({
            "top_seed": item.top_seed,
            "trajectory_index": item.trajectory_index,
            "input_sha256": canonical_array_hash(item.x),
            "teacher_sha256": canonical_array_hash(item.teacher),
        })
    write_csv(dirs["manifests"] / "input_teacher_hashes.csv", list(rows[0]), rows)


def prepare_dirs() -> dict[str, Path]:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {RUN_DIR}")
    dirs = {name: RUN_DIR / name for name in ("logs", "raw", "checkpoints", "manifests")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def tensor_data(items: list) -> tuple[torch.Tensor, torch.Tensor]:
    xs = np.stack([item.x for item in items])
    ys = np.stack([item.teacher[1:, 7] for item in items])
    return torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)


def fit_cell(items_train: list, items_val: list, dirs: dict[str, Path]) -> tuple[GatedCell, dict]:
    configure_torch()
    x_train, y_train = tensor_data(items_train)
    x_val, y_val = tensor_data(items_val)
    model = GatedCell(seed=6060, readout_seed=6064)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    order_gen = torch.Generator(device="cpu").manual_seed(6061)
    early = EarlyStoppingTracker(min_delta=1e-6, patience=15)
    history = []
    epoch_path = dirs["logs"] / "cell_training_epochs.csv"
    write_csv(epoch_path, ["epoch", "train_mse", "validation_mean_trajectory_mse"], [])
    start_time = time.perf_counter()
    for epoch in range(1, 201):
        model.train()
        order = torch.randperm(len(items_train), generator=order_gen)
        train_loss_sum = 0.0
        train_count = 0
        for offset in range(0, len(items_train), 32):
            indices = order[offset:offset + 32]
            _, prediction = rollout(model, x_train[indices])
            loss = (prediction - y_train[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_count += len(indices)
        model.eval()
        with torch.no_grad():
            _, val_prediction = rollout(model, x_val)
            per_trajectory_mse = (val_prediction - y_val).square().mean(dim=1)
            val_loss = float(per_trajectory_mse.mean())
        epoch_train = train_loss_sum / train_count
        if not np.isfinite(epoch_train):
            raise FloatingPointError(f"non-finite cell training loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_mse": epoch_train, "validation_mean_trajectory_mse": val_loss})
        with epoch_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=list(history[-1])).writerow(history[-1])
        should_stop = early.observe(val_loss, epoch, model.state_dict())
        if early.best_epoch == epoch:
            torch.save(early.best_state, dirs["checkpoints"] / "cell_state.pt")
        if should_stop:
            break
    if early.best_state is None:
        raise RuntimeError("cell training produced no finite validation checkpoint")
    model.load_state_dict(early.best_state)
    model.eval()
    torch.save(model.state_dict(), dirs["checkpoints"] / "cell_state.pt")
    metadata = {
        "best_epoch": early.best_epoch,
        "best_validation_mean_trajectory_mse": early.best_loss,
        "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - start_time,
        "checkpoint_sha256": file_sha256(dirs["checkpoints"] / "cell_state.pt"),
        "canonical_parameter_sha256": tensor_state_sha256(model.state_dict()),
        "loss_history_sha256": file_sha256(epoch_path),
    }
    return model, metadata


def evaluate_cell(model: GatedCell, test_items: list, train_items: list, dirs: dict[str, Path]) -> tuple[list[dict], dict]:
    _, y_train = tensor_data(train_items)
    constant = float(y_train.mean())
    rows = []
    torch_y = torch.as_tensor(np.stack([item.teacher[1:, 7] for item in test_items]), dtype=torch.float64)
    torch_x = torch.as_tensor(np.stack([item.x for item in test_items]), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        states_t, predictions_t = rollout(model, torch_x)
    states = states_t.cpu().numpy()
    predictions = predictions_t.cpu().numpy()
    targets = torch_y.cpu().numpy()
    for idx, item in enumerate(test_items):
        target = targets[idx]
        sd = float(np.std(target, ddof=0))
        mse = float(np.mean((predictions[idx] - target) ** 2))
        baseline_mse = float(np.mean((constant - target) ** 2))
        nrmse = float(np.sqrt(mse) / sd) if sd > 0 else float("nan")
        baseline_nrmse = float(np.sqrt(baseline_mse) / sd) if sd > 0 else float("nan")
        rows.append({
            "top_seed": item.top_seed,
            "trajectory_index": item.trajectory_index,
            "target_sd": sd,
            "mse": mse,
            "nrmse": nrmse,
            "constant_training_mean_mse": baseline_mse,
            "constant_training_mean_nrmse": baseline_nrmse,
            "status": "OK" if np.isfinite(nrmse) else "UNDEFINED_ZERO_VARIANCE",
            "input_sha256": canonical_array_hash(item.x),
        })
    write_csv(dirs["raw"] / "cell_test_trajectory_metrics.csv.gz", list(rows[0]), rows)
    finite_model = [row["nrmse"] for row in rows if np.isfinite(row["nrmse"])]
    finite_base = [row["constant_training_mean_nrmse"] for row in rows if np.isfinite(row["constant_training_mean_nrmse"])]
    macro_model = float(np.mean(finite_model)) if finite_model else float("nan")
    macro_base = float(np.mean(finite_base)) if finite_base else float("nan")
    nonconstant = [row for row in rows if row["target_sd"] > 0]
    finite_nonconstant = [row for row in nonconstant if np.isfinite(row["nrmse"])]
    learnability = "BEATS_BASELINE" if macro_model < macro_base and len(finite_nonconstant) == len(nonconstant) and len(nonconstant) > 0 else "DOES_NOT_BEAT_BASELINE"
    summary = {"macro_mean_test_nrmse": macro_model, "macro_mean_baseline_nrmse": macro_base, "learnability_descriptor": learnability, "test_trajectory_count": len(rows), "nonconstant_count": len(nonconstant), "undefined_count": sum(not np.isfinite(r["nrmse"]) for r in rows)}
    return rows, summary


def make_label_rows(model: GatedCell, items: list, split: str, dirs: dict[str, Path], provenance: dict[str, str], cell_hash: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rows = []
    xs, deltas, seeds, nodes, ticks = [], [], [], [], []
    model.eval()
    for item in items:
        x_t = torch.as_tensor(item.x[None], dtype=torch.float64)
        with torch.no_grad():
            states_t, _ = rollout(model, x_t)
        states = states_t[0].cpu().numpy()
        features, paired_losses, delta = counterfactual_labels(model, item.x, states, item.teacher)
        _, opportunities = feature_rows(item.x, states)
        if not (np.isfinite(features).all() and np.isfinite(paired_losses).all() and np.isfinite(delta).all()):
            raise FloatingPointError(f"non-finite counterfactual data for split={split}, seed={item.top_seed}, trajectory={item.trajectory_index}")
        isolated = np.asarray([node >= 8 for node, _ in opportunities])
        if np.any(np.abs(delta[isolated]) > 1e-12):
            raise AssertionError(f"isolated-node counterfactual Delta nonzero for split={split}, seed={item.top_seed}, trajectory={item.trajectory_index}")
        for row_idx, (node, tick) in enumerate(opportunities):
            xs.append(features[row_idx])
            deltas.append(delta[row_idx])
            seeds.append(item.top_seed)
            nodes.append(node)
            ticks.append(tick)
            row = {
                "split": split,
                "top_seed": item.top_seed,
                "trajectory_index": item.trajectory_index,
                "node": node,
                "tick": tick,
                "loss_update": float(paired_losses[row_idx, 0]),
                "loss_hold": float(paired_losses[row_idx, 1]),
                "signed_delta": float(delta[row_idx]),
                "input_sha256": canonical_array_hash(item.x),
                "teacher_sha256": canonical_array_hash(item.teacher),
                "snapshot_sha256": canonical_array_hash(states[tick]),
                "feature_sha256": canonical_array_hash(features[row_idx]),
                "trained_cell_sha256": cell_hash,
                "source_manifest_sha256": provenance["source_manifest_sha256"],
                "configuration_sha256": provenance["configuration_sha256"],
                "failure_status": "OK",
            }
            row.update({f"feature_{j:02d}": float(v) for j, v in enumerate(features[row_idx])})
            rows.append(row)
    label_path = dirs["raw"] / f"counterfactual_labels_{split}.csv.gz"
    write_csv(label_path, list(rows[0]), rows)
    return np.stack(xs), np.asarray(deltas), np.asarray(seeds), np.asarray(nodes), np.asarray(ticks), rows


def fit_predictor(x_train: np.ndarray, y_train: np.ndarray, seeds: np.ndarray, nodes: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, val_seeds: np.ndarray, val_nodes: np.ndarray, dirs: dict[str, Path]) -> tuple[InfluenceMLP, dict]:
    weights = balanced_row_weights(seeds, nodes)
    val_weights = balanced_row_weights(val_seeds, val_nodes)
    scaling = fit_standardization(x_train, y_train, weights)
    x_mean, x_std = scaling["x_mean"], scaling["x_std"]
    y_mean, y_std = scaling["y_mean"], scaling["y_std"]
    x_train_s = (x_train - x_mean) / x_std
    x_val_s = (x_val - x_mean) / x_std
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std
    np.savez(dirs["checkpoints"] / "predictor_standardization.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    model = InfluenceMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    order_gen = torch.Generator(device="cpu").manual_seed(6063)
    xt = torch.as_tensor(x_train_s, dtype=torch.float64)
    yt = torch.as_tensor(y_train_s, dtype=torch.float64)
    wt = torch.as_tensor(weights, dtype=torch.float64)
    xv = torch.as_tensor(x_val_s, dtype=torch.float64)
    yv = torch.as_tensor(y_val_s, dtype=torch.float64)
    wv = torch.as_tensor(val_weights, dtype=torch.float64)
    early = EarlyStoppingTracker(min_delta=1e-6, patience=10)
    history = []
    epoch_path = dirs["logs"] / "predictor_training_epochs.csv"
    write_csv(epoch_path, ["epoch", "train_weighted_mse", "validation_weighted_mse"], [])
    start_time = time.perf_counter()
    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(x_train), generator=order_gen)
        train_loss_total = 0.0
        train_weight_total = 0.0
        for offset in range(0, len(x_train), 256):
            idx = order[offset:offset + 256]
            pred = model(xt[idx])
            batch_weights = wt[idx]
            loss = torch.sum(batch_weights * (pred - yt[idx]).square()) / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_total += float(torch.sum(batch_weights * (pred.detach() - yt[idx]).square()))
            train_weight_total += float(torch.sum(batch_weights))
        model.eval()
        with torch.no_grad():
            val_pred = model(xv)
            val_loss = float(torch.sum(wv * (val_pred - yv).square()) / torch.sum(wv))
        train_weighted_mse = train_loss_total / train_weight_total
        if not np.isfinite(train_weighted_mse):
            raise FloatingPointError(f"non-finite predictor training loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_weighted_mse": train_weighted_mse, "validation_weighted_mse": val_loss})
        with epoch_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=list(history[-1])).writerow(history[-1])
        should_stop = early.observe(val_loss, epoch, model.state_dict())
        if early.best_epoch == epoch:
            torch.save(early.best_state, dirs["checkpoints"] / "influence_mlp_state.pt")
        if should_stop:
            break
    if early.best_state is None:
        raise RuntimeError("predictor training produced no finite validation checkpoint")
    model.load_state_dict(early.best_state)
    model.eval()
    torch.save(model.state_dict(), dirs["checkpoints"] / "influence_mlp_state.pt")
    metadata = {
        "best_epoch": early.best_epoch,
        "best_validation_weighted_mse": early.best_loss,
        "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - start_time,
        "checkpoint_sha256": file_sha256(dirs["checkpoints"] / "influence_mlp_state.pt"),
        "canonical_parameter_sha256": tensor_state_sha256(model.state_dict()),
        "loss_history_sha256": file_sha256(epoch_path),
        "standardization_sha256": arrays_sha256(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std),
    }
    return model, {**metadata, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def evaluate_predictor(model: InfluenceMLP, metadata: dict, x_test: np.ndarray, y_test: np.ndarray, test_seeds: np.ndarray, nodes: np.ndarray, ticks: np.ndarray, rows: list[dict], dirs: dict[str, Path]) -> tuple[list[dict], list[dict], dict]:
    xs = torch.as_tensor((x_test - metadata["x_mean"]) / metadata["x_std"], dtype=torch.float64)
    with torch.no_grad():
        prediction = model(xs).cpu().numpy() * metadata["y_std"] + metadata["y_mean"]
    scores = {
        "predictor": prediction,
        "input_change_abs": np.abs(x_test[:, 17]),
        "prior_residual": x_test[:, 19],
        "out_degree": x_test[:, 24],
        "distance_to_readout": -x_test[:, 25],
    }
    rand_scores = np.empty(len(y_test), dtype=np.float64)
    for seed in np.unique(test_seeds):
        for traj in np.unique([r["trajectory_index"] for r in rows if r["top_seed"] == seed]):
            mask = (test_seeds == seed) & (np.asarray([r["trajectory_index"] for r in rows]) == traj)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7070, int(seed), int(traj)])))
            rand_scores[mask] = rng.random(int(mask.sum()))
    scores["fixed_seed_random"] = rand_scores
    prediction_rows = []
    metric_rows = []
    predictor_hash = file_sha256(dirs["checkpoints"] / "influence_mlp_state.pt")
    trajectory_ids = [(r["top_seed"], r["trajectory_index"]) for r in rows]
    for seed, traj in sorted(set(trajectory_ids)):
        mask = np.asarray([(s == seed and t == traj) for s, t in trajectory_ids])
        y = y_test[mask]
        ns = nodes[mask]
        ts = ticks[mask]
        pred = prediction[mask]
        zero_mse = float(np.mean(y ** 2))
        predictor_mse = float(np.mean((pred - y) ** 2))
        valid_sign = np.abs(y) > 1e-12
        base = {
            "top_seed": seed,
            "trajectory_index": traj,
            "n_opportunities": int(mask.sum()),
            "predictor_signed_delta_mse": predictor_mse,
            "zero_delta_mse": zero_mse,
            "predictor_minus_zero_mse": predictor_mse - zero_mse,
            "predictor_sign_accuracy": float(np.mean(np.sign(pred[valid_sign]) == np.sign(y[valid_sign]))) if valid_sign.any() else float("nan"),
            "predictor_spearman": spearman(pred, y),
            "predictor_precision_top20": top_precision(pred, y, ns, ts),
            "status": "OK",
        }
        for name, all_score in scores.items():
            score = all_score[mask]
            base[f"{name}_spearman"] = spearman(score, y)
            base[f"{name}_precision_top20"] = top_precision(score, y, ns, ts)
        for stratum, stratum_mask in (("path", ns < 8), ("isolated", ns >= 8)):
            ys = y[stratum_mask]
            ns_s = ns[stratum_mask]
            ts_s = ts[stratum_mask]
            ps = pred[stratum_mask]
            base[f"{stratum}_predictor_mse"] = float(np.mean((ps - ys) ** 2))
            base[f"{stratum}_zero_delta_mse"] = float(np.mean(ys ** 2))
            sign_mask = np.abs(ys) > 1e-12
            base[f"{stratum}_predictor_sign_accuracy"] = float(np.mean(np.sign(ps[sign_mask]) == np.sign(ys[sign_mask]))) if sign_mask.any() else float("nan")
            base[f"{stratum}_predictor_spearman"] = spearman(ps, ys)
            base[f"{stratum}_predictor_precision_top20"] = top_precision(ps, ys, ns_s, ts_s)
            for name, all_score in scores.items():
                ss = all_score[mask][stratum_mask]
                base[f"{stratum}_{name}_spearman"] = spearman(ss, ys)
                base[f"{stratum}_{name}_precision_top20"] = top_precision(ss, ys, ns_s, ts_s)
        metric_rows.append(base)
        for local, global_idx in enumerate(np.flatnonzero(mask)):
            row = dict(rows[global_idx])
            row["predicted_delta"] = float(prediction[global_idx])
            row["predictor_checkpoint_sha256"] = predictor_hash
            for name, values in scores.items():
                row[f"score_{name}"] = float(values[global_idx])
            prediction_rows.append(row)
    write_csv(dirs["raw"] / "test_predictions.csv.gz", list(prediction_rows[0]), prediction_rows)
    write_csv(dirs["raw"] / "test_trajectory_metrics.csv.gz", list(metric_rows[0]), metric_rows)
    seed_rows, summary = aggregate_seed_metrics(metric_rows)
    write_csv(dirs["raw"] / "test_seed_metrics.csv.gz", list(seed_rows[0]), seed_rows)
    summary["predictor_minus_zero_mse_ci"] = list(paired_cluster_bootstrap(
        np.asarray([r["predictor_signed_delta_mse"] for r in seed_rows]),
        np.asarray([r["zero_delta_mse"] for r in seed_rows]),
    ))
    for name in scores:
        if name == "predictor":
            continue
        summary[f"predictor_minus_{name}_precision_ci"] = list(paired_cluster_bootstrap(
            np.asarray([r["predictor_precision_top20"] for r in seed_rows]),
            np.asarray([r[f"{name}_precision_top20"] for r in seed_rows]),
        ))
    return prediction_rows, seed_rows, summary


def main() -> None:
    configure_torch()
    dirs = prepare_dirs()
    prereg_path = MODEL_DIR / "pre_registration_v1.yaml"
    config = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    provenance = freeze_source_provenance()
    # Golden provenance is verified before training or label generation.
    sentinel_x = generate_input(424242, 0)
    sentinel_y = generate_teacher(sentinel_x)
    fixture = config["golden"]
    if canonical_array_hash(sentinel_x) != fixture["input_sha256"]:
        raise RuntimeError("frozen input golden hash mismatch")
    if canonical_array_hash(sentinel_y) != fixture["teacher_sha256"]:
        raise RuntimeError("frozen teacher golden hash mismatch")
    if canonical_array_hash(np.asarray(fixture["feature_fixture"]["vector"])) != fixture["feature_fixture"]["sha256"]:
        raise RuntimeError("frozen feature golden hash mismatch")

    train_items = trajectories(TRAIN_SEEDS)
    val_items = trajectories(VALIDATION_SEEDS)
    test_items = trajectories(TEST_SEEDS)
    write_input_provenance(train_items + val_items + test_items, dirs)
    model, cell_meta = fit_cell(train_items, val_items, dirs)
    _, cell_summary = evaluate_cell(model, test_items, train_items, dirs)

    # Labels for train/validation only are used to fit the predictor.
    x_train, y_train, s_train, n_train, t_train, train_label_rows = make_label_rows(model, train_items, "train", dirs, provenance, cell_meta["checkpoint_sha256"])
    x_val, y_val, s_val, n_val, t_val, val_label_rows = make_label_rows(model, val_items, "validation", dirs, provenance, cell_meta["checkpoint_sha256"])
    predictor, predictor_meta = fit_predictor(x_train, y_train, s_train, n_train, x_val, y_val, s_val, n_val, dirs)

    # Test labels are generated only after the frozen estimator is selected.
    x_test, y_test, s_test, n_test, t_test, test_label_rows = make_label_rows(model, test_items, "test", dirs, provenance, cell_meta["checkpoint_sha256"])
    test_label_path = dirs["raw"] / "counterfactual_labels_test.csv.gz"
    test_label_rows_only = test_label_rows
    write_csv(test_label_path, list(test_label_rows_only[0]), test_label_rows_only)
    _, seed_metrics, influence_summary = evaluate_predictor(predictor, predictor_meta, x_test, y_test, s_test, n_test, t_test, test_label_rows, dirs)

    write_csv(RUN_DIR / "v0.6_seed_results.csv", list(seed_metrics[0]), seed_metrics)
    summary = {
        "stage": "v0.6.0 synchronous trainability and influence-label bridge",
        "claim_boundary": "exploratory synthetic mechanism study; no scheduler, efficiency, connectome, or generalization claim",
        "cell_training": {k: v for k, v in cell_meta.items()},
        "cell_test": cell_summary,
        "predictor_training": {k: v for k, v in predictor_meta.items() if k not in ("x_mean", "x_std", "y_mean", "y_std")},
        "test_influence_metrics_by_seed": influence_summary,
        "counts": {"train_trajectories": len(train_items), "validation_trajectories": len(val_items), "test_trajectories": len(test_items), "train_labels": len(train_label_rows), "validation_labels": len(val_label_rows), "test_labels": len(test_label_rows)},
        "provenance": provenance,
    }
    summary["run_id"] = RUN_ID
    (RUN_DIR / "v0.6_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mps_available_but_disabled": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    (dirs["manifests"] / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    output_paths = [p for folder in (dirs["raw"], dirs["logs"], dirs["checkpoints"], dirs["manifests"]) for p in folder.rglob("*") if p.is_file() and p.name != "output_hashes.csv"]
    output_paths += [RUN_DIR / "v0.6_summary.json", RUN_DIR / "v0.6_seed_results.csv"]
    output_hash_rows = [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in sorted(output_paths)]
    write_csv(dirs["manifests"] / "output_hashes.csv", ["path", "sha256"], output_hash_rows)
    print(json.dumps({"run_id": RUN_ID, "run_dir": str(RUN_DIR.relative_to(ROOT)), "cell": cell_summary, "predictor_best_epoch": predictor_meta["best_epoch"], "train_labels": len(train_label_rows), "test_labels": len(test_label_rows)}, allow_nan=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_failure_record(error, RUN_ID)
        raise
