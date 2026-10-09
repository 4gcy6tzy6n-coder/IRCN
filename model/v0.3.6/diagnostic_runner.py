"""Single-instance, preregistered WP2 implementation-sensitivity diagnostic.

The official experiment is deliberately opt-in and requires a separate review
receipt. Importing this module and running its unit tests never executes a variant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set before importing NumPy/SciPy so the actual numerical libraries inherit it.
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_name] = "1"

import numpy as np
import scipy
import yaml

REPO = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO / "model/v0.3.6/DIAGNOSTIC_CONTRACT_v1.yaml"
PROTOCOL_PATH = REPO / "model/v0.3.6/DIAGNOSTIC_PROTOCOL_v1.md"
WP2 = REPO / "result/v0.3.5/WP2"
WP2_RUN = WP2 / "run_v1"
PARENT_MANIFEST = REPO / "result/v0.3.5/SHA256SUMS.csv"
SOURCE_ROOT = REPO / "model/v0.3.5"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def verify_frozen_inputs() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    immutability = contract["immutability"]
    # Audit each explicitly frozen parent artifact directly; the aggregate manifest
    # is an additional check, not a substitute for the declared file hashes.
    declared = {
        immutability["parent_wp2_contract"]: immutability["parent_wp2_contract_sha256"],
        immutability["parent_raw_labels_path"]: immutability["parent_raw_labels_sha256"],
        immutability["parent_instances_path"]: immutability["parent_instances_sha256"],
        immutability["candidate_scheduler_path"]: immutability["candidate_scheduler_sha256"],
        immutability["source_manifest"]: immutability["source_manifest_sha256"],
    }
    for relative, expected in declared.items():
        path = REPO / relative
        actual = sha256(path.read_bytes()) if path.is_file() else "MISSING"
        if actual != expected:
            raise RuntimeError(f"explicit frozen artifact hash mismatch: {relative}: {actual}")
    for relative, expected in contract["source_hashes"].items():
        path = REPO / relative
        actual = sha256(path.read_bytes()) if path.is_file() else "MISSING"
        if actual != expected:
            raise RuntimeError(f"frozen source hash mismatch: {relative}: {actual}")
    with PARENT_MANIFEST.open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    mismatches = []
    for row in rows:
        path = REPO / row["path"]
        actual = sha256(path.read_bytes()) if path.is_file() else "MISSING"
        if actual != row["sha256"]:
            mismatches.append({"path": row["path"], "expected": row["sha256"], "actual": actual})
    if len(rows) != immutability["required_parent_manifest_matches"] or mismatches:
        raise RuntimeError(f"parent manifest verification failed: {len(rows)} rows, {len(mismatches)} mismatches")
    return contract


def _load_parent_case(validate_input_fingerprint: bool = True):
    # Pin math-library threading before importing NumPy/SciPy-backed parent modules.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from dynamics.tasks import make_task_system
    from events.inputs import make_input
    from solvers.oracle import solve_oracle

    c = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    instance = c["instance"]
    ident = (instance["split"], instance["graph"], instance["population_size"],
             instance["task"], instance["seed"])
    rows = [json.loads(line) for line in (WP2_RUN / "instances.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in rows if (row["split"], row["graph"], row["n"], row["task"], row["seed"]) == ident]
    if len(matches) != 1:
        raise RuntimeError(f"frozen instance identity expected one row, got {len(matches)}")
    parent_meta = matches[0]
    x, driven = make_input(instance["task"], instance["population_size"], instance["seed"])
    times = np.linspace(0.0, instance["duration_seconds"],
                        round(instance["duration_seconds"] / instance["sample_dt_seconds"]) + 1)
    trace = np.asarray([x(float(t)) for t in times], dtype="<f8")
    # Validate the generated trace before experimental timing. Worker processes
    # reuse this preregistered fingerprint without rehashing it.
    if parent_meta.get("input_sha256") != instance["expected_input_trace_sha256"]:
        raise RuntimeError("frozen instance metadata disagrees with preregistered input fingerprint")
    input_hash = verify_input_trace_fingerprint(
        trace, instance["expected_input_trace_sha256"], validate_input_fingerprint)
    instance_hash = None
    system = make_task_system(instance["graph"], instance["population_size"], instance["seed"], driven)
    # Candidate scheduler is deliberately imported only after the generated
    # input bytes have passed the frozen fingerprint gate.
    from events.ode_segment_scheduler import solve_ode_segments

    return c, instance, parent_meta, x, system, times, trace, input_hash, instance_hash, solve_ode_segments, solve_oracle


def verify_input_trace_fingerprint(trace: np.ndarray, expected: str,
                                   calculate_hash: bool = True) -> str:
    if not calculate_hash:
        return expected
    actual = sha256(np.ascontiguousarray(trace, dtype="<f8").tobytes(order="C"))
    if actual != expected:
        raise RuntimeError(f"pre-run input trace hash mismatch: {actual}")
    return actual


def rms_nrmse(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.sqrt(np.mean(reference * reference))), 1e-12)
    return float(np.sqrt(np.mean((actual - reference) ** 2)) / scale)


def max_error(actual: np.ndarray, reference: np.ndarray) -> tuple[float, int, int]:
    delta = np.abs(np.asarray(actual) - np.asarray(reference))
    flat = int(np.argmax(delta))  # NumPy's first maximum gives earliest time, then lowest node.
    time_index, node_index = np.unravel_index(flat, delta.shape)
    return float(delta[time_index, node_index]), int(time_index), int(node_index)


def rk4_reference(system, x, times: np.ndarray, max_step: float,
                  input_breakpoints: list[float], left_limit_epsilon: float = 1e-9) -> np.ndarray:
    """Independent full-state classical RK4; steps terminate at samples/breakpoints."""
    times = np.asarray(times, dtype=np.float64)
    t0, t1 = float(times[0]), float(times[-1])
    boundaries = sorted(set([t0, t1, *map(float, times),
                             *(float(b) for b in input_breakpoints if t0 < b < t1)]))
    n = system.W.shape[0]
    state = np.zeros(n, dtype=np.float64)
    output = np.empty((len(times), n), dtype=np.float64)
    sample_index = {float(t): i for i, t in enumerate(times)}
    if t0 in sample_index:
        output[sample_index[t0]] = state

    def input_at(t: float, end_boundary: float, left_end: bool) -> np.ndarray:
        if left_end and t == end_boundary and end_boundary in input_breakpoints:
            t = end_boundary - left_limit_epsilon
        return np.asarray(x(float(t)), dtype=np.float64)

    def rhs(t: float, h: np.ndarray, end_boundary: float, left_end: bool) -> np.ndarray:
        ext = input_at(t, end_boundary, left_end)
        drive = system.W @ h + system.U @ ext + system.b
        return (-h + np.tanh(drive)) / system.tau

    current = t0
    for boundary in boundaries[1:]:
        while current < boundary:
            h = min(max_step, boundary - current)
            end = current + h
            # Ensure exact boundary landing when this is the shortened final step.
            if boundary - end <= 8 * np.finfo(float).eps * max(1.0, abs(boundary)):
                end = boundary
                h = end - current
            k1 = rhs(current, state, boundary, False)
            k2 = rhs(current + h / 2, state + h * k1 / 2, boundary, False)
            k3 = rhs(current + h / 2, state + h * k2 / 2, boundary, False)
            k4 = rhs(end, state + h * k3, boundary, end == boundary)
            state = state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            current = end
        if boundary in sample_index:
            output[sample_index[boundary]] = state
    return output


def _semantic_hash(trajectory: np.ndarray, event_bytes: bytes, counters: dict[str, Any],
                   resolved_config: dict[str, Any], schema_id: str) -> tuple[str, str, dict[str, str]]:
    normalized = _normalized_config(resolved_config)
    arr = np.ascontiguousarray(trajectory, dtype="<f8")
    header = canonical_json({"schema_id": schema_id, "dtype": "<f8", "shape": list(arr.shape)})
    counter_bytes = canonical_json(counters)
    config_bytes = canonical_json(normalized)
    pieces = [header, arr.tobytes(order="C"), event_bytes, counter_bytes, config_bytes]
    framed = b"".join(struct.pack("<Q", len(part)) + part for part in pieces)
    component_hashes = {"header": sha256(header), "trajectory": sha256(pieces[1]),
                        "event_jsonl": sha256(event_bytes), "counters": sha256(counter_bytes),
                        "normalized_config": sha256(config_bytes)}
    return sha256(framed), sha256(canonical_json(resolved_config)), component_hashes


def _normalized_config(resolved_config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in resolved_config.items()
            if k not in {"variant_id", "id", "repeat_of", "run_id", "output_path", "timestamp"}}


def _repeats_equal(run_dir: Path, first_variant_id: str, second_variant_id: str) -> bool:
    first = run_dir / first_variant_id
    second = run_dir / second_variant_id
    if (first / "trajectory.npy").read_bytes() != (second / "trajectory.npy").read_bytes():
        return False
    if (first / "events.jsonl").read_bytes() != (second / "events.jsonl").read_bytes():
        return False
    first_result = json.loads((first / "result.json").read_text(encoding="utf-8"))
    second_result = json.loads((second / "result.json").read_text(encoding="utf-8"))
    return (first_result["counters"] == second_result["counters"]
            and _normalized_config(first_result["config"]) == _normalized_config(second_result["config"]))


def _capture_worker_streams(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    stdout_file = (out / "stdout.log").open("ab", buffering=0)
    stderr_file = (out / "stderr.log").open("ab", buffering=0)
    os.dup2(stdout_file.fileno(), 1)
    os.dup2(stderr_file.fileno(), 2)
    return stdout_file, stderr_file


def _reference_worker(output_dir: str, validate_input_fingerprint: bool, queue) -> None:
    out = Path(output_dir)
    stream_handles = _capture_worker_streams(out)
    try:
        c, instance, _parent_meta, x, system, times, trace, input_hash, instance_hash, _scheduler, solve_oracle = _load_parent_case(validate_input_fingerprint)
        (out / "input_instance.json").write_bytes(canonical_json({
            "split": instance["split"], "graph": instance["graph"],
            "n": instance["population_size"], "task": instance["task"], "seed": instance["seed"],
            "duration_seconds": instance["duration_seconds"], "sample_dt_seconds": instance["sample_dt_seconds"],
            "input_trace_sha256": input_hash, "instance_sha256": instance_hash}) + b"\n")
        started = time.perf_counter()
        queue.put({"stage": "dop853_oracle"})
        oracle, oracle_info = solve_oracle(system, x, times,
                                          rtol=c["diagnostic_matrix"]["oracle"]["rtol"],
                                          atol=c["diagnostic_matrix"]["oracle"]["atol"])
        oracle_seconds = time.perf_counter() - started
        np.save(out / "oracle_dop853.npy", np.asarray(oracle, dtype="<f8"), allow_pickle=False)
        queue.put({"stage": "rk4_coarse"})
        coarse = rk4_reference(system, x, times, 0.001,
                               list(x.breakpoints(float(times[0]), float(times[-1]))),
                               c["diagnostic_matrix"]["independent_oracle_check"]["left_limit_epsilon_seconds"])
        coarse_seconds = time.perf_counter() - started - oracle_seconds
        np.save(out / "oracle_rk4_001.npy", np.asarray(coarse, dtype="<f8"), allow_pickle=False)
        queue.put({"stage": "rk4_fine"})
        fine = rk4_reference(system, x, times, 0.0005,
                             list(x.breakpoints(float(times[0]), float(times[-1]))),
                             c["diagnostic_matrix"]["independent_oracle_check"]["left_limit_epsilon_seconds"])
        if not all(np.isfinite(z).all() for z in (oracle, coarse, fine)):
            raise FloatingPointError("non-finite independent reference trajectory")
        fine_seconds = time.perf_counter() - started - oracle_seconds - coarse_seconds
        np.save(out / "oracle_rk4_0005.npy", np.asarray(fine, dtype="<f8"), allow_pickle=False)
        np.save(out / "input_trace.npy", trace, allow_pickle=False)
        cfg = c["diagnostic_matrix"]["independent_oracle_check"]
        coarse_fine = {"nrmse_ab": rms_nrmse(coarse, fine), "nrmse_ba": rms_nrmse(fine, coarse),
                       "max_abs": max_error(coarse, fine)[0]}
        each_vs_oracle = {}
        for label, arr in (("coarse", coarse), ("fine", fine)):
            each_vs_oracle[label] = {"nrmse_rk4_as_candidate": rms_nrmse(arr, oracle),
                                     "nrmse_dop853_as_candidate": rms_nrmse(oracle, arr),
                                     "max_abs": max_error(arr, oracle)[0]}
        pair_limit = cfg["refinement_pair_limits"]
        oracle_limit = cfg["each_refinement_vs_dop853_limits"]
        checks = {
            "rk4_refinements_converged": (coarse_fine["max_abs"] <= pair_limit["max_abs"]
                                           and coarse_fine["nrmse_ab"] <= pair_limit["nrmse"]
                                           and coarse_fine["nrmse_ba"] <= pair_limit["nrmse"]),
            "both_rk4_agree_with_dop853": all(
                data["max_abs"] <= oracle_limit["max_abs"]
                and data["nrmse_rk4_as_candidate"] <= oracle_limit["nrmse"]
                and data["nrmse_dop853_as_candidate"] <= oracle_limit["nrmse"]
                for data in each_vs_oracle.values())}
        summary = {"status": "complete", "oracle_nfev": int(oracle_info.nfev),
                   "input_trace_sha256": input_hash,
                   "instance_sha256": instance_hash,
                   "oracle_seconds": oracle_seconds, "rk4_coarse_seconds": coarse_seconds,
                   "rk4_fine_seconds": fine_seconds, "refinement_pair": coarse_fine,
                   "rk4_vs_dop853": each_vs_oracle, "checks": checks}
        (out / "oracle_check.json").write_bytes(canonical_json(summary) + b"\n")
        queue.put({"status": "complete", "summary": summary})
    except BaseException as exc:
        failure = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "stage": "reference"}
        if out.exists():
            (out / "failure.json").write_bytes(canonical_json(failure) + b"\n")
        queue.put(failure)


def _run_staged_process(target, args: tuple, stage_timeouts: dict[str, int], label: str) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue))
    process.start()
    stage = next(iter(stage_timeouts))
    deadline = time.monotonic() + stage_timeouts[stage]
    result = None
    while result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join()
            result = {"status": "timeout", "stage": stage, "timeout_seconds": stage_timeouts[stage],
                      "traceback": f"worker terminated after timeout in stage {stage}",
                      "exitcode": process.exitcode}
            break
        try:
            message = queue.get(timeout=min(0.5, remaining))
        except Exception:
            if not process.is_alive():
                result = {"status": "failed", "error": f"{label} worker exited with code {process.exitcode}"}
            continue
        if message.get("stage") in stage_timeouts:
            stage = message["stage"]
            deadline = time.monotonic() + stage_timeouts[stage]
        elif message.get("status") in {"complete", "failed"}:
            result = message
    process.join(timeout=1)
    queue.close()
    queue.join_thread()
    return result


def _variant_worker(config: dict[str, Any], output_dir: str, references_dir: str,
                    validate_input_fingerprint: bool, queue) -> None:
    out = Path(output_dir)
    stream_handles = _capture_worker_streams(out)
    try:
        c, instance, parent_meta, x, system, times, trace, input_hash, instance_hash, scheduler, _ = _load_parent_case(validate_input_fingerprint)
        oracle = np.load(Path(references_dir) / "oracle_dop853.npy", allow_pickle=False)
        rk4_coarse = np.load(Path(references_dir) / "oracle_rk4_001.npy", allow_pickle=False)
        rk4_fine = np.load(Path(references_dir) / "oracle_rk4_0005.npy", allow_pickle=False)
        queue.put({"stage": "candidate"})
        cfg = config.copy()
        ident = cfg["id"]
        kwargs = {"segment_tolerance": cfg.get("segment_tolerance", c["diagnostic_matrix"]["candidate_common"]["segment_tolerance"]),
                  "max_interval": cfg["max_interval_seconds"],
                  "input_mode": c["diagnostic_matrix"]["candidate_common"]["input_mode"],
                  "delivery_mode": cfg["delivery_mode"], "local_rtol": cfg["local_rtol"],
                  "local_atol": cfg["local_atol"], "local_batch": cfg["local_batch"],
                  "local_batch_size": cfg["local_batch_size"]}
        started = time.perf_counter()
        trajectory, meta = scheduler(system, x, times, **kwargs)
        scheduler_elapsed = time.perf_counter() - started
        events = meta.pop("event_log")
        event_bytes = b"".join(canonical_json(event) + b"\n" for event in events)
        np.save(out / "trajectory.npy", np.asarray(trajectory, dtype="<f8"), allow_pickle=False)
        (out / "events.jsonl").write_bytes(event_bytes)
        (out / "resolved_config.json").write_bytes(canonical_json(cfg) + b"\n")
        counters = {k: int(meta[k]) for k in ("event_count", "message_deliveries", "queue_pushes",
                                               "rhs_evaluations", "output_materializations")}
        if not np.isfinite(trajectory).all() or not np.isfinite(oracle).all():
            raise FloatingPointError("non-finite state in candidate or frozen oracle")
        if trajectory.shape != (201, 128):
            raise ValueError(f"trajectory shape must be [201,128], got {trajectory.shape}")
        result = {"variant_id": ident, "status": "complete", "config": cfg,
                  "input_sha256": input_hash, "instance_sha256": instance_hash,
                  "trace_shape": list(trace.shape),
                  "trajectory_shape": list(trajectory.shape), "trajectory_dtype": "<f8",
                  "counters": counters, "candidate_vs_dop853_nrmse": rms_nrmse(trajectory, oracle),
                  "candidate_vs_dop853_max_abs_error": max_error(trajectory, oracle)[0],
                  "candidate_vs_dop853_max_error_time_index": max_error(trajectory, oracle)[1],
                  "candidate_vs_dop853_max_error_time_seconds": float(times[max_error(trajectory, oracle)[1]]),
                  "candidate_vs_dop853_max_error_node": max_error(trajectory, oracle)[2],
                  "candidate_vs_rk4_coarse_nrmse": rms_nrmse(trajectory, rk4_coarse),
                  "candidate_vs_rk4_coarse_max_abs": max_error(trajectory, rk4_coarse)[0],
                  "candidate_vs_rk4_fine_nrmse": rms_nrmse(trajectory, rk4_fine),
                  "candidate_vs_rk4_fine_max_abs": max_error(trajectory, rk4_fine)[0],
                  "candidate_closer_to_rk4_coarse_than_dop853":
                      rms_nrmse(trajectory, rk4_coarse) < rms_nrmse(trajectory, oracle),
                  "candidate_closer_to_rk4_fine_than_dop853":
                      rms_nrmse(trajectory, rk4_fine) < rms_nrmse(trajectory, oracle),
                  "per_sample_nrmse": [rms_nrmse(trajectory[i], oracle[i]) for i in range(len(times))],
                  "scheduler_seconds": scheduler_elapsed, "metadata": meta,
                  "semantic_hash": None, "full_run_config_hash": None,
                  "normalized_config_hash": None, "component_hashes": None}
        (out / "result.json").write_bytes(canonical_json(result) + b"\n")
        queue.put({"status": "complete", "result": result})
    except BaseException as exc:
        failure = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                   "variant_id": config.get("id"), "stage": "candidate"}
        if "out" in locals() and out.exists():
            (out / "failure.json").write_bytes(canonical_json(failure) + b"\n")
        queue.put(failure)


def run_variant(config: dict[str, Any], output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    failure = _run_staged_process(_variant_worker,
                                  (config, str(output_dir), str(output_dir.parent / "reference"), False),
                                  {"candidate": timeout_seconds}, config["id"])
    if failure["status"] != "complete":
        failure.setdefault("variant_id", config["id"])
        failure.setdefault("traceback", failure.get("error", "worker failed without traceback"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure.json").write_bytes(canonical_json(failure) + b"\n")
    return failure


def _resolved_variants(contract: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    common = contract["diagnostic_matrix"]["candidate_common"]
    result = []
    for variant in contract["diagnostic_matrix"]["variants"]:
        item = {"variant_id": variant["id"], "id": variant["id"], "run_id": run_id,
                "repeat_of": variant.get("repeat_of"),
                "delivery_mode": variant.get("delivery_mode", "broadcast_batch"),
                "local_batch": variant.get("local_batch", True),
                "local_batch_size": variant.get("local_batch_size", 2),
                "local_rtol": variant.get("local_rtol", 1e-9),
                "local_atol": variant.get("local_atol", 1e-11),
                "segment_tolerance": variant.get("segment_tolerance", common["segment_tolerance"]),
                "max_interval_seconds": variant.get("max_interval_seconds", common["max_interval_seconds"]),
                "input_mode": common["input_mode"],
                "oracle_method": contract["diagnostic_matrix"]["oracle"]["method"],
                "oracle_rtol": contract["diagnostic_matrix"]["oracle"]["rtol"],
                "oracle_atol": contract["diagnostic_matrix"]["oracle"]["atol"],
                "duration_seconds": contract["instance"]["duration_seconds"],
                "sample_dt_seconds": contract["instance"]["sample_dt_seconds"],
                "population_size": contract["instance"]["population_size"],
                "graph": contract["instance"]["graph"], "task": contract["instance"]["task"],
                "seed": contract["instance"]["seed"], "initial_state": "all_zeros",
                "input_trace_sha256": contract["instance"]["expected_input_trace_sha256"],
                "array_dtype": "<f8", "cpu_only": True,
                "cpu_thread_environment": contract["environment"]["cpu_thread_environment"]}
        result.append(item)
    return result


def _failure_gate(result: dict[str, Any], contract: dict[str, Any]) -> bool:
    if result.get("status") != "complete":
        return False
    expected = contract["instance"]
    tolerance = expected["reproduction_tolerances"]
    counters = expected["expected_counters"]
    return (result.get("input_sha256") == expected["expected_input_trace_sha256"]
            and abs(result["candidate_vs_dop853_nrmse"] - expected["expected_candidate_nrmse"]) <= tolerance["nrmse_absolute"]
            and abs(result["candidate_vs_dop853_max_abs_error"] - expected["expected_candidate_max_abs_error"]) <= tolerance["max_abs_absolute"]
            and result["counters"] == counters)


def _validate_results_root(results_root: Path) -> Path:
    allowed = (REPO / "result/v0.3.6/diagnostics").resolve()
    actual = results_root.resolve()
    if actual != allowed:
        raise ValueError(f"results root must be the versioned v0.3.6 path: {allowed}")
    return allowed


def _validate_review_receipt(receipt: dict[str, Any]) -> dict[str, str]:
    reviewed = {
        "runner": sha256(Path(__file__).read_bytes()),
        "tests": sha256(Path(__file__).with_name("test_diagnostic_runner.py").read_bytes()),
        "contract": sha256(CONTRACT_PATH.read_bytes()),
        "protocol": sha256(PROTOCOL_PATH.read_bytes()),
    }
    if receipt.get("status") != "ACCEPT" or receipt.get("files_sha256") != reviewed:
        raise RuntimeError("review receipt does not ACCEPT the exact runner, tests, contract, and protocol")
    return reviewed


def _classify_final_status(reproduction_gate_passed: bool, failures: list[dict[str, Any]]) -> str:
    """A baseline timeout/error/mismatch blocks downstream interpretation."""
    if not reproduction_gate_passed:
        return "BLOCKED"
    return "COMPLETED_WITH_FAILURES" if failures else "COMPLETED_DIAGNOSTIC"


def _run_candidate_sequence(run_dir: Path, variants: list[dict[str, Any]],
                           contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reproduction_gate_passed = False
    timeout = contract["analysis"]["timeout_seconds"]["candidate_variant"]
    for index, variant in enumerate(variants):
        try:
            result = run_variant(variant, run_dir / variant["id"], timeout)
        except Exception as exc:
            result = {"variant_id": variant["id"], "status": "failed", "error": repr(exc),
                      "traceback": traceback.format_exc()}
        if result.get("status") != "complete":
            result.setdefault("variant_id", variant["id"])
            result.setdefault("traceback", result.get("error", "worker failed without traceback"))
            failures.append(result)
            rows.append({"variant_id": variant["id"], "status": result.get("status"),
                         "error": result.get("error", ""), "traceback": result.get("traceback", "")})
            if index < 2:
                break
            continue
        record = result["result"]
        rows.append({"variant_id": variant["id"], "status": "complete",
                     "candidate_vs_dop853_nrmse": record["candidate_vs_dop853_nrmse"],
                     "candidate_vs_dop853_max_abs_error": record["candidate_vs_dop853_max_abs_error"],
                     "candidate_vs_rk4_coarse_nrmse": record["candidate_vs_rk4_coarse_nrmse"],
                     "candidate_vs_rk4_coarse_max_abs": record["candidate_vs_rk4_coarse_max_abs"],
                     "candidate_vs_rk4_fine_nrmse": record["candidate_vs_rk4_fine_nrmse"],
                     "candidate_vs_rk4_fine_max_abs": record["candidate_vs_rk4_fine_max_abs"],
                     "scheduler_seconds": record["scheduler_seconds"],
                     "semantic_hash": record["semantic_hash"], **record["counters"]})
        if index == 0 and not _failure_gate(record, contract):
            failures.append({"variant_id": variant["id"], "status": "reproduction_gate_failed",
                             "result": record})
            break
        if index == 1:
            if (not _failure_gate(record, contract)
                    or not _repeats_equal(run_dir, variants[0]["id"], variants[1]["id"])):
                failures.append({"variant_id": variant["id"], "status": "reproduction_gate_failed",
                                 "result": record, "reason": "direct repeat output mismatch"})
                break
            reproduction_gate_passed = True
    return rows, failures, reproduction_gate_passed


def _write_output_manifest(run_dir: Path) -> str:
    entries = {}
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()
                       and p.name not in {"output_hash_manifest.json", "status.json"}):
        entries[str(path.relative_to(run_dir))] = sha256(path.read_bytes())
    payload = {"algorithm": "SHA-256", "excluded_self_and_mutable_status": True, "files": entries}
    data = canonical_json(payload) + b"\n"
    (run_dir / "output_hash_manifest.json").write_bytes(data)
    return sha256(data)


def _postrun_hash_pass(run_dir: Path, variants: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    """Hash experiment artifacts only after all diagnostic variants have stopped."""
    instance = contract["instance"]
    input_trace = np.load(run_dir / "reference/input_trace.npy", allow_pickle=False)
    input_hash = sha256(np.ascontiguousarray(input_trace, dtype="<f8").tobytes(order="C"))
    instance_hash = sha256(canonical_json({"split": instance["split"], "graph": instance["graph"],
                                           "n": instance["population_size"], "task": instance["task"],
                                           "seed": instance["seed"], "input_trace_sha256": input_hash}))
    expected = instance["expected_input_trace_sha256"]
    if input_hash != expected:
        raise RuntimeError(f"post-run input trace hash mismatch: {input_hash}")
    for variant in variants:
        out = run_dir / variant["id"]
        result_path = out / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trajectory = np.load(out / "trajectory.npy", allow_pickle=False)
        event_bytes = (out / "events.jsonl").read_bytes()
        semantic, full_config, components = _semantic_hash(
            trajectory, event_bytes, result["counters"], result["config"],
            contract["analysis"]["outputs"]["semantic_hash"]["schema_id"])
        result.update({"input_sha256": input_hash, "instance_sha256": instance_hash,
                       "semantic_hash": semantic, "full_run_config_hash": full_config,
                       "normalized_config_hash": components["normalized_config"],
                       "component_hashes": components})
        result_path.write_bytes(canonical_json(result) + b"\n")
    variants_csv = run_dir / "variants.csv"
    if variants_csv.is_file():
        with variants_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            result_path = run_dir / row["variant_id"] / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                row["semantic_hash"] = result["semantic_hash"]
                row["input_sha256"] = result["input_sha256"]
                row["instance_sha256"] = result["instance_sha256"]
        fields = sorted({key for row in rows for key in row})
        with variants_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary_path = run_dir / "oracle_validation.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["input_trace_sha256"] = input_hash
        summary["instance_sha256"] = instance_hash
        summary_path.write_bytes(canonical_json(summary) + b"\n")
    analysis_path = run_dir / "analysis.json"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        oracle_summary = analysis.get("oracle_validation", {})
        oracle_summary.update({"input_trace_sha256": input_hash, "instance_sha256": instance_hash})
        analysis["oracle_validation"] = oracle_summary
        analysis_path.write_bytes(canonical_json(analysis) + b"\n")


def execute(run_id: str, review_receipt: Path, results_root: Path) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", run_id):
        raise ValueError("run_id must be a safe 1–81 character identifier")
    safe_root = _validate_results_root(results_root)
    run_dir = safe_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_bytes(b"")
    (run_dir / "command.log").write_bytes(canonical_json({
        "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
        "started_utc": started_utc}) + b"\n")
    try:
        contract = verify_frozen_inputs()
        if not review_receipt.is_file():
            raise RuntimeError("official run requires independent code-review ACCEPT receipt")
        receipt = json.loads(review_receipt.read_text(encoding="utf-8"))
        reviewed = _validate_review_receipt(receipt)
        runtime = contract["environment"]
        if (platform.machine() != runtime["architecture"] or sys.version.split()[0] != runtime["python"]
                or np.__version__ != runtime["numpy"] or scipy.__version__ != runtime["scipy"]
                or platform.platform() != runtime["platform"]):
            raise RuntimeError("runtime differs from preregistered environment")
        lock = REPO / runtime["dependency_lock"]
        if sha256(lock.read_bytes()) != runtime["dependency_lock_sha256"]:
            raise RuntimeError("dependency lock hash differs from preregistered environment")
        for name, expected in runtime["cpu_thread_environment"].items():
            if os.environ.get(name) != expected:
                raise RuntimeError(f"CPU thread setting mismatch: {name}")
        # Validate the actual generated input trace before any reference or
        # candidate workload starts. Workers then reuse the frozen fingerprint.
        _load_parent_case(validate_input_fingerprint=True)
    except Exception as exc:
        failure = {"status": "BLOCKED", "stage": "preflight", "error": repr(exc),
                   "traceback": traceback.format_exc()}
        (run_dir / "failure.json").write_bytes(canonical_json(failure) + b"\n")
        manifest_hash = _write_output_manifest(run_dir)
        (run_dir / "status.json").write_bytes(canonical_json({
            "status": "BLOCKED", "started_utc": started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(), "reason": repr(exc),
            "output_hash_manifest_sha256": manifest_hash}) + b"\n")
        return 1
    shutil.copy2(CONTRACT_PATH, run_dir / "pre_registration.yaml")
    shutil.copy2(PROTOCOL_PATH, run_dir / "DIAGNOSTIC_PROTOCOL_v1.md")
    shutil.copy2(PARENT_MANIFEST, run_dir / "parent_source_manifest.csv")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                check=True, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                check=True, capture_output=True, text=True).stdout
    tracked_sources = [CONTRACT_PATH, PROTOCOL_PATH, Path(__file__), Path(__file__).with_name("test_diagnostic_runner.py")]
    (run_dir / "run_metadata.json").write_bytes(canonical_json({
        "run_id": run_id, "started_utc": started_utc,
        "git_commit": git_commit, "git_dirty": bool(git_status), "git_status_porcelain": git_status,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "architecture": platform.machine(),
        "dependency_lock": runtime["dependency_lock"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        "cpu_thread_environment": runtime["cpu_thread_environment"],
        "contract_sha256": sha256(CONTRACT_PATH.read_bytes()),
        "protocol_sha256": sha256(PROTOCOL_PATH.read_bytes()),
        "parent_manifest_sha256": sha256(PARENT_MANIFEST.read_bytes()),
        "implementation_sha256": {str(path.relative_to(REPO)): sha256(path.read_bytes()) for path in tracked_sources},
        "review_receipt_sha256": sha256(review_receipt.read_bytes())}) + b"\n")
    reference_dir = run_dir / "reference"
    reference_result = _run_staged_process(
        _reference_worker, (str(reference_dir), False),
        {"dop853_oracle": contract["analysis"]["timeout_seconds"]["dop853_oracle"],
         "rk4_coarse": contract["analysis"]["timeout_seconds"]["each_rk4_refinement"],
         "rk4_fine": contract["analysis"]["timeout_seconds"]["each_rk4_refinement"]},
        "independent_oracle")
    if reference_result.get("status") != "complete":
        reference_result.setdefault("variant_id", "independent_oracle")
        reference_dir.mkdir(parents=True, exist_ok=True)
        (reference_dir / "failure.json").write_bytes(canonical_json(reference_result) + b"\n")
        (run_dir / "failures.jsonl").write_bytes(canonical_json({"stage": "oracle_validation", **reference_result}) + b"\n")
        _finalize_run(run_dir, "BLOCKED", started_utc, "independent reference failed")
        return 1
    (run_dir / "oracle_validation.json").write_bytes(canonical_json(reference_result["summary"]) + b"\n")
    variants = _resolved_variants(contract, run_id)
    rows, failures, reproduction_gate_passed = _run_candidate_sequence(run_dir, variants, contract)
    if reproduction_gate_passed:
        _write_contrast_analysis(run_dir, variants, rows, contract, reference_result["summary"])
    (run_dir / "failures.jsonl").write_bytes(b"".join(canonical_json(x) + b"\n" for x in failures))
    with (run_dir / "variants.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    try:
        _postrun_hash_pass(run_dir, variants, contract)
    except Exception as exc:
        failure = {"stage": "postrun_hash_archive", "status": "failed", "error": repr(exc),
                   "traceback": traceback.format_exc()}
        failures.append(failure)
        with (run_dir / "failures.jsonl").open("ab") as f:
            f.write(canonical_json(failure) + b"\n")
    final_status = _classify_final_status(reproduction_gate_passed, failures)
    _finalize_run(run_dir, final_status, started_utc,
                  "diagnostic complete; no efficacy inference" if reproduction_gate_passed else "G0 reproduction gate did not pass")
    return 1 if failures else 0


def _finalize_run(run_dir: Path, status: str, started_utc: str, summary: str) -> None:
    run_meta_path = run_dir / "run_metadata.json"
    if run_meta_path.exists():
        meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta_path.write_bytes(canonical_json(meta) + b"\n")
    manifest_hash = _write_output_manifest(run_dir)
    (run_dir / "status.json").write_bytes(canonical_json({
        "status": status, "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "output_hash_manifest_sha256": manifest_hash}) + b"\n")


def _write_contrast_analysis(run_dir: Path, variants: list[dict[str, Any]],
                             rows: list[dict[str, Any]], contract: dict[str, Any],
                             oracle_check: dict[str, Any]) -> None:
    by_id = {row["variant_id"]: row for row in rows}
    trajectory_cache: dict[str, np.ndarray] = {}
    result_cache: dict[str, dict[str, Any]] = {}
    for variant in variants:
        vid = variant["id"]
        result_path = run_dir / vid / "result.json"
        if result_path.is_file():
            result_cache[vid] = json.loads(result_path.read_text(encoding="utf-8"))
            trajectory_cache[vid] = np.load(run_dir / vid / "trajectory.npy", allow_pickle=False)
    contrasts = []
    config_by_id = {variant["id"]: variant for variant in variants}
    for first, second in contract["analysis"]["preregistered_contrasts"]:
        if first not in trajectory_cache or second not in trajectory_cache:
            contrasts.append({"first": first, "second": second, "status": "incomplete"})
            continue
        a, b = trajectory_cache[first], trajectory_cache[second]
        d = result_cache[first]
        e = result_cache[second]
        pairwise = {"nrmse_a_as_candidate": rms_nrmse(a, b),
                    "nrmse_b_as_candidate": rms_nrmse(b, a),
                    "max_abs": max_error(a, b)[0],
                    "signed_oracle_nrmse_difference": e["candidate_vs_dop853_nrmse"] - d["candidate_vs_dop853_nrmse"],
                    "signed_oracle_max_abs_difference": e["candidate_vs_dop853_max_abs_error"] - d["candidate_vs_dop853_max_abs_error"]}
        pairwise["equivalence_type"] = ("delivery_mode"
                                          if config_by_id[first]["delivery_mode"] != config_by_id[second]["delivery_mode"]
                                          else "scalar_batch")
        pairwise["equivalence_pass"] = (pairwise["max_abs"] <= contract["analysis"]["mode_equivalence_limit_max_abs"]
                                         and pairwise["nrmse_a_as_candidate"] <= contract["analysis"]["mode_equivalence_limit_nrmse"]
                                         and pairwise["nrmse_b_as_candidate"] <= contract["analysis"]["mode_equivalence_limit_nrmse"])
        contrasts.append({"first": first, "second": second, "status": "complete", **pairwise})
    by_name = {v["id"]: result_cache.get(v["id"]) for v in variants}
    interaction = None
    required = {"edge_delivery_batch2", "frozen_baseline_repeat_1",
                "broadcast_delivery_scalar", "edge_delivery_scalar"}
    if required.issubset(by_name) and all(by_name[k] is not None for k in required):
        b2, e2 = by_name["frozen_baseline_repeat_1"], by_name["edge_delivery_batch2"]
        bs, es = by_name["broadcast_delivery_scalar"], by_name["edge_delivery_scalar"]
        interaction = {
            "oracle_nrmse_difference_in_differences":
                (e2["candidate_vs_dop853_nrmse"] - b2["candidate_vs_dop853_nrmse"])
                - (es["candidate_vs_dop853_nrmse"] - bs["candidate_vs_dop853_nrmse"]),
            "oracle_max_abs_difference_in_differences":
                (e2["candidate_vs_dop853_max_abs_error"] - b2["candidate_vs_dop853_max_abs_error"])
                - (es["candidate_vs_dop853_max_abs_error"] - bs["candidate_vs_dop853_max_abs_error"])}
    base_nrmse = result_cache.get("frozen_baseline_repeat_1", {}).get("candidate_vs_dop853_nrmse")
    tighter_local = result_cache.get("tighter_local_tolerance", {}).get("candidate_vs_dop853_nrmse")
    tighter_segment = result_cache.get("tighter_segment_tolerance", {}).get("candidate_vs_dop853_nrmse")
    expected_variant_ids = {v["id"] for v in variants}
    analysis = {"status": "complete" if expected_variant_ids.issubset(result_cache)
                and len(contrasts) == len(contract["analysis"]["preregistered_contrasts"])
                and all(item["status"] == "complete" for item in contrasts) else "partial",
                "oracle_validation": oracle_check,
                "contrasts": contrasts, "delivery_batch_interaction": interaction,
                "tolerance_sensitivity": {
                    "baseline_nrmse": base_nrmse,
                    "baseline_max_abs": result_cache.get("frozen_baseline_repeat_1", {}).get("candidate_vs_dop853_max_abs_error"),
                    "tighter_local_nrmse": tighter_local,
                    "tighter_local_max_abs": result_cache.get("tighter_local_tolerance", {}).get("candidate_vs_dop853_max_abs_error"),
                    "tighter_local_reduction_fold": (base_nrmse / tighter_local if base_nrmse is not None and tighter_local not in (None, 0) else None),
                    "tighter_local_10x_criterion_met": (base_nrmse / tighter_local >= contract["analysis"]["tighter_local_tolerance_error_reduction_minimum_fold"] if base_nrmse is not None and tighter_local not in (None, 0) else None),
                    "tighter_segment_nrmse": tighter_segment,
                    "tighter_segment_max_abs": result_cache.get("tighter_segment_tolerance", {}).get("candidate_vs_dop853_max_abs_error")}}
    (run_dir / "analysis.json").write_bytes(canonical_json(analysis) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--results-root", type=Path, default=REPO / "result/v0.3.6/diagnostics")
    args = parser.parse_args()
    if args.validate_only:
        contract = verify_frozen_inputs()
        print(json.dumps({"contract": contract["contract_id"], "source_hashes": "verified",
                          "parent_manifest": "96/96 verified"}, sort_keys=True))
        return
    if not args.run_id or not args.review_receipt:
        parser.error("official run requires --run-id and --review-receipt")
    raise SystemExit(execute(args.run_id, args.review_receipt, args.results_root))


if __name__ == "__main__":
    main()
