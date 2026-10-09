"""Registered, single-instance audit of the v0.3.6 RK4 terminal boundary."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
import yaml

REPO = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO / "model/v0.3.7/RK4_ENDPOINT_AUDIT_CONTRACT_v1.yaml"
PROTOCOL_PATH = REPO / "model/v0.3.7/RK4_ENDPOINT_AUDIT_PROTOCOL_v1.md"
TEST_PATH = REPO / "model/v0.3.7/test_audit.py"
PARENT = REPO / "result/v0.3.6/diagnostics/wp2-diagnostic-20261009-01"
OUT = REPO / "result/v0.3.7/rk4-endpoint-audit-20261009-01"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def terminal_jump(x, end: float, epsilon: float) -> bool:
    before = np.asarray(x(float(end - epsilon)), dtype=np.float64)
    earlier = np.asarray(x(float(end - 2 * epsilon)), dtype=np.float64)
    at = np.asarray(x(float(end)), dtype=np.float64)
    if before.shape != at.shape or earlier.shape != at.shape:
        raise ValueError("input shape changes at terminal boundary")
    jump_delta = np.abs(at - before)
    smooth_delta = np.abs(before - earlier)
    scale = np.maximum.reduce((np.abs(at), np.abs(before), np.abs(earlier),
                               np.ones_like(at)))
    numerical_floor = 64 * np.finfo(np.float64).eps * scale
    return bool(np.any(jump_delta > np.maximum(8.0 * smooth_delta, numerical_floor)))


def frozen_breakpoints(x, start: float, end: float, epsilon: float) -> tuple[list[float], bool]:
    interior = [float(q) for q in x.breakpoints(start, end) if start < float(q) < end]
    has_terminal_jump = terminal_jump(x, end, epsilon)
    points = sorted(set(interior + ([float(end)] if has_terminal_jump else [])))
    return points, has_terminal_jump


def corrected_rk4(system, x, times: np.ndarray, max_step: float,
                  breakpoints: list[float], terminal_has_jump: bool,
                  left_limit_epsilon: float) -> np.ndarray:
    """Independent full-state RK4, with a left-limit final stage at terminal jumps."""
    times = np.asarray(times, dtype=np.float64)
    start, end_time = float(times[0]), float(times[-1])
    boundaries = sorted(set([start, end_time, *map(float, times), *breakpoints]))
    state = np.zeros(system.W.shape[0], dtype=np.float64)
    result = np.empty((len(times), len(state)), dtype=np.float64)
    sample_index = {float(t): i for i, t in enumerate(times)}
    result[sample_index[start]] = state

    def derivative(t: float, h: np.ndarray, endpoint: float, left_end: bool) -> np.ndarray:
        eval_t = endpoint - left_limit_epsilon if left_end and t == endpoint else t
        external = np.asarray(x(float(eval_t)), dtype=np.float64)
        return (-h + np.tanh(system.W @ h + system.U @ external + system.b)) / system.tau

    current = start
    for boundary in boundaries[1:]:
        while current < boundary:
            step = min(max_step, boundary - current)
            next_t = current + step
            if boundary - next_t <= 8 * np.finfo(float).eps * max(1.0, abs(boundary)):
                next_t = boundary
                step = next_t - current
            left_endpoint = ((terminal_has_jump and boundary == end_time)
                             or (boundary in breakpoints))
            k1 = derivative(current, state, boundary, False)
            k2 = derivative(current + step / 2, state + step * k1 / 2, boundary, False)
            k3 = derivative(current + step / 2, state + step * k2 / 2, boundary, False)
            k4 = derivative(next_t, state + step * k3, boundary,
                            left_endpoint and next_t == boundary)
            state = state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            current = next_t
        if boundary in sample_index:
            result[sample_index[boundary]] = state
    return result


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=np.float64) ** 2))), 1e-12)
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)) / scale)


def error_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if np.shape(a) != np.shape(b) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("trajectory arrays must have matching shapes and finite values")
    return {"nrmse_a_as_candidate": nrmse(a, b),
            "nrmse_b_as_candidate": nrmse(b, a),
            "max_abs": float(np.max(np.abs(np.asarray(a) - np.asarray(b))))}


def load_frozen_case(contract: dict[str, Any]):
    source_root = REPO / "model/v0.3.5/src"
    sys.path.insert(0, str(source_root))
    from dynamics.tasks import make_task_system
    from events.inputs import make_input

    spec = contract["parent_run"]["frozen_instance"]
    x, driven = make_input(spec["task"], spec["population_size"], spec["seed"])
    system = make_task_system(spec["graph"], spec["population_size"], spec["seed"], driven)
    times = np.linspace(0.0, spec["duration_seconds"],
                        round(spec["duration_seconds"] / spec["sample_dt_seconds"]) + 1)
    trace = np.asarray([x(float(t)) for t in times], dtype="<f8")
    saved_trace = np.load(PARENT / "reference/input_trace.npy", allow_pickle=False)
    if not np.array_equal(trace, saved_trace):
        raise RuntimeError("reconstructed input trace differs from frozen parent trace")
    return x, system, times, trace, np.asarray(driven, dtype=int)


def check_terminal_locality(old: np.ndarray, new: np.ndarray, driven_nodes: np.ndarray,
                            tolerance: float) -> dict[str, Any]:
    if old.shape != new.shape or old.ndim != 2 or not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("old and corrected references must be finite, matching 2D arrays")
    delta = np.abs(new - old)
    changed = set(np.flatnonzero(delta[-1] > 0).astype(int).tolist())
    allowed = set(np.asarray(driven_nodes, dtype=int).tolist())
    unexpected = sorted(changed - allowed)
    return {"max_all": float(np.max(delta)),
            "max_nonterminal": float(np.max(delta[:-1])),
            "changed_terminal_nodes": sorted(changed),
            "unexpected_terminal_nodes": unexpected,
            "only_driven_terminal_nodes_changed": not unexpected,
            "nonterminal_within_limit": float(np.max(delta[:-1])) <= tolerance}


REVIEWED_FILES = {
    "model/v0.3.7/RK4_ENDPOINT_AUDIT_CONTRACT_v1.yaml",
    "model/v0.3.7/RK4_ENDPOINT_AUDIT_PROTOCOL_v1.md",
    "model/v0.3.7/audit_runner.py",
    "model/v0.3.7/test_audit.py",
    "model/v0.3.7/archive_after_completion.py",
}


def validate_review_receipt(receipt: dict[str, Any], current_commit: str,
                            working_tree_clean: bool) -> None:
    if receipt.get("status") != "ACCEPT" or not receipt.get("reviewer"):
        raise RuntimeError("independent code review receipt must be ACCEPT and identify reviewer")
    if receipt.get("reviewed_commit") != current_commit:
        raise RuntimeError("review receipt commit does not match executed commit")
    if set(receipt.get("reviewed_files", [])) != REVIEWED_FILES:
        raise RuntimeError("review receipt file scope does not match frozen review scope")
    if not working_tree_clean:
        raise RuntimeError("working tree must be clean after reviewed commit")


def write_failed_status(out: Path, error: BaseException, trace: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    failure = {"status": "FAILED_PRESERVED", "stage": "audit_execution",
               "error": repr(error), "traceback": trace}
    (out / "failure.json").write_bytes(canonical_json(failure) + b"\n")
    temporary = out / "status.json.tmp"
    temporary.write_bytes(canonical_json({"status": "FAILED_PRESERVED",
        "hashing": "deferred_until_post-report_archive",
        "partial_outputs_preserved": True}) + b"\n")
    temporary.replace(out / "status.json")


def _run_impl(audit_id: str, receipt_path: Path) -> int:
    if audit_id != "rk4-endpoint-audit-20261009-01":
        raise ValueError("audit_id must match the frozen contract")
    out = OUT
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        # Integrity hashing is intentionally deferred until all numerical work
        # and reports have been written, per the current project instruction.
        if not receipt_path.is_file():
            raise RuntimeError("independent code review receipt is required")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                        check=True, capture_output=True, text=True).stdout.strip()
        tree_clean = not subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=REPO, check=True, capture_output=True,
                                        text=True).stdout.strip()
        validate_review_receipt(receipt, current_commit, tree_clean)
        parent_meta = json.loads((PARENT / "run_metadata.json").read_text(encoding="utf-8"))
        if (sys.version.split()[0] != parent_meta["python"].split()[0]
                or np.__version__ != parent_meta["numpy"] or scipy.__version__ != parent_meta["scipy"]
                or platform.platform() != parent_meta["platform"]
                or platform.machine() != parent_meta["architecture"]):
            raise RuntimeError("runtime differs from frozen parent environment")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            if os.environ.get(name) != "1":
                raise RuntimeError(f"thread environment mismatch: {name}")
        x, system, times, trace, driven_nodes = load_frozen_case(contract)
    except Exception as exc:
        (out / "failure.json").write_bytes(canonical_json({"status": "BLOCKED", "error": repr(exc),
            "traceback": traceback.format_exc()}) + b"\n")
        (out / "status.json").write_bytes(canonical_json({"status": "BLOCKED", "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "hashing": "deferred_until_post-report_archive"}) + b"\n")
        return 1

    import shutil
    shutil.copy2(CONTRACT_PATH, out / "contract.yaml")
    shutil.copy2(PROTOCOL_PATH, out / "protocol.md")
    shutil.copy2(receipt_path, out / "review_receipt.json")
    shutil.copy2(PARENT / "pre_registration.yaml", out / "parent_contract.yaml")
    shutil.copy2(PARENT / "run_metadata.json", out / "parent_run_metadata.json")
    shutil.copy2(PARENT / "output_hash_manifest.json", out / "parent_output_hash_manifest.json")
    shutil.copy2(PARENT / "status.json", out / "parent_status.json")
    np.save(out / "input_trace.npy", trace, allow_pickle=False)
    epsilon = 1e-9
    breakpoints, terminal_has_jump = frozen_breakpoints(x, float(times[0]), float(times[-1]), epsilon)
    configs = (("coarse", 0.001), ("fine", 0.0005))
    corrected = {}
    timings = {}
    for name, step in configs:
        began = time.perf_counter()
        corrected[name] = corrected_rk4(system, x, times, step, breakpoints,
                                        terminal_has_jump, epsilon)
        timings[name] = time.perf_counter() - began
        np.save(out / f"corrected_rk4_{name}.npy", corrected[name], allow_pickle=False)
    dop = np.load(PARENT / "reference/oracle_dop853.npy", allow_pickle=False)
    old = {"coarse": np.load(PARENT / "reference/oracle_rk4_001.npy", allow_pickle=False),
           "fine": np.load(PARENT / "reference/oracle_rk4_0005.npy", allow_pickle=False)}
    criteria = contract["criteria"]
    refinement = error_metrics(corrected["coarse"], corrected["fine"])
    vs_dop = {name: error_metrics(value, dop) for name, value in corrected.items()}
    locality = {}
    for name in corrected:
        locality[name] = check_terminal_locality(old[name], corrected[name], driven_nodes,
            criteria["terminal_correction_locality"]["nonterminal_max_abs"])
    candidate_ids = contract["method"]["candidate_evaluation"]["variants"]
    candidate_rows = []
    for variant in candidate_ids:
        trajectory = np.load(PARENT / variant / "trajectory.npy", allow_pickle=False)
        candidate_rows.append({"variant_id": variant,
                               "vs_corrected_rk4_coarse": error_metrics(trajectory, corrected["coarse"]),
                               "vs_corrected_rk4_fine": error_metrics(trajectory, corrected["fine"])})
    refine_limit = criteria["rk4_refinement"]
    dop_limit = criteria["each_corrected_rk4_vs_dop853"]
    refinement_pass = (refinement["max_abs"] <= refine_limit["max_abs"]
                       and refinement["nrmse_a_as_candidate"] <= refine_limit["both_direction_nrmse"]
                       and refinement["nrmse_b_as_candidate"] <= refine_limit["both_direction_nrmse"])
    dop_pass = all(v["max_abs"] <= dop_limit["max_abs"]
                   and v["nrmse_a_as_candidate"] <= dop_limit["both_direction_nrmse"]
                   and v["nrmse_b_as_candidate"] <= dop_limit["both_direction_nrmse"]
                   for v in vs_dop.values())
    locality_pass = all(v["max_nonterminal"] <= criteria["terminal_correction_locality"]["nonterminal_max_abs"]
                        and v["only_driven_terminal_nodes_changed"]
                        for v in locality.values())
    result = {"status": "complete", "parent_verification": "PENDING_POST_REPORT_ARCHIVE",
              "parent_integrity_verified_post_calculation": False,
              "terminal_jump_detected": terminal_has_jump, "terminal_breakpoints": breakpoints,
              "terminal_left_limit_epsilon_seconds": epsilon,
              "rk4_timings_seconds": timings, "corrected_refinement": refinement,
              "corrected_rk4_vs_dop853": vs_dop, "terminal_correction_locality": locality,
              "frozen_driven_node_ids": driven_nodes.tolist(),
              "corrected_reference_checks": {"rk4_refinements_converged": refinement_pass,
                                             "both_corrected_rk4_agree_with_dop853": dop_pass,
                                             "correction_locality_passed": locality_pass},
              "candidate_trajectory_reanalysis": candidate_rows,
              "candidate_solver_reruns": 0,
              "parent_wp2_status": "INCONCLUSIVE", "wp3_wp6": "BLOCKED"}
    (out / "audit_results.json").write_bytes(canonical_json(result) + b"\n")
    with (out / "candidate_reference_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant_id", "coarse_nrmse", "coarse_max_abs",
                                               "fine_nrmse", "fine_max_abs"])
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow({"variant_id": row["variant_id"],
                "coarse_nrmse": row["vs_corrected_rk4_coarse"]["nrmse_a_as_candidate"],
                "coarse_max_abs": row["vs_corrected_rk4_coarse"]["max_abs"],
                "fine_nrmse": row["vs_corrected_rk4_fine"]["nrmse_a_as_candidate"],
                "fine_max_abs": row["vs_corrected_rk4_fine"]["max_abs"]})
    # Parent and implementation integrity are verified by the post-report
    # archive command, after the experiment and report are complete.
    result["parent_verification"] = "PENDING_POST_REPORT_ARCHIVE"
    result["parent_integrity_verified_post_calculation"] = False
    (out / "audit_results.json").write_bytes(canonical_json(result) + b"\n")
    metadata = {"audit_id": audit_id, "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                                     capture_output=True, text=True).stdout.strip(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "architecture": platform.machine(),
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "parent_manifest_sha256_expected": contract["parent_run"]["output_manifest_sha256"],
        "review_receipt": str(receipt_path),
        "candidate_solver_reruns": 0}
    (out / "run_metadata.json").write_bytes(canonical_json(metadata) + b"\n")
    audit_status = "COMPLETED_AUDIT" if refinement_pass and dop_pass and locality_pass else "COMPLETED_WITH_UNRESOLVED_CHECKS"
    (out / "status.json").write_bytes(canonical_json({"status": audit_status, "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "hashing": "deferred_until_post-report_archive"}) + b"\n")
    return 0


def run(audit_id: str, receipt_path: Path) -> int:
    if OUT.exists():
        raise RuntimeError(f"output path already exists; preserving prior run: {OUT}")
    try:
        return _run_impl(audit_id, receipt_path)
    except Exception as exc:
        write_failed_status(OUT, exc, traceback.format_exc())
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit-id")
    parser.add_argument("--review-receipt", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        print(json.dumps({"contract": contract["contract_id"],
                          "status": "SYNTAX_AND_SCHEMA_VALID; hashes deferred until audit completion"}, sort_keys=True))
        return
    if not args.audit_id or not args.review_receipt:
        parser.error("--audit-id and --review-receipt are required")
    raise SystemExit(run(args.audit_id, args.review_receipt))


if __name__ == "__main__":
    main()
