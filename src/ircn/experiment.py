"""Immutable E0/E1 execution, raw artifact capture, and gate calculation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from .core import integrate_adaptive, integrate_events, integrate_reference, integrate_rk4, make_circuit, nrmse, timed_call

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pre_registration_v1.yaml"
REPORTS = ROOT / "reports"
E0_REPORT = ROOT / "E0_report.md"
E1_REPORT = ROOT / "E1_report.md"
E1_RESULTS = ROOT / "E1_results.csv"
DECISION_REPORT = ROOT / "decision_v1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    result = {}
    paths = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py"), ROOT / "pyproject.toml", ROOT / "requirements-lock.txt"]
    for path in sorted(paths):
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def input_hash(circuit) -> str:
    digest = hashlib.sha256()
    for array in (circuit.w, circuit.tau, circuit.gain, circuit.workload.driven, circuit.workload.phases):
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    digest.update(json.dumps({"seed": circuit.seed, "kind": circuit.workload.kind, "pulse_times": circuit.workload.pulse_times}, sort_keys=True).encode())
    return digest.hexdigest()


def environment() -> dict[str, object]:
    import scipy
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


def _append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_e0(run_dir: Path | None = None) -> bool:
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir or REPORTS / f"e0_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_bytes(CONFIG.read_bytes())
    metadata = {"environment": environment(), "config_sha256": sha256_file(CONFIG), "pre_registration_sha256": sha256_file(ROOT / "pre_registration_v1.yaml"), "source_sha256": source_hashes()}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    test_env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    process = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=test_env, text=True, capture_output=True)
    (run_dir / "pytest.stdout.log").write_text(process.stdout, encoding="utf-8")
    (run_dir / "pytest.stderr.log").write_text(process.stderr, encoding="utf-8")
    passed = process.returncode == 0
    report = """# E0 correctness report\n\n""" + f"Status: **{'PASS' if passed else 'FAIL'}**\n\n" + "E0 executes the frozen T0–T6 automation suite. A failure blocks E1.\n\n```text\n" + (process.stdout[-6000:] or process.stderr[-6000:]) + "\n```\n"
    E0_REPORT.write_text(report, encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"status": "PASS" if passed else "FAIL", "returncode": process.returncode}, indent=2), encoding="utf-8")
    return passed


def _call(method: str, circuit, threshold: float):
    if method == "B0" or method == "B3":
        values, counts = integrate_rk4(circuit)
        return values, {"rhs_evaluations": counts.get("rhs_evaluations", 0), "queue_operations": 0, "node_refinements": 0, "fallbacks": 0}, []
    if method == "B1":
        values, counts = integrate_adaptive(circuit)
        return values, {"rhs_evaluations": counts.get("rhs_evaluations", 0), "queue_operations": 0, "node_refinements": 0, "fallbacks": 0}, []
    if method == "B2":
        values, counts, events = integrate_events(circuit, threshold=threshold, mode="input_event", log_events=True)
        return values, counts, events
    if method == "P":
        values, counts, events = integrate_events(circuit, threshold=threshold, mode="ircn", log_events=True)
        return values, counts, events
    raise ValueError(method)


def calibrate_threshold(run_dir: Path) -> dict[str, float | None]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seed = int(config["system"]["seeds"]["threshold_calibration"])
    candidate_values = sorted(config["system"]["threshold_calibration"]["candidates"], reverse=True)
    calibration_rows = []
    selected: dict[str, float | None] = {}
    for method, mode in config["system"]["threshold_calibration"]["modes"].items():
        selected[method] = None
        for threshold in candidate_values:
            max_error = 0.0
            for n in (64, 128, 256):
                for workload in ("sparse_pulses", "smooth_periodic"):
                    circuit = make_circuit(n, seed, workload)
                    reference = integrate_reference(circuit)
                    actual, counts, _events = integrate_events(circuit, threshold=threshold, mode=mode)
                    error = nrmse(actual, reference)
                    max_error = max(max_error, error)
                    calibration_rows.append({"method": method, "threshold": threshold, "n": n, "workload": workload, "input_sha256": input_hash(circuit), "nrmse": error, "node_refinements": counts["node_refinements"]})
                if max_error > 1e-3:
                    break
            _append_jsonl(run_dir / "threshold_calibration.jsonl", calibration_rows[-1:])
            if max_error <= 1e-3:
                selected[method] = threshold
                break
    (run_dir / "threshold_calibration_all.json").write_text(json.dumps(calibration_rows, indent=2), encoding="utf-8")
    (run_dir / "selected_threshold.json").write_text(json.dumps({"thresholds": selected, "seed": seed, "frozen_before_evaluation": True}, indent=2), encoding="utf-8")
    return selected


def _rss_bytes(raw: int) -> int:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    return raw if platform.system() == "Darwin" else raw * 1024


def run_e1() -> bool:
    REPORTS.mkdir(parents=True, exist_ok=True)
    e0 = E0_REPORT
    if not e0.exists() or "Status: **PASS**" not in e0.read_text(encoding="utf-8"):
        raise RuntimeError("E1 blocked: a passing reports/E0_report.md is required")
    run_dir = REPORTS / f"e1_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    (run_dir / "config.yaml").write_bytes(CONFIG.read_bytes())
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    metadata = {"environment": environment(), "config_sha256": sha256_file(CONFIG), "pre_registration_sha256": sha256_file(ROOT / "pre_registration_v1.yaml"), "source_sha256": source_hashes(), "e0_report_sha256": sha256_file(e0), "git_head": git_head}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    thresholds = calibrate_threshold(run_dir)
    threshold = thresholds.get("P")
    if threshold is None:
        (run_dir / "status.json").write_text(json.dumps({"status": "G1_NOT_RUN", "reason": "no frozen calibration threshold met the accuracy limit"}, indent=2), encoding="utf-8")
        E1_REPORT.write_text("# E1 report\n\nE1 evaluation was not run: no pre-registered IRCN threshold met the calibration accuracy limit. This is a negative feasibility result, not a performance comparison.\n", encoding="utf-8")
        _ensure_empty_results_csv()
        _write_decision("STOP", "No IRCN candidate threshold met the frozen NRMSE calibration limit; E1 was not opened and E2 is not recommended on this line.")
        return False
    (REPORTS / "calibrated_threshold.json").write_text(json.dumps({"thresholds": thresholds, "calibration_run": run_dir.name}, indent=2), encoding="utf-8")
    seed_list = config["system"]["seeds"]["evaluation"]
    methods = ["B0", "B1", "B2", "B3", "P"]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    event_path = run_dir / "events.jsonl"
    for n in (64, 128, 256):
        for workload in ("sparse_pulses", "smooth_periodic"):
            for seed in seed_list:
                circuit = make_circuit(n, int(seed), workload)
                reference = integrate_reference(circuit)
                for method in methods:
                    method_threshold = thresholds.get(method, threshold)
                    if method in {"B2", "P"} and method_threshold is None:
                        failure = {"n": n, "workload": workload, "seed": seed, "method": method, "reason": "no calibration threshold met NRMSE limit; method not evaluated"}
                        failures.append(failure)
                        _append_jsonl(run_dir / "failures.jsonl", [failure])
                        continue
                    for repetition in range(-3, 10):
                        try:
                            (result, counts, events), elapsed, rss = timed_call(lambda: _call(method, circuit, float(method_threshold or threshold)))
                            if repetition >= 0:
                                err = nrmse(result, reference)
                                row = {"n": n, "workload": workload, "seed": seed, "input_sha256": input_hash(circuit), "method": method, "repetition": repetition, "warmup": False, "wall_seconds": elapsed, "process_highwater_rss_bytes": _rss_bytes(rss), "nrmse": err, "max_abs_error": float(np.max(np.abs(result - reference))), **counts}
                                rows.append(row)
                            if method in {"B2", "P"}:
                                _append_jsonl(event_path, [{"n": n, "workload": workload, "seed": seed, "method": method, "repetition": repetition, **event} for event in events])
                        except BaseException as exc:
                            record = {"n": n, "workload": workload, "seed": seed, "method": method, "repetition": repetition, "exception": repr(exc), "traceback": traceback.format_exc()}
                            failures.append(record)
                            _append_jsonl(run_dir / "failures.jsonl", [record])
    if rows:
        canonical_results = E1_RESULTS
        attempt_results = canonical_results if not canonical_results.exists() else ROOT / f"E1_results_{run_dir.name}.csv"
        with attempt_results.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (run_dir / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        _ensure_empty_results_csv()
        (run_dir / "raw_results.csv").write_text("", encoding="utf-8")
    execution_failures = [item for item in failures if "no calibration threshold" not in str(item.get("reason", ""))]
    status = "COMPLETE" if not execution_failures and rows else "INCOMPLETE_WITH_FAILURES"
    (run_dir / "status.json").write_text(json.dumps({"status": status, "rows": len(rows), "failures": len(failures), "execution_failures": len(execution_failures), "thresholds": thresholds}, indent=2), encoding="utf-8")
    _summarize_e1(rows, failures, thresholds)
    return status == "COMPLETE"


def _ensure_empty_results_csv() -> None:
    if not E1_RESULTS.exists():
        with E1_RESULTS.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["n", "workload", "seed", "input_sha256", "method", "repetition", "warmup", "wall_seconds", "process_highwater_rss_bytes", "nrmse", "max_abs_error", "rhs_evaluations", "queue_operations", "node_refinements", "fallbacks"])


def _summarize_e1(rows: list[dict[str, object]], failures: list[dict[str, object]], thresholds: dict[str, float | None]) -> None:
    # Detailed gate statistics are calculated in stats.py; this report never hides failures.
    from .stats import analyze
    result = analyze(rows)
    (REPORTS / "E1_gate_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    overall = result["overall_pass"]
    decision = "CONTINUE" if overall else ("PIVOT" if sum(v["pass"] for v in result["workloads"].values()) == 1 else "STOP")
    reason = result["decision_reason"]
    report = "# E1 results\n\n" + f"Independent calibration thresholds: `{thresholds}`.\n\n" + f"Raw seed-level timed rows: {len(rows)}; retained failed runs: {len(failures)}.\n\n" + "| Workload | NRMSE max across cells | Latency ratio vs selected baseline | 95% CI | Holm p | G1 |\n|---|---:|---:|---:|---:|---|\n"
    for workload, item in result["workloads"].items():
        report += f"| {workload} | {item['max_nrmse']:.6g} | {item['ratio']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] | {item['holm_p']:.5g} | {'PASS' if item['pass'] else 'FAIL'} |\n"
    report += f"\nDecision: **{decision}** — {reason}\n\nFailures/timeouts are recorded in the run directory; this report is bounded to synthetic CPU simulation and does not establish biological specificity or AI transfer.\n"
    # Include method-level accuracy and runtime summaries, including weak results.
    report += "\n## Method summaries\n\n| Workload | Method | NRMSE max | Latency median (s) | p50 (s) | p95 (s) | Throughput (runs/s) | Process high-water RSS max (bytes) |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
    for workload in ("sparse_pulses", "smooth_periodic"):
        for method in ("B0", "B1", "B2", "B3", "P"):
            cells = [row for row in rows if row["workload"] == workload and row["method"] == method]
            if not cells:
                continue
            latencies = np.asarray([float(row["wall_seconds"]) for row in cells])
            throughput = 1.0 / float(np.median(latencies)) if latencies.size else float("nan")
            report += f"| {workload} | {method} | {max(float(row['nrmse']) for row in cells):.6g} | {np.median(latencies):.6g} | {np.percentile(latencies, 50):.6g} | {np.percentile(latencies, 95):.6g} | {throughput:.6g} | {max(int(row['process_highwater_rss_bytes']) for row in cells)} |\n"
    (E1_REPORT).write_text(report, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, workload in zip(axes, ("sparse_pulses", "smooth_periodic")):
            for method in ("B0", "B1", "B2", "B3", "P"):
                cells = [r for r in rows if r["workload"] == workload and r["method"] == method]
                if not cells:
                    continue
                x = float(np.median([float(r["wall_seconds"]) for r in cells]))
                y = float(np.median([float(r["nrmse"]) for r in cells]))
                axis.scatter(x, y, label=method)
                axis.annotate(method, (x, y), fontsize=8)
            axis.axhline(1e-3, color="black", linestyle="--", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(workload)
            axis.set_xlabel("Median end-to-end latency (s)")
            axis.set_ylabel("Median NRMSE")
            axis.legend()
        fig.savefig(REPORTS / "E1_pareto.png", dpi=160)
        fig.savefig(REPORTS / "E1_pareto.svg")
        plt.close(fig)
    except Exception as exc:
        _append_jsonl(REPORTS / "figure_failures.jsonl", [{"exception": repr(exc), "traceback": traceback.format_exc()}])
    _write_decision(decision, reason)


def _write_decision(decision: str, reason: str) -> None:
    DECISION_REPORT.write_text(
        "# IRCN decision v1\n\n"
        f"Decision: **{decision}**\n\n"
        f"Reason: {reason}\n\n"
        "## Facts\n\n"
        "Facts are the raw results and gate statuses in E0/E1 reports and run manifests.\n\n"
        "## Interpretation\n\n"
        "This decision applies only to the pre-registered synthetic-graph solver experiment on the named CPU. It does not establish connectome-specific computation, task learning, or biological mechanism.\n\n"
        "## Unresolved\n\n"
        "Biological topology contribution (E2), transfer (E3), and GPU/hybrid behavior (E4) remain untested. E2 is not executed by this run.\n\n"
        "## E2 recommendation\n\n"
        + ("Recommend opening a separate E2 preregistration; do not treat this as authorization to run E2." if decision == "CONTINUE" else "Do not enter E2 in this round; follow the frozen PIVOT/STOP rule above.")
        + "\n",
        encoding="utf-8",
    )
