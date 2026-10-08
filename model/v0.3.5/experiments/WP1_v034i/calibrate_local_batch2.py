"""Fail-fast calibration runner for the frozen v0.3.4i batch-size-2 candidate."""
import csv
import gzip
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from dynamics.tasks import make_task_system
from events.inputs import make_input
from events.ode_segment_scheduler import solve_ode_segments
from solvers.oracle import solve_oracle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/WP1_v034i"
OUT.mkdir(parents=True, exist_ok=True)
contract_path = ROOT / "contracts/V03_4I_LOCAL_BATCH2_CALIBRATION.yaml"
contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
contract = yaml.safe_load(contract_path.read_text())
limit = float(contract["accuracy"]["relative_nrmse_limit"])
candidate = contract["candidate"]
files = (
    "src/events/ode_segment_scheduler.py",
    "src/events/inputs.py",
    "src/dynamics/tasks.py",
    "src/solvers/oracle.py",
    "experiments/WP1_v034i/calibrate_local_batch2.py",
)
code_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
times = np.linspace(0.0, 2.0, 201)
raw_path = OUT / "calibration_raw.jsonl"
event_path = OUT / "events.jsonl.gz"
raw_path.write_text("")
event_path.write_bytes(b"")
rows = []
failures = []
start_all = time.perf_counter()
stop_reason = "FULL_CALIBRATION_SCOPE_PASSED"

for graph in contract["scope"]["graph_families"]:
    for n in contract["scope"]["population_sizes"]:
        for task in contract["scope"]["tasks"]:
            for seed in contract["scope"]["calibration_seeds"]:
                row = {"graph": graph, "n": n, "task": task, "seed": seed, "contract_sha256": contract_hash, "method": "causal_local_batch2"}
                try:
                    x, driven_ids = make_input(task, n, seed)
                    system = make_task_system(graph, n, seed, driven_ids)
                    trace = np.asarray([x(float(t)) for t in times], dtype="<f8")
                    row["input_sha256"] = hashlib.sha256(trace.tobytes()).hexdigest()
                    row["driven_count"] = int(len(driven_ids))
                    row["nonzero_U_rows"] = int(np.count_nonzero(np.any(system.U != 0, axis=1)))
                    if row["nonzero_U_rows"] != len(driven_ids):
                        raise ValueError("input support mismatch")
                    reference, oracle_info = solve_oracle(system, x, times, rtol=1e-13, atol=1e-15)
                    began = time.perf_counter()
                    prediction, meta = solve_ode_segments(
                        system,
                        x,
                        times,
                        segment_tolerance=float(candidate["segment_tolerance"]),
                        max_interval=float(candidate["max_interval_seconds"]),
                        input_mode=contract["input"]["external_input_integration"],
                        delivery_mode=candidate["delivery_mode"],
                        local_rtol=float(candidate["local_rtol"]),
                        local_atol=float(candidate["local_atol"]),
                        local_batch=bool(candidate["local_batch"]),
                        local_batch_size=int(candidate["local_batch_size"]),
                    )
                    row["wall_seconds"] = time.perf_counter() - began
                    scale = max(float(np.sqrt(np.mean(reference * reference))), 1e-12)
                    row["nrmse"] = float(np.sqrt(np.mean((prediction - reference) ** 2)) / scale)
                    row["max_abs"] = float(np.max(np.abs(prediction - reference)))
                    row["rhs_evaluations"] = int(meta["rhs_evaluations"])
                    row["event_count"] = int(meta["event_count"])
                    row["queue_pushes"] = int(meta["queue_pushes"])
                    row["message_deliveries"] = int(meta["message_deliveries"])
                    row["oracle_nfev"] = int(oracle_info.nfev)
                    row["failure"] = ""
                    with gzip.open(event_path, "at", encoding="utf-8", compresslevel=6) as stream:
                        for event in meta["event_log"]:
                            stream.write(json.dumps({"graph": graph, "n": n, "task": task, "seed": seed, **event}, sort_keys=True) + "\n")
                except Exception as exc:
                    row["failure"] = repr(exc)
                    failures.append({**row, "traceback": traceback.format_exc()})
                rows.append(row)
                with raw_path.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                passed = not row["failure"] and np.isfinite(row.get("nrmse", float("inf"))) and row["nrmse"] <= limit
                print(f"{len(rows)}/96 {graph} N={n} {task} seed={seed}: nrmse={row.get('nrmse', 'FAIL')} wall={row.get('wall_seconds', 'NA')}", flush=True)
                if not passed:
                    stop_reason = "FIRST_CALIBRATION_INSTANCE_FAILED_FROZEN_SCREEN"
                    break
            if stop_reason != "FULL_CALIBRATION_SCOPE_PASSED":
                break
        if stop_reason != "FULL_CALIBRATION_SCOPE_PASSED":
            break
    if stop_reason != "FULL_CALIBRATION_SCOPE_PASSED":
        break

with (OUT / "calibration.csv").open("w", newline="") as stream:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
(OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
valid = [r for r in rows if not r.get("failure") and "nrmse" in r]
result = {
    "contract": contract_path.relative_to(ROOT).as_posix(),
    "contract_sha256": contract_hash,
    "code_sha256": code_hashes,
    "status": stop_reason,
    "scope_instances": 96,
    "completed_instances": len(rows),
    "runtime_failures": len(failures),
    "accuracy_pass_rows": sum(r["nrmse"] <= limit for r in valid),
    "accuracy_fail_rows": sum(r["nrmse"] > limit for r in valid),
    "nrmse_limit": limit,
    "max_nrmse": max((r["nrmse"] for r in valid), default=None),
    "confirmatory_seeds_used": False,
    "downstream_science_executed": False,
    "elapsed_seconds": time.perf_counter() - start_all,
    "raw_rows": raw_path.name,
    "event_log": event_path.name,
}
(OUT / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
