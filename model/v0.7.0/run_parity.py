from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE / "src"))
sys.path.insert(0, str(ROOT / "model" / "v0.6.0" / "src"))

import ircn_v06 as v06  # noqa: E402
from ircn_v07 import (  # noqa: E402
    TickRuntime,
    canonical_trace_bytes,
    load_cell,
    semantic_output_hash,
    sparse_input_events,
)


CHECKPOINT_RELATIVE = "result/v0.6.0/runs/final_gz_02/checkpoints/cell_state.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype="<u8").tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    roots = (STAGE, ROOT / "model" / "v0.6.0" / "src")
    files: dict[str, str] = {}
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(ROOT))] = sha256(path)
    for relative in ("model/v0.6.0/MODEL_CONTRACT_v0.1.md",
                    "model/v0.6.0/requirements.lock"):
        path = ROOT / relative
        files[relative] = sha256(path)
    checkpoint = ROOT / CHECKPOINT_RELATIVE
    files[CHECKPOINT_RELATIVE] = sha256(checkpoint)
    return files


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def exact_repeat_match(first: dict[str, object], second: dict[str, object]) -> bool:
    """Compare deterministic semantic outputs and operation counters only."""
    return (
        first.get("status") == "PASS"
        and second.get("status") == "PASS"
        and first.get("semantic_output_sha256") not in (None, "MISSING")
        and first.get("semantic_output_sha256") == second.get("semantic_output_sha256")
        and first.get("operation_counts_json") == second.get("operation_counts_json")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None,
                        help="unique output directory name; existing IDs are rejected")
    args = parser.parse_args()
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    )
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("run-id may contain only letters, digits, underscore, and hyphen")

    config = yaml.safe_load((STAGE / "pre_registration.yaml").read_text())
    checkpoint = ROOT / CHECKPOINT_RELATIVE
    expected_checkpoint_hash = config["scope"]["cell_checkpoint_sha256"]
    params = load_cell(checkpoint, expected_checkpoint_hash)
    v06.configure_torch()
    model = v06.GatedCell()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    run_dir = ROOT / "result" / "v0.7.0" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = run_dir / "raw"
    trace_dir = raw_dir / "traces"
    error_dir = raw_dir / "errors"
    trace_dir.mkdir(parents=True)
    (raw_dir / "outputs").mkdir()
    error_dir.mkdir()

    fixtures: list[tuple[str, int | None, int | None, np.ndarray]] = []
    for seed in config["parity_fixtures"]["top_level_seeds"]:
        for trajectory in config["parity_fixtures"]["trajectory_indices"]:
            fixtures.append((f"seed{seed}_trajectory{trajectory}", seed, trajectory,
                             v06.generate_input(seed, trajectory)))
    for name in config["parity_fixtures"]["golden_fixtures"]:
        if name != "single_source_pulse_tick0":
            raise ValueError(f"unknown frozen golden fixture: {name}")
        golden = np.zeros((v06.N_TICKS, v06.N_NODES, 1), dtype=np.float64)
        golden[0, 0, 0] = 1.0
        fixtures.append((name, None, None, golden))

    rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    rerun_hashes: dict[str, list[str]] = {}
    for fixture, seed, trajectory, x in fixtures:
        input_hashes[fixture] = array_sha256(x)
        attempt_rows: list[dict[str, object]] = []
        for repeat in (1, 2):
            started = time.perf_counter()
            row: dict[str, object] = {"fixture": fixture, "seed": seed,
                                      "trajectory_index": trajectory, "repeat": repeat,
                                      "status": "FAIL", "input_sha256": input_hashes[fixture]}
            try:
                runtime = TickRuntime(params)
                states, readouts = runtime.run(
                    v06.N_TICKS, sparse_input_events(x), update_all=True
                )
                with torch.no_grad():
                    expected_states, expected_readouts = v06.rollout(
                        model, torch.as_tensor(x[None], dtype=torch.float64)
                    )
                ref_states = expected_states[0].cpu().numpy()
                ref_readouts = expected_readouts[0].cpu().numpy()
                state_error = np.abs(states - ref_states)
                readout_error = np.abs(readouts - ref_readouts)
                trace_bytes = canonical_trace_bytes(runtime.trace)
                trace_relative = Path("raw") / "traces" / f"{fixture}_repeat{repeat}.jsonl"
                (run_dir / trace_relative).write_bytes(trace_bytes)
                output_relative = Path("raw") / "outputs" / f"{fixture}_repeat{repeat}.npz"
                np.savez_compressed(run_dir / output_relative,
                                    states=states, readouts=readouts,
                                    reference_states=ref_states,
                                    reference_readouts=ref_readouts)
                semantic_hash = semantic_output_hash(states, readouts, trace_bytes)
                row.update({
                    "status": "PASS" if (float(state_error.max()) <= 1e-10
                                           and float(readout_error.max()) <= 1e-10) else "FAIL",
                    "max_abs_state_error": float(state_error.max()),
                    "max_abs_readout_error": float(readout_error.max()),
                    "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
                    "semantic_output_sha256": semantic_hash,
                    "trace_path": str(trace_relative),
                    "output_path": str(output_relative),
                    "transition_calls": runtime.counters["transitions"],
                    "messages_delivered": runtime.counters["messages_delivered"],
                    "full_state_scans": runtime.counters["full_state_scans"],
                    "operation_counts_json": json.dumps(runtime.counters, sort_keys=True),
                    "initialization_wall_seconds": runtime.cost_ledger["initialization_wall_seconds"],
                    "run_wall_seconds": runtime.cost_ledger["run_wall_seconds"],
                    "run_cpu_seconds": runtime.cost_ledger["run_cpu_seconds"],
                    "peak_rss_platform_units": runtime.cost_ledger["peak_rss_platform_units"],
                })
            except Exception as exc:  # preserve every failed fixture and rerun
                error_path = error_dir / f"{fixture}_repeat{repeat}.txt"
                error_path.write_text(traceback.format_exc())
                row.update({"error_type": type(exc).__name__, "error": str(exc),
                            "error_path": str(error_path.relative_to(run_dir))})
            row["driver_wall_seconds"] = time.perf_counter() - started
            attempt_rows.append(row)
        hashes = [str(row.get("semantic_output_sha256", "MISSING")) for row in attempt_rows]
        rerun_hashes[fixture] = hashes
        same = exact_repeat_match(attempt_rows[0], attempt_rows[1])
        for row in attempt_rows:
            row["exact_rerun_match"] = same
            if not same:
                row["status"] = "FAIL"
            rows.append(row)

    fields = sorted({key for row in rows for key in row})
    results_path = raw_dir / "parity_fixture_results.csv"
    with results_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "run_id": run_id,
        "git_head": git_head(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "source_file_sha256": source_manifest(),
        "input_array_sha256": input_hashes,
        "semantic_hashes_by_repeat": rerun_hashes,
        "output_file_sha256": {
            str(path.relative_to(run_dir)): sha256(path)
            for path in sorted(run_dir.rglob("*")) if path.is_file()
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    passed = sum(row["status"] == "PASS" for row in rows)
    failed = len(rows) - passed
    print(f"run_id={run_id} fixtures={len(fixtures)} attempts={len(rows)} "
          f"passed={passed} failed={failed} output={run_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
