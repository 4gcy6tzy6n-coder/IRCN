"""Deterministic, untimed replay to retain representative calibration event traces.

This is not an E1 evaluation and does not produce performance measurements.
It replays every pre-registered N=64 calibration input at the lowest frozen
candidate threshold so the failed event streams are auditable.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ircn.core import integrate_events, make_circuit  # noqa: E402
from ircn.experiment import environment, input_hash, sha256_file  # noqa: E402

OUT = Path(__file__).with_name("calibration_event_replay.jsonl.gz")
META = Path(__file__).with_name("calibration_event_replay.json")

with OUT.open("wb") as compressed:
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, compresslevel=9) as zipped:
        with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
            for method, mode in (("P", "ircn"), ("B2", "input_event")):
                for workload in ("sparse_pulses", "smooth_periodic"):
                    circuit = make_circuit(64, 8675309, workload)
                    _trajectory, counts, events = integrate_events(circuit, threshold=1e-5, mode=mode, log_events=True)
                    digest = input_hash(circuit)
                    for event in events:
                        row = {"method": method, "threshold": 1e-5, "n": 64, "seed": 8675309, "workload": workload, "input_sha256": digest, **event}
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(method, workload, digest, len(events), counts)

META.write_text(json.dumps({
    "purpose": "representative deterministic calibration trace replay; not E1 and not timed",
    "calibration_run": Path(__file__).parent.name,
    "source_hashes": json.loads((Path(__file__).parent / "metadata.json").read_text())['source_sha256'],
    "replay_script_sha256": sha256_file(Path(__file__)),
    "trace_sha256_compressed": sha256_file(OUT),
    "trace_sha256_uncompressed": hashlib.sha256(gzip.decompress(OUT.read_bytes())).hexdigest(),
    "trace_bytes_compressed": OUT.stat().st_size,
    "environment": environment(),
    "scope": "N=64, frozen calibration seed=8675309, every frozen workload and both event methods, threshold=lowest pre-registered candidate 1e-5",
    "event_trace_count": sum(1 for _ in gzip.open(OUT, "rt", encoding="utf-8")),
    "trace_path": OUT.name,
}, indent=2, sort_keys=True), encoding="utf-8")
