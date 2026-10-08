from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

RESULT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_latest_run_has_complete_provenance_and_self_consistent_outputs():
    runs = sorted(p for p in (RESULT_ROOT / "runs").glob("*") if (p / "v0.6_summary.json").is_file()) if (RESULT_ROOT / "runs").exists() else []
    if not runs:
        pytest.skip("no complete run exists yet")
    run = runs[-1]
    summary = json.loads((run / "v0.6_summary.json").read_text())
    assert summary["run_id"] == run.name
    labels = {split: read_csv(run / "raw" / f"counterfactual_labels_{split}.csv.gz") for split in ("train", "validation", "test")}
    expected_counts = {"train": 122_880, "validation": 61_440, "test": 122_880}
    for split, rows in labels.items():
        assert len(rows) == expected_counts[split]
        assert len({(row["top_seed"], row["trajectory_index"], row["node"], row["tick"]) for row in rows}) == len(rows)
        for row in rows:
            for key in ("snapshot_sha256", "input_sha256", "teacher_sha256", "feature_sha256", "trained_cell_sha256", "source_manifest_sha256", "configuration_sha256"):
                assert len(row[key]) == 64 and all(ch in "0123456789abcdef" for ch in row[key])
            assert row["failure_status"] == "OK"
            assert float(row["signed_delta"]) == pytest.approx(float(row["loss_hold"]) - float(row["loss_update"]), abs=1e-14)
            if int(row["node"]) >= 8:
                assert abs(float(row["signed_delta"])) <= 1e-12
    predictions = read_csv(run / "raw" / "test_predictions.csv.gz")
    assert len(predictions) == expected_counts["test"]
    assert all(row["predictor_checkpoint_sha256"] for row in predictions)
    manifest = read_csv(run / "manifests" / "output_hashes.csv")
    recorded = {row["path"]: row["sha256"] for row in manifest}
    actual = {str(path.relative_to(RESULT_ROOT.parent.parent)): sha256(path)
              for path in run.rglob("*") if path.is_file() and path.name != "output_hashes.csv"}
    assert recorded == actual
