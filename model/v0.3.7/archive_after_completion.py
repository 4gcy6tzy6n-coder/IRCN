"""Post-report integrity archive; must run only after report and audit outputs exist."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "model/v0.3.7/RK4_ENDPOINT_AUDIT_CONTRACT_v1.yaml"
OUT = REPO / "result/v0.3.7/rk4-endpoint-audit-20261009-01"
PARENT = REPO / "result/v0.3.6/diagnostics/wp2-diagnostic-20261009-01"
REPORT = OUT / "E0_report.md"
REQUIRED_OUTPUTS = ["contract.yaml", "protocol.md", "review_receipt.json",
                    "parent_contract.yaml", "parent_run_metadata.json",
                    "parent_output_hash_manifest.json", "parent_status.json",
                    "input_trace.npy", "corrected_rk4_coarse.npy", "corrected_rk4_fine.npy",
                    "audit_results.json", "candidate_reference_comparisons.csv", "run_metadata.json",
                    "command.log", "stdout.log", "stderr.log"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode()


def ensure_clean_execution_sources(source_paths, dirty_paths) -> None:
    dirty = set(source_paths) & set(dirty_paths)
    if dirty:
        raise RuntimeError(f"execution sources changed after review/run: {sorted(dirty)}")


def validate_parent_status(status: dict, expected_manifest_digest: str) -> None:
    if (status.get("status") != "COMPLETED_DIAGNOSTIC"
            or status.get("archive_revision") != 3
            or status.get("output_hash_manifest_sha256") != expected_manifest_digest):
        raise RuntimeError("parent status does not match frozen completion and manifest digest")


def validate_outputs(out: Path, report: Path, contract: dict) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"required audit outputs missing: {missing}")
    text = report.read_text(encoding="utf-8")
    if not text.strip() or "PENDING_POST_REPORT_ARCHIVE" not in text:
        raise RuntimeError("report must be non-empty and mark conclusions conditional on post-report archive")
    arrays = {name: np.load(out / name, allow_pickle=False)
              for name in ("corrected_rk4_coarse.npy", "corrected_rk4_fine.npy", "input_trace.npy")}
    coarse = arrays["corrected_rk4_coarse.npy"]
    fine = arrays["corrected_rk4_fine.npy"]
    trace = arrays["input_trace.npy"]
    instance = contract["parent_run"]["frozen_instance"]
    rows = round(instance["duration_seconds"] / instance["sample_dt_seconds"]) + 1
    expected_shape = (rows, instance["population_size"])
    if coarse.shape != expected_shape or fine.shape != expected_shape or trace.shape != expected_shape:
        raise RuntimeError(f"audit arrays must all have frozen shape {expected_shape}")
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"non-finite values in {name}")
    parent_status = json.loads((out / "parent_status.json").read_text(encoding="utf-8"))
    validate_parent_status(parent_status, contract["parent_run"]["output_manifest_sha256"])


def _archive() -> dict:
    if not REPORT.is_file() or not (OUT / "audit_results.json").is_file():
        raise SystemExit("refusing to hash: complete numerical outputs and E0_report.md are required")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    validate_outputs(OUT, REPORT, contract)
    status_before = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if not str(status_before.get("status", "")).startswith("COMPLETED"):
        raise SystemExit("refusing to hash: audit status is not completed")
    metadata = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "review_receipt.json").read_text(encoding="utf-8"))
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                    check=True, capture_output=True, text=True).stdout.strip()
    if metadata.get("git_commit") != current_commit or receipt.get("reviewed_commit") != current_commit:
        raise RuntimeError("reviewed/executed commit differs from post-run commit")
    reviewed_paths = ("model/v0.3.7/RK4_ENDPOINT_AUDIT_CONTRACT_v1.yaml",
         "model/v0.3.7/RK4_ENDPOINT_AUDIT_PROTOCOL_v1.md",
         "model/v0.3.7/audit_runner.py", "model/v0.3.7/test_audit.py",
         "model/v0.3.7/archive_after_completion.py",
         "model/v0.3.5/src/dynamics/tasks.py", "model/v0.3.5/src/dynamics/graphs.py",
         "model/v0.3.5/src/dynamics/system.py", "model/v0.3.5/src/events/inputs.py",
         "model/v0.3.5/requirements.lock")
    dirty_output = subprocess.run(["git", "status", "--porcelain", "--", *reviewed_paths],
                                  cwd=REPO, check=True, capture_output=True, text=True).stdout
    dirty_paths = [line[3:] for line in dirty_output.splitlines()]
    ensure_clean_execution_sources(reviewed_paths, dirty_paths)
    parent_manifest_path = PARENT / "output_hash_manifest.json"
    if (OUT / "parent_status.json").read_bytes() != (PARENT / "status.json").read_bytes():
        raise RuntimeError("frozen parent status changed after audit execution")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    actual_parent_files = {str(path.relative_to(PARENT)) for path in PARENT.rglob("*")
                           if path.is_file() and path.name not in {"output_hash_manifest.json", "status.json"}}
    extra_parent_files = sorted(actual_parent_files - set(parent_manifest["files"]))
    parent_mismatches = []
    for relative, expected in parent_manifest["files"].items():
        path = PARENT / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            parent_mismatches.append({"path": relative, "expected": expected, "actual": actual})
    manifest_digest = sha256(parent_manifest_path)
    parent_ok = (manifest_digest == contract["parent_run"]["output_manifest_sha256"]
                 and not parent_mismatches and not extra_parent_files)
    if extra_parent_files:
        parent_mismatches.extend({"path": path, "expected": "ABSENT_FROM_FROZEN_MANIFEST",
                                  "actual": "EXTRA_FILE"} for path in extra_parent_files)
    result_path = OUT / "audit_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["parent_verification"] = {"manifest_digest": manifest_digest,
                                     "integrity_verified": parent_ok,
                                     "mismatches": parent_mismatches}
    result["parent_integrity_verified_post_calculation"] = parent_ok
    if not parent_ok:
        result["status"] = "INVALID_PARENT_INTEGRITY"
    result_path.write_bytes(canonical(result))

    sources = [CONTRACT, REPO / "model/v0.3.7/RK4_ENDPOINT_AUDIT_PROTOCOL_v1.md",
               REPO / "model/v0.3.7/audit_runner.py", REPO / "model/v0.3.7/test_audit.py",
               REPO / "model/v0.3.5/src/dynamics/tasks.py",
               REPO / "model/v0.3.5/src/dynamics/graphs.py",
               REPO / "model/v0.3.5/src/dynamics/system.py",
               REPO / "model/v0.3.5/src/events/inputs.py",
               REPO / "model/v0.3.5/requirements.lock",
               REPO / "model/v0.3.7/archive_after_completion.py"]
    source_hashes = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    input_hashes = {f"parent/{relative}": expected
                    for relative, expected in parent_manifest["files"].items()}
    outputs = {str(path.relative_to(OUT)): sha256(path)
               for path in sorted(OUT.rglob("*")) if path.is_file()
               and path.name not in {"post_report_archive.json", "output_hash_manifest.json", "status.json"}}
    archive = {"hashing_phase": "POST_EXPERIMENT_AND_POST_REPORT",
               "parent_manifest_digest": manifest_digest,
               "parent_manifest_file_sha256": manifest_digest,
               "parent_integrity_verified": parent_ok,
               "parent_mismatches": parent_mismatches,
               "source_sha256": source_hashes,
               "input_sha256": input_hashes,
               "output_sha256": outputs}
    (OUT / "post_report_archive.json").write_bytes(canonical(archive))
    files = {str(path.relative_to(OUT)): sha256(path)
             for path in sorted(OUT.rglob("*")) if path.is_file()
             and path.name not in {"output_hash_manifest.json", "status.json"}}
    manifest_data = canonical({"algorithm": "SHA-256", "created_post_report": True,
                               "excluded_self_and_mutable_status": True, "files": files})
    (OUT / "output_hash_manifest.json").write_bytes(manifest_data)
    status_path = OUT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["parent_integrity"] = "PASS" if parent_ok else "FAIL"
    status["output_manifest_sha256"] = hashlib.sha256(manifest_data).hexdigest()
    if not parent_ok:
        status["status"] = "INVALID_PARENT_INTEGRITY"
    status_path.write_bytes(canonical(status))
    return {"parent_integrity": status["parent_integrity"], "status": status["status"],
            "manifest_sha256": status["output_manifest_sha256"]}


def main() -> None:
    try:
        result = _archive()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {"status": "INVALID_ARCHIVE", "error": repr(exc)}
        (OUT / "archive_failure.json").write_bytes(canonical(failure))
        status_path = OUT / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            status = {}
        status["status"] = "INVALID_ARCHIVE"
        status["archive_error"] = repr(exc)
        temporary = OUT / "status.json.tmp"
        temporary.write_bytes(canonical(status))
        temporary.replace(status_path)
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
