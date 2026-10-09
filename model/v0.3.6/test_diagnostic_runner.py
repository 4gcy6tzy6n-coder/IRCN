import sys
import inspect
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostic_runner as runner


def hanging_worker(output_path, queue):
    Path(output_path).write_text("partial", encoding="utf-8")
    queue.put({"stage": "slow"})
    time.sleep(2.0)


class DummySystem:
    def __init__(self):
        self.W = np.zeros((1, 1), dtype=np.float64)
        self.U = np.ones((1, 1), dtype=np.float64)
        self.b = np.zeros(1, dtype=np.float64)
        self.tau = np.ones(1, dtype=np.float64)


class DiagnosticRunnerTests(unittest.TestCase):
    def test_nrmse_uses_parent_wp2_rms_denominator(self):
        reference = np.array([[3.0, 4.0]])
        actual = np.array([[4.0, 4.0]])
        expected = np.sqrt(np.mean((actual - reference) ** 2)) / np.sqrt(np.mean(reference ** 2))
        self.assertEqual(runner.rms_nrmse(actual, reference), expected)

    def test_max_error_tie_breaks_by_time_then_node(self):
        reference = np.zeros((3, 2))
        actual = np.array([[0, 1], [1, 0], [0, 0]])
        self.assertEqual(runner.max_error(actual, reference), (1.0, 0, 1))

    def test_rk4_lands_on_sample_and_input_boundary_with_left_limit(self):
        x = lambda t: np.array([1.0 if t >= 0.005 else 0.0])
        x.breakpoints = lambda start, end: [0.005]
        times = np.array([0.0, 0.005, 0.01])
        result = runner.rk4_reference(DummySystem(), x, times, 0.01, [0.005])
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[1, 0], 0.0)
        self.assertGreater(result[2, 0], 0.0)
        self.assertAlmostEqual(result[2, 0], np.tanh(1.0) * (1.0 - np.exp(-0.005)), places=10)

    def test_semantic_hash_ignores_only_run_identity_fields(self):
        trajectory = np.array([[0.25]], dtype=np.float64)
        events = b'{"kind":"event"}\n'
        counters = {"events": 1}
        first = {"id": "baseline_1", "variant_id": "baseline_1", "run_id": "r1",
                 "repeat_of": None, "rtol": 1e-9, "task": "fixed"}
        second = {"id": "baseline_2", "variant_id": "baseline_2", "run_id": "r2",
                  "repeat_of": "baseline_1", "rtol": 1e-9, "task": "fixed"}
        hash1, full1, norm1 = runner._semantic_hash(trajectory, events, counters, first, "schema-v1")
        hash2, full2, norm2 = runner._semantic_hash(trajectory, events, counters, second, "schema-v1")
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(full1, full2)
        self.assertEqual(norm1["normalized_config"], norm2["normalized_config"])
        changed = dict(second, rtol=1e-8)
        hash3, _, norm3 = runner._semantic_hash(trajectory, events, counters, changed, "schema-v1")
        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(norm1["normalized_config"], norm3["normalized_config"])

    def test_postrun_archive_populates_semantic_and_instance_hashes_in_reports(self):
        contract = {"instance": {"split": "validation", "graph": "chain", "population_size": 1,
                                  "task": "test", "seed": 7, "expected_input_trace_sha256": ""},
                    "analysis": {"outputs": {"semantic_hash": {"schema_id": "test-schema"}}}}
        trace = np.zeros((2, 1), dtype="<f8")
        contract["instance"]["expected_input_trace_sha256"] = runner.sha256(trace.tobytes(order="C"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference").mkdir()
            np.save(root / "reference/input_trace.npy", trace, allow_pickle=False)
            variant = {"id": "v1"}
            out = root / "v1"
            out.mkdir()
            trajectory = np.zeros((2, 1), dtype="<f8")
            np.save(out / "trajectory.npy", trajectory, allow_pickle=False)
            (out / "events.jsonl").write_bytes(b"{}\n")
            counters = {"event_count": 0}
            config = {"id": "v1", "rtol": 1e-9}
            (out / "result.json").write_text(json.dumps({"counters": counters, "config": config}), encoding="utf-8")
            (root / "variants.csv").write_text("variant_id,status,semantic_hash\nv1,complete,\n", encoding="utf-8")
            (root / "analysis.json").write_text('{"oracle_validation":{}}', encoding="utf-8")
            runner._postrun_hash_pass(root, [variant], contract)
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            report = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
            with (root / "variants.csv").open(encoding="utf-8") as f:
                csv_row = next(__import__("csv").DictReader(f))
            self.assertTrue(result["semantic_hash"])
            self.assertTrue(result["instance_sha256"])
            self.assertEqual(csv_row["semantic_hash"], result["semantic_hash"])
            self.assertEqual(report["oracle_validation"]["instance_sha256"], result["instance_sha256"])

    def test_frozen_source_manifest_and_variant_normalization(self):
        contract = runner.verify_frozen_inputs()
        variants = runner._resolved_variants(contract, "unit-test-run")
        first = variants[0]
        second = variants[1]
        _, full_first, components_first = runner._semantic_hash(
            np.zeros((201, 128)), b"", {}, first, contract["analysis"]["outputs"]["semantic_hash"]["schema_id"])
        _, full_second, components_second = runner._semantic_hash(
            np.zeros((201, 128)), b"", {}, second, contract["analysis"]["outputs"]["semantic_hash"]["schema_id"])
        self.assertNotEqual(full_first, full_second)
        self.assertEqual(components_first["normalized_config"], components_second["normalized_config"])

    def test_reproduction_gate_checks_expected_metric_names(self):
        contract = runner.verify_frozen_inputs()
        expected = contract["instance"]
        result = {"status": "complete", "input_sha256": expected["expected_input_trace_sha256"],
                  "candidate_vs_dop853_nrmse": expected["expected_candidate_nrmse"],
                  "candidate_vs_dop853_max_abs_error": expected["expected_candidate_max_abs_error"],
                  "counters": expected["expected_counters"]}
        self.assertTrue(runner._failure_gate(result, contract))
        result["candidate_vs_dop853_max_abs_error"] += 1e-6
        self.assertFalse(runner._failure_gate(result, contract))

    def test_output_path_rejects_parent_version(self):
        parent_path = runner.REPO / "result/v0.3.5/should-never-be-written"
        with self.assertRaises(ValueError):
            runner._validate_results_root(parent_path)

    def test_failed_or_timed_out_baseline_is_blocked(self):
        self.assertEqual(runner._classify_final_status(False, [{"status": "timeout"}]), "BLOCKED")
        self.assertEqual(runner._classify_final_status(False, [{"status": "failed"}]), "BLOCKED")
        self.assertEqual(runner._classify_final_status(True, [{"status": "timeout"}]), "COMPLETED_WITH_FAILURES")

    def test_review_receipt_must_bind_all_four_reviewed_files(self):
        files = {
            "runner": runner.sha256(Path(runner.__file__).read_bytes()),
            "tests": runner.sha256(Path(runner.__file__).with_name("test_diagnostic_runner.py").read_bytes()),
            "contract": runner.sha256(runner.CONTRACT_PATH.read_bytes()),
            "protocol": runner.sha256(runner.PROTOCOL_PATH.read_bytes()),
        }
        receipt = {"status": "ACCEPT", "files_sha256": files}
        self.assertEqual(runner._validate_review_receipt(receipt), files)
        incomplete = {"status": "ACCEPT", "code_sha256": files["runner"]}
        with self.assertRaises(RuntimeError):
            runner._validate_review_receipt(incomplete)

    def test_input_fingerprint_mismatch_stops_before_scheduler_load(self):
        trace = np.array([[0.0], [1.0]], dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError, "pre-run input trace hash mismatch"):
            runner.verify_input_trace_fingerprint(trace, "not-the-input-hash")
        self.assertEqual(runner.verify_input_trace_fingerprint(trace, "frozen", calculate_hash=False), "frozen")
        source = inspect.getsource(runner._load_parent_case)
        self.assertLess(source.index("verify_input_trace_fingerprint"),
                        source.index("from events.ode_segment_scheduler import"))

    def test_preflight_failure_is_recorded_as_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = root / "result/v0.3.6/diagnostics"
            with patch.object(runner, "REPO", root):
                status = runner.execute("blocked-test", root / "missing-review.json", allowed)
            run_dir = allowed / "blocked-test"
            self.assertEqual(status, 1)
            record = __import__("json").loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "BLOCKED")
            self.assertTrue((run_dir / "failure.json").is_file())
            self.assertTrue((run_dir / "output_hash_manifest.json").is_file())
            manifest = __import__("json").loads((run_dir / "output_hash_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("command.log", manifest["files"])

    def test_timeout_preserves_partial_worker_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "partial.json"
            result = runner._run_staged_process(hanging_worker, (str(partial),), {"slow": 0.2}, "test-timeout")
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(partial.read_text(encoding="utf-8"), "partial")

    def test_incomplete_variant_analysis_is_never_marked_complete(self):
        contract = runner.verify_frozen_inputs()
        variants = runner._resolved_variants(contract, "analysis-test")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            baseline_dir = run_dir / "frozen_baseline_repeat_1"
            baseline_dir.mkdir()
            np.save(baseline_dir / "trajectory.npy", np.zeros((201, 128)))
            result = {"variant_id": "frozen_baseline_repeat_1", "candidate_vs_dop853_nrmse": 0.1,
                      "candidate_vs_dop853_max_abs_error": 0.2}
            (baseline_dir / "result.json").write_text(__import__("json").dumps(result), encoding="utf-8")
            runner._write_contrast_analysis(run_dir, variants, [], contract, {"checks": {}})
            analysis = __import__("json").loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis["status"], "partial")

    def test_baseline_reproduction_failure_stops_before_other_variants(self):
        contract = runner.verify_frozen_inputs()
        variants = runner._resolved_variants(contract, "gate-test")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def fake_run(config, output_dir, timeout):
                calls.append(config["id"])
                expected = contract["instance"]
                record = {"status": "complete", "input_sha256": expected["expected_input_trace_sha256"],
                          "candidate_vs_dop853_nrmse": expected["expected_candidate_nrmse"] + 1e-4,
                          "candidate_vs_dop853_max_abs_error": expected["expected_candidate_max_abs_error"],
                          "candidate_vs_rk4_coarse_nrmse": 0.0, "candidate_vs_rk4_coarse_max_abs": 0.0,
                          "candidate_vs_rk4_fine_nrmse": 0.0, "candidate_vs_rk4_fine_max_abs": 0.0,
                          "scheduler_seconds": 1.0, "semantic_hash": "hash",
                          "counters": expected["expected_counters"]}
                output_dir.mkdir(parents=True)
                (output_dir / "result.json").write_text(__import__("json").dumps(record), encoding="utf-8")
                np.save(output_dir / "trajectory.npy", np.zeros((201, 128)), allow_pickle=False)
                (output_dir / "events.jsonl").write_bytes(b"{\"event\":1}\n")
                return {"status": "complete", "result": record}
            with patch.object(runner, "run_variant", fake_run):
                rows, failures, passed = runner._run_candidate_sequence(Path(tmp), variants, contract)
            self.assertEqual(calls, ["frozen_baseline_repeat_1"])
            self.assertFalse(passed)
            self.assertEqual(failures[0]["status"], "reproduction_gate_failed")

    def test_post_baseline_variant_failure_does_not_skip_remaining_variants(self):
        contract = runner.verify_frozen_inputs()
        variants = runner._resolved_variants(contract, "continue-test")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def fake_run(config, output_dir, timeout):
                calls.append(config["id"])
                if config["id"] == "edge_delivery_batch2":
                    return {"status": "timeout", "stage": "candidate"}
                expected = contract["instance"]
                record = {"status": "complete", "variant_id": config["id"],
                          "config": config,
                          "input_sha256": expected["expected_input_trace_sha256"],
                          "candidate_vs_dop853_nrmse": expected["expected_candidate_nrmse"],
                          "candidate_vs_dop853_max_abs_error": expected["expected_candidate_max_abs_error"],
                          "candidate_vs_rk4_coarse_nrmse": 0.0, "candidate_vs_rk4_coarse_max_abs": 0.0,
                          "candidate_vs_rk4_fine_nrmse": 0.0, "candidate_vs_rk4_fine_max_abs": 0.0,
                          "scheduler_seconds": 1.0, "semantic_hash": "same",
                          "normalized_config_hash": "same-config", "counters": expected["expected_counters"]}
                output_dir.mkdir(parents=True)
                (output_dir / "result.json").write_text(__import__("json").dumps(record), encoding="utf-8")
                np.save(output_dir / "trajectory.npy", np.zeros((201, 128)), allow_pickle=False)
                (output_dir / "events.jsonl").write_bytes(b"{\"event\":1}\n")
                return {"status": "complete", "result": record}
            with patch.object(runner, "run_variant", fake_run):
                rows, failures, passed = runner._run_candidate_sequence(Path(tmp), variants, contract)
            self.assertTrue(passed)
            self.assertEqual(len(calls), len(variants))
            self.assertEqual(calls[-1], "tighter_segment_tolerance")
            self.assertEqual(failures[0]["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
