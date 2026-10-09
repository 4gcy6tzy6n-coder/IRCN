from types import SimpleNamespace

import numpy as np

from audit_runner import (REVIEWED_FILES, check_terminal_locality, corrected_rk4,
                          error_metrics, frozen_breakpoints, terminal_jump,
                          validate_review_receipt, write_failed_status)
from archive_after_completion import validate_outputs


class Input:
    def __init__(self, fn, points=()):
        self.fn = fn
        self.points = tuple(points)

    def __call__(self, t):
        return np.array([self.fn(t)], dtype=np.float64)

    def breakpoints(self, start, end):
        return [p for p in self.points if start < p < end]


def scalar_system():
    return SimpleNamespace(W=np.zeros((1, 1)), U=np.ones((1, 1)),
                           b=np.zeros(1), tau=np.ones(1))


def test_terminal_jump_detected_when_breakpoint_api_omits_endpoint():
    x = Input(lambda t: 1.0 if t < 1.0 else 0.0)
    points, has_jump = frozen_breakpoints(x, 0.0, 1.0, 1e-9)
    assert has_jump
    assert points == [1.0]


def test_terminal_rk4_stage_uses_left_limit_for_jump_endpoint():
    x = Input(lambda t: 1.0 if t < 1.0 else 0.0)
    times = np.array([0.0, 1.0])
    points, has_jump = frozen_breakpoints(x, 0.0, 1.0, 1e-9)
    actual = corrected_rk4(scalar_system(), x, times, 0.001, points,
                           has_jump, 1e-9)
    expected = np.tanh(1.0) * (1.0 - np.exp(-1.0))
    assert abs(actual[-1, 0] - expected) < 1e-11


def test_continuous_terminal_input_is_not_misclassified_as_jump():
    x = Input(lambda t: t)
    assert not terminal_jump(x, 1.0, 1e-9)
    points, has_jump = frozen_breakpoints(x, 0.0, 1.0, 1e-9)
    assert not has_jump
    assert points == []


def test_breakpoint_at_endpoint_uses_left_limit_on_final_stage():
    x = Input(lambda t: 1.0 if t < 1.0 else 0.0, points=[])
    points, has_jump = frozen_breakpoints(x, 0.0, 1.0, 1e-9)
    assert has_jump
    values = corrected_rk4(scalar_system(), x, np.array([0.0, 1.0]),
                           0.001, points, has_jump, 1e-9)
    assert abs(values[-1, 0] - np.tanh(1.0) * (1.0 - np.exp(-1.0))) < 1e-11


def test_terminal_locality_rejects_change_on_undriven_node():
    old = np.zeros((2, 3))
    new = old.copy()
    new[-1, 2] = 1e-6
    locality = check_terminal_locality(old, new, np.array([0, 1]), 1e-12)
    assert locality["unexpected_terminal_nodes"] == [2]
    assert not locality["only_driven_terminal_nodes_changed"]


def test_metric_rejects_mismatched_or_nonfinite_arrays():
    with np.testing.assert_raises(ValueError):
        error_metrics(np.zeros((2, 2)), np.zeros((2, 3)))
    invalid = np.zeros((2, 2))
    invalid[0, 0] = np.nan
    with np.testing.assert_raises(ValueError):
        error_metrics(invalid, np.zeros((2, 2)))


def test_review_receipt_binds_commit_scope_and_clean_tree():
    receipt = {"status": "ACCEPT", "reviewer": "independent reviewer",
               "reviewed_commit": "commit-123", "reviewed_files": sorted(REVIEWED_FILES)}
    validate_review_receipt(receipt, "commit-123", True)
    with np.testing.assert_raises(RuntimeError):
        validate_review_receipt(receipt, "commit-456", True)
    with np.testing.assert_raises(RuntimeError):
        validate_review_receipt(receipt, "commit-123", False)


def test_post_report_archive_rejects_missing_outputs_and_nonconditional_report(tmp_path):
    report = tmp_path / "E0_report.md"
    report.write_text("PENDING_POST_REPORT_ARCHIVE", encoding="utf-8")
    with np.testing.assert_raises(RuntimeError):
        validate_outputs(tmp_path, report)


def test_execution_failure_is_preserved_with_failure_and_atomic_status(tmp_path):
    error = RuntimeError("synthetic failure")
    write_failed_status(tmp_path, error, "trace")
    failure = __import__("json").loads((tmp_path / "failure.json").read_text())
    status = __import__("json").loads((tmp_path / "status.json").read_text())
    assert failure["status"] == status["status"] == "FAILED_PRESERVED"
    assert status["partial_outputs_preserved"]
    assert not (tmp_path / "status.json.tmp").exists()
