import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("icef_metrics_v042", Path(__file__).with_name("icef_metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


@pytest.mark.parametrize("invalid", [[np.nan, 0], [np.inf, 1], [2, 0], [-1, 1]])
def test_c1_rejects_nonbinary_or_nonfinite_masks(invalid):
    with pytest.raises(ValueError):
        metrics.compute_selectivity(invalid, [1, 0])


def test_c1_accepts_boolean_and_binary_masks():
    assert metrics.compute_selectivity([True, False], [1, 0])["selected_fraction"] == 0.5


@pytest.mark.parametrize("k", [1.5, 1.0, True, np.bool_(False)])
def test_c2_rejects_noninteger_k_with_contract_error(k):
    with pytest.raises(ValueError, match="k must be an integer"):
        metrics.influence_fidelity([1, 2], [1, 2], k=k)


@pytest.mark.parametrize("threshold,dwell", [(np.nan, 0), (1, np.nan), (np.inf, 0), (1, np.inf), (-1, 0), (1, -1)])
def test_c3_rejects_invalid_recovery_parameters(threshold, dwell):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        metrics.stable_recovery_time([0, 1], [1, 0], threshold=threshold, dwell=dwell)


@pytest.mark.parametrize("times", [[0, 0], [2, 1]])
def test_c4_rejects_duplicate_or_decreasing_timestamps(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.long_horizon_summary(times, np.zeros((2, 1)), [0, 0], state_norm_limit=1)


def test_valid_recovery_and_horizon_results_are_unchanged():
    assert metrics.stable_recovery_time([0, 1, 2], [2, 0.5, 0.4], threshold=1, dwell=1) == 1
    got = metrics.long_horizon_summary([0, 1], np.array([[0.0], [1.0]]), [1, 0], state_norm_limit=2)
    assert got["horizon"] == 1
    assert got["mean_task_loss"] == 0.5
