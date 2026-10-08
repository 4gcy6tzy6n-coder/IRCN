import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("icef_metrics", Path(__file__).with_name("icef_metrics.py"))
icef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icef)
task_spec = importlib.util.spec_from_file_location("diagnostic_tasks", Path(__file__).with_name("diagnostic_tasks.py"))
tasks = importlib.util.module_from_spec(task_spec)
task_spec.loader.exec_module(tasks)


def test_c1_known_selectivity_and_undefined_groups():
    got = icef.compute_selectivity([1, 0, 1, 0], [1, 1, 0, 0])
    assert got == {"selected_fraction": 0.5, "critical_recall": 0.5, "noncritical_update_rate": 0.5}
    assert np.isnan(icef.compute_selectivity([1, 0], [1, 1])["noncritical_update_rate"])


def test_c2_perfect_ranking_ties_and_constant_undefined():
    got = icef.influence_fidelity([0.1, 0.4, 0.2, -0.1], [1, 4, 2, -1], k=2)
    assert got["spearman"] == pytest.approx(1.0)
    assert got["sign_accuracy"] == 1.0
    assert got["top_k_overlap"] == 1.0
    assert np.isnan(icef.influence_fidelity([1, 1], [2, 2], k=1)["spearman"])
    assert icef.influence_fidelity([1, 3, 2], [1, 2, 3], k=1)["spearman"] == pytest.approx(0.5)


def test_c3_latency_censoring_and_stable_recovery():
    assert icef.event_latency(2.0, 2.25) == {"latency": 0.25, "missed": False}
    assert icef.event_latency(2.0, None)["missed"] is True
    assert icef.stable_recovery_time([0, 1, 2, 3, 4], [4, 2, 0.5, 0.4, 0.3], threshold=1, dwell=2) == 2
    assert np.isnan(icef.stable_recovery_time([0, 1, 2], [4, 0.5, 4], threshold=1, dwell=1))


def test_c4_long_horizon_reports_loss_and_state_bound_separately():
    got = icef.long_horizon_summary([0, 1, 3], np.array([[0, 0], [3, 4], [0, 2]]), [1, 0.5, 0.25], state_norm_limit=4)
    assert got == {"horizon": 3.0, "mean_task_loss": pytest.approx(0.5833333333), "final_task_loss": 0.25, "max_state_norm": 5.0, "state_bound_violation_fraction": 1 / 3}


def test_c5_pareto_keeps_ties_and_drops_dominated_points():
    assert icef.pareto_frontier([0.8, 0.9, 0.85, 0.9], [10, 12, 15, 11]).tolist() == [0, 3]


def test_c6_paired_structure_contrast():
    assert icef.paired_structure_contrast([0.8, 0.7, 0.9], [0.7, 0.75, 0.8]) == {
        "n_pairs": 3.0, "mean_difference": pytest.approx(0.05), "median_difference": pytest.approx(0.1)
    }


@pytest.mark.parametrize("args", [([1], [1, 0]), ([1, 0], [1])])
def test_rejects_shape_mismatch(args):
    with pytest.raises(ValueError):
        icef.compute_selectivity(*args)


def test_rejects_invalid_k_and_nonmonotonic_time():
    with pytest.raises(ValueError):
        icef.influence_fidelity([1], [1], k=2)
    with pytest.raises(ValueError):
        icef.stable_recovery_time([0, 0], [1, 0], threshold=0.5, dwell=0)


def test_sparse_task_known_answer_path_and_non_descendants():
    task = tasks.sparse_propagation()
    assert task["output"][8] == pytest.approx(0.8**7)
    assert task["output"][-1] > 0
    assert np.all(task["states"][:, 8:] == 0)


def test_multiscale_task_known_answer_recurrences():
    task = tasks.multiscale_integration(steps=8, fast_period=2, slow_period=4)
    assert task["fast"][1] == pytest.approx(0.5)
    assert task["fast"][2] == pytest.approx(0.75)
    assert task["fast"][3] == pytest.approx(-0.125)
    assert task["slow"][1] == pytest.approx(0.02)
    assert np.allclose(task["target"], task["slow"] + task["fast"])


@pytest.mark.parametrize("cue", [0, 1])
@pytest.mark.parametrize("delay", [5, 20, 50])
def test_quiet_memory_task_preserves_query(cue, delay):
    task = tasks.feedback_memory(cue, delay)
    assert task["query_output"] == task["target"]
    assert task["memory"].size == delay + 2


def test_task_generator_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        tasks.sparse_propagation(n_nodes=7)
    with pytest.raises(ValueError):
        tasks.feedback_memory(2, 4)
    with pytest.raises(ValueError):
        tasks.mean_squared_error(np.array([1.0]), np.array([1.0, 2.0]))


def test_budget_controls_are_exact_reproducible_and_oracle_discriminates():
    activity = np.zeros((4, 4))
    activity[2:, 2:] = 10
    critical = np.zeros((4, 4), dtype=bool)
    critical[0, 0] = critical[1, 0] = True
    random_a = tasks.budget_matched_schedule(activity, 2, method="random", seed=17)
    random_b = tasks.budget_matched_schedule(activity, 2, method="random", seed=17)
    oracle = tasks.budget_matched_schedule(activity, 2, method="oracle", critical=critical)
    activity_gate = tasks.budget_matched_schedule(activity, 2, method="activity")
    periodic = tasks.budget_matched_schedule(activity, 2, method="periodic")
    assert random_a.sum() == random_b.sum() == oracle.sum() == activity_gate.sum() == periodic.sum() == 2
    assert np.array_equal(random_a, random_b)
    assert np.all(oracle[critical])
    assert not np.any(activity_gate & critical)


def test_oracle_schedule_requires_explicit_known_answer():
    with pytest.raises(ValueError):
        tasks.budget_matched_schedule(np.zeros((2, 2)), 1, method="oracle")
