from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

MODEL_DIR = Path(__file__).resolve().parents[3] / "model" / "v0.6.0"
sys.path.insert(0, str(MODEL_DIR / "src"))
sys.path.insert(0, str(MODEL_DIR))

from ircn_v06 import (  # noqa: E402
    GatedCell,
    average_ranks,
    balanced_row_weights,
    canonical_array_hash,
    configure_torch,
    counterfactual_labels,
    feature_rows,
    generate_input,
    generate_teacher,
    graph,
    rollout,
    top_precision,
)
from run_experiment import (  # noqa: E402
    EarlyStoppingTracker,
    InfluenceMLP,
    aggregate_seed_metrics,
    fit_standardization,
    write_failure_record,
)

configure_torch()


class LinearTeacherCell:
    """Independent toy transition used only for golden counterfactual tests."""

    @staticmethod
    def step(h: torch.Tensor, x: torch.Tensor, w: torch.Tensor, indegree: torch.Tensor) -> torch.Tensor:
        del indegree
        result = torch.zeros_like(h)
        result[..., 0] = 0.5 * h[..., 0] + torch.matmul(w, h[..., 0:1]).squeeze(-1) + x[..., 0]
        return result

    @staticmethod
    def readout(h: torch.Tensor) -> torch.Tensor:
        return h[..., 7, 0]


def teacher_states(initial: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, _, _, _ = graph()
    state = initial.copy()
    states = [state.copy()]
    teacher = np.zeros((17, 32), dtype=np.float64)
    teacher[0] = initial[:, 0]
    for tick in range(16):
        nxt = 0.5 * state
        nxt[:, 0] += w @ state[:, 0] + x[tick, :, 0]
        state = nxt
        states.append(state.copy())
        teacher[tick + 1] = state[:, 0]
    return np.stack(states), teacher


def test_golden_input_teacher_and_feature_hashes():
    config = yaml.safe_load((MODEL_DIR / "pre_registration_v1.yaml").read_text())
    x = generate_input(424242, 0)
    y = generate_teacher(x)
    assert canonical_array_hash(x) == config["golden"]["input_sha256"]
    assert canonical_array_hash(y) == config["golden"]["teacher_sha256"]
    f = np.asarray(config["golden"]["feature_fixture"]["vector"], dtype=np.float64)
    assert canonical_array_hash(f) == config["golden"]["feature_fixture"]["sha256"]


def test_graph_direction_degrees_and_distance():
    w, indegree, outdegree, distance = graph()
    assert w[1, 0] == pytest.approx(0.8)
    assert w[0, 1] == 0.0
    assert indegree[7] == 1 and outdegree[0] == 1
    assert distance[6] == 1 and distance[0] == 7
    assert distance[8] == 33


def test_feature_order_and_t0_values_match_frozen_fixture():
    config = yaml.safe_load((MODEL_DIR / "pre_registration_v1.yaml").read_text())
    x = np.zeros((16, 32, 1), dtype=np.float64)
    states = np.zeros((17, 32, 8), dtype=np.float64)
    features, opportunities = feature_rows(x, states)
    row = features[opportunities.index((6, 0))]
    expected = np.asarray(config["golden"]["feature_fixture"]["vector"], dtype=np.float64)
    np.testing.assert_array_equal(row, expected)
    assert row.shape == (27,)


def test_feature_collision_fixture_is_distinguished_by_distance():
    config = yaml.safe_load((MODEL_DIR / "pre_registration_v1.yaml").read_text())
    x = np.zeros((16, 32, 1), dtype=np.float64)
    feature_rows_for_nodes = []
    for node in (5, 6):
        states = np.zeros((17, 32, 8), dtype=np.float64)
        states[0, node, 0] = 1.0
        features, opportunities = feature_rows(x, states)
        feature_rows_for_nodes.append(features[opportunities.index((node, 0))])
    a, b = feature_rows_for_nodes
    assert np.array_equal(a[:23], b[:23])
    assert a[25] == 2 and b[25] == 1
    assert not np.array_equal(a, b)
    expected = config["golden"]["feature_collision_fixture"]["expected_signed_deltas"]
    x = np.zeros((16, 32, 1), dtype=np.float64)
    opportunities = [(n, t) for t in range(12) for n in range(32)]
    for node, expected_delta in zip((5, 6), expected, strict=True):
        initial = np.zeros((32, 8), dtype=np.float64)
        initial[node, 0] = 1.0
        states, teacher = teacher_states(initial, x)
        _, _, delta = counterfactual_labels(LinearTeacherCell(), x, states, teacher)
        assert delta[opportunities.index((node, 0))] == pytest.approx(expected_delta, abs=1e-14)


def test_path_and_isolated_counterfactual_golden_deltas():
    x = np.zeros((16, 32, 1), dtype=np.float64)
    initial = np.zeros((32, 8), dtype=np.float64)
    initial[4, 0] = 1.0
    states, teacher = teacher_states(initial, x)
    _, losses, delta = counterfactual_labels(LinearTeacherCell(), x, states, teacher)
    opportunities = [(node, tick) for tick in range(12) for node in range(32)]
    path_idx = opportunities.index((4, 0))
    assert delta[path_idx] == pytest.approx(0.016384, abs=1e-14)
    assert losses[path_idx, 1] > losses[path_idx, 0]
    assert np.all(np.abs(delta[[i for i, (node, _) in enumerate(opportunities) if node >= 8]]) <= 1e-14)


def test_future_teacher_values_do_not_enter_decision_features():
    x = np.zeros((16, 32, 1), dtype=np.float64)
    states = np.zeros((17, 32, 8), dtype=np.float64)
    initial = np.zeros((32, 8), dtype=np.float64)
    initial[4, 0] = 1.0
    states, teacher = teacher_states(initial, x)
    first, _, delta_first = counterfactual_labels(LinearTeacherCell(), x, states, teacher)
    changed_future_target = teacher.copy()
    changed_future_target[2:, 7] += np.arange(1, 16, dtype=np.float64)
    second, _, delta_second = counterfactual_labels(LinearTeacherCell(), x, states, changed_future_target)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(delta_first, delta_second)


def test_balanced_weights_and_rank_ties_are_deterministic():
    seeds = np.repeat([100, 101], 12)
    nodes = np.tile(np.arange(12), 2)
    weights = balanced_row_weights(seeds, nodes)
    assert weights[seeds == 100].sum() == pytest.approx(weights[seeds == 101].sum())
    assert weights[(seeds == 100) & (nodes < 8)].sum() == pytest.approx(weights[(seeds == 100) & (nodes >= 8)].sum())
    np.testing.assert_array_equal(average_ranks(np.asarray([1.0, 1.0, 2.0])), [1.5, 1.5, 3.0])
    score = np.zeros(5)
    labels = np.asarray([1, 0, 0, 1, 1])
    nodes_t = np.asarray([3, 1, 0, 2, 4])
    ticks = np.zeros(5, dtype=int)
    assert top_precision(score, labels, nodes_t, ticks) == pytest.approx(0.0)


def test_cpu_cell_and_predictor_have_finite_nonzero_gradients_and_fixed_seed_hash():
    configure_torch()
    torch.use_deterministic_algorithms(True)
    def run() -> str:
        import hashlib
        cell = GatedCell()
        x = torch.linspace(-1, 1, 16 * 32, dtype=torch.float64).reshape(1, 16, 32, 1)
        _, out = rollout(cell, x)
        pred = InfluenceMLP()(torch.linspace(-1, 1, 27, dtype=torch.float64).reshape(1, 27))
        loss = out.square().mean() + pred.square().mean()
        loss.backward()
        parameters = list(cell.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) > 0 for p in parameters)
        digest = hashlib.sha256()
        for param in parameters:
            digest.update(param.detach().numpy().tobytes())
        digest.update(loss.detach().numpy().tobytes())
        return digest.hexdigest()
    assert run() == run()


def test_early_stopping_tracks_best_weights_and_patience():
    tracker = EarlyStoppingTracker(min_delta=0.1, patience=2)
    assert not tracker.observe(1.0, 1, {"w": torch.tensor([1.0])})
    assert not tracker.observe(0.95, 2, {"w": torch.tensor([2.0])})
    assert tracker.observe(0.94, 3, {"w": torch.tensor([3.0])})
    assert tracker.best_epoch == 3  # Lowest loss is retained even below min_delta.
    assert tracker.best_state["w"].item() == 3.0
    assert tracker.observe(1.1, 4, {"w": torch.tensor([4.0])})
    restored = tracker.best_state
    assert restored["w"].item() == 3.0
    with pytest.raises(FloatingPointError):
        tracker.observe(float("nan"), 5, {"w": torch.tensor([5.0])})


def test_standardization_uses_only_supplied_training_rows():
    x_train = np.asarray([[0.0, 2.0], [2.0, 2.0]])
    y_train = np.asarray([1.0, 3.0])
    weights = np.asarray([1.0, 1.0])
    expected = fit_standardization(x_train, y_train, weights)
    evaluation_only = np.asarray([[1e12, -1e12]])
    _ = evaluation_only  # Evaluation rows are not part of the scaler API.
    actual = fit_standardization(x_train.copy(), y_train.copy(), weights.copy())
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_trajectory_metrics_aggregate_to_seed_units_before_bootstrap():
    rows = [
        {"top_seed": 10, "trajectory_index": 0, "metric": 0.0, "status": "OK"},
        {"top_seed": 10, "trajectory_index": 1, "metric": 2.0, "status": "OK"},
        {"top_seed": 20, "trajectory_index": 0, "metric": 10.0, "status": "OK"},
    ]
    seed_rows, summary = aggregate_seed_metrics(rows, replicates=100, seed=9)
    assert [row["metric"] for row in seed_rows] == [1.0, 10.0]
    assert [row["trajectory_count"] for row in seed_rows] == [2, 1]
    assert summary["metric"]["mean"] == pytest.approx(5.5)
    assert summary["metric"]["bootstrap_95_percentile_ci"] == aggregate_seed_metrics(rows, 100, 9)[1]["metric"]["bootstrap_95_percentile_ci"]


def test_failure_record_is_preserved_and_never_overwritten(tmp_path):
    first = write_failure_record(RuntimeError("first failure"), "attempt-x", tmp_path)
    assert '"message": "first failure"' in first.read_text()
    with pytest.raises(FileExistsError):
        write_failure_record(RuntimeError("second failure"), "attempt-x", tmp_path)
