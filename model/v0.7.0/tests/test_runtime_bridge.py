from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "model" / "v0.6.0" / "src"))

import ircn_v06 as v06  # noqa: E402
from run_parity import exact_repeat_match  # noqa: E402
from ircn_v07 import (  # noqa: E402
    Decision,
    TickRuntime,
    canonical_trace_bytes,
    graph,
    load_cell,
    semantic_output_hash,
    sparse_input_events,
)
import ircn_v07  # noqa: E402


CHECKPOINT = ROOT / "result/v0.6.0/runs/final_gz_02/checkpoints/cell_state.pt"
CHECKPOINT_SHA = "b8df57150d9030f2690eb11a7977af8c7b629e43eec228678d186d2c102168c5"


@pytest.fixture(scope="module")
def params():
    return load_cell(CHECKPOINT, CHECKPOINT_SHA)


def test_frozen_checkpoint_hash_and_shape_validation():
    p = load_cell(CHECKPOINT, CHECKPOINT_SHA)
    assert set(p) == {"wz", "bz", "wc", "bc", "readout_w", "readout_b"}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_cell(CHECKPOINT, "0" * 64)


@pytest.mark.parametrize("seed", range(300, 310))
@pytest.mark.parametrize("trajectory", [0, 1])
def test_update_all_matches_v06_rollout(params, seed, trajectory):
    x = v06.generate_input(seed, trajectory)
    events = sparse_input_events(x)
    runtime = TickRuntime(params)
    states, readouts = runtime.run(v06.N_TICKS, events, update_all=True)

    v06.configure_torch()
    model = v06.GatedCell()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    with torch.no_grad():
        ref_states, ref_readouts = v06.rollout(
            model, torch.as_tensor(x[None], dtype=torch.float64)
        )
    np.testing.assert_allclose(states, ref_states[0].numpy(), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(readouts, ref_readouts[0].numpy(), rtol=0.0, atol=1e-10)


def test_deterministic_semantic_trace_and_hash(params):
    x = v06.generate_input(300, 0)
    outputs = []
    for _ in range(2):
        runtime = TickRuntime(params)
        states, readouts = runtime.run(v06.N_TICKS, sparse_input_events(x), update_all=True)
        trace = canonical_trace_bytes(runtime.trace)
        outputs.append((states, readouts, trace, semantic_output_hash(states, readouts, trace)))
    assert outputs[0][2] == outputs[1][2]
    assert outputs[0][3] == outputs[1][3]
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


def test_trace_phases_follow_frozen_tick_order(params):
    runtime = TickRuntime(params)
    x = np.zeros((2, 32, 1), dtype=np.float64)
    x[0, 0, 0] = 1.0
    runtime.run(2, sparse_input_events(x), update_all=True)
    ranks = {phase: i for i, phase in enumerate((
        "DELIVER", "INPUT", "CANDIDATE", "DECIDE", "COMMIT", "PUBLISH", "READOUT"
    ))}
    for tick in range(2):
        phases = [ranks[row["phase"]] for row in runtime.trace if row["tick"] == tick]
        assert phases == sorted(phases)


def test_candidate_is_removed_before_policy_callback(params):
    holder = {}
    seen = []

    def checking_policy(features):
        runtime = holder["runtime"]
        assert features.node_id not in runtime.candidates
        seen.append(features.node_id)
        return Decision("HOLD")

    runtime = TickRuntime(params, checking_policy)
    holder["runtime"] = runtime
    runtime.step(0, ((0, 0, 0.8), (0, 9, -0.2)))
    assert seen == [0, 9]


def test_selective_hold_does_not_transition_or_scan_full_state(params):
    policy_calls = []

    def hold_policy(features):
        policy_calls.append((features.node_id, features.tick, features.values.copy()))
        return Decision("HOLD")

    runtime = TickRuntime(params, hold_policy)
    events = ((0, 0, 1.0),)
    runtime.step(0, events)
    assert runtime.counters["transitions"] == 0
    assert runtime.counters["full_state_scans"] == 0
    assert runtime.counters["policy_calls"] == 1
    assert policy_calls[0][2].shape == (27,)
    assert np.count_nonzero(runtime.states) == 0

    for tick in range(1, 4):
        runtime.step(tick, ())
    assert runtime.counters["transitions"] == 0
    runtime.step(4, ())
    assert runtime.counters["transitions"] == 1  # max-hold forced update
    assert runtime.counters["forced_updates"] == 1
    assert runtime.counters["full_state_scans"] == 0


def test_selective_run_does_not_materialize_full_state_history(params):
    runtime = TickRuntime(params, lambda _features: Decision("HOLD"))
    states, readouts = runtime.run(4, ((0, 3, 0.5),))
    assert states is None
    assert readouts.shape == (4,)
    assert runtime.counters["full_state_scans"] == 0
    assert runtime.counters["full_state_materializations"] == 0


def test_message_is_visible_only_at_next_tick(params):
    runtime = TickRuntime(params)
    runtime.step(0, ((0, 0, 1.0),), update_all=True)
    assert runtime.counters["messages_sent"] > 0
    assert runtime.counters["messages_delivered"] == 0
    prior = runtime.aggregate[1].copy()
    runtime.step(1, (), update_all=True)
    assert runtime.counters["messages_delivered"] > 0
    assert not np.array_equal(runtime.aggregate[1], prior)


def test_same_tick_updates_use_pre_tick_states(params):
    runtime = TickRuntime(params)
    w, indegree, _ = graph()
    before = runtime.states.copy()
    # Update-all tick 0 must use the original zero aggregate, not source 0's
    # newly calculated state when it commits before node 1.
    expected_node_1 = v06.GatedCell()
    expected_node_1.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    with torch.no_grad():
        h = torch.as_tensor(before[None], dtype=torch.float64)
        x = torch.zeros((1, v06.N_NODES, 1), dtype=torch.float64)
        expected = expected_node_1.step(
            h, x, torch.as_tensor(w), torch.as_tensor(indegree)
        )[0, 1].numpy()
    runtime.step(0, ((0, 0, 1.0),), update_all=True)
    np.testing.assert_allclose(runtime.states[1], expected, rtol=0.0, atol=1e-12)


def test_clone_snapshot_restore_equivalence(params):
    runtime = TickRuntime(params)
    runtime.step(0, ((0, 0, 0.7),), update_all=True)
    branch = runtime.clone()
    assert runtime.snapshot_bytes() == branch.snapshot_bytes()
    runtime.step(1, (), update_all=True)
    branch.step(1, (), update_all=True)
    assert runtime.snapshot_bytes() == branch.snapshot_bytes()


def test_snapshot_and_clone_copy_costs_are_recorded(params):
    runtime = TickRuntime(params)
    runtime.snapshot()
    assert runtime.cost_ledger["snapshot_calls"] == 1
    assert runtime.cost_ledger["snapshot_array_bytes"] > 0
    assert runtime.cost_ledger["snapshot_wall_seconds"] >= 0.0
    encoded = runtime.snapshot_bytes()
    assert len(encoded) == runtime.cost_ledger["snapshot_serialized_bytes"]
    assert runtime.cost_ledger["snapshot_serializations"] == 1
    branch = runtime.clone()
    assert runtime.cost_ledger["clone_calls"] == 1
    assert runtime.cost_ledger["clone_array_bytes"] > 0
    assert runtime.cost_ledger["clone_wall_seconds"] >= 0.0
    assert branch.snapshot_bytes() == runtime.snapshot_bytes()


def test_clone_counterfactual_intervention_is_branch_local(params):
    root = TickRuntime(params)
    root_snapshot = root.snapshot_bytes()
    hold_branch = root.clone()
    update_branch = root.clone()
    assert hold_branch.snapshot_bytes() == update_branch.snapshot_bytes() == root_snapshot

    pulse = ((0, 0, 1.0),)
    hold_branch.step(0, pulse, one_shot_override=(0, Decision("HOLD")))
    update_branch.step(0, pulse, one_shot_override=(0, Decision("UPDATE")))
    assert root.snapshot_bytes() == root_snapshot
    assert not np.array_equal(hold_branch.states, update_branch.states)
    assert len(hold_branch.deadlines) == 1
    assert len(update_branch.pending_messages) == 1

    held_replay = hold_branch.clone()
    updated_replay = update_branch.clone()
    hold_branch.step(1, ())
    held_replay.step(1, ())
    update_branch.step(1, ())
    updated_replay.step(1, ())
    assert hold_branch.snapshot_bytes() == held_replay.snapshot_bytes()
    assert update_branch.snapshot_bytes() == updated_replay.snapshot_bytes()


def test_invalid_inputs_and_policy_output_fail_closed(params):
    runtime = TickRuntime(params)
    with pytest.raises(ValueError, match="finite"):
        sparse_input_events(np.full((2, 32, 1), np.nan))
    with pytest.raises(ValueError, match="shape"):
        sparse_input_events(np.zeros((2, 31, 1)))
    with pytest.raises(ValueError, match="nonnegative integer"):
        runtime.run(-1, ())
    with pytest.raises(ValueError, match="node"):
        runtime.step(0, ((0, 32, 1.0),))
    assert runtime.tick == -1
    assert runtime.counters["candidate_enqueues"] == 0
    with pytest.raises(ValueError, match="finite"):
        runtime.step(0, ((0, 0, float("inf")),))
    assert runtime.tick == -1

    bad_runtime = TickRuntime(params, lambda _features: "UPDATE")
    with pytest.raises(TypeError, match="Decision"):
        bad_runtime.step(0, ((0, 2, 0.5),))
    assert bad_runtime.failed
    with pytest.raises(RuntimeError, match="failed"):
        bad_runtime.step(1, ())
    with pytest.raises(ValueError, match="callable"):
        TickRuntime(params, policy="not callable")


def test_equal_payload_message_does_not_create_candidate_or_cancel_deadline(params):
    runtime = TickRuntime(params)
    runtime.deadline_generation[1] = 3
    runtime.deadlines[1] = [(1, 3)]
    runtime.pending_messages[0] = [(0, 0, 0, 1, np.zeros(8, dtype=np.float64))]
    runtime.step(0, ())
    assert runtime.counters["policy_calls"] == 0
    assert runtime.deadline_generation[1] == 3
    assert runtime.deadlines[1] == [(1, 3)]


def test_transition_failure_marks_runtime_unusable(params, monkeypatch):
    runtime = TickRuntime(params)

    def fail_transition(*_args, **_kwargs):
        raise FloatingPointError("injected transition failure")

    monkeypatch.setattr(ircn_v07, "local_transition", fail_transition)
    with pytest.raises(FloatingPointError, match="injected"):
        runtime.step(0, ((0, 0, 1.0),))
    assert runtime.failed
    assert runtime.counters["transition_failures"] == 1
    with pytest.raises(RuntimeError, match="failed"):
        runtime.run(1, ())


def test_repeat_gate_detects_counter_differences():
    first = {"status": "PASS", "semantic_output_sha256": "abc",
             "operation_counts_json": '{"transitions":1}'}
    same = dict(first)
    changed = {**first, "operation_counts_json": '{"transitions":2}'}
    assert exact_repeat_match(first, same)
    assert not exact_repeat_match(first, changed)


def test_input_events_are_sparse_and_include_zero_reset():
    x = np.zeros((3, 32, 1), dtype=np.float64)
    x[0, 5, 0] = 0.8
    x[1, 5, 0] = 0.0
    x[2, 9, 0] = -0.3
    assert sparse_input_events(x) == ((0, 5, 0.8), (1, 5, 0.0), (2, 9, -0.3))
