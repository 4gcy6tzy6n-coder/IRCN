import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("event_runtime", Path(__file__).with_name("event_runtime.py"))
runtime_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime_module
spec.loader.exec_module(runtime_module)
Action = runtime_module.Action
Decision = runtime_module.Decision
EventLimitError = runtime_module.EventLimitError
EventRuntime = runtime_module.EventRuntime


def add_transition(node, own, aggregate, value, elapsed):
    return own + aggregate + value + elapsed * 0


def update_policy(node, features):
    return Decision(Action.UPDATE)


def hold_policy(node, features):
    return Decision(Action.HOLD)


def make_runtime(**kwargs):
    return EventRuntime(transition=add_transition, policy=update_policy, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"node_count": 0, "state_dim": 1, "input_dim": 1, "edges": []},
    {"node_count": 2, "state_dim": 1, "input_dim": 1, "edges": [(0, 0, 1)]},
    {"node_count": 2, "state_dim": 1, "input_dim": 1, "edges": [(0, 1, np.nan)]},
    {"node_count": 2, "state_dim": 1, "input_dim": 1, "edges": [(0, 1, 1), (0, 1, 1)]},
])
def test_r0_rejects_invalid_runtime_graph(kwargs):
    params = dict(transition=add_transition, policy=update_policy)
    params.update(kwargs)
    with pytest.raises(ValueError):
        EventRuntime(**params)


@pytest.mark.parametrize("value", [0, -1, True, np.nan, "1"])
def test_r0_rejects_invalid_time_configuration(value):
    with pytest.raises(ValueError):
        make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[], message_delay=value)


def test_r0_rejected_input_does_not_mutate_external_time_cursor():
    runtime = make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[])
    with pytest.raises(ValueError):
        runtime.push_input(2, 0, [1, 2])
    runtime.push_input(1, 0, [3])
    assert runtime._last_external_time == 1


def test_r0_rejects_non_numeric_vectors_without_mutating_time():
    runtime = make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[])
    with pytest.raises(ValueError):
        runtime.push_input(2, 0, ["3"])
    runtime.push_input(1, 0, [3])
    assert runtime._last_external_time == 1
    with pytest.raises(ValueError):
        make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[], initial_states=[["0"]])


@pytest.mark.parametrize("action,steps", [("update", 1), (Action.UPDATE, True), (Action.REFINE, 1.5)])
def test_r0_rejects_malformed_policy_decisions(action, steps):
    with pytest.raises(ValueError):
        Decision(action, steps)


def test_r1_equal_time_order_and_runs_are_deterministic():
    def run_once():
        runtime = make_runtime(node_count=2, state_dim=1, input_dim=1, edges=[])
        runtime.push_input(0, 1, [2])
        runtime.push_input(0, 0, [1])
        runtime.run(0)
        return runtime.states, runtime.trace

    states_a, trace_a = run_once()
    states_b, trace_b = run_once()
    assert np.array_equal(states_a, states_b)
    assert trace_a == trace_b
    assert [(x["kind"], x.get("node")) for x in trace_a[:2]] == [("input", 1), ("input", 0)]


def test_r2_only_active_node_transition_is_evaluated():
    calls = []

    def transition(node, own, aggregate, value, elapsed):
        calls.append(node)
        return own + value

    runtime = EventRuntime(3, 1, 1, [], transition, update_policy)
    runtime.push_input(0, 1, [5])
    runtime.run(0)
    assert calls == [1]
    assert runtime.counters["transition_calls"] == 1
    assert np.array_equal(runtime.states[:, 0], [0, 5, 0])


def test_r3_message_cannot_change_destination_before_delay_or_beyond_reachability():
    runtime = make_runtime(node_count=3, state_dim=1, input_dim=1,
                           edges=[(0, 1, 1), (1, 2, 1)], message_delay=1)
    runtime.push_input(0, 0, [1])
    runtime.run(0)
    assert np.array_equal(runtime.states[:, 0], [1, 0, 0])
    runtime.run(1)
    assert np.array_equal(runtime.states[:, 0], [1, 1, 0])
    runtime.run(2)
    assert np.array_equal(runtime.states[:, 0], [1, 1, 1])
    assert runtime.counters["messages_sent"] == 2


def test_r4_stale_message_version_is_rejected():
    runtime = make_runtime(node_count=2, state_dim=1, input_dim=1, edges=[(0, 1, 1)])
    runtime._enqueue(1, "message", (0, 1, 2, np.array([2.0])))
    runtime._enqueue(2, "message", (0, 1, 1, np.array([1.0])))
    runtime.run(2)
    assert runtime._received_versions[1][0] == 2
    assert runtime.counters["stale_messages"] == 1
    assert runtime._messages[1][0][0] == 2


def test_r4_new_input_invalidates_superseded_deadline():
    runtime = EventRuntime(1, 1, 1, [], add_transition, hold_policy,
                           max_wait=1, max_hold=4, max_consecutive_holds=10)
    runtime.push_input(0, 0, [1])
    runtime.push_input(0.5, 0, [2])
    runtime.run(2)
    assert runtime.counters["stale_deadlines"] >= 1
    assert runtime.counters["transition_calls"] == 0


def test_r5_hold_update_and_refine_have_distinct_transition_and_commit_counts():
    calls = []
    deltas = []

    def transition(node, own, aggregate, value, elapsed):
        calls.append(node)
        deltas.append(elapsed)
        return own + 1

    for decision, expected_calls in ((Decision(Action.HOLD), 0),
                                     (Decision(Action.UPDATE), 1),
                                     (Decision(Action.REFINE, 3), 3)):
        calls.clear()
        deltas.clear()
        runtime = EventRuntime(1, 1, 1, [], transition, lambda n, f: decision,
                               max_wait=2, max_hold=5)
        runtime.push_input(0, 0, [0])
        runtime.run(0)
        assert len(calls) == expected_calls
        if decision.action is Action.HOLD:
            assert runtime.counters["commits"] == 0
        else:
            assert runtime.counters["commits"] == 1
            assert runtime._versions[0] == 1
        if decision.action is Action.REFINE:
            assert deltas == [0.0, 0.0, 0.0]
            assert runtime._states[0, 0] == 3


def test_r5_refine_uses_elapsed_once_then_zero():
    deltas = []

    def transition(node, own, aggregate, value, elapsed):
        deltas.append(elapsed)
        return own + elapsed + 1

    runtime = EventRuntime(1, 1, 1, [], transition,
                           lambda n, f: Decision(Action.REFINE, 3))
    runtime.push_input(2, 0, [0])
    runtime.run(2)
    assert deltas == [2.0, 0.0, 0.0]
    assert runtime._versions[0] == 1


def test_r6_hold_policy_is_forced_to_update_by_liveness_limit():
    runtime = EventRuntime(1, 1, 1, [], add_transition, hold_policy,
                           max_wait=1, max_hold=2, max_consecutive_holds=10)
    runtime.push_input(0, 0, [1])
    runtime.run(2)
    assert runtime.counters["policy_calls"] == 2
    assert runtime.counters["forced_updates"] == 1
    assert runtime.counters["commits"] == 1


def test_r6_event_cap_raises_and_preserves_partial_state():
    runtime = make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[], max_events=1)
    runtime.push_input(0, 0, [1])
    with pytest.raises(EventLimitError):
        runtime.run(0)
    assert runtime.counters["event_limit_failures"] == 1
    assert np.array_equal(runtime.states, [[0]])
    assert runtime.counters["queue_pushes"] - runtime.counters["queue_pops"] == 1
    assert runtime.trace[-1]["kind"] == "event_limit_error"


def test_failed_transition_is_logged_without_partial_commit():
    def broken_transition(*args):
        raise RuntimeError("deliberate test failure")

    runtime = EventRuntime(1, 1, 1, [], broken_transition, update_policy)
    runtime.push_input(0, 0, [1])
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        runtime.run(0)
    assert np.array_equal(runtime.states, [[0]])
    assert runtime.counters["transition_failures"] == 1
    assert runtime.trace[-1]["kind"] == "transition_error"


def test_invalid_policy_output_is_logged_before_transition():
    runtime = EventRuntime(1, 1, 1, [], add_transition, lambda node, features: "update")
    runtime.push_input(0, 0, [1])
    with pytest.raises(TypeError, match="policy must return a Decision"):
        runtime.run(0)
    assert np.array_equal(runtime.states, [[0]])
    assert runtime.counters["policy_failures"] == 1
    assert runtime.trace[-1]["kind"] == "policy_error"


def test_readout_failure_is_logged_without_state_mutation():
    def broken_readout(states):
        raise RuntimeError("readout failure")

    runtime = EventRuntime(1, 1, 1, [], add_transition, update_policy, readout=broken_readout)
    runtime.push_input(0, 0, [2])
    runtime.push_readout(1)
    with pytest.raises(RuntimeError, match="readout failure"):
        runtime.run(1)
    assert runtime.states[0, 0] == 2
    assert runtime.counters["readout_failures"] == 1
    assert runtime.trace[-1]["kind"] == "readout_error"


def test_r7_readout_observes_committed_snapshot_without_updating_nodes():
    runtime = EventRuntime(1, 1, 1, [], add_transition, update_policy,
                           readout=lambda states: float(states[0][0]))
    runtime.push_input(0, 0, [4])
    runtime.push_readout(1)
    outputs = runtime.run(1)
    assert outputs == [(1.0, 4.0)]
    assert runtime.counters["transition_calls"] == 1
    assert runtime.counters["readout_calls"] == 1


def test_r7_readout_mutation_cannot_change_runtime_state():
    def readout(states):
        with pytest.raises(ValueError):
            states[0][0] = 100
        return states[0][0]

    runtime = EventRuntime(1, 1, 1, [], add_transition, update_policy, readout=readout)
    runtime.push_input(0, 0, [2])
    runtime.push_readout(1)
    runtime.run(1)
    assert runtime.states[0, 0] == 2


def test_r8_counters_reconcile_for_simple_trace():
    runtime = make_runtime(node_count=1, state_dim=1, input_dim=1, edges=[])
    runtime.push_input(0, 0, [3])
    runtime.push_readout(1)
    runtime.run(1)
    assert runtime.counters["queue_pushes"] == runtime.counters["queue_pops"]
    assert runtime.counters["events_processed"] == runtime.counters["queue_pops"]
    assert runtime.counters["transition_calls"] == runtime.counters["commits"] == 1
    assert runtime.counters["policy_calls"] == 1
