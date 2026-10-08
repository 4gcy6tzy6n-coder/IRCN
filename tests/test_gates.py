from __future__ import annotations

import time

import numpy as np

from ircn.core import integrate_events, integrate_reference, integrate_rk4, make_circuit, needs_refinement, nrmse, report_times, timed_call


def test_t0_determinism_same_seed_recreates_inputs_and_trajectory():
    a = make_circuit(16, 101, "smooth_periodic")
    b = make_circuit(16, 101, "smooth_periodic")
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.tau, b.tau)
    dense = make_circuit(16, 101, "smooth_periodic", dense=True)
    assert np.array_equal(a.tau, dense.tau)
    assert np.array_equal(a.gain, dense.gain)
    assert np.array_equal(a.workload.driven, dense.workload.driven)
    assert np.array_equal(a.workload.phases, dense.workload.phases)
    out_a, _ = integrate_rk4(a)
    out_b, _ = integrate_rk4(b)
    assert np.array_equal(out_a, out_b)


def test_t1_all_node_rk4_matches_independent_scalar_update():
    circuit = make_circuit(16, 211, "smooth_periodic")
    vector, _ = integrate_rk4(circuit, step=0.005)
    times = report_times()
    state = np.zeros(circuit.n)
    scalar = np.empty_like(vector)
    scalar[0] = state
    for sample in range(1, times.size):
        count = 2  # two 5 ms RK4 steps per 10 ms report interval
        dt = 0.005
        for j in range(count):
            t = float(times[sample - 1] + j * dt)
            k1 = circuit.rhs(t, state)
            k2 = circuit.rhs(t + dt / 2, state + dt * k1 / 2)
            k3 = circuit.rhs(t + dt / 2, state + dt * k2 / 2)
            k4 = circuit.rhs(t + dt, state + dt * k3)
            state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        scalar[sample] = state
    assert np.allclose(vector, scalar, rtol=0.0, atol=2e-13)


def test_t2_events_are_causal_deterministic_and_sorted():
    circuit = make_circuit(16, 307, "sparse_pulses")
    out_a, _, events_a = integrate_events(circuit, threshold=0.01, mode="ircn", log_events=True)
    out_b, _, events_b = integrate_events(circuit, threshold=0.01, mode="ircn", log_events=True)
    assert np.array_equal(out_a, out_b)
    assert events_a == events_b
    assert all(float(event["time"]) >= 0.0 for event in events_a)
    assert [float(event["time"]) for event in events_a] == sorted(float(event["time"]) for event in events_a)


def test_t3_tighter_event_threshold_reduces_oracle_error():
    circuit = make_circuit(16, 401, "smooth_periodic")
    reference = integrate_reference(circuit)
    errors = [nrmse(integrate_events(circuit, threshold=value, mode="ircn")[0], reference) for value in (0.01, 0.003, 0.001)]
    assert errors[2] < errors[0], errors


def test_t4_long_run_remains_finite_and_queue_bounded():
    circuit = make_circuit(64, 503, "sparse_pulses")
    result, counts, events = integrate_events(circuit, threshold=0.01, mode="ircn")
    assert result.shape == (1001, 64)
    assert np.isfinite(result).all()
    assert counts["queue_operations"] < 10_000_000
    assert len(events) == 0  # detailed event logs are opt-in


def test_t5_timer_covers_work_performed_inside_call():
    def operation():
        time.sleep(0.02)
        return "done"

    result, elapsed, _ = timed_call(operation)
    assert result == "done"
    assert elapsed >= 0.015


def test_t6_large_local_drift_uses_fallback_refinement():
    assert needs_refinement(0.02, 0.001)
    assert not needs_refinement(0.0001, 0.001)
    circuit = make_circuit(16, 601, "sparse_pulses")
    _result, counts, _events = integrate_events(circuit, threshold=0.0001, mode="ircn")
    assert counts["fallbacks"] > 0
