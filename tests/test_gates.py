from __future__ import annotations

import time
import os
import subprocess
import sys

import numpy as np

from ircn.core import Circuit, Workload, integrate_adaptive, integrate_events, integrate_reference, integrate_rk4, make_circuit, needs_refinement, nrmse, report_times, timed_call


def test_t0_determinism_same_seed_recreates_inputs_and_trajectory():
    a = make_circuit(16, 101, "smooth_periodic")
    b = make_circuit(16, 101, "smooth_periodic")
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.tau, b.tau)
    dense = make_circuit(16, 101, "smooth_periodic", dense=True)
    assert np.count_nonzero(dense.w) == 16 * 15
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


def test_t1_single_node_event_degeneracy_matches_oracle():
    workload = Workload("sparse_pulses", np.array([0]), np.array([1.0]), np.array([0.0]), ((0.002,),), pulse_width=0.005)
    circuit = Circuit(1, np.zeros((1, 1)), np.ones(1), np.ones(1), workload, 78)
    reference = integrate_reference(circuit)
    event_result, _counts, _events = integrate_events(circuit, threshold=1e-3, mode="ircn")
    assert np.max(np.abs(event_result - reference)) < 1e-8


def test_t1_coupled_full_update_limit_tracks_synchronous_rk4():
    workload = Workload("smooth_periodic", np.array([0]), np.array([0.2, 0.2]), np.array([0.7]), (), smooth_amplitude=0.2)
    circuit = Circuit(2, np.array([[0.0, 0.2], [0.2, 0.0]]), np.ones(2), np.array([0.2, 0.2]), workload, 781)
    event_result, counts, events = integrate_events(circuit, threshold=1e-5, mode="ircn", log_events=True)
    synchronous, _counts = integrate_rk4(circuit)
    assert counts["node_refinements"] > 0
    assert {int(event["node"]) for event in events} == {0, 1}
    assert np.max(np.abs(event_result - synchronous)) < 3e-6


def test_t2_events_are_causal_deterministic_and_sorted():
    circuit = make_circuit(16, 307, "sparse_pulses")
    out_a, _, events_a = integrate_events(circuit, threshold=0.01, mode="ircn", log_events=True)
    out_b, _, events_b = integrate_events(circuit, threshold=0.01, mode="ircn", log_events=True)
    assert np.array_equal(out_a, out_b)
    assert events_a == events_b
    assert all(float(event["time"]) >= 0.0 for event in events_a)
    assert [float(event["time"]) for event in events_a] == sorted(float(event["time"]) for event in events_a)
    assert any(event["kind"] == "input" for event in events_a)


def test_t2_pulse_oracle_advances_through_segments_without_output_samples():
    workload = Workload("sparse_pulses", np.array([0]), np.array([1.0]), np.array([0.0]), ((0.002,),), pulse_width=0.005)
    circuit = Circuit(1, np.zeros((1, 1)), np.ones(1), np.ones(1), workload, 77)
    expected = np.tanh(1.0) * (1.0 - np.exp(-0.005)) * np.exp(-0.003)
    oracle = integrate_reference(circuit)
    adaptive, _ = integrate_adaptive(circuit, rtol=1e-11, atol=1e-13)
    fixed, _ = integrate_rk4(circuit, step=0.00025)
    assert abs(oracle[1, 0] - expected) < 1e-12
    assert abs(adaptive[1, 0] - expected) < 1e-10
    assert abs(fixed[1, 0] - expected) < 1e-9


def test_t2_neighbor_message_is_applied_after_causal_arrival():
    workload = Workload("sparse_pulses", np.array([0]), np.array([1.0, 0.0]), np.array([0.0]), ((0.002,),), pulse_width=0.05)
    circuit = Circuit(2, np.array([[0.0, 0.0], [0.5, 0.0]]), np.ones(2), np.array([1.0, 0.0]), workload, 308)
    _result, _counts, events = integrate_events(circuit, threshold=0.01, mode="ircn", log_events=True)
    source_publications = [float(e["time"]) for e in events if e["node"] == 0 and float(e["delta"]) != 0.0]
    target_updates = [e for e in events if e["node"] == 1 and float(e["pending"]) != 0.0]
    assert source_publications and target_updates
    first_source_publication = min(source_publications)
    assert all(float(e["time"]) >= first_source_publication for e in target_updates)
    for event in target_updates:
        assert abs((float(event["held_drive_after"]) - float(event["held_drive_before"])) - float(event["pending"])) < 1e-12


def test_t3_tighter_event_threshold_reduces_oracle_error():
    for workload in ("sparse_pulses", "smooth_periodic"):
        circuit = make_circuit(16, 401, workload)
        reference = integrate_reference(circuit)
        errors = [nrmse(integrate_events(circuit, threshold=value, mode="ircn")[0], reference) for value in (0.003, 0.001, 0.0003)]
        assert errors[0] >= errors[1] >= errors[2], (workload, errors)


def test_t4_long_run_remains_finite_and_queue_bounded():
    circuit = make_circuit(64, 503, "sparse_pulses")
    result, counts, events = integrate_events(circuit, threshold=0.01, mode="ircn", horizon=60.0)
    assert result.shape == (6001, 64)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1.0
    assert counts["queue_operations"] < 10_000_000
    assert len(events) == 0  # detailed event logs are opt-in


def test_t4_quiescent_nodes_do_not_receive_periodic_fake_events():
    workload = Workload("sparse_pulses", np.array([], dtype=int), np.array([0.0, 0.0]), np.array([]), ())
    circuit = Circuit(2, np.zeros((2, 2)), np.ones(2), np.zeros(2), workload, 79)
    result, counts, _ = integrate_events(circuit, threshold=0.001, mode="ircn")
    assert np.array_equal(result, np.zeros_like(result))
    assert counts["node_refinements"] == 0


def test_t5_timer_covers_work_performed_inside_call():
    def operation():
        time.sleep(0.02)
        return "done"

    result, elapsed, _ = timed_call(operation)
    assert result == "done"
    assert elapsed >= 0.015
    circuit = make_circuit(8, 601, "smooth_periodic")
    (trajectory, _counts), solver_elapsed, _rss = timed_call(lambda: integrate_rk4(circuit))
    assert trajectory.shape == (1001, 8)  # result collection is inside the timed call
    assert solver_elapsed > 0.0
    env = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")}
    probe = subprocess.run(
        [sys.executable, "-c", "from ircn import cli; import os; print('|'.join(os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')))"],
        env=env, text=True, capture_output=True, check=True,
    )
    assert probe.stdout.strip() == "1|1|1"


def test_t6_large_local_drift_uses_fallback_refinement():
    assert needs_refinement(0.02, 0.001)
    assert not needs_refinement(0.0001, 0.001)
    workload = Workload("smooth_periodic", np.array([0]), np.array([1.0]), np.array([np.pi / 2]), (), smooth_frequency=10.0, smooth_amplitude=0.2)
    circuit = Circuit(1, np.zeros((1, 1)), np.ones(1), np.ones(1), workload, 601)
    reference = integrate_reference(circuit)
    result, counts, events = integrate_events(circuit, threshold=0.0001, mode="ircn", max_step=0.01, log_events=True)
    baseline, _baseline_counts, _baseline_events = integrate_events(circuit, threshold=0.0001, mode="input_event", max_step=0.01)
    assert counts["fallbacks"] > 0
    assert any(int(event["fallback_refinement"]) == 1 and int(event["rhs_evaluations"]) == 8 for event in events)
    assert np.max(np.abs(result - reference)) < np.max(np.abs(baseline - reference))
