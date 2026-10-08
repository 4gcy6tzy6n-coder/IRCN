import numpy as np

from dynamics.tasks import make_task_system
from events.inputs import make_input
from events.ode_segment_scheduler import solve_ode_segments
from solvers.oracle import solve_oracle


def test_expired_segment_successor_is_published_to_downstream_nodes():
    x, driven_ids = make_input("sparse_pulse", 16, 101)
    system = make_task_system("chain", 16, 101, driven_ids)
    times = np.linspace(0.0, 0.2, 21)
    oracle, _ = solve_oracle(system, x, times, rtol=1e-13, atol=1e-15)

    predicted, meta = solve_ode_segments(
        system, x, times, segment_tolerance=1e-3, max_interval=0.002
    )
    scale = max(float(np.sqrt(np.mean(oracle**2))), 1e-12)
    nrmse = float(np.sqrt(np.mean((predicted - oracle) ** 2)) / scale)

    assert nrmse < 1e-2
    assert meta["event_count"] > 16


def test_feedback_loop_does_not_republish_zero_length_segments_at_terminal_time():
    x, driven_ids = make_input("sparse_pulse", 16, 101)
    system = make_task_system("feedback_ring", 16, 101, driven_ids)
    times = np.linspace(0.0, 0.2, 21)

    predicted, meta = solve_ode_segments(
        system, x, times, segment_tolerance=1e-3, max_interval=0.05
    )

    assert np.isfinite(predicted).all()
    assert meta["event_count"] < 20_000
