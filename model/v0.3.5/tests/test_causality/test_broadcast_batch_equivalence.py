import numpy as np

from dynamics.tasks import make_task_system
from events.inputs import make_input
from events.ode_segment_scheduler import solve_ode_segments
from solvers.oracle import solve_oracle


def test_batched_broadcast_preserves_states_and_reduces_heap_operations():
    x, driven_ids = make_input("smooth", 16, 271)
    system = make_task_system("feedback_ring", 16, 271, driven_ids)
    times = np.linspace(0.0, 0.2, 21)
    options = dict(
        segment_tolerance=1e-7,
        max_interval=0.02,
        input_mode="exact_exogenous",
    )

    edge_states, edge_meta = solve_ode_segments(
        system, x, times, delivery_mode="edge_messages", **options
    )
    batch_states, batch_meta = solve_ode_segments(
        system, x, times, delivery_mode="broadcast_batch", **options
    )

    np.testing.assert_allclose(batch_states, edge_states, rtol=1e-12, atol=1e-14)
    assert batch_meta["message_deliveries"] == edge_meta["message_deliveries"]
    assert batch_meta["queue_pushes"] < edge_meta["queue_pushes"]
    delivered = [e for e in batch_meta["event_log"] if e["kind"] == "message"]
    assert delivered
    assert all(e["parent_sequence"] is not None for e in delivered)


def test_batched_local_integrations_match_scalar_local_integrations():
    x, driven_ids = make_input("smooth", 16, 293)
    system = make_task_system("chain", 16, 293, driven_ids)
    times = np.linspace(0.0, 0.2, 21)
    options = dict(
        segment_tolerance=1e-7,
        max_interval=0.02,
        input_mode="exact_exogenous",
        delivery_mode="broadcast_batch",
        local_rtol=1e-9,
        local_atol=1e-11,
    )

    scalar_states, scalar_meta = solve_ode_segments(system, x, times, **options)
    batched_states, batched_meta = solve_ode_segments(
        system, x, times, local_batch=True, **options
    )

    np.testing.assert_allclose(batched_states, scalar_states, rtol=1e-8, atol=2e-12)
    assert batched_meta["rhs_evaluations"] < scalar_meta["rhs_evaluations"]


def test_small_local_batches_retain_accuracy_on_dense_burst_graph():
    x, driven_ids = make_input("dense_burst", 64, 293)
    system = make_task_system("feedback_ring", 64, 293, driven_ids)
    times = np.linspace(0.0, 0.2, 21)
    reference, _ = solve_oracle(system, x, times, rtol=1e-13, atol=1e-15)
    prediction, _ = solve_ode_segments(
        system,
        x,
        times,
        segment_tolerance=1e-7,
        max_interval=0.02,
        input_mode="exact_exogenous",
        delivery_mode="broadcast_batch",
        local_rtol=1e-9,
        local_atol=1e-11,
        local_batch=True,
        local_batch_size=2,
    )

    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    nrmse = float(np.sqrt(np.mean((prediction - reference) ** 2)) / scale)
    assert nrmse < 6.837551017609374e-8
