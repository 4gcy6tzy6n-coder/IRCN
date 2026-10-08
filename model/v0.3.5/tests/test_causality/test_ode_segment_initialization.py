import numpy as np

from dynamics.system import System
from events.ode_segment_scheduler import solve_ode_segments


def test_unforced_nodes_are_initialized_for_bias_dynamics():
    bias = np.array([0.1, -0.2, 0.3])
    system = System(
        W=np.zeros((3, 3)),
        U=np.zeros((3, 3)),
        b=bias,
        tau=np.ones(3),
    )
    zero_input = lambda t: np.zeros(3)
    times = np.array([0.0, 0.01, 0.02])

    predicted, meta = solve_ode_segments(system, zero_input, times, max_interval=0.02)
    expected = np.tanh(bias)[None, :] * (1.0 - np.exp(-times[:, None]))

    np.testing.assert_allclose(predicted, expected, rtol=1e-9, atol=1e-11)
    assert meta["event_count"] >= len(bias)
