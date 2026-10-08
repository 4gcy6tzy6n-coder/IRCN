import numpy as np

from dynamics.system import System
from solvers.oracle import solve_oracle


class StepInput:
    def __call__(self, t):
        return np.array([1.0 if t < 0.5 else 0.0])

    def breakpoints(self, t0, t1):
        return [0.5] if t0 < 0.5 < t1 else []


def test_oracle_uses_left_limit_at_segment_right_endpoint():
    system = System(
        W=np.zeros((1, 1)),
        U=np.ones((1, 1)),
        b=np.zeros(1),
        tau=np.ones(1),
    )
    times = np.array([0.0, 0.5, 1.0])
    values, _ = solve_oracle(system, StepInput(), times, rtol=1e-12, atol=1e-14)

    target = np.tanh(1.0)
    at_switch = target * (1.0 - np.exp(-0.5))
    at_end = at_switch * np.exp(-0.5)
    np.testing.assert_allclose(values[:, 0], [0.0, at_switch, at_end], rtol=1e-10, atol=1e-12)
