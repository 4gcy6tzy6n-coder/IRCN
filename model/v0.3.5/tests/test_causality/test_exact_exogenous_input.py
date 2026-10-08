import numpy as np

from dynamics.system import System
from events.ode_segment_scheduler import solve_ode_segments
from solvers.oracle import solve_oracle


class SmoothInput:
    kind = "smooth"

    def __call__(self, t):
        return np.array([np.sin(np.pi * t)])

    def breakpoints(self, t0, t1):
        return []


class PulseInput:
    kind = "sparse_pulse"

    def __call__(self, t):
        return np.array([1.0 if t < 0.02 else 0.0])

    def breakpoints(self, t0, t1):
        return [0.02] if t0 < 0.02 < t1 else []


def _system():
    return System(
        W=np.zeros((1, 1)),
        U=np.ones((1, 1)),
        b=np.array([0.1]),
        tau=np.ones(1),
    )


def _assert_exact_candidate_matches_oracle(signal, times):
    system = _system()
    reference, _ = solve_oracle(system, signal, times, rtol=1e-13, atol=1e-15)
    prediction, _ = solve_ode_segments(
        system,
        signal,
        times,
        segment_tolerance=1e-9,
        max_interval=0.02,
        input_mode="exact_exogenous",
    )
    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    nrmse = float(np.sqrt(np.mean((prediction - reference) ** 2)) / scale)
    assert nrmse < 1e-8


def test_smooth_exogenous_signal_is_evaluated_at_local_solver_stages():
    _assert_exact_candidate_matches_oracle(SmoothInput(), np.linspace(0.0, 1.0, 101))


def test_discontinuous_input_is_split_at_the_exact_falling_edge():
    _assert_exact_candidate_matches_oracle(PulseInput(), np.linspace(0.0, 0.1, 11))
