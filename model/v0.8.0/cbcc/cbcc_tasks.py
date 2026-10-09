"""Small deterministic synthetic task families for circuit-budget studies."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

N_NODES = 32
N_CIRCUITS = 4
N_TICKS = 24


@dataclass(frozen=True)
class Sequence:
    x: np.ndarray  # (tick, node)
    y: np.ndarray  # (tick,)
    key_circuit: int
    cue_tick: int
    switch_tick: int | None
    target_onset: int


def generate(family: str, seed: int, index: int) -> Sequence:
    """Return one independent sequence; key signal is weak, distractors are strong."""
    if family not in {"sparse_key", "feedback_switch"}:
        raise ValueError(f"unknown task family: {family}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
    x = np.zeros((N_TICKS, N_NODES), dtype=np.float64)
    y = np.zeros(N_TICKS, dtype=np.float64)
    circuit = int(rng.integers(N_CIRCUITS))
    key = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.35))
    cue = int(rng.integers(2, 6))
    start = circuit * 8
    x[cue, start:start + 8] = key
    if family == "sparse_key":
        # Strong activity is task-irrelevant; the weak cue determines a delayed output.
        for _ in range(8):
            t, c = int(rng.integers(N_TICKS)), int(rng.integers(N_CIRCUITS))
            if c != circuit:
                x[t, c * 8:(c + 1) * 8] += rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.0)
        y[cue + 6:cue + 10] = key
    else:
        # A context switch makes a feedback-dependent latent sign relevant later.
        switch = int(rng.integers(8, 12))
        x[switch, circuit * 8:(circuit + 1) * 8] += 0.25
        y[switch + 5:] = key
        for _ in range(10):
            t, c = int(rng.integers(N_TICKS)), int(rng.integers(N_CIRCUITS))
            if c != circuit:
                x[t, c * 8:(c + 1) * 8] += rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
    switch_tick = None if family == "sparse_key" else switch
    target_onset = cue + 6 if family == "sparse_key" else switch + 5
    return Sequence(x, y, circuit, cue, switch_tick, target_onset)


def dataset(family: str, seeds: tuple[int, ...], n_per_seed: int = 8) -> list[tuple[int, Sequence]]:
    return [(seed, generate(family, seed, i)) for seed in seeds for i in range(n_per_seed)]
