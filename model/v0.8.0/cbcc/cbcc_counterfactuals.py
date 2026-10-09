"""Paired circuit update/skip labels; each pair shares an identical start state."""
from __future__ import annotations

import numpy as np
from cbcc_model import N_CIRCUITS, N_TICKS, WIDTH, rollout_full

CF_HORIZON = 15


def online_features(model, seq, states: np.ndarray, tick: int, circuit: int) -> np.ndarray:
    lo, hi = circuit * 8, (circuit + 1) * 8
    h = states[tick - 1] if tick else np.zeros_like(states[0])
    prev_x = seq.x[tick - 1] if tick else np.zeros_like(seq.x[0])
    block = h[lo:hi]
    return np.r_[block.mean(), block.std(), np.linalg.norm(block),
                 model.circuit_readout_norms[circuit], np.mean(np.abs(seq.x[tick, lo:hi])),
                 np.mean(np.abs(seq.x[tick, lo:hi] - prev_x[lo:hi])),
                 np.linalg.norm(block - (states[tick - 2, lo:hi] if tick > 1 else 0)),
                 tick / N_TICKS, circuit / N_CIRCUITS]


def paired_labels(model, seq, horizon: int = CF_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    states, _ = rollout_full(model, seq)
    rows, labels = [], []
    for t in range(N_TICKS):
        lookahead = min(horizon, N_TICKS - t)
        for c in range(N_CIRCUITS):
            start = (states[t - 1] if t else np.zeros_like(states[0])).copy()
            branch_means = []
            for update in (True, False):
                h = start.copy()
                branch_losses = []
                for k in range(t, t + lookahead):
                    if k == t:
                        circuits = range(N_CIRCUITS) if update else (j for j in range(N_CIRCUITS) if j != c)
                    else:
                        circuits = range(N_CIRCUITS)
                    old = h.copy()
                    for j in circuits:
                        lo, hi = j * 8, (j + 1) * 8
                        old[lo:hi] = model.step_circuit(h, seq.x[k], j)
                    h = old
                    branch_losses.append((model.predict(h) - seq.y[k]) ** 2)
                branch_means.append(float(np.mean(branch_losses)))
            rows.append(online_features(model, seq, states, t, c))
            labels.append(branch_means[1] - branch_means[0])
    return np.asarray(rows), np.asarray(labels)


def joint_benefit(model, seq, tick: int, circuits: tuple[int, ...], horizon: int = CF_HORIZON, states: np.ndarray | None = None) -> float:
    """Paired benefit from updating a circuit set together at one decision tick."""
    if states is None:
        states, _ = rollout_full(model, seq)
    start = (states[tick - 1] if tick else np.zeros_like(states[0])).copy()
    losses = []
    for update in (True, False):
        h = start.copy()
        branch_losses = []
        for k in range(tick, tick + horizon):
            old = h.copy()
            selected = range(N_CIRCUITS) if update or k > tick else (c for c in range(N_CIRCUITS) if c not in circuits)
            for c in selected:
                lo, hi = c * 8, (c + 1) * 8
                old[lo:hi] = model.step_circuit(h, seq.x[k], c)
            h = old
            branch_losses.append((model.predict(h) - seq.y[k]) ** 2)
        losses.append(float(np.mean(branch_losses)))
    return losses[1] - losses[0]
