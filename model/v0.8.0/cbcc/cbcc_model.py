"""Trainable recurrent reservoir with a fitted task readout."""
from __future__ import annotations

import numpy as np
from cbcc_tasks import N_NODES, N_CIRCUITS, N_TICKS

WIDTH = 8


class RecurrentModel:
    def __init__(self, seed: int = 1901) -> None:
        rng = np.random.default_rng(seed)
        self.adjacency = np.zeros((N_NODES, N_NODES), dtype=np.float64)
        # A leaky memory ring preserves weak cues across delayed targets while
        # retaining directed local communication between neighboring nodes.
        for i in range(N_NODES):
            self.adjacency[i, i] = 0.90
            self.adjacency[i, (i - 1) % N_NODES] = 0.04
            self.adjacency[i, (i + 1) % N_NODES] = 0.04
        self.rec = np.eye(WIDTH) + rng.normal(0.0, 0.02, (WIDTH, WIDTH))
        self.input = rng.normal(0.0, 0.55, (1, WIDTH))
        self.readout = np.zeros(N_NODES * WIDTH + 1)
        self.readout_mean = np.zeros(N_NODES * WIDTH + 1)
        self.readout_scale = np.ones(N_NODES * WIDTH + 1)
        self.circuit_readout_norms = np.zeros(N_CIRCUITS)
        self.readout_ridge = 1.0

    def step_circuit(self, h: np.ndarray, x: np.ndarray, circuit: int) -> np.ndarray:
        """Compute only one circuit's candidate state."""
        lo, hi = circuit * 8, (circuit + 1) * 8
        agg = self.adjacency[lo:hi] @ h
        return np.tanh(agg @ self.rec + x[lo:hi, None] @ self.input)

    def features(self, h: np.ndarray) -> np.ndarray:
        return np.r_[h.reshape(-1), 1.0]

    def predict(self, h: np.ndarray) -> float:
        return float(((self.features(h) - self.readout_mean) / self.readout_scale) @ self.readout)

    def fit_readout(self, sequences: list, ridge: float = 1.0) -> None:
        rows, targets = [], []
        for _, seq in sequences:
            h = np.zeros((N_NODES, WIDTH))
            for t in range(N_TICKS):
                old = h.copy()
                for c in range(N_CIRCUITS):
                    lo, hi = c * 8, (c + 1) * 8
                    old[lo:hi] = self.step_circuit(h, seq.x[t], c)
                h = old
                rows.append(self.features(h))
                targets.append(seq.y[t])
        x = np.asarray(rows)
        self.readout_mean = x.mean(axis=0)
        self.readout_scale = x.std(axis=0)
        self.readout_scale[self.readout_scale < 1e-10] = 1.0
        self.readout_mean[-1] = 0.0
        self.readout_scale[-1] = 1.0
        z = (x - self.readout_mean) / self.readout_scale
        self.readout = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ np.asarray(targets))
        self.readout_ridge = float(ridge)
        self.circuit_readout_norms = np.asarray([
            np.linalg.norm(self.readout[c * 8 * WIDTH:(c + 1) * 8 * WIDTH])
            for c in range(N_CIRCUITS)
        ])


def rollout_full(model: RecurrentModel, seq) -> tuple[np.ndarray, np.ndarray]:
    h = np.zeros((N_NODES, WIDTH))
    states, preds = [], []
    for t in range(N_TICKS):
        h = np.vstack([model.step_circuit(h, seq.x[t], c) for c in range(N_CIRCUITS)])
        states.append(h.copy())
        preds.append(model.predict(h))
    return np.asarray(states), np.asarray(preds)
