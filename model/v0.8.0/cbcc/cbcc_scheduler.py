"""Budgeted circuit scheduler; scoring uses decision-time features only."""
from __future__ import annotations

import numpy as np
from cbcc_model import N_CIRCUITS, N_TICKS
from cbcc_counterfactuals import online_features

POLICIES = ("cbcc", "random", "fixed_frequency", "activity", "no_influence")


def fit_influence(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    mean, scale = features.mean(0), features.std(0)
    scale[scale < 1e-10] = 1.0
    z = (features - mean) / scale
    coef = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ labels)
    return np.r_[mean, scale, coef]


def score(model, estimator: np.ndarray, seq, states: np.ndarray, tick: int, circuit: int) -> float:
    f = online_features(model, seq, states, tick, circuit)
    d = len(f)
    mean, scale, coef = estimator[:d], estimator[d:2*d], estimator[2*d:]
    return float(((f - mean) / scale) @ coef)


def mac_costs(model, estimator: np.ndarray) -> tuple[int, int]:
    # One block: adjacency @ state, recurrent projection, and input projection.
    update = 8 * (32 * 8 + 8 * 8 + 8)
    # Count 3 squared-vector summaries per circuit (std, norm, residual) plus
    # one 9-feature linear score per circuit; static readout norms are precomputed.
    estimate = N_CIRCUITS * (3 * 8 + len(estimator) // 3)
    return update, estimate


def choose(policy: str, model, estimator, seq, states, tick, budget, rng):
    """Return circuit ids and measured-MAC proxy spent; no candidate state precomputation."""
    update_mac, estimator_mac = mac_costs(model, estimator)
    if policy == "cbcc":
        overhead = estimator_mac
        if budget < overhead + update_mac:
            return [], 0
        values = [score(model, estimator, seq, states, tick, c) for c in range(N_CIRCUITS)]
        order = sorted(range(N_CIRCUITS), key=lambda c: (-values[c], c))
    elif policy == "activity":
        # Cheap observable input/change activity proxy; no candidate states are computed.
        activity = [np.mean(np.abs(seq.x[tick, c*8:(c+1)*8])) +
                    (np.linalg.norm(states[tick-1, c*8:(c+1)*8] - states[tick-2, c*8:(c+1)*8]) if tick > 1 else 0.0)
                    for c in range(N_CIRCUITS)]
        order = sorted(range(N_CIRCUITS), key=lambda c: (-activity[c], c))
        overhead = 4 * 8  # 32 square MACs in four circuit residual norms.
    elif policy == "fixed_frequency":
        start = tick % N_CIRCUITS
        order = [(start + offset) % N_CIRCUITS for offset in range(N_CIRCUITS)]
        overhead = 0
    elif policy == "random":
        order = list(rng.permutation(N_CIRCUITS))
        overhead = 0
    elif policy == "no_influence":
        order = sorted(
            range(N_CIRCUITS), key=lambda c: (
                -(model.circuit_readout_norms[c] * np.linalg.norm(states[tick-1, c*8:(c+1)*8])) if tick else 0.0, c))
        # Four 8-value squared-norm summaries and four scalar readout products.
        overhead = 4 * 8 + N_CIRCUITS
    else:
        raise ValueError(policy)
    available_updates = max(0, (budget - overhead) // update_mac)
    selected = [c for c in order if values[c] > 0][:available_updates] if policy == "cbcc" else order[:available_updates]
    if policy == "activity":
        selected = [order[0]] if activity[order[0]] > 0.05 and available_updates > 0 else []
    return selected, overhead + len(selected) * update_mac
