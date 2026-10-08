"""Deterministic generators for ICEF known-answer diagnostic environments.

These generate reference trajectories only; they do not implement IRCN or a
scheduler and must not be used as capability evidence.
"""
from __future__ import annotations

import numpy as np


def sparse_propagation(steps: int = 16, n_nodes: int = 32) -> dict[str, np.ndarray]:
    if steps < 2 or n_nodes < 8:
        raise ValueError("steps >= 2 and n_nodes >= 8 are required")
    weights = np.zeros((n_nodes, n_nodes), dtype=float)  # target, source
    for i in range(7):
        weights[i + 1, i] = 0.8
    inputs = np.zeros((steps, n_nodes), dtype=float)
    inputs[0, 0] = 1.0
    states = np.zeros((steps + 1, n_nodes), dtype=float)
    for t in range(steps):
        states[t + 1] = 0.5 * states[t] + weights @ states[t] + inputs[t]
    return {"weights": weights, "inputs": inputs, "states": states, "output": states[:, 7].copy()}


def multiscale_integration(steps: int = 200, *, fast_period: int = 2, slow_period: int = 20) -> dict[str, np.ndarray]:
    if steps < 2 or fast_period < 1 or slow_period < 1:
        raise ValueError("steps and periods must be positive, with steps >= 2")
    t = np.arange(steps)
    a = np.where((t // slow_period) % 2 == 0, 1.0, -1.0)
    b = np.where((t // fast_period) % 2 == 0, 1.0, -1.0)
    slow = np.zeros(steps + 1)
    fast = np.zeros(steps + 1)
    for i in range(steps):
        slow[i + 1] = 0.98 * slow[i] + 0.02 * a[i]
        fast[i + 1] = 0.5 * fast[i] + 0.5 * b[i]
    return {"slow_input": a, "fast_input": b, "slow": slow, "fast": fast, "target": slow + fast}


def feedback_memory(cue: int, delay: int) -> dict[str, np.ndarray]:
    if cue not in (0, 1) or delay < 1:
        raise ValueError("cue must be binary and delay >= 1")
    memory = np.full(delay + 2, float(cue))
    times = np.arange(delay + 2, dtype=int)
    query_index = delay + 1
    return {
        "memory": memory,
        "times": times,
        "query_index": np.array(query_index),
        "target": np.array(float(cue)),
        "query_output": np.array(memory[query_index]),
    }


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    p, y = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if p.shape != y.shape or p.size == 0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
        raise ValueError("prediction and target must be finite, nonempty, and have equal shape")
    return float(np.mean((p - y) ** 2))


def budget_matched_schedule(
    activity: np.ndarray,
    budget: int,
    *,
    method: str,
    seed: int = 0,
    critical: np.ndarray | None = None,
) -> np.ndarray:
    """Build opportunity masks for diagnostic controls, not a runtime scheduler.

    Methods are periodic, random, activity, and oracle. Every method selects
    exactly ``budget`` opportunities; oracle requires a supplied known-answer
    critical mask and is an upper-bound control only.
    """
    a = np.asarray(activity, dtype=float)
    if a.ndim != 2 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("activity must be a finite nonempty (time, node) matrix")
    total = a.size
    if not 0 <= budget <= total:
        raise ValueError("budget must be between zero and all opportunities")
    if method not in {"periodic", "random", "activity", "oracle"}:
        raise ValueError("unknown method")
    if method == "oracle":
        c = np.asarray(critical, dtype=bool) if critical is not None else None
        if c is None or c.shape != a.shape:
            raise ValueError("oracle requires a same-shaped known-answer critical mask")
        score = c.astype(float).ravel()
    else:
        score = a.ravel()
    if method == "periodic":
        selected = np.floor((np.arange(budget) + 0.5) * total / max(1, budget)).astype(int) if budget else np.array([], dtype=int)
    elif method == "random":
        selected = np.random.default_rng(seed).choice(total, size=budget, replace=False)
    else:
        selected = np.argsort(-score, kind="mergesort")[:budget]
    mask = np.zeros(total, dtype=bool)
    mask[selected] = True
    return mask.reshape(a.shape)
