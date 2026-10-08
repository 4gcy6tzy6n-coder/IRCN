"""Stateful discrete diagnostic runners for the ICEF toy environments.

This is a metric-validation harness, not the proposed learned IRCN model. All
methods share task transitions and exact opportunity budgets. No wall-clock or
cross-hardware claim is made from Python operation counts.
"""
from __future__ import annotations

import numpy as np

from diagnostic_tasks import (
    budget_matched_schedule,
    feedback_memory,
    mean_squared_error,
    multiscale_integration,
    sparse_propagation,
)


def run_sparse(mask: np.ndarray, *, steps: int = 16, n_nodes: int = 32) -> dict[str, object]:
    """Apply selected node transitions from a per-step state snapshot."""
    task = sparse_propagation(steps, n_nodes)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != task["inputs"].shape:
        raise ValueError("mask must have shape (steps, nodes)")
    states = np.zeros_like(task["states"])
    for t in range(steps):
        old = states[t]
        nxt = old.copy()
        for i in np.flatnonzero(selected[t]):
            nxt[i] = 0.5 * old[i] + task["weights"][i] @ old + task["inputs"][t, i]
        states[t + 1] = nxt
    return {
        "states": states,
        "output": states[:, 7],
        "loss": mean_squared_error(states[:, 7], task["output"]),
        "updates": int(selected.sum()),
        "selected": selected,
    }


def run_multiscale(mask: np.ndarray, *, steps: int = 200) -> dict[str, object]:
    task = multiscale_integration(steps)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (steps, 2):
        raise ValueError("mask must have shape (steps, 2): slow then fast")
    slow = np.zeros(steps + 1)
    fast = np.zeros(steps + 1)
    for t in range(steps):
        slow[t + 1] = slow[t]
        fast[t + 1] = fast[t]
        if selected[t, 0]:
            slow[t + 1] = 0.98 * slow[t] + 0.02 * task["slow_input"][t]
        if selected[t, 1]:
            fast[t + 1] = 0.5 * fast[t] + 0.5 * task["fast_input"][t]
    target = slow + fast
    return {
        "slow": slow,
        "fast": fast,
        "prediction": target,
        "loss": mean_squared_error(target, task["target"]),
        "updates": int(selected.sum()),
        "selected": selected,
    }


def run_memory(mask: np.ndarray, cue: int, delay: int) -> dict[str, object]:
    task = feedback_memory(cue, delay)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (delay + 2, 1):
        raise ValueError("mask must have shape (delay + 2, 1)")
    memory = np.zeros(delay + 2)
    for t in range(delay + 2):
        if selected[t, 0] and t == 0:
            memory[t] = cue
        elif t > 0:
            memory[t] = memory[t - 1]
    output = memory[int(task["query_index"])]
    return {
        "memory": memory,
        "output": float(output),
        "loss": float((output - float(task["target"])) ** 2),
        "updates": int(selected.sum()),
        "selected": selected,
    }


def masks_for_methods(
    activity: np.ndarray, critical: np.ndarray, budget: int, *, seed: int = 0
) -> dict[str, np.ndarray]:
    """Return equal-budget controls and an explicitly non-deployable oracle."""
    return {
        method: budget_matched_schedule(
            activity, budget, method=method, seed=seed,
            critical=critical if method == "oracle" else None,
        )
        for method in ("periodic", "random", "activity", "oracle")
    }


def compare_sparse_methods(*, budget: int = 128, seed: int = 0) -> dict[str, dict[str, object]]:
    task = sparse_propagation()
    critical = np.zeros_like(task["inputs"], dtype=bool)
    critical[:, :8] = True
    masks = masks_for_methods(np.abs(task["inputs"]), critical, budget, seed=seed)
    masks["update_all"] = np.ones_like(critical)
    return {name: run_sparse(mask) for name, mask in masks.items()}


def compare_memory_methods(cue: int = 1, delay: int = 20, *, budget: int = 1, seed: int = 0) -> dict[str, dict[str, object]]:
    n = delay + 2
    activity = np.zeros((n, 1))
    activity[0, 0] = float(cue)
    critical = np.zeros((n, 1), dtype=bool)
    critical[0, 0] = True
    masks = masks_for_methods(activity, critical, budget, seed=seed)
    masks["update_all"] = np.ones_like(critical)
    return {name: run_memory(mask, cue, delay) for name, mask in masks.items()}
