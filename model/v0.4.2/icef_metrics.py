"""Small, explicit ICEF metric functions for Phase II measurement prototyping.

These functions summarize supplied per-opportunity/per-trajectory records. They
do not schedule computation, create oracle labels, or certify scientific validity.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _vector(name: str, x: Sequence[float], *, finite: bool = True) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if finite and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    return a


def _binary_mask(name: str, x: Sequence[bool]) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if a.dtype.kind == "b":
        return a
    if a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must contain only booleans or finite 0/1 values")
    return a.astype(bool)


def compute_selectivity(selected: Sequence[bool], critical: Sequence[bool]) -> dict[str, float]:
    """C1 fractions at the decision-opportunity level; undefined denominators -> NaN."""
    s = _binary_mask("selected", selected)
    c = _binary_mask("critical", critical)
    if s.ndim != 1 or c.ndim != 1 or s.shape != c.shape or s.size == 0:
        raise ValueError("selected and critical must be nonempty, equally sized vectors")
    return {
        "selected_fraction": float(s.mean()),
        "critical_recall": float(s[c].mean()) if c.any() else float("nan"),
        "noncritical_update_rate": float(s[~c].mean()) if (~c).any() else float("nan"),
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def influence_fidelity(
    predicted_benefit: Sequence[float],
    observed_benefit: Sequence[float],
    *,
    k: int,
) -> dict[str, float]:
    """C2 rank/sign/top-k metrics; benefit > 0 means UPDATE reduces task loss."""
    p = _vector("predicted_benefit", predicted_benefit)
    o = _vector("observed_benefit", observed_benefit)
    if p.size == 0 or p.shape != o.shape:
        raise ValueError("predicted and observed benefit must be nonempty and equally sized")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 1 <= k <= p.size:
        raise ValueError("k must be within [1, number of opportunities]")
    pr, orank = _average_ranks(p), _average_ranks(o)
    pred_top = set(np.argsort(-p, kind="mergesort")[:k].tolist())
    true_top = set(np.argsort(-o, kind="mergesort")[:k].tolist())
    actual_sign = np.sign(o)
    predicted_sign = np.sign(p)
    return {
        "spearman": _corr(pr, orank),
        "sign_accuracy": float(np.mean(predicted_sign == actual_sign)),
        "beneficial_sign_accuracy": float(np.mean(predicted_sign[o != 0] == actual_sign[o != 0])) if np.any(o != 0) else float("nan"),
        "top_k_overlap": len(pred_top & true_top) / k,
        "n_opportunities": float(p.size),
    }


def event_latency(event_time: float, first_update_time: float | None) -> dict[str, float | bool]:
    """C3 event-to-first-update delay; missing updates are explicitly censored."""
    if not np.isfinite(event_time):
        raise ValueError("event_time must be finite")
    if first_update_time is None:
        return {"latency": float("nan"), "missed": True}
    if not np.isfinite(first_update_time) or first_update_time < event_time:
        raise ValueError("first update must be finite and no earlier than event")
    return {"latency": float(first_update_time - event_time), "missed": False}


def stable_recovery_time(
    times: Sequence[float], losses: Sequence[float], *, threshold: float, dwell: float
) -> float:
    """C3 first threshold crossing that remains below threshold for dwell; NaN if censored."""
    t, l = _vector("times", times), _vector("losses", losses)
    if t.size == 0 or t.shape != l.shape or np.any(np.diff(t) <= 0):
        raise ValueError("times/losses must be nonempty, equal length, and strictly increasing")
    if not np.isfinite(threshold) or not np.isfinite(dwell) or threshold < 0 or dwell < 0:
        raise ValueError("threshold and dwell must be finite and nonnegative")
    for i in np.flatnonzero(l <= threshold):
        end = np.searchsorted(t, t[i] + dwell, side="left")
        if end <= t.size - 1 and np.all(l[i : end + 1] <= threshold):
            return float(t[i] - t[0])
    return float("nan")


def long_horizon_summary(
    times: Sequence[float], states: np.ndarray, task_loss: Sequence[float], *, state_norm_limit: float
) -> dict[str, float]:
    """C4 diagnostic summary; state distance is not itself a success criterion."""
    t = _vector("times", times)
    loss = _vector("task_loss", task_loss)
    h = np.asarray(states, dtype=float)
    if t.size == 0 or t.shape != loss.shape or h.ndim != 2 or h.shape[0] != t.size or h.shape[1] == 0:
        raise ValueError("states must have shape (len(times), state_width), matching task_loss")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    if not np.all(np.isfinite(h)) or not np.isfinite(state_norm_limit) or state_norm_limit <= 0:
        raise ValueError("states must be finite and state_norm_limit finite and positive")
    norms = np.linalg.norm(h, axis=1)
    return {
        "horizon": float(t[-1] - t[0]),
        "mean_task_loss": float(loss.mean()),
        "final_task_loss": float(loss[-1]),
        "max_state_norm": float(norms.max()),
        "state_bound_violation_fraction": float(np.mean(norms > state_norm_limit)),
    }


def pareto_frontier(quality: Sequence[float], cost: Sequence[float]) -> np.ndarray:
    """C5 indices not dominated when quality is maximized and cost minimized."""
    q, c = _vector("quality", quality), _vector("cost", cost)
    if q.size == 0 or q.shape != c.shape or np.any(c < 0):
        raise ValueError("quality/cost must be nonempty equal vectors and cost nonnegative")
    keep = np.ones(q.size, dtype=bool)
    for i in range(q.size):
        dominates_i = (q >= q[i]) & (c <= c[i]) & ((q > q[i]) | (c < c[i]))
        if dominates_i.any():
            keep[i] = False
    return np.flatnonzero(keep)


def paired_structure_contrast(target: Sequence[float], control: Sequence[float]) -> dict[str, float]:
    """C6 paired per-seed outcome contrast, target minus control."""
    a, b = _vector("target", target), _vector("control", control)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError("target and control must be nonempty paired vectors")
    d = a - b
    return {"n_pairs": float(d.size), "mean_difference": float(d.mean()), "median_difference": float(np.median(d))}
