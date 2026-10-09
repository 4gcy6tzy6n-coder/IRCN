"""Seed-level metrics and paired uncertainty for CBCC experiments."""
from __future__ import annotations

import math
import numpy as np
from statistics import NormalDist


def paired_summary(a: np.ndarray, b: np.ndarray, seed: int = 8686, reps: int = 10000) -> dict:
    """Paired cluster bootstrap over task seeds; positive means method a is better."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boot = np.mean(d[rng.integers(0, len(d), (reps, len(d)))], axis=1)
    return {"mean_improvement": float(d.mean()), "relative_improvement": float(d.mean() / max(abs(b.mean()), 1e-12)),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "n_seeds": int(len(d))}


def power_n(sd: float, margin: float = 0.05, target_effect: float = 0.10,
            alpha: float = 0.00625, power: float = 0.8,
            minimum: int = 8) -> int:
    """Paired normal approximation for H1 target_effect against the margin null."""
    gap = target_effect - margin
    if gap <= 0:
        raise ValueError("target_effect must exceed the tested success margin")
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("pilot SD must be finite and nonnegative")
    z = NormalDist()
    n = int(math.ceil(((z.inv_cdf(1 - alpha) + z.inv_cdf(power)) * sd / gap) ** 2))
    return max(minimum, n)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(values):
        values = np.asarray(values, float)
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), float)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            out[order[i:j]] = (i + j - 1) / 2 + 1
            i = j
        return out
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
