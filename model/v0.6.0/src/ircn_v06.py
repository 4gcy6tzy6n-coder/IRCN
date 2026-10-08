"""Frozen v0.6.0 synchronous trainability and influence-label bridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


N_NODES = 32
N_TICKS = 16
WIDTH = 8
N_FEATURES = 27
PATH_NODES = 8
TICKS_LABELED = tuple(range(12))
TRAIN_SEEDS = tuple(range(100, 110))
VALIDATION_SEEDS = tuple(range(200, 205))
TEST_SEEDS = tuple(range(300, 310))


def configure_torch() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def graph() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return W[target,source], in/out degree, and directed distance to node 7."""
    w = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for source in range(7):
        w[source + 1, source] = 0.8
    indegree = np.count_nonzero(w, axis=1).astype(np.float64)
    outdegree = np.count_nonzero(w, axis=0).astype(np.float64)
    # Follow source->target edges from i to readout 7.
    distance = np.full(N_NODES, 33, dtype=np.float64)
    for node in range(8):
        distance[node] = 7 - node
    return w, indegree, outdegree, distance


def canonical_array_hash(array: np.ndarray) -> str:
    import hashlib
    import struct

    arr = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", arr.ndim))
    digest.update(struct.pack("<" + "Q" * arr.ndim, *arr.shape))
    digest.update(arr.dtype.str.encode("ascii") + b"\0")
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def generate_input(top_seed: int, trajectory_index: int) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(top_seed), int(trajectory_index)]))
    )
    x = np.zeros((N_TICKS, N_NODES, 1), dtype=np.float64)
    source_count = int(rng.integers(1, 3))
    source_ticks = np.sort(rng.choice(9, size=source_count, replace=False))
    source_amplitudes = rng.uniform(0.25, 1.25, size=source_count)
    x[source_ticks, 0, 0] += source_amplitudes
    distractor_count = int(rng.integers(0, 3))
    for _ in range(distractor_count):
        node = int(rng.integers(8, 32))
        tick = int(rng.integers(0, 16))
        amplitude = float(rng.uniform(-1.0, 1.0))
        x[tick, node, 0] += amplitude
    return x


def generate_teacher(x: np.ndarray) -> np.ndarray:
    w, _, _, _ = graph()
    h = np.zeros((N_TICKS + 1, N_NODES), dtype=np.float64)
    for tick in range(N_TICKS):
        h[tick + 1] = 0.5 * h[tick] + w @ h[tick] + x[tick, :, 0]
    return h


class GatedCell(torch.nn.Module):
    """Shared gated local recurrent cell; aggregate uses W[target,source]."""

    def __init__(self, seed: int = 6060, readout_seed: int = 6064) -> None:
        super().__init__()
        cell_gen = torch.Generator(device="cpu").manual_seed(seed)
        readout_gen = torch.Generator(device="cpu").manual_seed(readout_seed)
        self.wz = torch.nn.Parameter(torch.empty((WIDTH, 2 * WIDTH + 2), dtype=torch.float64))
        self.bz = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float64))
        self.wc = torch.nn.Parameter(torch.empty((WIDTH, 2 * WIDTH + 2), dtype=torch.float64))
        self.bc = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float64))
        self.readout_w = torch.nn.Parameter(torch.empty((WIDTH,), dtype=torch.float64))
        self.readout_b = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.wz, generator=cell_gen)
        torch.nn.init.xavier_uniform_(self.wc, generator=cell_gen)
        torch.nn.init.xavier_uniform_(self.readout_w.reshape(1, -1), generator=readout_gen)

    def step(self, h: torch.Tensor, x: torch.Tensor, w: torch.Tensor, indegree: torch.Tensor) -> torch.Tensor:
        # h: (..., node, width); x: (..., node, 1)
        norm = torch.sqrt(torch.clamp(indegree, min=1.0)).reshape((1,) * (h.ndim - 2) + (-1, 1))
        agg = torch.matmul(w, h) / norm
        log_dt = torch.full_like(x, float(np.log(2.0)))
        q = torch.cat((h, agg, x, log_dt), dim=-1)
        z = torch.sigmoid(torch.nn.functional.linear(q, self.wz, self.bz))
        c = torch.tanh(torch.nn.functional.linear(q, self.wc, self.bc))
        return (1.0 - z) * h + z * c

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sum(h[..., 7, :] * self.readout_w, dim=-1) + self.readout_b


def torch_graph(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    w, indegree, _, _ = graph()
    return (
        torch.as_tensor(w, dtype=torch.float64, device=device),
        torch.as_tensor(indegree, dtype=torch.float64, device=device),
    )


def rollout(model: GatedCell, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All-node synchronous rollout. x shape is (batch,tick,node,1)."""
    if x.ndim != 4 or tuple(x.shape[1:]) != (N_TICKS, N_NODES, 1):
        raise ValueError(f"expected (batch,{N_TICKS},{N_NODES},1), got {tuple(x.shape)}")
    w, indegree = torch_graph(x.device)
    h = torch.zeros((x.shape[0], N_NODES, WIDTH), dtype=torch.float64, device=x.device)
    states = [h]
    readouts = []
    for tick in range(N_TICKS):
        h = model.step(h, x[:, tick], w, indegree)
        states.append(h)
        readouts.append(model.readout(h))
    return torch.stack(states, dim=1), torch.stack(readouts, dim=1)


def feature_rows(
    x: np.ndarray,
    states: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build all 32*12 decision features using current/past values only."""
    if x.shape != (N_TICKS, N_NODES, 1) or states.shape != (N_TICKS + 1, N_NODES, WIDTH):
        raise ValueError("unexpected input or state shape")
    w, indegree, outdegree, distance = graph()
    features: list[np.ndarray] = []
    opportunities: list[tuple[int, int]] = []
    for tick in TICKS_LABELED:
        h = states[tick]
        h_prev = states[tick - 1] if tick else np.zeros_like(h)
        inp_prev = x[tick - 1, :, 0] if tick else np.zeros(N_NODES, dtype=np.float64)
        aggregate = (w @ h) / np.sqrt(np.maximum(1.0, indegree))[:, None]
        msg_change = np.zeros(N_NODES, dtype=np.float64)
        if tick:
            change_norm = np.linalg.norm(h - h_prev, axis=1)
            msg_change = np.abs(w) @ change_norm
        input_change = np.abs(x[tick, :, 0] - inp_prev)
        residual = np.linalg.norm(h - h_prev, axis=1)
        for node in range(N_NODES):
            row = np.concatenate(
                (
                    h[node],
                    aggregate[node],
                    np.asarray(
                        [
                            x[tick, node, 0],
                            input_change[node],
                            msg_change[node],
                            residual[node],
                            1.0,
                            0.0,
                            0.0,
                            indegree[node],
                            outdegree[node],
                            distance[node],
                            float(distance[node] <= 32),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
            if row.shape != (N_FEATURES,):
                raise AssertionError(f"feature shape must be {N_FEATURES}, got {row.shape}")
            features.append(row)
            opportunities.append((node, tick))
    return np.stack(features), opportunities


@torch.no_grad()
def counterfactual_labels(
    model: GatedCell,
    x: np.ndarray,
    states: np.ndarray,
    teacher: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired loss rows and signed single-node UPDATE benefit labels."""
    features, opportunities = feature_rows(x, states)
    starts = np.asarray([tick for _, tick in opportunities], dtype=np.int64)
    nodes = np.asarray([node for node, _ in opportunities], dtype=np.int64)
    base = torch.as_tensor(states[starts], dtype=torch.float64)
    branch = base[:, None, :, :].repeat(1, 2, 1, 1)
    w, indegree = torch_graph()
    x_t = torch.as_tensor(x, dtype=torch.float64)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float64)
    losses = torch.zeros((len(opportunities), 2), dtype=torch.float64)
    hold_mask = torch.zeros((len(opportunities), 1, N_NODES, 1), dtype=torch.bool)
    hold_mask[torch.arange(len(nodes)), 0, torch.as_tensor(nodes), 0] = True
    for horizon_offset in range(4):
        tick_indices = torch.as_tensor(starts + horizon_offset, dtype=torch.long)
        inputs = x_t[tick_indices]  # opportunity,node,1
        old = branch
        updated = model.step(
            branch.reshape(-1, N_NODES, WIDTH),
            inputs[:, None].expand(-1, 2, -1, -1).reshape(-1, N_NODES, 1),
            w,
            indegree,
        ).reshape_as(branch)
        if horizon_offset == 0:
            keep = hold_mask.expand(-1, 2, -1, -1).clone()
            keep[:, 1] = hold_mask[:, 0]
            keep[:, 0] = False
            updated = torch.where(keep, old, updated)
        branch = updated
        predictions = model.readout(branch)
        target = teacher_t[tick_indices + 1, 7]
        losses += (predictions - target[:, None]).square()
    losses /= 4.0
    delta = losses[:, 1] - losses[:, 0]
    return features, losses.cpu().numpy(), delta.cpu().numpy()


def balanced_row_weights(seed_ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Equal total weight per seed, then equal path/isolated weight per seed."""
    seed_ids = np.asarray(seed_ids)
    nodes = np.asarray(nodes)
    seeds = np.unique(seed_ids)
    weights = np.zeros(len(seed_ids), dtype=np.float64)
    path = nodes < PATH_NODES
    for seed in seeds:
        in_seed = seed_ids == seed
        for group, mask in ((True, path), (False, ~path)):
            selected = in_seed & (mask if group else mask)
            if selected.any():
                weights[selected] = 0.5 / (len(seeds) * int(selected.sum()))
    # Normalize to average weight 1 for mini-batch stochastic optimization.
    return weights * (len(weights) / weights.sum())


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def top_precision(scores: np.ndarray, labels: np.ndarray, nodes: np.ndarray, ticks: np.ndarray) -> float:
    n_top = int(np.ceil(0.20 * len(scores)))
    order = np.lexsort((ticks, nodes, -np.asarray(scores)))
    return float(np.mean(np.asarray(labels)[order[:n_top]] > 0.0))


def paired_cluster_bootstrap(
    by_seed_a: np.ndarray,
    by_seed_b: np.ndarray | None = None,
    *,
    replicates: int = 10_000,
    seed: int = 8686,
) -> tuple[float, float]:
    values = np.asarray(by_seed_a, dtype=np.float64)
    if by_seed_b is not None:
        values = values - np.asarray(by_seed_b, dtype=np.float64)
    valid = np.isfinite(values)
    values = values[valid]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Trajectory:
    top_seed: int
    trajectory_index: int
    x: np.ndarray
    teacher: np.ndarray


def trajectories(seeds: Iterable[int]) -> list[Trajectory]:
    return [
        Trajectory(seed, index, x, generate_teacher(x))
        for seed in seeds
        for index in range(32)
        for x in [generate_input(seed, index)]
    ]
