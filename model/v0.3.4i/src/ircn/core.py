"""Synthetic ODE, frozen workloads, solvers, and local event schedulers."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def needs_refinement(predicted_error: float, tolerance: float) -> bool:
    """Return whether a local extrapolation bound requires a smaller RK4 step."""
    return not math.isfinite(predicted_error) or predicted_error > tolerance


@dataclass(frozen=True)
class Workload:
    kind: str
    driven: NDArray[np.int64]
    gain: FloatArray
    phases: FloatArray
    pulse_times: tuple[tuple[float, ...], ...]
    pulse_width: float = 0.05
    pulse_amplitude: float = 1.0
    smooth_frequency: float = 0.25
    smooth_amplitude: float = 0.2

    def value(self, t: float) -> FloatArray:
        x = np.zeros_like(self.gain)
        if self.kind == "smooth_periodic":
            x[self.driven] = self.smooth_amplitude * np.sin(
                2.0 * math.pi * self.smooth_frequency * t + self.phases
            )
        else:
            for k, node in enumerate(self.driven):
                active = any(start <= t < start + self.pulse_width for start in self.pulse_times[k])
                if active:
                    x[node] = self.pulse_amplitude
        return x

    def node_value(self, t: float, node: int) -> float:
        k = int(np.searchsorted(self.driven, node))
        if k >= self.driven.size or int(self.driven[k]) != node:
            return 0.0
        if self.kind == "smooth_periodic":
            return self.smooth_amplitude * math.sin(2.0 * math.pi * self.smooth_frequency * t + float(self.phases[k]))
        return self.pulse_amplitude if any(start <= t < start + self.pulse_width for start in self.pulse_times[k]) else 0.0

    def breakpoints(self, horizon: float) -> tuple[float, ...]:
        points = {0.0, horizon}
        if self.kind == "sparse_pulses":
            for starts in self.pulse_times:
                for start in starts:
                    points.add(start)
                    points.add(min(horizon, start + self.pulse_width))
        return tuple(sorted(points))


@dataclass(frozen=True)
class Circuit:
    n: int
    w: FloatArray  # row i receives from column j
    tau: FloatArray
    gain: FloatArray
    workload: Workload
    seed: int

    def rhs(self, t: float, h: FloatArray) -> FloatArray:
        return (-h + np.tanh(self.w @ h + self.gain * self.workload.value(t))) / self.tau

    def rhs_node(self, node: int, t: float, h: FloatArray) -> float:
        recurrent = float(self.w[node] @ h)
        return (-h[node] + math.tanh(recurrent + self.gain[node] * self.workload.node_value(t, node))) / self.tau[node]


def make_circuit(n: int, seed: int, workload_kind: str, dense: bool = False) -> Circuit:
    graph_seed, node_seed, input_seed = np.random.SeedSequence(seed).spawn(3)
    graph_rng = np.random.default_rng(graph_seed)
    node_rng = np.random.default_rng(node_seed)
    input_rng = np.random.default_rng(input_seed)
    probability = 1.0 if dense else 4.0 / n
    mask = graph_rng.random((n, n)) < probability
    np.fill_diagonal(mask, False)
    w = np.zeros((n, n), dtype=np.float64)
    w[mask] = graph_rng.normal(0.0, 1.0 / math.sqrt(4.0), int(mask.sum()))
    # Scale by spectral radius to keep all generated systems in a bounded regime.
    radius = float(np.max(np.abs(np.linalg.eigvals(w)))) if np.any(w) else 0.0
    if radius > 0.0:
        w *= 0.65 / radius
    tau = np.exp(node_rng.uniform(math.log(0.5), math.log(2.0), n))
    gain = node_rng.normal(0.0, 0.2, n)
    driven = np.sort(input_rng.choice(n, size=max(1, math.ceil(0.05 * n)), replace=False))
    phases = input_rng.uniform(0.0, 2.0 * math.pi, driven.size)
    pulse_times: list[tuple[float, ...]] = []
    for _node in driven:
        count = input_rng.poisson(0.25 * 10.0)
        pulse_times.append(tuple(sorted(input_rng.uniform(0.0, 9.95, count).tolist())))
    workload = Workload(workload_kind, driven, gain, phases, tuple(pulse_times))
    return Circuit(n, w, tau, gain, workload, seed)


def report_times(horizon: float = 10.0, interval: float = 0.01) -> FloatArray:
    count = int(round(horizon / interval))
    return np.linspace(0.0, horizon, count + 1, dtype=np.float64)


def integrate_reference(circuit: Circuit, *, rtol: float = 1e-12, atol: float = 1e-14) -> FloatArray:
    """High-accuracy oracle, segmented at pulse discontinuities."""
    times = report_times()
    h = np.zeros(circuit.n, dtype=np.float64)
    out = np.empty((times.size, circuit.n), dtype=np.float64)
    out[0] = h
    breaks = circuit.workload.breakpoints(float(times[-1]))
    for left, right in zip(breaks[:-1], breaks[1:]):
        ix = np.flatnonzero((times > left) & (times <= right))
        eval_times = times[ix]
        if eval_times.size == 0 or eval_times[-1] != right:
            eval_times = np.append(eval_times, right)

        def segment_rhs(t: float, y: FloatArray) -> FloatArray:
            # The segment ending at a pulse edge uses the left limit there.
            eval_t = np.nextafter(right, left) if circuit.workload.kind == "sparse_pulses" and t >= right else t
            return circuit.rhs(eval_t, y)

        sol = solve_ivp(segment_rhs, (left, right), h, method="DOP853", t_eval=eval_times, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"oracle failed: {sol.message}")
        if ix.size:
            out[ix] = sol.y[:, :ix.size].T
        h = sol.y[:, -1]
    return out


def integrate_adaptive(circuit: Circuit, *, rtol: float = 1e-7, atol: float = 1e-9) -> tuple[FloatArray, dict[str, int]]:
    times = report_times()
    h = np.zeros(circuit.n, dtype=np.float64)
    out = np.empty((times.size, circuit.n), dtype=np.float64)
    out[0] = h
    nfev = 0
    breaks = circuit.workload.breakpoints(float(times[-1]))
    for left, right in zip(breaks[:-1], breaks[1:]):
        ix = np.flatnonzero((times > left) & (times <= right))
        eval_times = times[ix]
        if eval_times.size == 0 or eval_times[-1] != right:
            eval_times = np.append(eval_times, right)

        def segment_rhs(t: float, y: FloatArray) -> FloatArray:
            eval_t = np.nextafter(right, left) if circuit.workload.kind == "sparse_pulses" and t >= right else t
            return circuit.rhs(eval_t, y)

        sol = solve_ivp(segment_rhs, (left, right), h, method="DOP853", t_eval=eval_times, rtol=rtol, atol=atol)
        nfev += sol.nfev
        if not sol.success:
            raise RuntimeError(f"adaptive baseline failed: {sol.message}")
        if ix.size:
            out[ix] = sol.y[:, :ix.size].T
        h = sol.y[:, -1]
    return out, {"rhs_evaluations": nfev}


def integrate_rk4(circuit: Circuit, *, step: float = 0.005) -> tuple[FloatArray, dict[str, int]]:
    times = report_times()
    h = np.zeros(circuit.n, dtype=np.float64)
    out = np.empty((times.size, circuit.n), dtype=np.float64)
    out[0] = h
    nfev = 0
    for k in range(1, times.size):
        left, right = float(times[k - 1]), float(times[k])
        # Split report intervals at pulse edges, preserving discontinuity semantics.
        cuts = [left, *(b for b in circuit.workload.breakpoints(right) if left < b < right), right]
        for a, b in zip(cuts[:-1], cuts[1:]):
            count = max(1, int(math.ceil((b - a) / step)))
            dt = (b - a) / count
            for j in range(count):
                t = a + j * dt
                k1 = circuit.rhs(t, h)
                k2 = circuit.rhs(t + dt / 2, h + dt * k1 / 2)
                k3 = circuit.rhs(t + dt / 2, h + dt * k2 / 2)
                end_t = t + dt
                if circuit.workload.kind == "sparse_pulses" and any(abs(end_t - edge) <= 1e-14 for edge in circuit.workload.breakpoints(float(times[-1]))):
                    end_t = np.nextafter(end_t, t)
                k4 = circuit.rhs(end_t, h + dt * k3)
                h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                nfev += 4
        out[k] = h
    return out, {"rhs_evaluations": nfev}


def _local_rk4(circuit: Circuit, node: int, start: float, end: float, h0: float, drive: float, max_step: float) -> tuple[float, int]:
    if end <= start:
        return h0, 0
    driven_ix = int(np.searchsorted(circuit.workload.driven, node))
    is_driven = driven_ix < circuit.workload.driven.size and int(circuit.workload.driven[driven_ix]) == node
    if circuit.workload.kind == "sparse_pulses" or not is_driven:
        value = float(h0)
        if circuit.workload.kind == "sparse_pulses":
            cuts = [start, *(edge for edge in circuit.workload.breakpoints(end) if start < edge < end), end]
            for left, right in zip(cuts[:-1], cuts[1:]):
                x = circuit.workload.node_value((left + right) / 2, node)
                equilibrium = math.tanh(drive + circuit.gain[node] * x)
                value = equilibrium + (value - equilibrium) * math.exp(-(right - left) / circuit.tau[node])
        else:
            equilibrium = math.tanh(drive)
            value = equilibrium + (value - equilibrium) * math.exp(-(end - start) / circuit.tau[node])
        return value, 0
    count = max(1, int(math.ceil((end - start) / max_step)))
    dt = (end - start) / count
    y = float(h0)
    # Neighbor input is held between its local events; only this node is refined.
    for index in range(count):
        t = start + index * dt
        def f(tt: float, yy: float) -> float:
            x = circuit.workload.node_value(tt, node)
            return (-yy + math.tanh(float(drive) + circuit.gain[node] * x)) / circuit.tau[node]
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)
        y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y, 4 * count


def integrate_events(circuit: Circuit, *, threshold: float, mode: str, max_step: float = 0.005, horizon: float = 10.0, log_events: bool = False) -> tuple[FloatArray, dict[str, int], list[dict[str, float | int | str]]]:
    """Local lazy scheduler. It never forms the all-node RHS as a trigger prefilter."""
    if mode not in {"ircn", "input_event"}:
        raise ValueError(f"unknown event mode {mode}")
    n = circuit.n
    times = report_times(horizon)
    state = np.zeros(n, dtype=np.float64)
    last = np.zeros(n, dtype=np.float64)
    held_drive = np.zeros(n, dtype=np.float64)
    pending_input = np.zeros(n, dtype=np.float64)
    self_token = np.zeros(n, dtype=np.int64)
    event_sequence = 0
    heap: list[tuple[float, int, int, str]] = []
    events: list[dict[str, float | int | str]] = []
    # Build sparse outgoing adjacency once; propagation touches only actual neighbors.
    outgoing = [np.flatnonzero(circuit.w[:, node]).astype(int) for node in range(n)]
    edge_weight = [circuit.w[outgoing[node], node] for node in range(n)]
    rhs_evals = queue_ops = updates = 0
    fallbacks = 0
    snapshots = np.empty((times.size, n), dtype=np.float64)
    snapshots[0] = state

    def push(t: float, node: int, kind: str) -> None:
        nonlocal queue_ops, event_sequence
        event_sequence += 1
        if kind == "self":
            self_token[node] = event_sequence
        heapq.heappush(heap, (float(t), int(node), int(event_sequence), kind))
        queue_ops += 1

    # External forcing is delivered only to the small driven subset.
    if circuit.workload.kind == "sparse_pulses":
        for k, node in enumerate(circuit.workload.driven):
            for start in circuit.workload.pulse_times[k]:
                push(start, int(node), "input")
                push(min(horizon, start + circuit.workload.pulse_width), int(node), "input")
    else:
        source_dt = 0.01
        for node in circuit.workload.driven:
            for t in np.arange(source_dt, horizon + 1e-12, source_dt):
                push(float(t), int(node), "input")

    sample_index = 1
    for sample_t in times[1:]:
        # Sample after all local events at the same timestamp.
        heapq.heappush(heap, (float(sample_t), n, int(sample_index), "sample"))
        queue_ops += 1
        sample_index += 1
    sample_index = 1
    while heap:
        t, node, token, kind = heapq.heappop(heap)
        queue_ops += 1
        if kind == "sample":
            for output_node in range(n):
                snapshots[sample_index, output_node], used = _local_rk4(
                    circuit, output_node, float(last[output_node]), float(t), float(state[output_node]), float(held_drive[output_node]), max_step
                )
                rhs_evals += used
            sample_index += 1
            continue
        if (kind == "self" and token != self_token[node]) or t > horizon + 1e-12:
            continue
        # Materialize only this node under the recurrent drive held since its last event.
        predicted_error = 0.0
        if mode == "ircn":
            local_f_at_start = abs((-state[node] + math.tanh(held_drive[node] + circuit.gain[node] * circuit.workload.node_value(last[node], node))) / circuit.tau[node])
            predicted_error = local_f_at_start * max(0.0, t - last[node])
        refine = mode == "ircn" and needs_refinement(predicted_error, threshold)
        if refine:
            fallbacks += 1
        pending_before = float(pending_input[node])
        held_drive_before = float(held_drive[node])
        new_value, used = _local_rk4(circuit, node, last[node], t, state[node], held_drive[node], max_step / 2 if refine else max_step)
        rhs_evals += used
        delta = new_value - state[node]
        state[node] = new_value
        last[node] = t
        held_drive[node] += pending_input[node]
        pending_input[node] = 0.0
        updates += 1
        if log_events:
            events.append({
                "time": t,
                "node": node,
                "kind": kind,
                "delta": float(delta),
                "pending": pending_before,
                "held_drive_before": held_drive_before,
                "held_drive_after": float(held_drive[node]),
                "fallback_refinement": int(refine),
                "rhs_evaluations": int(used),
            })
        if kind == "input":
            # Input arrivals must update their source even if the recurrent delta is tiny.
            pass
        # Publish this node's state delta to only its outgoing neighbors.
        for target, weight in zip(outgoing[node], edge_weight[node]):
            contribution = float(weight * delta)
            pending_input[target] += contribution
            trigger_value = abs(pending_input[target])
            if mode == "ircn":
                # Local drift estimate from the single target's local state and cached input.
                local_drive = held_drive[target] + pending_input[target]
                local_f = abs((-state[target] + math.tanh(local_drive + circuit.gain[target] * circuit.workload.node_value(t, target))) / circuit.tau[target])
                rhs_evals += 1
                trigger_value += local_f * max(0.0, t - last[target])
            if trigger_value >= threshold:
                push(t, int(target), "neighbor")
        # Schedule only when estimated local drift reaches tolerance. Inactive nodes
        # remain analytically predictable instead of receiving periodic fake events.
        if mode == "ircn":
            local_drive = held_drive[node]
            f = (-state[node] + math.tanh(local_drive + circuit.gain[node] * circuit.workload.node_value(t, node))) / circuit.tau[node]
            rhs_evals += 1
            delay = threshold / max(abs(f), 1e-12)
            if t + delay <= horizon + 1e-12:
                push(t + delay, node, "self")
    if sample_index < times.size:
        snapshots[sample_index:] = state
    return snapshots, {"rhs_evaluations": rhs_evals, "queue_operations": queue_ops, "node_refinements": updates, "fallbacks": fallbacks}, events


def nrmse(actual: FloatArray, reference: FloatArray) -> float:
    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-12)
    return float(np.sqrt(np.mean((actual - reference) ** 2)) / scale)


def timed_call(fn: Callable[[], object]) -> tuple[object, float, int]:
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    return result, elapsed, int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)
