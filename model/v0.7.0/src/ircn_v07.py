"""Tick-synchronous local event runtime bridge for the frozen IRCN v0.6 cell.

This module establishes execution semantics only. It does not train a policy
or claim a runtime advantage.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from numbers import Real
import resource
import time
from typing import Callable, Mapping, Sequence

import numpy as np


NODE_COUNT = 32
WIDTH = 8
READOUT_NODE = 7
MAX_WAIT_TICKS = 1
MAX_HOLD_TICKS = 5
MAX_CONSECUTIVE_HOLDS = 5
PHASES = ("DELIVER", "INPUT", "CANDIDATE", "DECIDE", "COMMIT", "PUBLISH", "READOUT")
FEATURE_NAMES = (
    *(f"state_{i}" for i in range(WIDTH)),
    *(f"aggregate_{i}" for i in range(WIDTH)),
    "input", "input_change", "aggregate_change", "prior_residual", "elapsed",
    "holds", "deadline", "in_degree", "out_degree", "distance", "reachable",
)


def _float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen W[target, source] and directed degree vectors."""
    w = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    for src in range(7):
        w[src + 1, src] = 0.8
    return w, np.count_nonzero(w, axis=1), np.count_nonzero(w, axis=0)


def sparse_input_events(inputs: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    """Convert a frozen dense trajectory into explicit sparse value changes.

    Conversion belongs to the input provider, outside runtime timing. Zero
    reset events are included whenever a previously nonzero input returns zero.
    """
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (NODE_COUNT, 1) or not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite with shape (ticks, 32, 1)")
    previous = np.zeros(NODE_COUNT, dtype=np.float64)
    events: list[tuple[int, int, float]] = []
    for tick in range(x.shape[0]):
        row = x[tick, :, 0]
        changed = np.flatnonzero(row != previous)
        for node in changed:
            events.append((tick, int(node), float(row[node])))
        previous = row.copy()
    return tuple(events)


@dataclass(frozen=True)
class Decision:
    action: str

    def __post_init__(self) -> None:
        if self.action not in ("HOLD", "UPDATE"):
            raise ValueError("action must be HOLD or UPDATE")


@dataclass(frozen=True)
class RuntimePolicyFeaturesV1:
    node_id: int
    tick: int
    values: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("node_id", self.node_id), ("tick", self.tick)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.shape != (27,) or not np.all(np.isfinite(arr)):
            raise ValueError("policy features must be a finite vector of length 27")
        arr = arr.copy()
        arr.setflags(write=False)
        object.__setattr__(self, "values", arr)


Policy = Callable[[RuntimePolicyFeaturesV1], Decision]


def load_cell(checkpoint: str | Path, expected_sha256: str) -> dict[str, np.ndarray]:
    """Load the frozen Torch state dict after validating its file hash."""
    path = Path(checkpoint)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_shapes = {"wz": (8, 18), "bz": (8,), "wc": (8, 18), "bc": (8,),
                       "readout_w": (8,), "readout_b": ()}
    if set(state) != set(expected_shapes):
        raise ValueError("checkpoint has unexpected parameter names")
    result: dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        arr = state[key].detach().cpu().numpy().astype(np.float64, copy=True)
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"checkpoint parameter {key} has invalid shape or values")
        result[key] = arr
    return result


def local_transition(state: np.ndarray, aggregate: np.ndarray, value: float,
                     params: Mapping[str, np.ndarray]) -> np.ndarray:
    """Exact v0.6 local GatedCell operation for one node and dt=1."""
    h = np.asarray(state, dtype=np.float64)
    agg = np.asarray(aggregate, dtype=np.float64)
    x = _float(value, "input")
    if h.shape != (WIDTH,) or agg.shape != (WIDTH,) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(agg)):
        raise ValueError("state and aggregate must be finite width-8 vectors")
    q = np.concatenate((h, agg, np.asarray([x, math.log(2.0)])))
    z = 1.0 / (1.0 + np.exp(-(params["wz"] @ q + params["bz"])))
    c = np.tanh(params["wc"] @ q + params["bc"])
    result = (1.0 - z) * h + z * c
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("cell transition produced a non-finite state")
    return result


class TickRuntime:
    """Synchronous commit runtime with event-local candidate selection."""

    def __init__(self, params: Mapping[str, np.ndarray], policy: Policy | None = None,
                 *, node_count: int = NODE_COUNT, initial_states: np.ndarray | None = None):
        init_started = time.perf_counter()
        if node_count != NODE_COUNT:
            raise ValueError("v0.7 frozen runtime requires 32 nodes")
        if policy is not None and not callable(policy):
            raise ValueError("policy must be callable")
        shapes = {"wz": (8, 18), "bz": (8,), "wc": (8, 18), "bc": (8,),
                  "readout_w": (8,), "readout_b": ()}
        if set(params) != set(shapes):
            raise ValueError("cell parameters have unexpected names")
        self.params = {}
        for key, shape in shapes.items():
            value = np.asarray(params[key], dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"cell parameter {key} has invalid shape or values")
            self.params[key] = value.copy()
        self.policy = policy or (lambda _features: Decision("UPDATE"))
        self.w, self.in_degree, self.out_degree = graph()
        self.incoming = [np.flatnonzero(self.w[n]).tolist() for n in range(NODE_COUNT)]
        self.outgoing = [np.flatnonzero(self.w[:, n]).tolist() for n in range(NODE_COUNT)]
        if initial_states is None:
            self.states = np.zeros((NODE_COUNT, WIDTH), dtype=np.float64)
        else:
            arr = np.asarray(initial_states, dtype=np.float64)
            if arr.shape != (NODE_COUNT, WIDTH) or not np.all(np.isfinite(arr)):
                raise ValueError("initial_states must be finite with shape (32, 8)")
            self.states = arr.copy()
        self.inputs = np.zeros(NODE_COUNT, dtype=np.float64)
        self.input_change = np.zeros(NODE_COUNT, dtype=np.float64)
        self.aggregate = (self.w @ self.states) / np.sqrt(np.maximum(1, self.in_degree))[:, None]
        self.aggregate_change = np.zeros(NODE_COUNT, dtype=np.float64)
        self.prior_residual = np.zeros(NODE_COUNT, dtype=np.float64)
        self.last_update_tick = np.full(NODE_COUNT, -1, dtype=np.int64)
        self.holds = np.zeros(NODE_COUNT, dtype=np.int64)
        self.versions = np.zeros(NODE_COUNT, dtype=np.int64)
        self.received_versions = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.int64)
        self.latest_message = np.zeros((NODE_COUNT, NODE_COUNT, WIDTH), dtype=np.float64)
        for dst in range(NODE_COUNT):
            for src in self.incoming[dst]:
                self.latest_message[dst, src] = self.states[src]
        self.pending_messages: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {}
        self.deadline_generation = np.zeros(NODE_COUNT, dtype=np.int64)
        # Bucket deadlines by due tick: selective execution must not inspect
        # one deadline entry per graph node on every tick.
        self.deadlines: dict[int, list[tuple[int, int]]] = {}
        self.candidates: OrderedDict[int, int] = OrderedDict()
        self._candidate_requests: OrderedDict[int, bool] = OrderedDict()
        self.sequence = 0
        self.tick = -1
        self.failed = False
        self.intervention_used = False
        self.readouts: list[float] = []
        self.trace: list[dict[str, object]] = []
        self.counters = {name: 0 for name in (
            "candidate_enqueues", "candidate_coalesces", "policy_calls", "forced_updates",
            "holds", "transitions", "aggregate_edge_updates", "messages_delivered",
            "messages_sent", "stale_deadlines", "readouts", "full_state_scans",
            "full_state_materializations",
            "queue_pushes", "queue_pops", "deadline_pushes", "deadline_pops",
            "input_events_seen", "readout_calls", "trace_records", "policy_failures",
            "transition_failures", "output_bytes_materialized",
        )}
        self.cost_ledger = {
            "initialization_wall_seconds": time.perf_counter() - init_started,
            "run_wall_seconds": 0.0,
            "run_cpu_seconds": 0.0,
            "peak_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "peak_rss_platform": "macOS_bytes",
            "snapshot_calls": 0,
            "snapshot_array_bytes": 0,
            "snapshot_wall_seconds": 0.0,
            "snapshot_serializations": 0,
            "snapshot_serialized_bytes": 0,
            "snapshot_serialization_wall_seconds": 0.0,
            "clone_calls": 0,
            "clone_array_bytes": 0,
            "clone_wall_seconds": 0.0,
        }

    def _record(self, phase: str, kind: str, **fields: object) -> None:
        if phase not in PHASES:
            raise ValueError("invalid phase")
        row: dict[str, object] = {"sequence": len(self.trace), "tick": self.tick,
                                  "phase": phase, "kind": kind}
        row.update(fields)
        self.trace.append(row)
        self.counters["trace_records"] += 1

    def _request_candidate(self, node: int, *, deadline: bool = False) -> None:
        if node in self._candidate_requests:
            self._candidate_requests[node] = self._candidate_requests[node] or deadline
            return
        self._candidate_requests[node] = deadline

    def _enqueue_candidate(self, node: int, *, deadline: bool = False) -> None:
        if node in self.candidates:
            self.counters["candidate_coalesces"] += 1
            self._record("CANDIDATE", "coalesce", node=node)
            return
        self.candidates[node] = self.sequence
        self.sequence += 1
        self.counters["candidate_enqueues"] += 1
        self._record("CANDIDATE", "enqueue", node=node, deadline=deadline,
                     insertion_sequence=self.candidates[node])

    def _invalidate_deadline(self, node: int) -> None:
        self.deadline_generation[node] += 1

    def _features(self, node: int, tick: int, deadline: bool) -> RuntimePolicyFeaturesV1:
        distance = float(7 - node) if node <= 7 else 33.0
        reachable = float(node <= 7)
        values = np.concatenate((
            self.states[node], self.aggregate[node], np.asarray([
                self.inputs[node], self.input_change[node], self.aggregate_change[node],
                self.prior_residual[node], tick - int(self.last_update_tick[node]),
                float(self.holds[node]), float(deadline), float(self.in_degree[node]),
                float(self.out_degree[node]), distance, reachable,
            ], dtype=np.float64),
        ))
        return RuntimePolicyFeaturesV1(node, tick, values)

    def snapshot(self) -> dict[str, object]:
        """Return a deep, complete runtime snapshot for diagnostics."""
        started = time.perf_counter()
        result = copy.deepcopy({k: v for k, v in self.__dict__.items()
                                if k not in ("policy", "cost_ledger", "_candidate_requests")})
        self.cost_ledger["snapshot_calls"] += 1
        self.cost_ledger["snapshot_array_bytes"] += self._snapshot_array_bytes(result)
        self.cost_ledger["snapshot_wall_seconds"] += time.perf_counter() - started
        return result

    @classmethod
    def _snapshot_array_bytes(cls, value: object) -> int:
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, Mapping):
            return sum(cls._snapshot_array_bytes(k) + cls._snapshot_array_bytes(v)
                       for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return sum(cls._snapshot_array_bytes(item) for item in value)
        return 0

    def restore(self, snapshot: Mapping[str, object]) -> None:
        policy = self.policy
        restored = copy.deepcopy(dict(snapshot))
        expected = {k for k in self.__dict__ if k not in ("policy", "cost_ledger", "_candidate_requests")}
        if set(restored) != expected:
            raise ValueError("snapshot fields do not match runtime schema")
        for key, value in restored.items():
            setattr(self, key, value)
        self.policy = policy

    def clone(self) -> "TickRuntime":
        started = time.perf_counter()
        clone = TickRuntime(self.params, self.policy, initial_states=self.states)
        snapshot = self.snapshot()
        clone.restore(snapshot)
        copied_bytes = self._snapshot_array_bytes(snapshot)
        self.cost_ledger["clone_calls"] += 1
        self.cost_ledger["clone_array_bytes"] += copied_bytes
        self.cost_ledger["clone_wall_seconds"] += time.perf_counter() - started
        return clone

    def snapshot_bytes(self) -> bytes:
        snap = self.snapshot()
        started = time.perf_counter()

        def encode(value: object) -> object:
            if isinstance(value, np.ndarray):
                return {"dtype": value.dtype.str, "shape": list(value.shape),
                        "values": [float(v).hex() for v in value.ravel()]}
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value).hex()
            if isinstance(value, float):
                return value.hex()
            if isinstance(value, tuple):
                return [encode(v) for v in value]
            if isinstance(value, list):
                return [encode(v) for v in value]
            if isinstance(value, OrderedDict):
                return [[encode(k), encode(v)] for k, v in value.items()]
            if isinstance(value, dict):
                return [[encode(k), encode(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
            if value is None or isinstance(value, (str, int, bool)):
                return value
            raise TypeError(f"cannot serialize snapshot value {type(value).__name__}")

        result = json.dumps(encode(snap), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.cost_ledger["snapshot_serializations"] += 1
        self.cost_ledger["snapshot_serialized_bytes"] += len(result)
        self.cost_ledger["snapshot_serialization_wall_seconds"] += time.perf_counter() - started
        return result

    def _deliver(self, tick: int) -> None:
        for _seq, src, dst, version, payload in self.pending_messages.pop(tick, []):
            self.counters["queue_pops"] += 1
            if version <= self.received_versions[dst, src]:
                self._record("DELIVER", "stale_message", src=src, dst=dst, version=version)
                continue
            old = self.latest_message[dst, src].copy()
            delta = (payload - old) * self.w[dst, src]
            norm = math.sqrt(max(1, int(self.in_degree[dst])))
            self.aggregate[dst] += delta / norm
            change = float(np.linalg.norm(delta / norm))
            self.aggregate_change[dst] += change
            self.latest_message[dst, src] = payload
            self.received_versions[dst, src] = version
            self.counters["aggregate_edge_updates"] += 1
            self.counters["messages_delivered"] += 1
            self._record("DELIVER", "message", src=src, dst=dst, version=version,
                         aggregate_change=float(change))
            if change != 0.0:
                self._request_candidate(dst)
                self._invalidate_deadline(dst)

    def _apply_inputs(self, tick: int, events: Sequence[tuple[int, int, float]]) -> None:
        for event_tick, node, value in events:
            if event_tick != tick:
                raise ValueError("step accepts only input events for the current tick")
            if isinstance(node, bool) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) < NODE_COUNT:
                raise ValueError("input event node is invalid")
            new_value = _float(value, "input event value")
            node = int(node)
            change = abs(new_value - float(self.inputs[node]))
            self.inputs[node] = new_value
            self.input_change[node] += change
            if change != 0.0:
                self._request_candidate(node)
                self._invalidate_deadline(node)
            self.counters["input_events_seen"] += 1
            self._record("INPUT", "input", node=node, value=new_value.hex(), change=change.hex())

    def _expire_deadlines(self, tick: int) -> set[int]:
        due_nodes: set[int] = set()
        for node, generation in self.deadlines.pop(tick, []):
            self.counters["deadline_pops"] += 1
            if generation == int(self.deadline_generation[node]):
                due_nodes.add(node)
                self._request_candidate(node, deadline=True)
            else:
                self.counters["stale_deadlines"] += 1
                self._record("CANDIDATE", "stale_deadline", node=node)
        return due_nodes

    def _flush_candidate_requests(self) -> None:
        for node, deadline in self._candidate_requests.items():
            self._enqueue_candidate(node, deadline=deadline)
        self._candidate_requests.clear()

    def step(self, tick: int, input_events: Sequence[tuple[int, int, float]], *,
             update_all: bool = False,
             one_shot_override: tuple[int, Decision] | None = None) -> float:
        if self.failed:
            raise RuntimeError("runtime is failed and cannot be resumed")
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or int(tick) != self.tick + 1:
            raise ValueError("tick must be the next consecutive nonnegative integer")
        tick = int(tick)
        if tick < 0:
            raise ValueError("tick must be nonnegative")
        for event in input_events:
            if (len(event) != 3 or isinstance(event[0], bool)
                    or not isinstance(event[0], (int, np.integer)) or event[0] != tick):
                raise ValueError("step accepts only (current_tick, node, value) input events")
            node = event[1]
            if isinstance(node, bool) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) < NODE_COUNT:
                raise ValueError("input event node is invalid")
            _float(event[2], "input event value")
        if one_shot_override is not None:
            override_node, override_action = one_shot_override
            if self.intervention_used:
                raise ValueError("only one action intervention is permitted per runtime")
            if isinstance(override_node, bool) or not isinstance(override_node, (int, np.integer)) or not 0 <= int(override_node) < NODE_COUNT:
                raise ValueError("intervention node is invalid")
            if not isinstance(override_action, Decision):
                raise ValueError("intervention action must be a Decision")
        self.tick = tick
        self._candidate_requests.clear()
        self._deliver(tick)
        self._apply_inputs(tick, input_events)
        deadline_nodes = self._expire_deadlines(tick)
        self._flush_candidate_requests()

        if update_all:
            self.counters["full_state_scans"] += 1
            selected = list(range(NODE_COUNT))
            actions = {node: Decision("UPDATE") for node in selected}
        else:
            selected = list(self.candidates)
            actions: dict[int, Decision] = {}
            deadline_by_node = deadline_nodes
            for node in selected:
                self.candidates.pop(node, None)
                forced = (tick - int(self.last_update_tick[node]) >= MAX_HOLD_TICKS
                          or int(self.holds[node]) >= MAX_CONSECUTIVE_HOLDS)
                if forced:
                    self.counters["forced_updates"] += 1
                    actions[node] = Decision("UPDATE")
                else:
                    features = self._features(node, tick, node in deadline_by_node)
                    self.counters["policy_calls"] += 1
                    try:
                        decision = self.policy(features)
                    except Exception as exc:
                        self.failed = True
                        self.counters["policy_failures"] += 1
                        self._record("DECIDE", "failure", node=node,
                                     error_type=type(exc).__name__, error=str(exc))
                        raise
                    if not isinstance(decision, Decision):
                        self.failed = True
                        self.counters["policy_failures"] += 1
                        self._record("DECIDE", "failure", node=node,
                                     error_type="TypeError", error="policy must return Decision")
                        raise TypeError("policy must return Decision")
                    actions[node] = decision
                self._record("DECIDE", "decision", node=node, action=actions[node].action)
        if one_shot_override is not None:
            override_node, override_action = one_shot_override
            override_node = int(override_node)
            if override_node not in actions:
                self.failed = True
                self._record("DECIDE", "failure", node=override_node,
                             error_type="ValueError", error="intervention node is not a candidate")
                raise ValueError("intervention node is not a current candidate")
            actions[override_node] = override_action
            self.intervention_used = True
            self._record("DECIDE", "intervention", node=override_node,
                         action=override_action.action)
        self.candidates.clear()

        old_states: dict[int, np.ndarray] = {}
        updates: dict[int, np.ndarray] = {}
        for node in selected:
            if actions[node].action == "HOLD":
                self.holds[node] += 1
                self.counters["holds"] += 1
                self._invalidate_deadline(node)
                target = min(tick + MAX_WAIT_TICKS,
                            int(self.last_update_tick[node]) + MAX_HOLD_TICKS)
                if target <= tick:
                    raise RuntimeError("forced-update rule failed before deadline scheduling")
                self.deadlines.setdefault(target, []).append(
                    (node, int(self.deadline_generation[node]))
                )
                self.counters["deadline_pushes"] += 1
                self._record("COMMIT", "hold", node=node, holds=int(self.holds[node]))
                continue
            if actions[node].action != "UPDATE":
                raise ValueError("invalid policy action")
            old = self.states[node].copy()
            old_states[node] = old
            agg = self.aggregate[node].copy()
            try:
                updates[node] = local_transition(old, agg, self.inputs[node], self.params)
            except Exception as exc:
                self.failed = True
                self.counters["transition_failures"] += 1
                self._record("COMMIT", "failure", node=node,
                             error_type=type(exc).__name__, error=str(exc))
                raise
            self.counters["transitions"] += 1
            self._record("COMMIT", "update", node=node)

        changed_sources: list[int] = []
        for node, next_state in updates.items():
            residual = float(np.linalg.norm(next_state - old_states[node]))
            self.states[node] = next_state
            self.prior_residual[node] = residual
            self.last_update_tick[node] = tick
            self.holds[node] = 0
            self.input_change[node] = 0.0
            self.aggregate_change[node] = 0.0
            self._invalidate_deadline(node)
            changed_sources.append(node)
        self._record("COMMIT", "barrier", updated=len(changed_sources))

        for src in changed_sources:
            self.versions[src] += 1
            payload = self.states[src].copy()
            for dst in self.outgoing[src]:
                self.pending_messages.setdefault(tick + 1, []).append(
                    (self.sequence, src, dst, int(self.versions[src]), payload.copy())
                )
                self.sequence += 1
                self.counters["messages_sent"] += 1
                self.counters["queue_pushes"] += 1
                self._record("PUBLISH", "message_queued", src=src, dst=dst,
                             delivery_tick=tick + 1, version=int(self.versions[src]))

        result = float(self.states[READOUT_NODE] @ self.params["readout_w"] + self.params["readout_b"])
        self.readouts.append(result)
        self.counters["readouts"] += 1
        self.counters["readout_calls"] += 1
        self.counters["output_bytes_materialized"] += np.dtype(np.float64).itemsize
        self._record("READOUT", "readout", value=result.hex())
        return result

    def run(self, ticks: int, input_events: Sequence[tuple[int, int, float]], *,
            update_all: bool = False, capture_full_states: bool = False
            ) -> tuple[np.ndarray | None, np.ndarray]:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            return self._run(ticks, input_events, update_all=update_all,
                             capture_full_states=capture_full_states)
        finally:
            self.cost_ledger["run_wall_seconds"] += time.perf_counter() - wall_started
            self.cost_ledger["run_cpu_seconds"] += time.process_time() - cpu_started
            self.cost_ledger["peak_rss_platform_units"] = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )

    def _run(self, ticks: int, input_events: Sequence[tuple[int, int, float]], *,
             update_all: bool = False, capture_full_states: bool = False
             ) -> tuple[np.ndarray | None, np.ndarray]:
        if self.failed:
            raise RuntimeError("runtime is failed and cannot be resumed")
        if isinstance(ticks, bool) or not isinstance(ticks, (int, np.integer)) or ticks < 0:
            raise ValueError("ticks must be a nonnegative integer")
        first_tick = self.tick + 1
        stop_tick = first_tick + ticks
        capture = update_all or capture_full_states
        state_rows: list[np.ndarray] | None = [self.states.copy()] if capture else None
        if capture:
            self.counters["full_state_materializations"] += 1
            self.counters["output_bytes_materialized"] += self.states.nbytes
        first_readout = len(self.readouts)
        events_by_tick: dict[int, list[tuple[int, int, float]]] = {}
        for event in input_events:
            if len(event) != 3:
                raise ValueError("each input event must be (tick, node, value)")
            event_tick, _node, _value = event
            if isinstance(event_tick, bool) or not isinstance(event_tick, (int, np.integer)) or not first_tick <= int(event_tick) < stop_tick:
                raise ValueError("input event tick is outside the run horizon")
            events_by_tick.setdefault(int(event_tick), []).append(event)
        for tick in range(first_tick, stop_tick):
            self.step(tick, events_by_tick.get(tick, ()), update_all=update_all)
            if state_rows is not None:
                state_rows.append(self.states.copy())
                self.counters["full_state_materializations"] += 1
                self.counters["output_bytes_materialized"] += self.states.nbytes
        state_array = np.stack(state_rows) if state_rows is not None else None
        return state_array, np.asarray(self.readouts[first_readout:], dtype=np.float64)


def canonical_trace_bytes(trace: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize trace in actual occurrence order with canonical float strings."""
    records = []
    for expected_sequence, item in enumerate(trace):
        if item.get("sequence") != expected_sequence or item.get("phase") not in PHASES:
            raise ValueError("trace sequence or phase is invalid")
        row = {}
        for key, value in item.items():
            if isinstance(value, float):
                row[key] = value.hex()
            else:
                row[key] = value
        records.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(records) + ("\n" if records else "")).encode("utf-8")


def semantic_output_hash(states: np.ndarray, readouts: np.ndarray, trace_bytes: bytes) -> str:
    digest = hashlib.sha256()
    for arr in (np.asarray(states, dtype="<f8"), np.asarray(readouts, dtype="<f8")):
        digest.update(np.asarray(arr.shape, dtype="<u8").tobytes())
        digest.update(np.ascontiguousarray(arr).tobytes())
    digest.update(trace_bytes)
    return digest.hexdigest()
