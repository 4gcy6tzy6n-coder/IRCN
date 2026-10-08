"""Deterministic CPU event runtime for IRCN v0.5.0 prototype.

The scheduler invokes policy and transition callbacks only for the active node.
This module defines execution mechanics; it does not learn a policy or establish
task quality or runtime advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
import time
from numbers import Integral, Real
from typing import Callable, Iterable, Sequence

import numpy as np


class Action(str, Enum):
    HOLD = "hold"
    UPDATE = "update"
    REFINE = "refine"


@dataclass(frozen=True)
class Decision:
    action: Action
    steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise ValueError("action must be an Action enum")
        if isinstance(self.steps, (bool, np.bool_)) or not isinstance(self.steps, Integral):
            raise ValueError("steps must be an integer")
        if self.action is Action.REFINE:
            if self.steps < 2:
                raise ValueError("REFINE requires an integer step count >= 2")
        elif self.steps != 1:
            raise ValueError("HOLD and UPDATE must use steps=1")


@dataclass(frozen=True)
class PolicyFeatures:
    time: float
    elapsed: float
    input_change: float
    message_change: float
    prior_residual: float
    consecutive_holds: int
    deadline: bool


@dataclass(order=True)
class _Event:
    time: float
    sequence: int
    kind: str = field(compare=False)
    payload: tuple = field(compare=False, default=())


class EventLimitError(RuntimeError):
    """Raised when the event cap is reached while eligible events remain."""


Transition = Callable[[int, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
Policy = Callable[[int, PolicyFeatures], Decision]
Readout = Callable[[tuple[np.ndarray, ...]], object]


def _finite_vector(value: Sequence[float], width: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must have a real numeric dtype")
    arr = raw.astype(np.float64, copy=False)
    if arr.ndim != 1 or arr.shape[0] != width or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector of width {width}")
    return arr.copy()


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


class EventRuntime:
    """Local-update runtime with versioned messages and deterministic ordering.

    Edges are `(source, destination, weight)`. All events have nonnegative
    logical times. `run(until)` is horizon-bounded because recurrent graphs can
    otherwise generate an unending message stream.
    """

    def __init__(
        self,
        node_count: int,
        state_dim: int,
        input_dim: int,
        edges: Iterable[tuple[int, int, float]],
        transition: Transition,
        policy: Policy,
        *,
        message_delay: float = 1.0,
        max_wait: float = 1.0,
        max_hold: float = 5.0,
        max_consecutive_holds: int = 5,
        max_events: int = 1_000_000,
        initial_states: np.ndarray | None = None,
        readout: Readout | None = None,
    ) -> None:
        for name, value in (("node_count", node_count), ("state_dim", state_dim), ("input_dim", input_dim),
                            ("max_consecutive_holds", max_consecutive_holds), ("max_events", max_events)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not callable(transition) or not callable(policy):
            raise ValueError("transition and policy must be callable")

        message_delay = _finite_real(message_delay, "message_delay")
        max_wait = _finite_real(max_wait, "max_wait")
        max_hold = _finite_real(max_hold, "max_hold")
        if message_delay <= 0 or max_wait <= 0 or max_hold <= 0:
            raise ValueError("message_delay, max_wait, and max_hold must be positive")

        self.node_count = int(node_count)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.message_delay = float(message_delay)
        self.max_wait = float(max_wait)
        self.max_hold = float(max_hold)
        self.max_consecutive_holds = int(max_consecutive_holds)
        self.max_events = int(max_events)
        self.transition = transition
        self.policy = policy
        self.readout_fn = readout

        self.edge_weights: dict[tuple[int, int], float] = {}
        self.incoming: list[list[int]] = [[] for _ in range(node_count)]
        self.outgoing: list[list[int]] = [[] for _ in range(node_count)]
        for edge in edges:
            if len(edge) != 3:
                raise ValueError("each edge must be (source, destination, weight)")
            src, dst, weight = edge
            if (isinstance(src, bool) or not isinstance(src, (int, np.integer))
                    or isinstance(dst, bool) or not isinstance(dst, (int, np.integer))):
                raise ValueError("edge endpoints must be integer node ids")
            src, dst = int(src), int(dst)
            weight = _finite_real(weight, "edge weight")
            if not (0 <= src < node_count and 0 <= dst < node_count) or src == dst:
                raise ValueError("edge endpoints must be valid distinct node ids")
            if (src, dst) in self.edge_weights:
                raise ValueError("duplicate directed edge")
            self.edge_weights[(src, dst)] = weight
            self.incoming[dst].append(src)
            self.outgoing[src].append(dst)
        for neighbors in self.incoming + self.outgoing:
            neighbors.sort()

        if initial_states is None:
            self._states = np.zeros((node_count, state_dim), dtype=np.float64)
        else:
            raw_states = np.asarray(initial_states)
            if raw_states.dtype.kind not in "biuf":
                raise ValueError("initial_states must have a real numeric dtype")
            states = raw_states.astype(np.float64, copy=False)
            if states.shape != (node_count, state_dim) or not np.all(np.isfinite(states)):
                raise ValueError("initial_states must be a finite (node_count, state_dim) array")
            self._states = states.copy()

        self._inputs = np.zeros((node_count, input_dim), dtype=np.float64)
        self._messages: list[dict[int, np.ndarray]] = [dict() for _ in range(node_count)]
        self._received_versions: list[dict[int, int]] = [
            {src: 0 for src in self.incoming[node]} for node in range(node_count)
        ]
        self._last_commit_time = np.zeros(node_count, dtype=np.float64)
        self._versions = np.zeros(node_count, dtype=np.int64)
        self._input_change = np.zeros(node_count, dtype=np.float64)
        self._message_change = np.zeros(node_count, dtype=np.float64)
        self._prior_residual = np.zeros(node_count, dtype=np.float64)
        self._holds = np.zeros(node_count, dtype=np.int64)
        self._deadline_generation = np.zeros(node_count, dtype=np.int64)

        self._queue: list[_Event] = []
        self._next_sequence = 0
        self._pending_decisions: dict[tuple[int, float], bool] = {}
        self._now = 0.0
        self._last_external_time = 0.0
        self._closed = False
        self.trace: list[dict[str, object]] = []
        self.readouts: list[tuple[float, object]] = []
        self.counters: dict[str, int] = {
            "queue_pushes": 0, "queue_pops": 0, "events_processed": 0,
            "input_events": 0, "decision_events": 0, "deadline_events": 0,
            "readout_events": 0, "policy_calls": 0, "forced_updates": 0,
            "transition_calls": 0, "commits": 0, "messages_sent": 0,
            "messages_delivered": 0, "stale_messages": 0,
            "stale_deadlines": 0, "coalesced_decisions": 0,
            "aggregated_edges": 0, "readout_calls": 0, "event_limit_failures": 0,
            "transition_failures": 0, "policy_failures": 0, "readout_failures": 0,
        }
        self.run_seconds = 0.0

    @property
    def current_time(self) -> float:
        return self._now

    @property
    def states(self) -> np.ndarray:
        """Return a defensive snapshot; selection code never calls this property."""
        return self._states.copy()

    def _enqueue(self, timestamp: float, kind: str, payload: tuple = ()) -> None:
        heapq.heappush(self._queue, _Event(timestamp, self._next_sequence, kind, payload))
        self._next_sequence += 1
        self.counters["queue_pushes"] += 1

    def _coerce_external_time(self, timestamp: float) -> float:
        timestamp = _finite_real(timestamp, "event timestamp")
        if timestamp < 0:
            raise ValueError("event timestamp must be finite and nonnegative")
        if timestamp < self._last_external_time or timestamp < self._now:
            raise ValueError("external event timestamps must be nondecreasing and not in the past")
        return timestamp

    def push_input(self, timestamp: float, node: int, value: Sequence[float]) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)) or not 0 <= int(node) < self.node_count:
            raise ValueError("node must be a valid integer node id")
        timestamp = self._coerce_external_time(timestamp)
        vector = _finite_vector(value, self.input_dim, "input")
        self._last_external_time = timestamp
        self._enqueue(timestamp, "input", (int(node), vector))

    def push_readout(self, timestamp: float) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        timestamp = self._coerce_external_time(timestamp)
        self._last_external_time = timestamp
        self._enqueue(timestamp, "readout")

    def _schedule_decision(self, node: int, timestamp: float, *, deadline: bool = False) -> None:
        key = (node, timestamp)
        if key in self._pending_decisions:
            self._pending_decisions[key] = self._pending_decisions[key] or deadline
            self.counters["coalesced_decisions"] += 1
            return
        self._pending_decisions[key] = deadline
        self._enqueue(timestamp, "decision", (node,))

    def _invalidate_deadline(self, node: int) -> None:
        self._deadline_generation[node] += 1

    def _schedule_deadline(self, node: int, timestamp: float) -> None:
        target = min(timestamp + self.max_wait, float(self._last_commit_time[node] + self.max_hold))
        if target <= timestamp:
            target = timestamp + min(self.max_wait, self.max_hold)
        self._deadline_generation[node] += 1
        token = int(self._deadline_generation[node])
        self._enqueue(target, "deadline", (node, token))

    def _aggregate(self, node: int) -> np.ndarray:
        aggregate = np.zeros(self.state_dim, dtype=np.float64)
        for src in self.incoming[node]:
            message = self._messages[node].get(src)
            if message is not None:
                aggregate += self.edge_weights[(src, node)] * message
            self.counters["aggregated_edges"] += 1
        return aggregate / math.sqrt(max(1, len(self.incoming[node])))

    def _policy_features(self, node: int, timestamp: float, deadline: bool) -> PolicyFeatures:
        return PolicyFeatures(
            time=timestamp,
            elapsed=max(0.0, timestamp - float(self._last_commit_time[node])),
            input_change=float(self._input_change[node]),
            message_change=float(self._message_change[node]),
            prior_residual=float(self._prior_residual[node]),
            consecutive_holds=int(self._holds[node]),
            deadline=deadline,
        )

    def _deliver_message(self, timestamp: float, src: int, dst: int, version: int, payload: np.ndarray) -> None:
        if version <= self._received_versions[dst][src]:
            self.counters["stale_messages"] += 1
            self.trace.append({"time": timestamp, "kind": "stale_message", "src": src, "dst": dst, "version": version})
            return
        old = self._messages[dst].get(src)
        if old is not None:
            self._message_change[dst] += abs(self.edge_weights[(src, dst)]) * float(np.linalg.norm(payload - old))
        else:
            self._message_change[dst] += abs(self.edge_weights[(src, dst)]) * float(np.linalg.norm(payload))
        self._messages[dst][src] = payload.copy()
        self._received_versions[dst][src] = version
        self.counters["messages_delivered"] += 1
        self._invalidate_deadline(dst)
        self._schedule_decision(dst, timestamp)
        self.trace.append({"time": timestamp, "kind": "message", "src": src, "dst": dst, "version": version})

    def _commit(self, node: int, timestamp: float, decision: Decision) -> None:
        old = self._states[node].copy()
        aggregate = self._aggregate(node)
        elapsed = max(0.0, timestamp - float(self._last_commit_time[node]))
        steps = decision.steps if decision.action is Action.REFINE else 1
        candidate = old
        for step in range(steps):
            dt = elapsed if step == 0 else 0.0
            self.counters["transition_calls"] += 1
            try:
                new_state = self.transition(
                    node,
                    candidate.copy(),
                    aggregate.copy(),
                    self._inputs[node].copy(),
                    dt,
                )
            except Exception as exc:
                self.counters["transition_failures"] += 1
                self.trace.append({"time": timestamp, "kind": "transition_error", "node": node,
                                   "error_type": type(exc).__name__, "message": str(exc)})
                raise
            try:
                arr = _finite_vector(new_state, self.state_dim, "transition state")
            except (TypeError, ValueError) as exc:
                self.counters["transition_failures"] += 1
                self.trace.append({"time": timestamp, "kind": "transition_error", "node": node,
                                   "error_type": type(exc).__name__, "message": str(exc)})
                raise ValueError("transition must return a finite state vector of state_dim")
            candidate = arr

        self._states[node] = candidate
        self._last_commit_time[node] = timestamp
        self._versions[node] += 1
        self._prior_residual[node] = float(np.linalg.norm(candidate - old))
        self._input_change[node] = 0.0
        self._message_change[node] = 0.0
        self._holds[node] = 0
        self._invalidate_deadline(node)
        self.counters["commits"] += 1
        self.trace.append({"time": timestamp, "kind": "commit", "node": node,
                           "version": int(self._versions[node]), "action": decision.action.value,
                           "transition_calls": steps})

        for dst in self.outgoing[node]:
            self._enqueue(timestamp + self.message_delay, "message",
                          (node, dst, int(self._versions[node]), candidate.copy()))
            self.counters["messages_sent"] += 1

    def _handle_decision(self, timestamp: float, node: int) -> None:
        deadline = self._pending_decisions.pop((node, timestamp), False)
        self.counters["decision_events"] += 1
        elapsed = max(0.0, timestamp - float(self._last_commit_time[node]))
        forced = (elapsed >= self.max_hold or int(self._holds[node]) >= self.max_consecutive_holds)
        if forced:
            decision = Decision(Action.UPDATE)
            self.counters["forced_updates"] += 1
        else:
            self.counters["policy_calls"] += 1
            try:
                decision = self.policy(node, self._policy_features(node, timestamp, deadline))
            except Exception as exc:
                self.counters["policy_failures"] += 1
                self.trace.append({"time": timestamp, "kind": "policy_error", "node": node,
                                   "error_type": type(exc).__name__, "message": str(exc)})
                raise
            if not isinstance(decision, Decision):
                self.counters["policy_failures"] += 1
                self.trace.append({"time": timestamp, "kind": "policy_error", "node": node,
                                   "error_type": "TypeError", "message": "policy must return a Decision"})
                raise TypeError("policy must return a Decision")
        if decision.action is Action.HOLD:
            self._holds[node] += 1
            self.trace.append({"time": timestamp, "kind": "hold", "node": node,
                               "consecutive_holds": int(self._holds[node]), "deadline": deadline})
            self._schedule_deadline(node, timestamp)
            return
        self._commit(node, timestamp, decision)

    def _handle_readout(self, timestamp: float) -> None:
        snapshot = tuple(row.copy() for row in self._states)
        for row in snapshot:
            row.setflags(write=False)
        try:
            value = self.readout_fn(snapshot) if self.readout_fn else snapshot
        except Exception as exc:
            self.counters["readout_failures"] += 1
            self.trace.append({"time": timestamp, "kind": "readout_error",
                               "error_type": type(exc).__name__, "message": str(exc)})
            raise
        self.readouts.append((timestamp, value))
        self.counters["readout_events"] += 1
        self.counters["readout_calls"] += 1
        self.trace.append({"time": timestamp, "kind": "readout"})

    def run(self, until: float) -> list[tuple[float, object]]:
        """Process queued events through an inclusive finite logical-time horizon."""
        horizon = _finite_real(until, "until")
        if horizon < self._now:
            raise ValueError("until must be finite and no earlier than current time")
        started = time.perf_counter()
        processed = 0
        try:
            while self._queue and self._queue[0].time <= horizon:
                if processed >= self.max_events:
                    self.counters["event_limit_failures"] += 1
                    self.trace.append({"time": self._now, "kind": "event_limit_error",
                                       "limit": self.max_events, "until": horizon})
                    raise EventLimitError(f"event cap {self.max_events} reached before time {horizon}")
                event = heapq.heappop(self._queue)
                self.counters["queue_pops"] += 1
                processed += 1
                self.counters["events_processed"] += 1
                if event.time < self._now:
                    raise RuntimeError("internal event queue time regression")
                self._now = event.time

                if event.kind == "input":
                    node, value = event.payload
                    old = self._inputs[node]
                    self._input_change[node] += float(np.linalg.norm(value - old))
                    self._inputs[node] = value.copy()
                    self.counters["input_events"] += 1
                    self._invalidate_deadline(node)
                    self._schedule_decision(node, event.time)
                    self.trace.append({"time": event.time, "kind": "input", "node": node})
                elif event.kind == "decision":
                    (node,) = event.payload
                    self._handle_decision(event.time, node)
                elif event.kind == "deadline":
                    node, token = event.payload
                    self.counters["deadline_events"] += 1
                    if token != int(self._deadline_generation[node]):
                        self.counters["stale_deadlines"] += 1
                        self.trace.append({"time": event.time, "kind": "stale_deadline", "node": node})
                    else:
                        self._schedule_decision(node, event.time, deadline=True)
                elif event.kind == "message":
                    src, dst, version, payload = event.payload
                    self._deliver_message(event.time, src, dst, version, payload)
                elif event.kind == "readout":
                    self._handle_readout(event.time)
                else:
                    raise RuntimeError(f"unknown internal event kind: {event.kind}")
            self._now = horizon
            return self.readouts.copy()
        finally:
            self.run_seconds += time.perf_counter() - started

    def close(self) -> None:
        self._closed = True
