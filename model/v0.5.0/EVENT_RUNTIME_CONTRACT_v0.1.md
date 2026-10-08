# IRCN v0.5.0 event runtime contract v0.1

**Stage:** executable runtime prototype; no trained model or capability result is implied. This contract narrows the v0.4 architecture semantics for a first CPU event-engine implementation.

## Frozen runtime semantics

1. The graph is a finite directed graph with unique, non-self edges `(source, destination, weight)`; weights must be finite `float64` values. Node states are finite one-dimensional vectors of configured width `state_dim`; input vectors use configured width `input_dim`. Initial state is zero for every node and is treated as committed at logical time `0` with version `0`. The runtime uses `float64` in this prototype.
2. External input/readout timestamps are finite, nonnegative, and nondecreasing. Internal events are ordered by `(timestamp, monotonically increasing sequence_id)`. Input events carry a valid node id and a copied finite `input_dim` vector. The latest input starts at zero and remains cached until replaced.
3. Each node caches its latest input, latest delivered message per predecessor, highest received version per edge (initially `0`), last committed state/time/version, input-change summary, received-message delta summary, prior state residual, consecutive HOLD count, and deadline generation token. Input-change summary accumulates the L2 norm of every input difference since the last commit. Message-change summary accumulates `abs(edge_weight) * L2(message_new - message_old)` for every accepted message since the last commit. Both summaries reset after a commit. Prior state residual is `L2(h_new - h_old)` from the previous commit.
4. An input/message arrival schedules a local decision at that timestamp. Duplicate pending decisions for the same node and timestamp are coalesced. A message is delivered only after a fixed positive delay; zero-delay cycles are forbidden in v0.1.
5. The policy sees only a `PolicyFeatures` record for the active node (elapsed time, input-change summary, message-change summary, previous transition residual, skip count, and whether this is a deadline). It cannot receive a full state array or invoke the transition function to decide whether to update.
6. Actions are `HOLD`, `UPDATE`, and `REFINE(k)`. `REFINE(k)` evaluates the same local transition `k` times with the current cached input and incoming aggregate held fixed; the first call receives elapsed time since the previous commit, later calls receive zero elapsed time. Intermediate states are private. One commit increments the visible node version once and sends one message version.
7. `HOLD` evaluates no transition. It schedules a retry at `min(time + max_wait, last_commit_time + max_hold)`. At a deadline equal to `last_commit_time + max_hold`, or when the consecutive-HOLD limit has already been reached, the runtime forces UPDATE without another policy call. Both liveness limits are positive finite configuration values. A new input/message invalidates the node's previous deadline token.
8. On UPDATE/REFINE, the local transition receives only `(node_id, own_state, incoming_aggregate, latest_input, elapsed)`. The aggregate is the weighted sum of latest delivered predecessor messages divided by `sqrt(max(1, in_degree))`. The transition cannot mutate other node states; runtime-owned arrays passed to callbacks are defensive copies.
9. A commit increments the source version, records the event time, updates the cached residual, and schedules copies of the state only to direct outgoing neighbors at `time + message_delay`. Each destination accepts a message only if its source version exceeds the version already received on that edge; stale messages are logged and discarded.
10. A readout event may inspect every committed node state, but the callback/readout operation is counted separately. The callback receives an immutable snapshot. A readout does not update any node.
11. Initial state must be `(node_count, state_dim)` zeros or a finite array with exactly that shape. Edges require valid integer node ids and finite weights. `max_events` is a positive integer stop guard. If reached while eligible events remain, execution raises an explicit event-limit error and preserves the partial trace/counters.

## Scope and known implementation boundaries

- The runtime is a discrete logical-event state machine. It does not approximate a continuous-time ODE.
- Policies and local transitions are injected callables; learning either is out of scope for this engine task.
- The queue is a binary heap, so insertion/removal costs are `O(log E)`; message aggregation costs `O(in_degree)` when a node actually evaluates. v0.1 exposes counters and wall time but does not claim an asymptotic or hardware speedup.
- Wall time from `run()` covers queue processing and callbacks after construction. A separate end-to-end benchmark must include graph/model construction, input ingestion, output collection, and synchronization.
- REFINE currently means repeated transition applications with frozen input/message cache; it is not a solver accuracy refinement and must not be described as one.

## Acceptance tests (R0–R8)

- **R0 validation:** reject invalid graphs, vectors, non-finite/decreasing timestamps, zero/negative message delay, and invalid liveness limits before execution.
- **R1 deterministic order:** identical events/config produce identical event traces and states; equal-time events use stable insertion order.
- **R2 local-only work:** one node decision invokes only that node's transition and outgoing message sends; no transition prepass or hidden full-state scan.
- **R3 causality:** a source update cannot alter a destination before its delayed message; nodes outside the reachable directed set remain unchanged absent external input.
- **R4 message versions:** newer message commits; older/equal versions cannot roll back cached message or state; superseded deadlines are ignored.
- **R5 action semantics:** HOLD calls no transition; UPDATE calls once; REFINE(k) calls k times but commits/sends one version.
- **R6 liveness:** persistent HOLD is eventually forced by configured limits; event cap produces an explicit failure rather than an infinite loop.
- **R7 readout isolation:** readout observes current states and adds readout cost but never mutates/updates nodes.
- **R8 accounting:** event, queue, policy, transition, message, stale-drop, readout counts reconcile with the trace; timer scope is explicit.

## Review boundary

Before implementation, review this contract for contradictions and missing corner cases. Before task completion, inspect the implementation against every R0–R8 item, fix all Blocker/Major findings, re-run targeted and repository checks, and record unresolved limitations. Code checks do not establish useful task behavior or publication readiness.
