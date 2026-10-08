# v0.5.0 runtime — pre-implementation contract review

**Review mode:** separate read-only pass by the implementation owner; not independent.

## Contract risks checked before coding

1. **Zero-time cycles:** allowing zero message latency in a directed cycle can make an event loop non-terminating. Contract v0.1 forbids zero delay.
2. **Same-time ordering:** without a stable sequence key and duplicate-evaluation coalescing, equal-time messages can create platform-dependent traces or repeated updates. Contract freezes `(time, sequence_id)` ordering and per-node/per-time coalescing.
3. **Loss of input during HOLD:** using only the event payload would drop a sample when a node holds. Contract requires caching latest input and accumulated change summary; a future task must state whether latest-value or accumulated-input semantics are intended.
4. **Stale event rollback:** delayed messages and superseded deadlines can arrive after newer state. Contract requires edge-local version rejection and deadline generation tokens.
5. **REFINE ambiguity:** repeated updates could accidentally consume stale/new messages differently or leak intermediate state. Contract freezes cached input/message semantics, first/later elapsed values, one visible version, one outgoing message.
6. **Liveness and unbounded work:** a policy that always holds can starve a node, while a zero-delay cycle can grow the queue indefinitely. Contract requires positive retry/max-hold limits, max consecutive holds, positive message delay, and an event cap.
7. **Hidden full pass and readout cost:** selection must not calculate every transition. Contract confines decisions to the active node and charges global readout separately.
8. **Timing overclaim:** `run()` begins after construction, so it cannot by itself prove end-to-end savings. The contract explicitly narrows timer scope and defers complete wall-clock comparison.

## Gate 0 acceptance scope

Implement only the event-engine contract in `model/v0.5.0/EVENT_RUNTIME_CONTRACT_v0.1.md`, with tests R0–R8. Do not train a policy, run a task-performance study, or claim efficiency. All implementation and evidence files stay under `model/` and `result/`. A completion package is allowed only after each runtime invariant has direct tests and the full existing suite passes.

No code has been changed at this pre-review point.
