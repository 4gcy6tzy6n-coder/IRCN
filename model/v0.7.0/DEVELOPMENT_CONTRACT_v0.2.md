# IRCN v0.7.0 — Synchronous runtime bridge contract

**Status:** pre-implementation contract, revision 0.2. v0.3–v0.6 artifacts remain immutable. This stage tests execution semantics only; it does not make performance, dynamic-policy, connectome, biological, novelty, or publication claims.

## 1. Research question and boundary

Can the v0.5 event runtime mechanics be adapted to execute the frozen v0.6 cell under simultaneous tick semantics, with local candidate updates, complete clone/restore, and no hidden full-network computation in selective mode?

Scope is synthetic, CPU-only, and limited to the frozen v0.6 32-node chain task. No new task family, learned policy training, performance comparison, E2/E3/E4, connectome input, NeuroConverge modification, or NMI manuscript edit is in scope. C1–C6 values may be emitted as descriptive instrumentation, but none is a validated score or scientific result.

## 2. Frozen anchor and provenance

- Graph, input generator, cell equation, train/test split, and tick semantics are exactly those in `model/v0.6.0/MODEL_CONTRACT_v0.1.md`.
- Frozen cell checkpoint: `result/v0.6.0/runs/final_gz_02/checkpoints/cell_state.pt`, SHA-256 `b8df57150d9030f2690eb11a7977af8c7b629e43eec228678d186d2c102168c5`.
- Frozen predictor checkpoint is retained only as a provenance artifact and is not deployed: `result/v0.6.0/runs/final_gz_02/checkpoints/influence_mlp_state.pt`, SHA-256 `d380f5718e5b3167bb973153a2b326c211d10c98dc85a56f08ec37f02cec0f53`.
- Parity fixtures are the v0.6 test seeds and golden hand-built cases; they are implementation fixtures, not policy-evaluation test data.
- No v0.6 weight, task, input, threshold, or test result may be tuned in this stage.

## 3. Tick execution contract

At integer tick `t`, apply these phases in exact order: `DELIVER`, `INPUT`, `CANDIDATE`, `DECIDE`, `COMMIT`, `PUBLISH`, `READOUT`. The input API is a pre-generated sparse stream of `(tick,node,new_value)` changes, including explicit zero-valued reset events; the runtime never scans the dense `(tick,node)` input matrix to discover changes. The identical sparse stream is used by parity and selective paths. `DELIVER` applies messages scheduled for `t` from prior commits and incrementally refreshes only affected destination aggregates/change summaries. `INPUT` applies a changed value and accumulates `abs(new_value-old_value)` as input-change summary. Both phases enqueue a candidate only when an event changes a node's input or incoming aggregate, or when its deadline expires. Persistent nonzero summaries do not independently re-enqueue candidates. Candidate insertion uses a monotonic sequence; a same-tick duplicate coalesces without changing its first insertion sequence. `DECIDE` removes the candidate before calling the policy.

Before any policy call, force UPDATE if `t-last_update_tick >= MAX_HOLD_TICKS` or holds have reached `MAX_CONSECUTIVE_HOLDS`, regardless of whether the candidate came from input, message, or deadline. If the policy returns HOLD, leave input/message summaries pending, increment consecutive holds, and schedule exactly one deadline at `min(t + MAX_WAIT_TICKS, last_update_tick + MAX_HOLD_TICKS)`; a target not strictly greater than `t` is an internal error because the forced-update rule should have fired. A new input value or message that changes the tracked input/aggregate invalidates the old deadline token and enqueues the node once; equal-valued events do neither. If policy returns UPDATE, reset pending summaries, hold count and deadline; set `last_update_tick=t`. On an expired deadline, force UPDATE under the same condition; otherwise re-enqueue as a normal candidate. Frozen values are `MAX_WAIT_TICKS=1`, `MAX_HOLD_TICKS=5`, `MAX_CONSECUTIVE_HOLDS=5`. Initial `last_update_tick=-1`, so elapsed at tick 0 is 1. These counters only govern selective-mode candidate eligibility; update-all mode updates every node each tick.

Read policy features from read-only cached local state. The API passes `RuntimePolicyFeaturesV1`; callbacks are trusted code, not a Python security sandbox. The built-in v0.7 policy must be stateless/deterministic, receive no runtime/full-state/future-input references, and pass source inspection and call-path tests. In selective mode, invoke policy only for candidates. HOLD performs no transition, all-node candidate-state construction, or full-state scan. Candidate maintenance and queue operations are counted. In update-all parity mode only, mark every node UPDATE; this explicit reference path may traverse all nodes and is never presented as a selective-compute method. Compute accepted transitions from immutable `h[t]`; commit selected states simultaneously as `h[t+1]`. Update affected destination aggregates incrementally from changed source states. New state messages become visible at `t+1`. Compute readout after the simultaneous commit.

For the frozen cell, node `i` update is the exact v0.6 `GatedCell.step` local formula with `a_i[t] = sum_j W[i,j] h_j[t] / sqrt(max(1, indegree(i)))`; all nodes use `dt=1` as encoded by `log(2)`. A HOLD copies `h_i[t]` to `h_i[t+1]` and emits no new state message. Duplicate same-tick candidate causes coalesce by node with stable insertion sequence. `REFINE` is disabled in v0.7.

Update-all parity gate: for every frozen fixture and tick, maximum absolute state and readout difference from v0.6 synchronous `rollout` is at most `1e-10`. Trace phase is one of the enum values listed above and records actual occurrence order; do not sort events after execution. Canonical trace serialization is UTF-8 JSON Lines, fixed schema and field order per event kind, compact JSON separators `(',', ':')`, one `\n` per record, and every float encoded as a quoted Python `float.hex()` string. Require identical state/readout arrays, trace bytes, deterministic counters, and semantic-output hashes across exact reruns. Wall time, CPU time, RSS, timestamps, and environment fields are preserved but excluded from deterministic output hashes.

## 4. Versioned feature and accounting contract

`RuntimePolicyFeaturesV1` separates metadata from model features. Metadata is `node_id` and `tick`; model inputs are exactly 27 values in this order: own state (8); cached incoming aggregate (8); current scalar input (1); then 10 scalars: input-change norm; incoming-aggregate-change norm; previous committed update residual; elapsed ticks; consecutive holds; deadline flag; in-degree; out-degree; directed distance to readout capped at 33; reachability. All vectors are immutable float64 copies. Every value is timestamped at the start of tick `t` and may depend only on data visible by that phase. Static graph fields are precomputed once and counted as initialization. Own state, aggregate, input, and event summaries are maintained incrementally; policy code may not recompute them by scanning all nodes/edges. The v0.7 parity policy is stateless and deterministic; no hidden callback state is allowed. A future stateful policy must define an explicit cloneable state protocol before use.

Measure initialization wall time, run wall time (including sparse-event indexing, callbacks, queue work, readout, and requested output collection), run CPU time, and process high-water RSS (reported in platform units; it is not attributable to an individual runtime instance). Record separate operation counts for candidate enqueue/coalescing, policy calls, transitions, affected-edge aggregate updates, message queue pushes/pops, deadline pushes/pops, readout calls, trace records, and output materialization bytes. These operation counts do not substitute for elapsed time. Full snapshots/clones are allowed only for correctness tests and counterfactual semantics diagnostics; snapshot copy time and bytes are charged and these branches are excluded from deployment claims. In this stage no deployment latency comparison is made.

## 5. Clone/restore correctness diagnostics

The snapshot must include every runtime state component that can affect later output: committed states; input and input history summaries; cached aggregates; candidate set and insertion sequence counter; pending messages and versions; deadlines and generation tokens; policy-visible hold/residual metadata; current tick; and deterministic counters. Since the v0.7 parity policy is stateless and deterministic, there is no policy state or RNG to snapshot. Clones must have identical canonical runtime-snapshot serialization before intervention. A test intervention changes one node's action once; subsequent actions use the same frozen policy and branch-local state. This diagnostic is not used to train or evaluate a predictor in v0.7. No joint-action oracle is computed.

## 6. Acceptance gates and required defect workflow

1. Before implementation, an independent reviewer must ACCEPT this contract. Fix all blocker/major findings and obtain an independent re-review before coding.
2. Implement only after Gate 1. Self-review the complete diff against this contract and inspect state ordering, candidate invalidation, local-only HOLD, clone completeness, error handling, and cost counters.
3. Before completion packaging, obtain independent implementation defect review. Fix every blocker/major issue, add regression checks, and re-review changed behavior.
4. Run targeted tests for tick phase ordering; update-all parity; candidate lifecycle, deadlines, and invalidation; local candidate behavior; no-transition HOLD; deterministic ordering; clone/restore; and invalid inputs including non-finite values, wrong dimensions, invalid checkpoint hash, illegal tick/deadline, and invalid policy output. Then run repository regression tests, static checks, reproducibility checks, and source/input hashes; preserve exact outputs.
5. After tests, independently verify result/manifest/hash/report consistency before writing the completion package. Completion package is NOT READY if any required review, fix, test, or evidence check is missing.

Tests passing means software semantics passed within this scope. It does not mean ICEF validation, dynamic allocation benefit, learned-policy utility, speedup, or scientific validation passed.

## 7. Deliverable paths and decision

- Source, tests, lock file, frozen config, and source hashes: `model/v0.7.0/`.
- Logs, raw parity outputs, review records, verification manifest, and scoped decision: `result/v0.7.0/`.
- Stage source/deliverables are restricted to those paths.
- `READY` means the runtime bridge contract and software checks passed only. `NOT READY` means any blocker remains. Dynamic-policy/ICEF evaluation is explicitly deferred to a separately reviewed and preregistered task contract; this stage cannot recommend E2 or claim that future work is scientifically justified by efficacy evidence.

## 8. Explicit unresolved research questions

The v0.6 training task has one path to readout and does not identify state-dependent allocation value beyond topology. The v0.6 predictor was trained on synchronous single-node counterfactuals and has no validated event-runtime estimand. Future capability evaluation therefore needs a new preregistered task with identifiable same-topology, state-dependent action effects, runtime feature distribution, policy initialization units, label horizon, budget-matched controls, and prospective statistical precision. None may be inferred from v0.7 parity results.
