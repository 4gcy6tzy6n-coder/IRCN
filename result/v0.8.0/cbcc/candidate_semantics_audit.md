# IRCN v0.3.6/v0.3.7 candidate execution semantics audit

## Status and scope

This is a read-only audit of the archived v0.3.6 single-instance diagnostic and the v0.3.7 RK4 endpoint audit. It uses saved trajectories, event logs, configurations, reports, and frozen source from Git commits `2973ba1` and `82e3b45`. No solver was rerun, no frozen artifact was modified, and no new parameter value was tested.

The evidence covers one validation instance only: `chain / N=128 / dense_burst / seed=128012`, duration 2.0 s, output interval 0.01 s. It cannot establish behavior on another graph, task, seed, duration, or machine. WP2 remains `INCONCLUSIVE`; WP3–WP6 remain `BLOCKED`.

## Audit result

The saved evidence confirms that scalar and batch candidates execute different numerical paths even when their delivery mode is held fixed. Their saved output trajectories first differ at the 0.01 s sample by floating-point-scale amounts, and their semantic event sets first differ at 0.02 s. The first difference larger than `1e-9` occurs at 0.53 s on driven node 61, immediately after the dense-burst input falling edge at 0.52 s. This temporal alignment is confirmed; its causal mechanism is not.

The delivery representation is not the source of the saved trajectory discrepancy on this instance: batch edge delivery and batch broadcast have bitwise-identical trajectories, as do scalar edge delivery and scalar broadcast. Their event logs and event counts differ because broadcast delivery records a broadcast plus expanded message records, while edge delivery directly queues messages.

The local tolerance controls SciPy DOP853's local vector error estimate for each scalar or small vector integration. It does not directly control full-network trajectory error or the segment publication decision. The separate `segment_tolerance` controls whether a newly integrated segment is published, using a nine-point maximum difference against the previously published segment. Consequently, changing local integration tolerances can change later event order and publication decisions. The saved tighter-tolerance run becoming less accurate is evidence of sensitivity in this hybrid integration-and-publication system, not evidence that stricter DOP853 tolerances intrinsically reduce numerical accuracy.

## Evidence inventory

The current working branch does not contain the later v0.3.6/v0.3.7 archive directories, so this audit read them directly from the immutable Git objects listed below.

| Evidence | Git object and path | Use in this audit |
|---|---|---|
| v0.3.6 frozen contract | `2973ba1:model/v0.3.6/DIAGNOSTIC_CONTRACT_v1.yaml` | Instance, variants, limits, fixed factors |
| v0.3.6 protocol | `2973ba1:model/v0.3.6/DIAGNOSTIC_PROTOCOL_v1.md` | Claim boundary and comparison definitions |
| Candidate scheduler source | `2973ba1:model/v0.3.5/src/events/ode_segment_scheduler.py` | Batch, scalar, tolerance, publication, and delivery semantics |
| Saved v0.3.6 run | `2973ba1:result/v0.3.6/diagnostics/wp2-diagnostic-20261009-01/` | Trajectories, event logs, resolved configs, counters, analysis |
| v0.3.7 audit source | `82e3b45:model/v0.3.7/audit_runner.py` | Corrected terminal-boundary reference calculation |
| Saved v0.3.7 audit | `82e3b45:result/v0.3.7/rk4-endpoint-audit-20261009-01/` | Corrected reference checks and candidate reanalysis |

## Confirmed execution semantics

### Event processing and causal state

Events are ordered by `(time, insertion sequence)` in a heap. All events at one time are removed as a group. Message, broadcast, input, initialization, and refresh events update inboxes or mark nodes as touched; touched nodes are then sorted and rebuilt. A node's local right-hand side reads only its causally received inbox segments, current exogenous input, bias, and its own integrated state. It does not first compute the full current network state.

Published source segments are held in each destination inbox. A node rebuild therefore uses the most recently delivered source segments, which can differ from a source node's newly computed but unpublished segment. This is the operative local-event approximation.

### Scalar versus batch integration

Touched nodes with the same segment end time are grouped. With `local_batch=false`, each node is passed to a separate DOP853 solve. With `local_batch=true`, groups are split into chunks of at most `local_batch_size`; the frozen batch size is two.

The components inside a batch share one adaptive DOP853 solve, accepted steps, and dense-output object. Their right-hand-side components remain locally separable because each component reads delivered inbox segments rather than another component's evolving state inside that same solve. The material semantic difference is therefore shared adaptive step and error control plus shared dense-output construction, followed by thresholded publication. The saved evidence does not show that batching changes the declared differential equation itself.

### What the two tolerances control

For a chunk of `m` nodes, the scheduler passes `rtol / sqrt(m)` and `atol / sqrt(m)` to `solve_ivp(..., method="DOP853")`. In the frozen SciPy 1.17.1 runtime, the Runge–Kutta error norm is the RMS norm of `estimated_error / (atol + rtol * scale)`. A scalar solve uses the unscaled configured tolerances; a two-node batch uses both tolerances divided by `sqrt(2)`.

This local error control applies to one local integration interval. It is not a bound on the global 128-node trajectory, on downstream event timing, or on oracle-relative NRMSE.

`segment_tolerance` is a different control. After a rebuild, the scheduler samples the overlap of the old and new segments at nine evenly spaced times. It publishes when the maximum sampled difference is at least the threshold, or when no usable prior publication exists. The frozen baseline uses `1e-7`; the tighter segment variant uses `1e-8`.

### Edge delivery versus broadcast delivery

`edge_messages` queues one message per outgoing edge. `broadcast_batch` queues one broadcast event and expands it into destination message records when processed. Because these are different log representations, raw event position and total event count are not valid equivalence criteria across delivery modes. Full saved trajectories are valid for that comparison.

On the frozen instance:

- batch broadcast and batch edge delivery are bitwise identical over all `201 x 128` saved states;
- scalar broadcast and scalar edge delivery are bitwise identical over all saved states;
- message-delivery counts match within each pair, although total event counts differ by representation.

## Offline divergence replay

The following results were computed from the saved `.npy` trajectories and JSONL event logs. "Semantic event" comparison removes only `sequence` and `parent_sequence`, whose values necessarily shift when a representation inserts a broadcast record; all event kind, time, node, source, version, and horizon fields remain.

| Comparison against frozen batch/broadcast baseline | First unequal saved state | First saved difference > `1e-12` | First saved difference > `1e-9` | First semantic event-set difference |
|---|---:|---:|---:|---:|
| scalar/broadcast | 0.01 s, node 1, `4.34e-19` | 0.03 s, node 22, `1.63e-12` | 0.53 s, node 61, `6.08e-3` | 0.02 s |
| tighter local tolerance | 0.01 s, node 1, `1.63e-18` | 0.02 s, node 17, `2.56e-12` | 0.20 s, node 43, `6.87e-9` | 0.04 s |
| tighter segment tolerance | 0.01 s, node 4, `1.01e-11` | 0.01 s, node 4, `1.01e-11` | 0.53 s, node 61, `7.16e-7` | 0.00 s |
| batch/edge delivery | no difference | no difference | no difference | raw representation differs at 0.00 s |

At 0.01 s, 110 of 128 scalar/batch state entries differ exactly, but the largest difference is only `1.73e-18`. The scalar/batch difference first exceeds `1e-15` at 0.02 s on node 39. At 0.02 s their semantic event sets also diverge: the batch run contains publications and broadcasts absent from the scalar run, including a publication for node 20. The logs do not store the unpublished segment values or the nine-point comparison values, so they cannot prove which threshold comparison first caused that event-set difference.

The `dense_burst` input is on for 0.02 s every 0.125 s. The 0.53 s large scalar/batch difference follows the 0.52 s falling edge, and node 61 is one of the seven driven nodes recorded by v0.3.7. This is a useful localization fact. It does not distinguish endpoint evaluation, adaptive-step coupling, dense-output interpolation, or a prior publication difference as the cause.

## Saved outcome summary

All values below are relative to the frozen DOP853 oracle and describe this one instance.

| Variant | NRMSE | Maximum absolute error | Maximum-error location | RHS evaluations | Scheduler seconds |
|---|---:|---:|---|---:|---:|
| batch/broadcast baseline | `8.6401e-3` | `6.0754e-3` | 0.53 s, node 61 | 506,341 | 17.81 |
| batch/edge delivery | `8.6401e-3` | `6.0754e-3` | 0.53 s, node 61 | 506,341 | 17.48 |
| scalar/broadcast | `1.4816e-9` | `7.3408e-10` | 1.80 s, node 100 | 798,525 | 23.27 |
| scalar/edge delivery | `1.4816e-9` | `7.3408e-10` | 1.80 s, node 100 | 798,525 | 23.24 |
| tighter local tolerance | `1.6325e-2` | `1.0763e-2` | 1.53 s, node 61 | 679,079 | 21.85 |
| tighter segment tolerance | `1.2132e-2` | `9.1394e-3` | 1.53 s, node 61 | 513,862 | 17.88 |

Repeated frozen baselines had identical trajectories, event streams, normalized configurations, and counters. These wall times are diagnostic observations from one CPU run, not independent samples or evidence of an efficiency advantage.

## Effect of the v0.3.7 reference audit

v0.3.7 found a separate defect in the independent RK4 check: the input breakpoint API omitted a discontinuity exactly at the 2.0 s terminal endpoint, so the old RK4 final stage evaluated the right-continuous input instead of the left limit. The correction changed only the terminal sample and only the seven driven nodes. Corrected coarse and fine RK4 agreed with each other and DOP853 near floating-point precision.

This corroborates the DOP853 reference for the one frozen instance and invalidates the earlier claim that its reference was independently unresolved. It does not resolve the candidate scalar/batch discrepancy: v0.3.7 reran no candidate solver, and the batch, scalar, and tolerance comparisons against corrected RK4 remain materially the same.

## Confirmed facts

1. Delivery representation changes event-log structure but not the saved trajectory within either the scalar or batch pair on this instance.
2. Scalar and two-node batch integration produce different saved states and later different publication/event sets under otherwise matched broadcast delivery.
3. The scalar trajectory is much closer to the independently corroborated reference on this instance, but uses more RHS evaluations and wall time.
4. Tightening local DOP853 tolerances changes event counts, update order, and the final trajectory; it does not monotonically reduce oracle error in the saved run.
5. Tightening the publication threshold changes the event set immediately at time zero and increases publications and deliveries.
6. The v0.3.7 terminal-boundary defect belongs to the independent RK4 harness. It does not explain away the saved candidate discrepancies.

## Hypotheses requiring new evidence

1. Shared adaptive steps or dense output in a two-node DOP853 solve may perturb predicted segments enough to cross the discontinuous publication threshold. The first 0.02 s event-set difference is consistent with this mechanism, but the necessary local error estimates and nine-point comparison values were not logged.
2. Right-continuous exogenous input at a local segment endpoint may interact with a DOP853 step ending at an input discontinuity. The large 0.53 s error on driven node 61 after the 0.52 s falling edge makes this plausible, but no saved local-step trace establishes it.
3. Non-monotonic behavior under tighter local tolerance may result from threshold crossings and changed event order rather than larger local truncation error. This cannot be separated with the current logs.

These hypotheses are alternatives, not cumulative findings. The current evidence cannot assign their relative contribution.

## Unresolved limitations

- Only one graph, task, seed, population size, and two-node batch size were audited. No statistical inference or generalization is valid.
- Saved trajectories are sampled every 0.01 s. They locate the first saved-state difference, not the first continuous-time numerical difference.
- Event logs contain event metadata but not segment coefficients, integrated local states, batch membership, accepted/rejected DOP853 steps, scaled error norms, or the nine sampled publication differences. Exact event-level numerical replay is therefore impossible from the archive alone.
- Position-by-position event-log comparison is invalid across delivery representations because sequence IDs and inserted broadcast events differ by design.
- The v0.3.6 parent labels manifest and analysis execution manifest retain their documented source-hash provenance discrepancy. This audit does not modify either artifact.
- The evidence measures solver behavior. It does not establish task utility, model-prediction ability, biological structure contribution, or an efficiency advantage.

## Track A decision

The historical evidence is sufficient to confirm an implementation-sensitive scalar/batch semantic divergence and to localize its first saved and logged manifestations. It is insufficient to confirm a single root cause. A future diagnostic would need to log batch membership, local interval endpoints, accepted/rejected steps or final scaled error norm, old/new nine-point segment differences, and left/right input values at local endpoints. That work requires a separately frozen protocol and is outside this read-only audit.

No project gate changes follow from this report: WP2 stays `INCONCLUSIVE`, WP3–WP6 stay `BLOCKED`, and this audit does not authorize E2.
