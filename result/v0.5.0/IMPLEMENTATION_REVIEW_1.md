# v0.5.0 event runtime — implementation review pass 1

Review mode: separate read-only self-review by implementation owner; not independent.

## Findings before targeted tests

| ID | Severity | Finding | Risk | Required fix/test |
|---|---|---|---|---|
| I1 | Major | `Decision` accepted a raw string action; identity checks treated it as UPDATE, and trace serialization could fail after state mutation. | Partial commit with exception; invalid policy output can corrupt run status. | Validate enum and integer step count in `Decision.__post_init__`; invalid policy action fails before transition. |
| I2 | Major | `push_input` advanced `_last_external_time` before validating vector shape/finiteness. | Rejected input leaves hidden scheduler state changed and can reject later valid input. | Make timestamp validation pure, commit monotonic-time state only after all input validation passes. |
| I3 | Minor | bool/string values could pass or leak low-level type errors in positive time configuration or run horizon. | Configuration errors are inconsistent and less reproducible. | Accept only real non-bool finite numbers; normalize invalid values to `ValueError`; add tests. |

All three findings were fixed before this review pass and covered by regression tests. The second pass is a read-only owner self-review, not an independent review.

## Post-fix review pass 2

Reviewed the complete runtime against contract R0–R8, focusing on validation-before-mutation, event ordering/coalescing, generation-token invalidation, partial failures, local callback boundaries, and counter/timer behavior.

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| R2-1 | Informational | `max_events` caps queue events per `run()` call, not total transition calls; `REFINE(k)` executes all requested steps synchronously and has no independent step cap. A policy can therefore request an extremely large `k`. | No contract-defined maximum exists. Record as a resource-boundary limitation; do not describe `max_events` as a total compute budget. |
| R2-2 | Informational | The policy API receives only local features, but Python callbacks are not sandboxed and may close over external state. | API boundary is not a security boundary; callers remain responsible for callback purity. |

No Blocker or Major defect was found in this pass. Event ordering, stale-version rejection, deadline invalidation, HOLD liveness, local-only transitions, immutable state snapshots, and error accounting match the contract within the declared callback trust boundary. This remains an implementation-owner self-review and does not establish absence of all defects.

## Verification evidence

- Targeted: `python -m pytest -q model/v0.5.0/test_event_runtime.py model/v0.4.1/test_icef_metrics.py model/v0.4.2/test_metric_hardening.py` — 70 passed.
- Repository suite: `python -m pytest -q` — 47 passed.
- `git diff --check -- model/v0.5.0 result/v0.5.0` — passed.
- No scientific performance, task-capability, or model-training claim is made.

**Disposition:** implementation review gate passes for this scoped runtime prototype, with the two informational boundaries above disclosed. The stage can be prepared for publication after artifact/hash and repository-path review.
