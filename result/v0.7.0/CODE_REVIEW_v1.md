# v0.7.0 independent defect review

## Scope

Reviewed `model/v0.7.0/DEVELOPMENT_CONTRACT_v0.2.md`, `src/ircn_v07.py`, `tests/test_runtime_bridge.py`, `run_parity.py`, `pre_registration.yaml`, `requirements.lock`, and the final parity output directory `runs/final_candidate_review_01/`.

## Review history

1. Contract review v0.1: **NOT READY**. Findings included local candidate vs global top-k conflict, non-identifiable task, undefined event-runtime estimand, and mismatch between the v0.6 task and altered pulse distribution. Scope was reduced to a synchronous runtime semantics bridge, and the task was frozen to v0.6 inputs/checkpoint.
2. Contract review v0.2, first pass: **NOT READY**. One blocker and three major findings concerned incomplete candidate/deadline lifecycle, callback isolation claims, deterministic trace/hash details, and feature metadata. These were revised; the independent reviewer returned **ACCEPT** before implementation.
3. Implementation review, first pass: **NOT READY**. Six major findings covered trace phase ordering, zero-change messages, failed runtime reuse, incomplete operation accounting, missing one-action clone intervention test, and overwritable/incomplete parity provenance. Fixes and regression tests were added.
4. Implementation re-review: **NOT READY**. Three major findings remained: candidate removal timing, snapshot/clone copy cost, and deterministic operation-counter comparison. Fixes and negative/regression tests were added.
5. Final independent re-review: **ACCEPT**. No known blocker or major defect remained within the reviewed scope.
6. Completion-package audit, first pass: **NOT READY**. Reviewer confirmed code and parity evidence, but found missing exact verification logs and stale superseded-run status text. Final verification logs and an independent 95-file hash audit were added under `verification/`; the stale status was corrected. The original raw output for the two early failing test iterations was not recoverable and is explicitly disclosed rather than reconstructed.
7. Completion-package re-review: **ACCEPT**. Reviewer verified all 20 completion-index entries, all 95 independent source/output hashes, saved command outputs and exit codes, final parity numbers, superseded-run indexing, and the disclosed historical log gap. No blocker or major finding remains.

## Findings and fixes

| Finding | Fix | Regression/evidence |
| --- | --- | --- |
| Tick trace phase order did not match contract | Buffer local candidate requests until the CANDIDATE phase; record events in actual phase order | `test_trace_phases_follow_frozen_tick_order`; all final traces validated |
| Zero aggregate-change messages caused needless decisions and invalidated deadlines | Candidate/deadline invalidation now occurs only when a tracked input or aggregate changes | `test_equal_payload_message_does_not_create_candidate_or_cancel_deadline` |
| A failed callback/transition left a resumable runtime | Mark runtime failed, record failure, and reject later `step`/`run`; reject non-callable policy at construction | invalid-policy and injected transition-failure tests |
| Deadline and message handling could scan unrelated work; cost ledger was incomplete | Tick-bucket message/deadline queues and explicit operation/materialization counters; time and RSS are reported separately | counters in raw CSV; selective mode has no full-state scan/materialization |
| Snapshot/clone copy work was uncharged | Record snapshot calls, array bytes, wall time, serialized bytes, and clone bytes/time in a diagnostic ledger excluded from semantic snapshots | `test_snapshot_and_clone_copy_costs_are_recorded` |
| Clone test lacked a one-action counterfactual | Fork the same state and apply one HOLD vs UPDATE override; verify untouched source branch and deterministic branch-local continuation | `test_clone_counterfactual_intervention_is_branch_local` |
| Repeated runs could overwrite prior evidence and counter differences could pass | Unique run IDs, refuse existing paths, save exact two-repeat hashes and compare operation-count JSON; hash sources, inputs, and all output files | `test_repeat_gate_detects_counter_differences`; overwrite guard returned `FileExistsError` without altering prior files |

## Review conclusion

**ACCEPT for the scoped v0.7.0 software-semantics milestone.** This does not establish that no bug exists outside the reviewed tests and code paths. It does not validate ICEF as a scientific framework and does not establish task benefit, dynamic policy utility, wall-clock speedup, biological validity, novelty, or publication readiness.

## Completion-package evidence gap

The two initial failed test iterations were retained as a textual repair history, but their original command stdout/stderr were not captured. This is a historical provenance gap. Final verification commands were rerun after the implementation review and their exact outputs, stderr streams, and exit codes are preserved in `verification/`; this does not recreate the missing initial logs.
