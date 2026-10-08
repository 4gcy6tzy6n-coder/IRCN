# IRCN v0.7.0 — E0 runtime semantics report

## Status

**Software semantics gate: PASS within the frozen v0.7.0 scope.**

This stage implements a tick-synchronous adapter for the frozen v0.6 recurrent cell and tests its update-all path against the v0.6 synchronous rollout. It does not evaluate model quality, a learned event policy, compute allocation benefit, or runtime advantage.

## Facts

- The frozen checkpoint SHA-256 is `b8df57150d9030f2690eb11a7977af8c7b629e43eec228678d186d2c102168c5`.
- The parity run used 20 generated fixtures (10 held-out top-level seeds × 2 trajectory indices) plus one hand-built single-source pulse fixture.
- Each fixture ran twice independently: 21 fixtures × 2 = 42 attempts; all 42 passed.
- Maximum absolute state difference from the v0.6 Torch rollout was `2.831068712794149e-15`; maximum absolute readout difference was `1.3322676295501878e-14`. Both are below the frozen `1e-10` criteria.
- All 21 repeated semantic-output hashes and deterministic operation-count records matched exactly.
- The final run manifest verified 10 source/contract/dependency/checkpoint hashes and 85 output hashes against current files with no mismatch.
- The final v0.7 targeted suite passed: 36 tests. The repository root suite passed: 47 tests. The versioned v0.7 test directory is run separately from the root test configuration.
- Python compilation passed for v0.7 source and parity driver.
- The parity driver refused a duplicate run ID with `FileExistsError`; the existing run directory remained intact.
- Initial test iterations had two test failures caused by an incorrectly expected max-hold tick and a fixture that manually changed state without updating cached aggregates. The test fixture/expectation was corrected. Earlier implementation review rounds found and fixed major defects; full review history is recorded in `CODE_REVIEW_v1.md`.
- Exact outputs for the final targeted tests, repository regression, syntax check, duplicate-ID guard, and independent hash audit are preserved in [`verification/`](verification/). Raw stdout, stderr, and exit codes are retained. Original raw outputs from the initial two failing test iterations were not captured; the available record is the contemporaneous textual summary above, not a reconstructed log.
- Earlier parity output and runs are retained under `runs/superseded_pre_review/` and `runs/defectfix_01/`, `runs/defectfix_02/`; they are not pooled into the final gate. `runs/final_candidate_review_01/` is the final evidence directory.

## Interpretation

The adapter reproduces the frozen v0.6 synchronous transition and readout on the specified parity fixtures within float64 numerical tolerance. The selective candidate path is event-local by construction: event changes and due deadlines create candidates; deadline buckets avoid scanning all nodes; selected-node state is copied locally; selective `run()` does not materialize full-state history unless explicitly requested. Update-all mode is a parity reference and performs a full update by design.

These findings establish execution-semantic parity for the frozen fixtures. They do not show that selective execution is faster or preserves task quality, and operation counts are not a substitute for end-to-end timing.

## Unresolved limits

- v0.6 uses one directed path to readout and does not identify state-dependent compute-allocation value beyond topology.
- The v0.6 influence predictor is not deployed, retrained, or evaluated by this stage.
- The adapter uses a discrete tick barrier and is not a continuous-time asynchronous neural simulator.
- The numerical parity oracle is the existing v0.6 Torch rollout, while the adapter uses a separate NumPy local transition implementation; this is useful differential evidence but does not rule out a shared equation/specification error.
- This synthetic bridge does not validate ICEF, real-task utility, connectome contribution, biological interpretation, publication novelty, or cross-hardware behavior.

## Evidence

Raw per-fixture rows, semantic outputs, event traces, run metadata, and manifests are in [`runs/final_candidate_review_01/`](runs/final_candidate_review_01/). Independent verification logs and hashes are in [`verification/`](verification/). Frozen scope and thresholds are in [`../../model/v0.7.0/pre_registration.yaml`](../../model/v0.7.0/pre_registration.yaml).
