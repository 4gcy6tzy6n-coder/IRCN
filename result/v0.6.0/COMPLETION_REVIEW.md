# v0.6.0 Completion Review Gate

This gate is required before preparing the stage as complete. Passing tests alone does not mean that the implementation is defect-free; the report records the review scope, findings, fixes, and remaining limits.

## Before implementation

1. Freeze the versioned model contract, pre-registration, dependency lock, and golden fixtures.
2. Obtain an independent read-only contract review. Resolve all Blocker and Major findings before writing model code or generating train/test labels.
3. Reproduce the dependency environment from the lock and run the deterministic CPU / finite-gradient smoke check.

## After implementation

1. Review every changed source and test file against the frozen contract. Check tensor shapes, graph orientation, time/readout alignment, training/test separation, feature and label leakage, counterfactual branch independence, weighting, aggregation, and failure retention.
2. Run targeted tests for each finding and contract invariant, then the complete v0.6 test suite and available static/package checks. Inspect the actual outputs and logs; do not rely only on exit status summaries.
3. Request an independent code review after tests. Record each finding with severity, file/line, reproduction evidence, fix, and re-test result. Fix all Blocker/Major findings and all confirmed correctness bugs; rerun relevant tests after every fix and the full suite after the final fix.
4. Recompute source, configuration, input, and result hashes. Cross-check CSV aggregates and report claims against raw rows, including failures and undefined values.
5. Only after the review has no unresolved correctness findings and the final verification passes, prepare the stage completion report and publish the scoped `model/v0.6.0/` and `result/v0.6.0/` artifacts.

## Required completion record

The final review record must state reviewer role and independence limits; changed-file scope; checks run and outcomes; every finding and its disposition; any remaining non-correctness limitations; and the exact evidence supporting the stage decision. “No known defect found after the stated review and checks” is acceptable. An absolute claim that software is bug-free is not.

This process does not replace scientific gates or convert exploratory evidence into confirmatory evidence.
