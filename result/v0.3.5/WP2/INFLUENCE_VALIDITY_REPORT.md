# WP2 influence validity report — v1 frozen execution

## Facts

- The prespecified contract `contracts/V03_WP2_INFLUENCE_v1.yaml` remained unchanged (SHA-256 `f81da88afa5bc6d3a5ae204132f7afa74b32a00afdf7ecc6a1e2fc0f43574328`). No downstream WP3/WP4/WP5/WP6 computation ran.
- The completed run contains all 528 expected graph/task/seed instances and 50,176 expected node-snapshot labels. Coverage verification found 0 duplicate, missing, or unexpected rows. The run failure list is empty. Raw-label SHA-256: `683b3fb5182dc86420ba9f1a545344bc33b685f9bd20732eb404b37d4d758ed9`.
- Full automated test suite passed: 47 tests. The WP2-focused semantic and deterministic-sampling tests passed: 6 tests.
- Candidate full-trace accuracy did **not** meet the frozen WP1 bound. One of 528 instances exceeded the `6.837551017609374e-08` NRMSE limit: validation / chain / N=128 / dense_burst / seed 128012 had NRMSE `0.00864007689194927` and maximum absolute error `0.006075382462433044`. The largest of the other 527 cases was `1.1441322096568026e-08`.
- Therefore the contract-mapped outcome is `INCONCLUSIVE`, not PASS or FAIL. The frozen contract explicitly assigns this accuracy-gate failure to INCONCLUSIVE.
- Descriptive downstream metrics were computed from the complete data set, but they are not accepted as evidence for the predictor claim because the feature-state accuracy gate failed:
  - Validation selected `activity` as the strongest cheap baseline. Candidate validation Recall@20% was `0.86699`; activity was `0.78940`.
  - In-distribution test candidate-minus-selected-baseline Recall@20% was `0.05059`, with the prespecified 10,000-replicate cluster bootstrap 95% interval `[0.02981, 0.07254]` over 45 `(N, task, seed)` clusters.
  - Median feature-plus-scoring cost was `0.00007381 s` versus median saved local-update cost `0.0114839 s`; training wall time was `3.5035 s` versus `4.5203 s` total estimated saved update cost. These are descriptive and gated, not a WP2 pass.
  - OOD `state_switch` candidate Recall@20% was `0.86538` over 15 `(N, seed)` groups; it is reported separately and cannot remedy the failed accuracy gate.
  - Oracle label-generation wall time was `10,911.0 s` and is offline labeling expense, not inference efficiency.
- One initial implementation attempt (7 errors due a missing snapshot-time argument) remains preserved in `run_v1_attempt1_failed_implementation/` and is excluded from the final run. A later first analysis launch had a Python import-path failure after label generation completed. The analysis import path and run-entrypoint recovery path were fixed without changing the contract or raw labels; the analysis then completed and its outputs passed dataset integrity checks.

## Interpretation

WP2 does not establish influence-predictor validity. The observed ranking and cost numbers are promising descriptive observations, but the frozen candidate trajectory did not satisfy its required accuracy bound for one validation instance. Reporting only the favorable ranking or cost numbers would violate the frozen gate. WP3 therefore remains blocked.

The outlier is isolated to one frozen validation case; this does not identify the cause. The current evidence cannot distinguish a scheduler boundary-condition defect, a difficult interaction between the dense-burst input and this graph instance, or another numerical implementation issue. No test-specific fix or rerun is represented as confirmatory evidence.

## Open questions and limitations

1. What causes the single chain/N=128/dense_burst validation trajectory error, and can it be reproduced with a minimal numerical diagnostic?
2. Does a general, test-independent implementation repair preserve the WP1 contract and improve the entire frozen scope, or is this candidate family insufficient for the dense-burst regime?
3. The estimand remains algorithmic influence on an identity all-node trajectory. It is not biological causality or task utility.
4. Predictor feature/scoring timings are CPU measurements on this host and do not establish end-to-end scheduler acceleration or cross-hardware performance.

## Decision

**WP2: INCONCLUSIVE. WP3: BLOCKED. Do not proceed to WP3 under this result.** A future attempt needs a documented root cause and a separately frozen repair/validation protocol before any new confirmatory data are produced. This run is retained unchanged, including its negative accuracy result and earlier implementation/analysis failures.

## Reproducibility

- Frozen contract: `contracts/V03_WP2_INFLUENCE_v1.yaml`
- Run data and outputs: `reports/WP2/run_v1/`
- Label manifest: `reports/WP2/run_v1/labels_manifest.json`
- Analysis provenance for the post-generation import-path repair: `reports/WP2/run_v1/analysis_execution_manifest.json`
- Test command: `PYTHONPATH=.:src pytest -q` (47 passed)
