# E1 — solver Pareto comparison

## Status

**G1 was not opened.** The preregistered independent threshold calibration failed its NRMSE eligibility condition for IRCN P on the first graph size, `N=64`. Per the frozen rule, no evaluation-seed timing comparison was started and no efficiency claim is made. B2 also had no qualifying calibration threshold and would have been excluded as an accuracy-ineligible comparator.

The calibration run and raw logs are under `reports/e1_20261008_110424/`. `threshold_calibration_all.json` contains every evaluated candidate row. `calibration_event_replay.jsonl.gz` contains a deterministic untimed replay of the lowest pre-registered threshold for both methods and both workloads; its SHA-256 and input hashes are recorded in `calibration_event_replay.json`. That replay is an audit trace, not an E1 benchmark run.

## Calibration facts

Frozen NRMSE ceiling: `0.001`. The calibration scan is stopped early for a candidate once it fails at N=64, because a threshold failing one required workload cannot satisfy the frozen rule for all workloads and graph sizes.

| Method | Threshold | Pulse NRMSE (N=64) | Smooth NRMSE (N=64) | Calibration eligibility |
|---|---:|---:|---:|---|
| P | 0.01 | 0.151663 | 0.036825 | Fail |
| P | 0.003 | 0.144940 | 0.020015 | Fail |
| P | 0.001 | 0.159477 | 0.007704 | Fail |
| P | 0.0003 | 0.160646 | 0.006098 | Fail |
| P | 0.0001 | 0.156302 | 0.002452 | Fail |
| P | 0.00003 | 0.130473 | 0.000921 | Fail |
| P | 0.00001 | 0.127955 | 0.000373 | Fail |
| B2 | 0.01 | 0.665239 | 1.000000 | Fail |
| B2 | 0.003 | 0.665311 | 1.000000 | Fail |
| B2 | 0.001 | 0.665096 | 1.000000 | Fail |
| B2 | 0.0003 | 0.665076 | 1.000000 | Fail |
| B2 | 0.0001 | 0.464813 | 1.000000 | Fail |
| B2 | 0.00003 | 0.464607 | 1.000000 | Fail |
| B2 | 0.00001 | 0.464360 | 1.000000 | Fail |

All figures above are from the single frozen calibration seed `8675309`; they are gate-calibration observations, not independent statistical evidence. The ten evaluation seeds were not opened. B0/B1/B3 and P/B2 were not timed for E1.

## Interpretation

The current event implementations do not meet the preregistered trajectory-accuracy requirement on the pulse-driven N=64 calibration case. Tightening P's threshold improved smooth-input accuracy but did not bring the pulse workload close to the NRMSE ceiling. This is evidence against this frozen implementation/configuration, not evidence that all local event solvers are impossible. The results do not identify whether the main cause is pulse-boundary handling, asynchronous coupling error, the local predictor, or another implementation limitation; no post-result parameter change was made.

Because no candidate P threshold qualified, there is no valid same-accuracy runtime comparison. The `E1_results.csv` file is intentionally header-only; it does not contain fabricated timing rows.

## Decision

**STOP** the current acceleration/connectome-extension line under the frozen decision rule. Do not proceed to E2. No profiling-based implementation repair was justified by these calibration outputs, so the single allowed repair was not used.

## Unresolved

- E1 real wall-clock, latency distributions, memory comparisons, and Pareto results remain unmeasured.
- This test used synthetic graphs only; biological topology and AI transfer were not tested.
- The source-level legacy AI repository audit remains incomplete because the referenced checkout was not available locally or through the restricted network. See `legacy_reuse_audit.md`.
