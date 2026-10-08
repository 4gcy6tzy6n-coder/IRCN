# E1 calibration report

**G1 was not opened.** No IRCN threshold met the frozen NRMSE ceiling; evaluation-seed timing was not started.

Calibration run: `e1_20261008_112905`. Selected thresholds: `{'P': None, 'B2': None}`.

The corrected run includes an untimed deterministic event replay at `reports/e1_20261008_112905/calibration_event_replay.jsonl.gz`; hashes and row count are in its adjacent JSON manifest. The first attempt is invalidated and preserved under `reports/e1_20261008_110424/`.

| Method | Threshold | N | Workload | NRMSE | Node refinements |
|---|---:|---:|---|---:|---:|
| P | 0.01 | 64 | sparse_pulses | 0.45044306 | 78 |
| P | 0.01 | 64 | smooth_periodic | 0.15528728 | 4144 |
| P | 0.003 | 64 | sparse_pulses | 0.1929268 | 198 |
| P | 0.003 | 64 | smooth_periodic | 0.07935227 | 4557 |
| P | 0.001 | 64 | sparse_pulses | 0.094235385 | 545 |
| P | 0.001 | 64 | smooth_periodic | 0.03704003 | 5596 |
| P | 0.0003 | 64 | sparse_pulses | 0.04304284 | 1715 |
| P | 0.0003 | 64 | smooth_periodic | 0.013895456 | 10261 |
| P | 0.0001 | 64 | sparse_pulses | 0.01869493 | 5000 |
| P | 0.0001 | 64 | smooth_periodic | 0.0053846764 | 24623 |
| P | 3e-05 | 64 | sparse_pulses | 0.0068049326 | 16388 |
| P | 3e-05 | 64 | smooth_periodic | 0.0016954643 | 75626 |
| P | 1e-05 | 64 | sparse_pulses | 0.0025033506 | 48948 |
| P | 1e-05 | 64 | smooth_periodic | 0.00057889043 | 222327 |
| B2 | 0.01 | 64 | sparse_pulses | 0.40086223 | 22 |
| B2 | 0.01 | 64 | smooth_periodic | 0.21571999 | 4022 |
| B2 | 0.003 | 64 | sparse_pulses | 0.91918019 | 51 |
| B2 | 0.003 | 64 | smooth_periodic | 0.10871869 | 4203 |
| B2 | 0.001 | 64 | sparse_pulses | 0.9512937 | 75 |
| B2 | 0.001 | 64 | smooth_periodic | 0.05301934 | 4803 |
| B2 | 0.0003 | 64 | sparse_pulses | 0.94910738 | 165 |
| B2 | 0.0003 | 64 | smooth_periodic | 0.02420805 | 6757 |
| B2 | 0.0001 | 64 | sparse_pulses | 0.73571875 | 342 |
| B2 | 0.0001 | 64 | smooth_periodic | 0.010676811 | 11723 |
| B2 | 3e-05 | 64 | sparse_pulses | 0.66584103 | 752 |
| B2 | 3e-05 | 64 | smooth_periodic | 0.0043828832 | 20898 |
| B2 | 1e-05 | 64 | sparse_pulses | 0.36070976 | 1688 |
| B2 | 1e-05 | 64 | smooth_periodic | 0.0028982355 | 37290 |

These are calibration-seed eligibility observations, not independent inferential evidence. `E1_results.csv` remains header-only. No wall-clock or efficiency conclusion is available.

## Interpretation

The corrected local event implementation failed calibration eligibility. This does not establish that all event-driven solvers fail; it blocks this configuration from G1 and E2.

## Protocol deviation

The smooth-input event cadence is hardcoded at 10 ms. The preregistration sets a 10 ms output interval but does not explicitly freeze this cadence for continuous input. As documented in [protocol_deviations.md](/Users/yyl/Desktop/workshop/NMI/IRCN/protocol_deviations.md), this scan is an implementation-level calibration probe rather than fully confirmatory preregistered evidence. No latency conclusion is available either way.

## Unresolved

The per-method RSS field uses process-lifetime `ru_maxrss` and is not attributable to a single method in a shared process. B1/B3 dense versus CSR kernel choice also needs to be frozen before any timed comparison. Since G1 was not opened, no memory or efficiency comparison is reported. The legacy source-level audit remains incomplete.
