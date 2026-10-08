# Decision — v0.3.4e exact-exogenous calibration candidate

## Status: calibration accuracy PASS; no confirmatory evaluation

### Facts

- The prospective contract is `contracts/V03_4E_EXACT_INPUT_CALIBRATION.yaml` (SHA-256 `647988deb29a0dda054ba8a286e185a7e76be33c43e72d700cfa4d94ca605cd5`).
- All 96 frozen calibration instances completed, with 96 accuracy passes, zero accuracy failures and zero runtime failures. Maximum relative NRMSE was `4.8716e-9`, below the frozen limit `6.8376e-8`.
- Total elapsed time was 2384.22 seconds. Per-instance wall time averaged 24.46 seconds (median 12.15, maximum 230.15) for a 2-second simulated interval. These are single calibration-run measurements, not a controlled comparator experiment.
- No confirmatory seeds were used. No E2 or downstream science was run.
- The test suite available in the current checkout passes 38 tests; this is a later repository-wide check and does not replace this candidate's frozen calibration evidence.

### Interpretation

This establishes the v0.3.4e candidate's calibration accuracy on the specified screen only. The observed wall time is too costly to claim an efficiency benefit and there is no matched baseline comparison. Passing calibration accuracy is necessary but insufficient for WP1 confirmation or E2.

### Unresolved

Performance cause, repeatability, confirmatory accuracy, and matched end-to-end cost remain unestablished. The later v0.3.4h screen attempted a looser local integrator tolerance and failed its own frozen calibration gate; it does not invalidate the v0.3.4e measurements but provides no eligible faster replacement.
