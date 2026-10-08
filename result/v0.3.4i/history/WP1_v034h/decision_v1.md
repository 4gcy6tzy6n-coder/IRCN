# Decision — v0.3.4h fast local-tolerance candidate

## Decision: STOP this candidate; no confirmatory WP1 or E2

### Facts

- Frozen contract: `contracts/V03_4H_FAST_LOCAL_TOLERANCE_CALIBRATION.yaml`, SHA-256 `2b031c90b4ddba6e476e3fcc1a0c05d96f8669def3647942d14f0cdb77286ad7`.
- The candidate used exact exogenous input evaluation, broadcast-batch delivery, 20 ms maximum segments, and local DOP853 tolerances `rtol=1e-7`, `atol=1e-9`.
- The frozen rule required every calibration row to meet relative NRMSE `6.837551017609374e-08`, stopping at the first decisive failure. Of 96 planned instances, 61 completed: 60 passed and one failed; no runtime failures were recorded.
- The failing instance was `feedback_ring/N64/dense_burst/seed293`, with NRMSE `7.034060227662225e-08` (about 2.9% over the limit). The partial-screen maximum is this value. Mean candidate wall time for the 61 completed rows was 10.89 s (median 9.84, maximum 43.93) for 2 s simulated time; single-run calibration timing is diagnostic only.
- The preserved CSV, JSONL raw rows, compressed event log, schema-failure record, and run metadata are in this directory. The earlier schema error occurred before any candidate integration; the corrected runner's code hash is recorded in `calibration.json`.
- No confirmatory seeds, E2, or downstream work-package science were run.

### Interpretation

The frozen all-rows criterion failed, so this candidate is ineligible for confirmatory evaluation. The faster observed partial-screen timing does not support an efficiency claim and cannot compensate for the accuracy failure. This rejects this configuration, not all causal event-driven methods.

### Unresolved

The speed/accuracy tradeoff and the algorithmic source of runtime remain open. Any materially revised candidate needs a new prospective contract and untouched evaluation seeds.
