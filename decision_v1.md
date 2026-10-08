# IRCN decision v1

**Decision: STOP** — do not enter E2 on the current evidence.

## Facts

- Corrected-source E0 T0–T6 passed for run `e0_20261008_112849`.
- Calibration run `e1_20261008_112905` found no IRCN threshold at or below the frozen NRMSE limit across required workloads and graph sizes.
- Evaluation seeds, E1 timing, G1 statistics, and E2/E3/E4 were not run. `E1_results.csv` is header-only.
- An earlier calibration attempt was invalidated after code review found a broken segmented oracle and event queue semantics; its raw artifacts are retained and marked in the audit record.
- The 10 ms smooth-input event cadence was used by implementation but was not explicitly frozen in v1. The corrected calibration is therefore an implementation-level probe, not fully confirmatory evidence; see `protocol_deviations.md`.

## Interpretation

The corrected implementation still fails calibration eligibility, and the cadence omission limits the calibration's confirmatory status. No efficiency claim is supportable. The operational decision is to stop this route and not enter E2; this does not reject all local event solvers.

## Unresolved

The specific sources of remaining trajectory error are not isolated. Per-method RSS is not attributable with the current shared-process high-water measurement. Dense versus CSR kernels for B1/B3 also need a pre-outcome choice before any future timed E1. The legacy source-level audit remains incomplete.

## E2 recommendation

Not worth proceeding to E2 on this evidence. A future attempt requires numerical diagnosis and a new preregistration; do not carry these failed runs forward as validation.
