# E1 protocol deviations and unresolved freeze items

## Confirmed deviation

The implementation schedules smooth external inputs to driven nodes every 0.01 s (`src/ircn/core.py`, `integrate_events`). The pre-registration freezes the 0.01 s output sampling interval but does not explicitly freeze that interval as the event arrival cadence for a continuous smooth signal. This cadence materially affects B2/P event counts and trajectory accuracy. It was chosen by implementation convention and not as a separately documented pre-outcome protocol parameter.

Therefore the corrected threshold scan in `reports/e1_20261008_112905/` is an implementation-level calibration probe, not fully compliant confirmatory evidence under `pre_registration_v1.yaml`. Its no-qualified-threshold result is reported transparently and supports not proceeding to E2, but it cannot establish a preregistered G1 conclusion. Any future experiment must define the smooth-input event-generation rule and freeze it before new evaluation data are collected; this run's observed values must not be used to tune that rule.

## Unresolved comparator and cost details

- B1/B3 use dense ndarray storage and dense matvecs for a sparse topology. Dense kernels can be faster at N=64/128 while CSR can be faster at N=256, so the reasonable kernel choice should be established and frozen before any future timed E1.
- `ru_maxrss` is a process-lifetime high-water measurement. It cannot be attributed to individual methods in the shared-process run path. No method-level memory comparison is claimed.
- The legacy AI source checkout was unavailable, so the source-level reuse audit is incomplete. No legacy code or result was used as evidence.
