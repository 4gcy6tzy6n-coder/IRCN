# IRCN decision v1

**Decision: STOP** — stop the current acceleration/connectome-extension line. Do not proceed to E2 in this round.

## Facts

- E0 T0–T6 passed in a fresh run: 8 tests passed.
- The preregistered threshold scan evaluated all seven fixed thresholds at the calibration seed `8675309` on the first required graph size, N=64. Every P threshold exceeded the NRMSE limit of `1e-3` on the sparse-pulse workload (observed NRMSE 0.127955–0.160646). Every B2 threshold also failed calibration.
- P met the NRMSE limit for the smooth workload only at `3e-5` and `1e-5`; the co-primary pulse workload still failed. Under the frozen rule, no IRCN threshold was eligible.
- E1 evaluation-seed runs, comparator timing, and G1 inferential tests did not run. `E1_results.csv` is header-only. E2/E3/E4 did not run.
- The E1 calibration artifact includes input hashes, raw threshold metrics, scheduler counters, environment and code hashes. Representative untimed event traces are also retained.

## Interpretation

The tested event scheduler configuration failed the frozen trajectory-quality gate before efficiency could be assessed. Therefore no speed or Pareto claim is supportable. This result rejects continuing this exact implementation/configuration into the connectome stage; it does not establish a universal negative result for event-driven local solvers.

## Unresolved

The calibration errors do not isolate the error source. Pulse-boundary semantics, asynchronous neighbor-state staleness, and the local prediction/refinement rule remain possible contributors. No permitted profiling-based repair was established, so no repair-and-retest cycle was performed. The legacy code audit is limited to the audit notes supplied in the v0.2 plan because its source checkout was unavailable.

## E2 recommendation

**Not worth proceeding to E2 on the current evidence.** E2 requires an accuracy-qualified computation first; adding real topology now would confound structural interpretation with an already-failing solver. A future proposal would need a new, pre-outcome freeze after diagnosing the numerical failure and would not inherit this run's results as validation.
