# Decision v1 — IRCN v0.3.3 local ODE segment candidate

## Decision: STOP this candidate; do not run confirmatory WP1 or E2

### Facts

- The candidate used the single frozen configuration: segment tolerance `1e-7`, maximum interval `0.05 s`, smooth-input gate `0.001 s`.
- Of 96 preregistered calibration instances, 33 were completed before stopping. There were zero solver exceptions; 16 completed instances exceeded the frozen `6.837551017609374e-08` relative NRMSE limit and 17 were below it.
- The highest observed NRMSE was `0.07487`. A N=128 dense-burst instance had `0.01262`; a N=64 star/smooth instance had `2.81e-7`, already above the limit.
- Mean single-run candidate wall time among completed instances was `7.14 s` for a 2 s simulation; the maximum was `37.53 s`. These calibration measurements were not warmup/repetition-controlled and cannot support a formal performance comparison.
- All completed per-instance rows include input hashes; events are retained in `events.jsonl.gz`. No confirmatory seeds, E2, or downstream science were run.

### Interpretation

The frozen rule requires every calibration instance to pass. The observed accuracy failures are sufficient to reject eligibility, and the measured runtime is poor. The remaining scan was stopped because no further row could change the all-instances-required decision. This rejects this candidate configuration, not the entire family of event-driven solvers.

### Unresolved

The prototype still has no demonstrated full-scope accuracy-cost path. Any materially different method requires its own prospective contract and untouched evaluation seeds. No efficiency or publication claim is supported.
