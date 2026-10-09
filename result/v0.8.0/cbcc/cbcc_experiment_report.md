# CBCC development and pilot report (v2)

## Status

**Confirmatory evaluation was not run.** The preregistered pilot completed on eight top-level seeds per task family but found no scheduler baseline within the frozen 1% per-seed realized-MAC eligibility bound. It therefore could not select a primary baseline or determine confirmatory sample size. This report is a development and pilot report, not a confirmation of CBCC benefit.

The code review found no blocker in the final reviewed patch; its formal verdict was COMMENT because Python LSP diagnostics were unavailable. The automated test suite passed 17 tests, and Python compilation passed. The final report, data split record, and experiment evidence are in `research/cbcc/`.

## Frozen protocol

The v2 preregistration is [pre_registration_v2.yaml](pre_registration_v2.yaml). It uses a 32-node, four-circuit recurrent reservoir, grouped training-seed CV for ridge selection, a sparse-key memory task, and a feedback-switch task. The sole primary endpoint is seed-level task MSE under a realized compute budget matched to within 1% for every paired seed. The primary success margin is 5% relative loss reduction. Confirmatory inference would use a paired seed-level analysis and Holm correction over eight planned comparisons.

The v1 development validation exposed poor retention of delayed weak cues. v2 changed the reservoir to a leaky-memory ring and declared fresh validation seeds before rerunning the learnability gate. v1 validation seeds 200–207 were retired from all v2 evaluation splits.

## Results

### Fresh validation gate

The full-update model beat the zero-output baseline in both task families on fresh validation seeds 1200–1207:

| Task family | Full-update MSE | Zero-output MSE | Gate |
|---|---:|---:|---|
| sparse_key | 0.010234 | 0.011306 | Pass |
| feedback_switch | 0.019627 | 0.026394 | Pass |

These values establish task learnability for the revised model on this split; they do not establish scheduler benefit.

### Pilot outcomes

Means below are descriptive summaries across the eight pilot seeds, with eight sequences per seed. They are not confirmatory statistical estimates. MAC is the counted policy-dependent multiply-accumulate proxy. Full update is shown as a quality reference and exceeds the scheduler budget.

| Family | Policy | Mean task MSE | Mean MAC per seed | Budget utilization | Mean wall seconds per sequence | Cue recall |
|---|---|---:|---:|---:|---:|---:|
| sparse_key | CBCC | 0.010840 | 504,552 | 95.35% | 0.00325 | 0.828 |
| sparse_key | Random | 0.010719 | 527,424 | 99.67% | 0.00045 | 0.312 |
| sparse_key | Fixed frequency | 0.010746 | 527,424 | 99.67% | 0.00037 | 0.312 |
| sparse_key | Activity | 0.010533 | 445,336 | 84.16% | 0.00078 | 0.500 |
| sparse_key | No influence | 0.010366 | 529,088 | 99.99% | 0.00051 | 0.234 |
| sparse_key | Full update | 0.009708 | 2,015,232 | 380.84% | 0.00058 | 1.000 |
| sparse_key | Zero output | 0.010839 | 0 | 0.00% | 0.0000049 | 0.000 |
| feedback_switch | CBCC | 0.026122 | 25,344 | 4.79% | 0.00328 | 0.000 |
| feedback_switch | Random | 0.025339 | 527,424 | 99.67% | 0.00049 | 0.312 |
| feedback_switch | Fixed frequency | 0.024916 | 527,424 | 99.67% | 0.00039 | 0.312 |
| feedback_switch | Activity | 0.026572 | 435,824 | 82.36% | 0.00086 | 0.547 |
| feedback_switch | No influence | 0.024327 | 529,088 | 99.99% | 0.00053 | 0.234 |
| feedback_switch | Full update | 0.016904 | 2,015,232 | 380.84% | 0.00059 | 1.000 |
| feedback_switch | Zero output | 0.026014 | 0 | 0.00% | 0.0000056 | 0.000 |

No scheduler baseline was eligible under the frozen per-seed budget rule in either family. For sparse_key, CBCC spent 3.8–5.4% less MAC than random/fixed-frequency on each pilot seed; for feedback_switch, CBCC spent dramatically less because its influence scores did not cause updates. The pilot freeze records every per-seed discrepancy and marks the confirmatory sample size infeasible.

At full precision, CBCC and zero-output MSE were 0.01083973 and 0.01083902 on sparse_key, and 0.02612243 and 0.02601414 on feedback_switch. CBCC therefore did not beat even the zero-output reference descriptively. Its feedback-switch cue recall was zero. Its measured runtime includes influence feature computation and scheduler overhead, and was slower than the non-CBCC schedulers in this small run. Runtime values are noisy single-run measurements at microsecond scale and are not inferential evidence.

## Facts, interpretation, and limits

### Facts

- The fresh v2 learnability validation passed for both task families.
- The pilot ran all five schedulers, full update, and zero output on eight pilot seeds per task family; per-seed metrics, per-sequence metrics, and per-tick event records were retained.
- No baseline met the frozen per-seed 1% realized-MAC criterion. The freeze file therefore contains no primary baseline and no feasible confirmatory sample size.
- No confirmatory evaluation, E2, or real-world task was run.
- An earlier preflight smoke attempt failed JSONL serialization on a NumPy integer. The serializer was corrected, and the failure record remains in `results/failures.jsonl`; it was not discarded as a successful run.
- The pilot's `failures.jsonl` entry was reconstructed after the metric/freeze files were written (metrics at 17:27:52; logging code edit at 17:28:23; failure record at 17:28:26, all +0800). It documents the infeasible budget gate from the saved freeze; it is not contemporaneous runner output and does not change the metrics or events.
- The separate historical engine audit covers one saved 128-node v0.3.7 case only; see [candidate_semantics_audit.md](../../reports/candidate_semantics_audit.md). It leaves WP2 `INCONCLUSIVE` and does not alter WP3–WP6 status.

### Interpretation

The pilot does not support the claim that the current future-influence scheduler improves task performance at equal realized compute. The zero feedback-switch recall and low MAC use suggest that the estimator's scores or decision rule fail to activate the task-relevant circuit in that family. In sparse_key, cue recall is higher, but this did not translate into lower MSE or runtime benefit.

The confirmatory question is currently not testable under the frozen comparison rule because CBCC and all four controls did not realize matching compute on paired seeds. Changing to a common compute cap or modifying scheduler decisions would define a new comparison contract and requires a new preregistration; it cannot be retroactively applied to these pilot data.

Final SHA-256 provenance is recorded after the experiment and report files in [provenance_manifest.sha256](provenance_manifest.sha256).

### Unresolved issues

- Why does the influence estimator fail to trigger on feedback-switch critical events?
- Can a fair equal-realized-compute comparison be designed without giving controls future knowledge of CBCC's per-seed consumption?
- Does influence scoring have any net value after its wall-clock overhead is included?
- Would another independent task family reproduce the observed learnability and gating behavior?
- A MAC proxy does not measure all memory traffic, Python overhead, or hardware execution cost; the pilot wall times are too short for robust latency conclusions.

## Decision

**Do not proceed to confirmatory evaluation under v2, and do not enter E2.** Preserve the pilot as an infeasibility and negative-result record. Any next algorithm study must first define a new compute-matching estimand, improve feedback-switch event activation using training data only, and preregister fresh seeds and comparators. These are recommendations for a separately scoped follow-up, not changes made in this run.
