# CBCC v2 data split and provenance

## Frozen split

`pre_registration_v2.yaml` was frozen on 2026-10-09 before the fresh validation run.
The independent unit is the top-level task seed; each seed generates eight sequences.

| Split | Top-level seeds | Use |
|---|---:|---|
| Training | 100–109 | Family readout fitting and sparse-key influence-estimator fitting; readout ridge selected by grouped 5-fold CV over these seeds. |
| Fresh validation | 1200–1207 | Full-update versus zero-output learnability gate. |
| Pilot | 1300–1307 | Select a per-seed budget-matched baseline and estimate sample size. The pilot found no eligible baseline. |
| Confirmatory | 3000 onward | Not run because no pilot baseline met the frozen 1% per-seed realized-MAC match. |
| Retired development validation | 200–207 | Used only to diagnose v1 failure. Not reused in v2 validation, pilot, or confirmatory splits. |

Task generation is deterministic from `(family, top_level_seed, sequence_index)` using NumPy `SeedSequence`. The two task families are `sparse_key` and `feedback_switch`. The influence estimator is trained only on sparse-key training sequences; feedback-switch evaluation is zero-shot with respect to estimator fitting.

## Provenance and retention

No external or participant data are used. Inputs and targets are generated in code by `cbcc_tasks.py`. The v2 fresh-validation run is retained under `results/validation_v2_20261009T092706.571860Z/`; the pilot and its infeasibility freeze are retained under `results/pilot_20261009T092726.879189Z/`. These directories contain per-seed metrics, per-sequence metrics, per-tick event JSONL, gate/freeze metadata, and the pilot failure record.

The v1 development-gate failure is retained under `results/validation_gate_20261009T092156.827901Z/`. An earlier serialization failure during preflight is retained in `results/failures.jsonl`.

The pilot `failures.jsonl` entry was reconstructed after the run at 2026-10-09 17:28:26 +0800, after the pilot metrics and freeze were written at 17:27:52. The runner's explicit logging fix was saved at 17:28:23. This is a provenance correction documenting the already-observed pilot stop (no baseline passed per-seed budget eligibility); it does not alter or regenerate scientific metrics, event records, or the freeze. The error text records the reason from the stored infeasible freeze. No confirmatory outcomes exist.
