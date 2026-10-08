# Pre-implementation defect review — ICEF metric hardening

Review target: `model/v0.4.1/icef_metrics.py` at parent `origin/main` (`42ffcf87c400cd7007df3749db08aa1eac51b2db` includes the later Phase II merge chain; verify actual `git merge-base` before release).

Review mode: separate read-only review pass by the implementation owner; not independent.

## Findings reproduced before edits

| ID | Severity | Finding | Reproduction evidence | Required acceptance check |
|---|---|---|---|---|
| F1 | Major | `compute_selectivity` coerces values to bool; NaN becomes `True`, corrupting selected fraction/recall without an error. | `compute_selectivity([NaN,0],[1,0])` returned selected fraction 0.5 and critical recall 1.0. | Reject NaN/infinity/non-binary masks; accept bool and finite 0/1 inputs. |
| F2 | Major | `long_horizon_summary` accepts duplicate or decreasing timestamps and can return a negative horizon. | `times=[2,1]` returned horizon -1.0. | Require finite, strictly increasing, nonempty timestamps; add invalid-input tests. |
| F3 | Major | `stable_recovery_time` accepts NaN threshold/dwell, returning NaN as though recovery were censored. | `threshold=NaN` returned NaN instead of a validation error. | Require finite threshold and dwell and reject negative values. |
| F4 | Minor | `influence_fidelity` accepts a non-integer `k` through validation, then leaks a Python slice `TypeError`. | `k=1.5` raised `TypeError: slice indices must be integers...`. | Validate integer-valued k up front; bool is not a valid k. |

## Scope and stop condition

Fix only these input-contract defects in the ICEF metric helpers, add regression tests for each, and preserve existing valid bool/0-1 inputs. Do not alter metric estimands, thresholds, diagnostic tasks, or prior results. The patch is not ready for release until targeted tests, the existing regression suite, diff review, and result-hash checks pass. This audit makes no claim about defects outside the reviewed module and cases.
