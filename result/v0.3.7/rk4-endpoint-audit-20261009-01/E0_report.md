# v0.3.7 RK4 terminal-boundary audit report

**Audit:** `rk4-endpoint-audit-20261009-01`
**Execution status:** `COMPLETED_AUDIT`; post-report parent integrity/archive verification passed.
**Claim boundary:** one frozen 128-node `dense_burst` instance only. No candidate solver reruns, no efficiency claim, no WP2 upgrade, and no E2 authorization.

> `POST_REPORT_ARCHIVE_VERIFIED` — the complete frozen parent package was verified after the report and raw numerical outputs were complete. Source, input, and output hashes are recorded in the archive manifest.

**Gate history:** the pre-archive report used `PENDING_POST_REPORT_ARCHIVE`; after verification passed, this final report replaced that pending conclusion with the verified status above, and the final manifest was regenerated.

## Facts

- The frozen instance is the v0.3.6 validation case: 128 nodes, `dense_burst`, seed 128012, 2.0 s duration, and 0.01 s reporting interval. The seven driven node IDs were `[17, 37, 38, 40, 61, 95, 97]`.
- The terminal input discontinuity was detected at 2.0 s even though the input breakpoint API omitted that endpoint. Corrected RK4 changed only the terminal sample and only those seven driven nodes. The maximum difference outside the terminal sample was exactly 0 for both step sizes.
- The corrected coarse/fine RK4 pair (steps 0.001/0.0005 s) had maximum absolute difference `9.21e-15`; both directional NRMSE values were `3.30e-14`.
- Corrected RK4 versus the frozen DOP853 trajectory had maximum absolute differences `4.71e-14` (coarse) and `4.12e-14` (fine). Both directional NRMSE values were below `7.0e-14` for both runs. Both corrected refinements passed the unchanged v0.3.6 thresholds.
- The original v0.3.6 reference check recorded coarse/fine RK4 maximum differences of `1.6762e-5` and `8.3810e-6` against DOP853, and a coarse/fine refinement difference of `8.3810e-6`; these failed its frozen checks. Those values are retained in the v0.3.6 `reference/oracle_check.json`.
- The correction was localized: maximum terminal changes were `1.6762e-5` for coarse RK4 and `8.3810e-6` for fine RK4. No nonterminal sample changed.
- Candidate trajectory reanalysis used the seven frozen arrays without rerunning solvers. Against corrected fine RK4, the repeated baseline and batched-edge variant had NRMSE about `8.64e-3`; the two scalar-delivery variants had NRMSE about `1.48e-9`; tighter local and segment tolerance variants had NRMSE about `1.63e-2` and `1.21e-2`, respectively. Full values are in `candidate_reference_comparisons.csv`.
- Corrected RK4 wall times were approximately 0.082 s (coarse) and 0.162 s (fine), measured only for this audit computation. They are not an E1 speed comparison.
- Frozen v0.3.6 status remains `COMPLETED_DIAGNOSTIC`; its WP2 finding remains `INCONCLUSIVE`; WP3–WP6 remain blocked. No files in the v0.3.6 package were intentionally modified.

## Interpretation

The corrected independent RK4 refinements agree with the frozen DOP853 trajectory to near floating-point precision, while their difference from the original RK4 arrays is confined to the terminal sample and exactly the externally driven nodes. This strongly supports the interpretation that the earlier RK4/DOP853 discrepancy for this instance came from the independent RK4 harness evaluating the right-continuous input at its final stage instead of using the left limit at a terminal jump.

This corroborates the DOP853 reference for this one frozen instance. It does not establish correctness for other tasks, graphs, seeds, or solver paths. The candidate reanalysis is only an offline comparison to the corrected reference; it does not establish task utility, general solver quality, or efficiency.

## Unresolved / pending

- The archive verified every v0.3.6 parent-manifest entry, the parent status snapshot, source cleanliness, and the expected environment provenance. The final archive manifest was regenerated after updating this report.
- The discrepancy root cause is strongly localized but has only been audited on one terminal pulse boundary and one frozen instance.
- The large candidate differences for several frozen variants remain unexplained by this audit; no new candidate executions or parameter changes were made.
- No statistical inference is applicable to this single-instance audit.

## Decision boundary

This audit does not alter the parent decision: WP2 stays `INCONCLUSIVE`, WP3–WP6 stay `BLOCKED`, and entering E2 is not supported by this work. Any future broader oracle validation needs a separately reviewed and frozen protocol.
