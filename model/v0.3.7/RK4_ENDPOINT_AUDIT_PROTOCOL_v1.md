# IRCN v0.3.7 terminal input discontinuity audit

**Status:** draft pending independent review. The contract and protocol are jointly normative. No audit run may start until an independent reviewer accepts the contract, protocol, runner, and tests. Hash calculation is deferred until the experiment and report are complete.

The accepted receipt must name the reviewed Git commit, reviewer, and exact five-file scope (contract, protocol, runner, tests, and post-report archive utility). Before execution, the runner verifies that commit identity, that scope, and a clean worktree; these checks use Git metadata and do not calculate experiment hashes. Execution exceptions preserve partial outputs with a failure record and atomic FAILED_PRESERVED status; an existing audit directory is never overwritten or rerun.

## Scope

This follow-up addresses one narrow discrepancy found in the completed v0.3.6 diagnostic. It will load the frozen DOP853 and RK4 arrays plus seven saved candidate trajectories. It will not rerun any candidate solver, edit the v0.3.6 run, change its preregistered thresholds, or upgrade WP2 status.

## Forensic observation motivating the audit

In the frozen run, coarse/fine RK4 errors against DOP853 are at numerical noise for samples through 1.99 s, then jump only at 2.00 s on the seven externally driven nodes. The observed fine-RK4 terminal difference is predicted by the omitted final RK4 stage's right-vs-left input contribution to within approximately 1e-13 in the forensic calculation. This is a hypothesis to test with the predeclared corrected reference, not a new success criterion.

## Corrected reference calculation

Reconstruct the frozen graph, system, and exact input from the parent contract and its verified v0.3.5 source snapshot. Independently implement classical full-state RK4 using the matrix RHS directly. Integrate at maximum steps 0.001 s and 0.0005 s. The boundary set is the union of start, end, all declared input breakpoints, and every 0.01 s report time. No step may cross a boundary; do not interpolate.

A generator may omit a discontinuity exactly at the terminal time because its breakpoint API returns only strict interior points. A raw inequality check would falsely classify every continuously varying signal as discontinuous. Therefore compare the componentwise increments over [t_end - 2e-9, t_end - 1e-9] and [t_end - 1e-9, t_end]. Classify a jump only when the final increment exceeds both eight times the preceding increment and a 64-machine-epsilon scale floor. If any component qualifies, treat t_end as a discontinuity endpoint: use the left-limit input for the final RK4 stage and record the continuous terminal state once. This rule is frozen and covered by jump and smooth-signal tests.

Save corrected coarse and fine RK4 arrays. Compare them in both NRMSE orientations and maximum absolute error. Apply the unchanged parent limits: refinement max absolute error and both directional NRMSE ≤ 1e-9; each corrected RK4 vs DOP853 max absolute error and both directional NRMSE ≤ 1e-8. Report the previously frozen and corrected values side by side.

## Frozen candidate re-evaluation

Load the seven exact candidate trajectories from v0.3.6 and calculate each candidate's NRMSE and maximum absolute error against each corrected RK4 reference. This is a deterministic secondary analysis of already completed candidates; do not rerun, retune, or replace any candidate.

Compare each corrected RK4 array with its corresponding frozen raw RK4 array. Any change must be confined to the terminal sample and driven nodes, with maximum nonterminal difference ≤ 1e-12. A violation blocks attribution and is reported as an implementation error. Corrected, oracle, and candidate arrays must have matching shapes and finite values.

## Decision and limits

If corrected RK4 refinements meet the parent limits and both agree with frozen DOP853, conclude only that DOP853 is independently corroborated on this one instance and that the earlier RK4 discrepancy came from terminal-boundary handling. If not, keep oracle validity unresolved. In either case WP2 remains INCONCLUSIVE; WP3–WP6 remain BLOCKED; the audit does not authorize E2 or any manuscript claim.

All parent artifacts are read-only. Save raw corrected arrays, direct comparisons, command/environment logs, review receipt, and a post-run manifest in the new v0.3.7 result directory. Do not calculate hashes during setup, review, or numerical execution. After all arrays, `E0_report.md`, and status files are complete, run `archive_after_completion.py` to verify the frozen parent manifest and calculate source, input, and output hashes. The report must state `PENDING_POST_REPORT_ARCHIVE` and make all attribution conditional until this check completes. Archive validation checks the full required output list, array dimensions/finiteness, parent status/revision and byte-for-byte preservation, commit identity, and source cleanliness. Any archive error changes status to `INVALID_ARCHIVE`; a parent integrity failure changes it to `INVALID_PARENT_INTEGRITY`.
