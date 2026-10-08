# WP1 v0.3.4i stage report

## Facts

- Frozen calibration: 96/96 cases pass; maximum NRMSE 5.5742639352578454e-09 against 6.837551017609374e-08.
- Untouched-seed confirmation: 240/240 cases pass; maximum NRMSE 4.2634849884279e-09; zero runtime failures.
- All 40 automated tests pass.
- Confirmation event log is losslessly recompressed; see `event_compression_manifest.json`.
- No WP2 influence science, E1/G1 comparison, E2/E3/E4, or connectome analysis ran.

## Interpretation

The v0.3.4i candidate passes WP1 accuracy/correctness for the frozen synthetic scope. WP2 protocol design is the next eligible step. This is not evidence of speedup, efficiency, influence prediction, biological validity, or generalization beyond the tested scope.

## Unresolved

WP2 estimand and protocol need prospective freeze and execution. Fair matched solver baselines, repeated timing, independent verifier signoff, and downstream biological relevance remain open.

## Decision

WP1: PASS for accuracy scope. Proceed to WP2 design only. E1/G1 and E2 remain unevaluated/not justified by this stage.
