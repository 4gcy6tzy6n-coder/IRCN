# v0.3 work package status

| WP | Status | Evidence / reason |
|---|---|---|
| WP0 | COMPLETE WITH AUDIT LIMITATIONS | Version lineage, contract revisions, source/dependency manifests and threat model recorded. v0.2 history remains unchanged. |
| WP1 | PASS (accuracy/correctness scope only) | v0.3.4i passed 96/96 calibration and 240/240 untouched-seed confirmation cases. Max NRMSE `4.2635e-9` against `6.8376e-8`; zero runtime failures. This does not establish efficiency or G1. |
| WP2 | INCONCLUSIVE | All 528 instances and 50,176 rows are present with zero label-generation failures, but one frozen candidate trajectory exceeded the WP1 accuracy bound (NRMSE `0.0086401`). See `reports/WP2/INFLUENCE_VALIDITY_REPORT.md`. |
| WP3 | BLOCKED | WP2 is inconclusive; no scheduler-level outcome/efficiency experiment was run. |
| WP4 | BLOCKED | WP1–WP3 prerequisites are not all passed; no connectome data analyzed. |
| WP5 | BLOCKED | WP1–WP3 prerequisites are not all passed; no transfer task run. |
| WP6 | BLOCKED | Algorithm and WP1–WP3 prerequisites are not all passed; no hardware matrix run. |
| WP7 | IN PROGRESS | Historical failures and the WP2 negative accuracy result are retained. No manuscript claims added. |

No WP3–WP6 scientific computation or NMI manuscript change was made.
