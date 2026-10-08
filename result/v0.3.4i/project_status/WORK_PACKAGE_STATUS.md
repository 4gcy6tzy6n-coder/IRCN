# v0.3 work package status

| WP | Status | Evidence / reason |
|---|---|---|
| WP0 | COMPLETE WITH AUDIT LIMITATIONS | Version lineage, contract revisions, source/dependency manifests and threat model recorded. v0.2 history remains unchanged. |
| WP1 | PASS (accuracy/correctness scope only) | v0.3.4i passed 96/96 calibration and 240/240 untouched-seed confirmation cases. Max NRMSE `4.2635e-9` against `6.8376e-8`; zero runtime failures. This does not establish efficiency or G1. |
| WP2 | READY TO DESIGN; NOT RUN | WP1 prerequisite now passes. Freeze WP2 design and contract before generating influence labels or running science. |
| WP3 | BLOCKED | WP2 has not been scientifically run or passed. |
| WP4 | BLOCKED | WP1–WP3 prerequisites are not all passed; no connectome data analyzed. |
| WP5 | BLOCKED | WP1–WP3 prerequisites are not all passed; no transfer task run. |
| WP6 | BLOCKED | Algorithm and WP1–WP3 prerequisites are not all passed; no hardware matrix run. |
| WP7 | IN PROGRESS | Failed and invalidated candidates are retained. v0.3.4i now passes WP1 accuracy confirmation; fair efficiency comparison remains outstanding. |

WP2 entrypoint reports readiness only. No WP2 scientific computation, E2/E3/E4, or manuscript claim was produced.
