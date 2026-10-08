# IRCN version lineage

## Frozen parent
- v0.2 source commit: `2e9896e4abc632428fa6884663036383bb92b419` (`2e9896e`), on branch `main` before v0.3 branch creation.
- v0.2 decision: `STOP` the acceleration/connectome expansion route; no E2 authorization. Preserve v0.2 reports and all failed attempts unchanged.
- v0.3 contract source: `IRCN_v0.3_%E5%AE%8C%E6%95%B4%E7%A0%94%E5%8F%91%E4%B8%8E%E9%AA%8C%E8%AF%81%E6%96%B9%E6%A1%88.md`, SHA-256 `7f7e695c8ba1fa4f3d71808a8512496b93898b326efa0cea662c046a7cd4d5bf`.
- v0.3 branch: `v0.3-development`, branched from the frozen commit above.

## Historical evidence (not v0.3 evidence)
- E0 latest v0.2 run: `reports/e0_20261008_112849` (13 passed).
- E1 latest v0.2 calibration: `reports/e1_20261008_112905`; thresholds were not selected and confirmatory evaluation did not run. Pulse minimum P NRMSE was 0.002503 at N=64, above the old v0.2 0.001 ceiling.
- Earlier calibration attempt was invalidated and retained. See `protocol_deviations.md` and the original run directories.
- These runs establish only what v0.2 did; they do not establish or refute v0.3.

## Boundary
No v0.2 scientific result is imported as v0.3 evidence. `NeuroConverge` and `nmi_feedback_routing` remain outside this development line and are not modified.
