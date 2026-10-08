# Calibration decision — v0.3.4i local batch size 2

## Status: calibration accuracy PASS; proceed to frozen confirmation only

### Facts

- Frozen contract: `contracts/V03_4I_LOCAL_BATCH2_CALIBRATION.yaml`, SHA-256 `0587a37039ddeb785d5c14b035ca5415d5fda0634598880172f2df0d06dc8db8`.
- All 96 calibration rows completed across four graph families, three sizes, four tasks and seeds 401/409. There were 96 accuracy passes, zero accuracy failures and zero runtime failures.
- Maximum relative NRMSE was `5.5743e-9`, below the frozen `6.8376e-8` limit. Mean NRMSE was `9.5821e-10`.
- Single-run timing averaged 19.19 s (median 9.22 s, maximum 188.40 s) for a 2-second simulated interval. Mean timing was 3.26 s at N16, 15.77 s at N64, and 38.55 s at N128. These measurements are calibration diagnostics only.
- The raw 96 rows, per-instance input hashes, event stream, summary and code hashes are retained in this directory.
- No confirmatory seeds or downstream work-package science were used in calibration.

### Interpretation

The candidate is eligible for its separately frozen confirmation on seeds 419, 421, 431, 433 and 439. Calibration does not establish WP1 confirmation, trajectory generalization, comparative efficiency, G1 or permission to run E2. Runtime remains very high on modular sparse N128 dense-burst cases.

### Unresolved

The measured runtime has no paired baseline and is not repeated-timing controlled. Whether accuracy generalizes and whether local batching reduces end-to-end cost under a fair comparator remain open.
