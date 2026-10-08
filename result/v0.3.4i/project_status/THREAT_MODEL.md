# v0.3 threat model

| Threat | Mitigation / evidence |
|---|---|
| Oracle error mislabeled as scheduler error | Independent SciPy DOP853 path, tolerance audit against tighter settings, preserve per-trace errors |
| Future information or exact oracle labels leak into online policy | Event queue only sees timestamped arrived messages; labels generated offline; API separation and causality tests |
| Full state/derivative computed before skipping | Instrument RHS/node evaluations; local event update computes only reached nodes and their outgoing messages |
| Baseline intentionally weakened | Compare standard SciPy DOP853, fixed RK4 and event baseline at frozen settings; report every baseline and noninferiority |
| Scheduler overhead omitted | End-to-end wall time wraps initialization through output collection; costs separately instrumented |
| Seed/trajectory leakage | Independent task instances and seeds for calibration/evaluation; no post-test threshold adjustment |
| Graph confounding | Synthetic motif tests first; connectome gate later requires matched structural controls and randomized task mapping |
| Memory measurement attribution | Record process peak RSS and label its scope; don't claim per-method peak without isolated process |
| Selective reporting | Preserve all raw records including failures/timeouts; publish per-seed and aggregate tables |
| Self-review presented as independent | Mark verification as same-agent or independent; no false independent sign-off |
