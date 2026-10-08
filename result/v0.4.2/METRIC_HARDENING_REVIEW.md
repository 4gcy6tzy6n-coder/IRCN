# ICEF metric hardening — completion review record

**Status:** READY for review/publish as a scoped code correction. This is not certification that the whole ICEF framework or IRCN model is defect-free.

## Acceptance contract and reviewed scope

The pre-implementation audit and reproduced findings are in `PRE_IMPLEMENTATION_REVIEW.md`. The patch is isolated to a versioned copy at `model/v0.4.2/`; the published `model/v0.4.1/` source and tests remain byte-identical to `origin/main` (verified by `git diff --exit-code origin/main -- ...`). Metric estimands, data, tasks, thresholds, and prior result files were not changed.

## Review and fixes

Review mode: separate read-only self-review by the implementation owner; **not independent**. Reviewed code and tests line-by-line after implementation.

- **F1 (Major):** binary masks now accept booleans or finite numeric 0/1 only. NaN, infinity, and other numeric values raise `ValueError`. Regression coverage added.
- **F2 (Major):** long-horizon summaries now reject duplicate/decreasing timestamps; `_vector` already rejects non-finite timestamps. Regression coverage added.
- **F3 (Major):** recovery threshold/dwell must be finite and nonnegative; invalid values no longer masquerade as censored recovery. Regression coverage added.
- **F4 (Minor):** `k` must be an integer type and cannot be bool; invalid values raise the documented `ValueError` rather than leaking slice `TypeError`. Regression coverage added.

No blocker, major, or minor findings remain in the reviewed changes and reproduced cases. Residual uncertainty remains outside those input paths and the tested metric behaviors.

## Verification evidence

- `python -m pytest -q model/v0.4.2/test_metric_hardening.py` — 18 passed.
- `python -m pytest -q model/v0.4.1/test_icef_metrics.py` — 23 passed against the preserved published snapshot.
- `python -m pytest -q` — 47 passed.
- `git diff --check -- model/v0.4.2 result/v0.4.2` — passed.
- `git diff --exit-code origin/main -- model/v0.4.1/icef_metrics.py model/v0.4.1/test_icef_metrics.py` — passed; old version unchanged.
- File hashes are recorded in `SHA256SUMS.csv`.

## Scientific status and limitations

This is software input-validation hardening only. It does not close Phase II construct-validity gaps, add wall-clock benchmarking, train IRCN, establish task utility, or authorize Phase III/IV claims. Passing tests establish only the tested behaviors.
