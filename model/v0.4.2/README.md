# IRCN v0.4.2 — ICEF metric input-contract hardening

This patch preserves the published `model/v0.4.1/` snapshot and carries forward only validated input-contract fixes for ICEF metric helpers. It does not change metric definitions or scientific results.

Run the version-specific regression tests from the repository root:

```sh
python -m pytest -q model/v0.4.2/test_metric_hardening.py
```

Review findings reproduced before edits are recorded in `result/v0.4.2/PRE_IMPLEMENTATION_REVIEW.md`.
