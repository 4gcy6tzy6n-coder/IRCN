# IRCN project decision status — v0.3.4i WP1

## WP1 decision: PASS for correctness and trajectory accuracy only

### Facts

- The frozen calibration contract SHA-256 is `0587a37039ca`; 96/96 cases passed with zero runtime failures and maximum NRMSE `5.5743e-9` (limit `6.8376e-8`).
- The separately frozen confirmatory contract SHA-256 is `98e734a02a02`; all 240 untouched-seed cases passed with zero runtime failures and maximum NRMSE `4.2635e-9`.
- The full automated suite passed 40 tests in the recorded run.
- No WP2 science, E1/G1 baseline comparison, E2/E3/E4, connectome analysis, or manuscript result was generated.

### Interpretation

The v0.3.4i implementation clears the project's WP1 correctness/accuracy gate for the frozen synthetic scope. This supports proceeding to freeze a WP2 influence-validity protocol. It does not support an efficiency claim or justify jumping directly to WP3/E2.

### Unresolved

WP2 influence validity must be prospectively specified and evaluated. Fair matched cost comparisons, independent review, broader robustness, and downstream real-data relevance remain unestablished.

### Decision

Proceed to WP2 protocol design only. Do not claim E1/G1 or algorithmic speedup. E2 is not currently authorized by evidence; reconsider only after WP2 passes and its dependency contract is met.
