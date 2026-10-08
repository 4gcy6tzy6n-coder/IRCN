# ICEF diagnostic task specifications v0.1

**Status:** task-design candidates for Phase II prototype validation. They are not frozen confirmatory tasks and were not executed. Values below define deterministic toy cases; confirmatory distributions, seeds, dimensions, and sample sizes still require a separate pre-registration.

All tasks use the same event-indexed runtime contract in `model/v0.4.0/architecture_semantics_v0.1.md`. Report task loss, update count, policy/scheduler/message/readout wall-clock cost, and event trace. Compare update-all, fixed periodic, budget-matched random, activity threshold, learned policy, and offline oracle (upper bound only). Keep model capacity, initialization, training data, and training compute matched. The task generator's target is a known-answer mechanism, not an external claim of general intelligence.

## A. Sparse perturbation propagation

**Question:** can event scheduling follow a sparse causal path without updating disconnected or weakly coupled regions?

Generate a directed graph with `N=32` nodes, a directed chain `0 -> 1 -> ... -> 7`, plus 24 isolated nodes. Set scalar edge gain to `0.8`, node decay to `0.5`, and all other edge weights to zero. At integer times `t=0..15`, inject `u_0(0)=1`, then zero input. Reference dynamics at each synchronous step are `h_i[t+1] = 0.5 h_i[t] + sum_j W_ji h_j[t]`; output is `y[t]=h_7[t]`. State and target are exactly computable by dense recurrence. Use squared output error and report false updates among nodes 8..31, path coverage, quality, and complete cost.

**Known-answer checks:** with all updates enabled, event trace follows the frozen node order and output matches the recurrence; with updates restricted to non-descendants, output stays zero; an offline path oracle updates only nodes 0..7. The oracle is not deployable.

**Adversarial cases:** a high-activity dead-end branch and a low-amplitude but high-gain path should distinguish activity thresholding from downstream influence. These cases test metric discrimination, not model capability.

## B. Multi-timescale latent integration

**Question:** can update allocation track fast and slow task components under a fixed total update budget?

Provide two observed scalar channels and a target `y(t)=s(t)+f(t)`. The slow latent is `s[t+1]=0.98 s[t]+0.02 a[t]`; fast latent is `f[t+1]=0.5 f[t]+0.5 b[t]`. Inputs `a,b` are piecewise constant; `b` alternates sign every 2 steps and `a` every 20 steps. Run 200 steps. A predictor must emit both components; loss is mean squared error over the full sequence and separately over transition windows. At each budget, count each local state transition equally and charge scheduler overhead separately.

**Known-answer checks:** a fixed-frequency oracle with periods 1 for the fast channel and 10 for the slow channel reproduces exact latent states when updates occur at input transitions and declared decay steps; a scheduler that never updates the fast channel has elevated transition-window error. This oracle only tests the metric/task generator contract.

**Adversarial cases:** shift the fast period at an unannounced midpoint and add fast-channel noise without changing the target; this distinguishes true demand tracking from frequency/activity chasing.

## C. Feedback-dependent hidden memory

**Question:** can the scheduler preserve a low-activity state that is necessary for a future output?

Use two recurrent units, memory `m` and readout context `r`. At cue time 0, input bit `q` is written into memory by `m <- q`. During delay steps 1..K, there is no external input and the memory update is an identity transition; at query step K+1, output is `y=m`, with target `q`. Evaluate `q` in `{0,1}`, delays `K` in `{5,20,50}`, and both positive/negative cue pairs from identical initial state. Task loss is binary cross entropy on the query output. The identity memory is deliberately quiet: skipping it is safe only if its stored state is preserved exactly.

**Known-answer checks:** update-all and a scheduler that holds the identity memory without erasing it achieve the same query output; a reset/decay ablation that overwrites memory fails on at least one cue. Measure response latency from cue and query events, memory state integrity, total updates, and query loss.

**Adversarial cases:** add distractor activity to a separate unit during delay; the task-relevant memory remains silent. This tests whether activity-only gating makes a false skip. Vary the query time without announcing it to test the scheduler's declared deadline/input assumptions.

## Limitations

These tasks are intentionally small and mechanism-targeted. They can validate that metrics detect specified failure modes but cannot establish general task advantage. Phase III must add a predeclared independent realistic task and report all baselines; no task's hand-designed oracle may be presented as an external benchmark result.
