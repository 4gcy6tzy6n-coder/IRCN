# IRCN v0.4.1 Phase II — ICEF metric prototype

**Status:** partial measurement prototype implemented; no scientific task evaluation or capability comparison was run.

## Facts

- Implemented metric helpers for C1 compute selectivity, C2 influence ranking/sign, C3 event latency and stable recovery, C4 long-horizon loss/state-bound summaries, C5 quality-cost Pareto filtering, and C6 paired structure contrasts.
- Twenty known-answer / input-validation tests pass for the Phase II metrics and task generators. The existing project regression suite passes 47 tests.
- The C2 output uses a single top-k overlap statistic; quality and cost remain separate. Undefined correlation and censored recovery are preserved as NaN rather than assigned favorable or unfavorable values.
- Phase II used only deterministic synthetic known-answer fixtures. Three reference task generators and exact-budget periodic/random/activity/oracle opportunity-mask controls are implemented and tested. They do not execute stateful schedulers or compare algorithms.
- It did not train a network, generate counterfactual task labels, measure runtime, use connectome data, or change v0.3 results. The task specifications are diagnostic prototypes, not confirmatory tasks or independent real-world validation.

## Interpretation

The prototype verifies basic arithmetic, input contracts, sign conventions, tie behavior, censoring, Pareto dominance, and paired contrasts. It does not establish construct validity or that the six axes measure distinct abilities. Current helpers are small aggregation primitives, not a benchmark runner or a complete evaluation framework.

## Incomplete work and risks

1. C1 allocation variation across input/state conditions is not yet calculated; add it only after freezing the opportunity unit and condition strata.
2. C2 estimator runtime and full deployment costs are supplied by the eventual experiment harness, not calculated here. The labels require paired counterfactual task runs with shared starting state and future path.
3. C3 extra recovery work and survival-analysis intervals for censored recovery are not implemented; the helper preserves censoring as NaN.
4. C4 perturbation recovery and reference deviation need declared task-specific protocols.
5. C5 returns a Pareto frontier but has no budget sweep runner, bootstrap uncertainty, or oracle regret computation.
6. C6 does not generate or validate degree/module matched graphs; it summarizes paired outcomes only.
7. No task thresholds, sample sizes, or acceptance margins were frozen. Accordingly no metric can be used for a confirmatory pass/fail decision yet.

These gaps prevent claiming a validated ICEF framework. The Phase II deliverable is an executable measurement prototype with deterministic task/control generators and explicit remaining work.

## Decision

Continue only measurement-system development. Before any confirmatory Phase III run, implement a stateful matched baseline harness, complete missing metric/cost components, expand prior-art review, define one independent realistic task, freeze budgets and statistical units, and assess discriminant behavior using known-answer and adversarial controls. No Phase III capability or Phase IV connectome experiment was run in this phase.

## Artifacts

- Metric contract: `model/v0.4.1/ICEF_PROTOCOL_v0.1.md`
- Candidate diagnostic tasks: `model/v0.4.1/DIAGNOSTIC_TASKS_v0.1.md`
- Deterministic generators and budget control masks: `model/v0.4.1/diagnostic_tasks.py`
- Executable functions: `model/v0.4.1/icef_metrics.py`
- Known-answer tests: `model/v0.4.1/test_icef_metrics.py`
- Test summary: `result/v0.4.1/EVALUATION.csv`
- Hashes and provenance: `result/v0.4.1/PHASE_II_MANIFEST.json`, `result/v0.4.1/SHA256SUMS.csv`
