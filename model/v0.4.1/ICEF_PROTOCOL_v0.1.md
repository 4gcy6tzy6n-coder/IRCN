# ICEF Phase II metric contract v0.1

**Status:** executable metric prototype; not a frozen capability experiment and not scientifically validated. No composite score is defined. Every reported metric must be paired with task quality and complete deployment cost.

## Shared reporting contract

Use independent task-seed/model-training-run pairs as statistical units; repeated timing runs are technical replicates, not independent samples. Store per-unit records before aggregation. Report failures, timeouts, missingness, compute hardware, warmup/repetition policy, wall-clock including policy/scheduler/communication/readout costs, and update counts separately. Thresholds and task-specific acceptance margins must be set before any confirmatory evaluation. These functions do not infer missing costs or repair incomplete logs.

## Six axes and operational quantities

| Axis | Primary quantities | Required discriminating controls | What it cannot establish alone |
|---|---|---|---|
| C1 Compute selectivity | selected/available update fraction; critical-node recall; noncritical update rate; per-input allocation variation | update-all, fixed-frequency, random budget-matched, activity-threshold | Quality preservation or wall-clock savings |
| C2 Future influence fidelity | Spearman rank correlation between decision-time prediction and paired task-loss benefit; sign accuracy; top-k precision/recall; estimator cost | random ranking, activity/degree ranking, deployable learned predictor, offline oracle as upper bound only | Biological causality; useful selection if estimator cost exceeds saved work |
| C3 Event responsiveness | first relevant update latency from event; missed-event fraction; task-loss recovery duration and extra work | periodic scheduler, random budget-matched, update-all reference | Robustness to untested event distributions |
| C4 Long-horizon stability | horizon-wise task loss; state norm/boundedness violations; perturbation recovery time; deviation from same-model update-all reference | update-all same transition, fixed-frequency, no-recovery ablation | Task utility from hidden-state similarity alone |
| C5 Budget adaptability | per-budget task quality, complete cost, stability; nondominated quality-cost points; regret to budget-specific oracle | fixed policy, random, activity threshold, offline oracle upper bound | Smooth degradation outside tested budgets |
| C6 Structural contribution | paired task-quality/cost contrast for target graph against degree-matched, module-matched, and rewired controls | same model capacity, training budget, seeds, task and update budget; multiple graph randomizations | Biological explanation from a single graph or unmatched controls |

## Estimands and sign conventions

- C1 selected fraction is `updates / opportunities`; critical recall is selected critical opportunities / all critical opportunities; noncritical rate is selected noncritical opportunities / all noncritical opportunities. Report per trajectory and then aggregate by independent seed.
- C2 benefit is `loss(skip) - loss(update)` under identical starting runtime state and shared future input/target path. Positive means update helps. This prototype ranks signed benefit (which update is expected to help most); a separate absolute-impact estimand must be declared if detecting harm is the goal. Ties use average ranks; top-k ties resolve by stable input order. Constant vectors yield undefined correlation (`NaN`), never silently zero.
- C3 latency is first qualifying target-node update time minus exogenous event time. Recovery time is the first post-event sample below a predeclared task-loss threshold that stays below it for a predeclared dwell interval. Missing recovery is right-censored and must be retained; this prototype returns no imputed value.
- C4 includes task-level quality and internal-state bounds. State trajectory difference is diagnostic and is not a pass criterion by itself. Compare like-for-like transition functions and inputs.
- C5 returns the full per-budget records and nondominated points. Cost and quality are separate axes; no arbitrary weighted sum. The oracle is an upper bound and its information/computation is unavailable to deployment.
- C6 contrasts are paired by task seed and initialization. Graph controls must be generated and validated against declared degree/module constraints; graph identity alone is not a sample-size unit.

## Prototype boundary and validation

The functions in `icef_metrics.py` implement C1/C2 summaries, C4 trajectory summaries, C5 Pareto filtering, and paired C6 contrasts. C3 timestamp censoring is represented directly. They validate shapes and preserve undefined values. The tests use synthetic known-answer arrays to check arithmetic, direction, ties, dominated points, censoring, and malformed inputs. This verifies code behavior only; construct validity, task suitability, statistical power, and algorithmic performance remain open.

Before Phase III, expand prior-art review around per-node asynchronous recurrent scheduling and influence-based computation, define task-specific C3 recovery rules, freeze each diagnostic task and matched graph generator, predefine budgets and inferential methods, and add an independent realistic task. Do not use metric development data for confirmatory claims.
