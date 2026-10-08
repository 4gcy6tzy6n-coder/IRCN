# IRCN v0.4 Phase I report — architecture and computational semantics

**Stage status:** DESIGN COMPLETE. Implementation and scientific validation have not started.

## Facts

- The candidate architecture is defined as an event-indexed, directed recurrent state machine. Node states are held between events; selected nodes execute a learned local transition and send versioned messages only to outgoing neighbors.
- `HOLD`, `UPDATE`, and `REFINE(k)` have separate semantics. The contract prohibits computing all node transitions before scheduling decisions. Local transition, policy, queue, messages, synchronization, and readout costs are all in scope.
- The first candidate cell, cached policy features, task-conditioned counterfactual influence estimand, event ordering, and seven required invariants are specified in `model/v0.4.0/architecture_semantics_v0.1.md`.
- The initial literature pass documents adaptive computation time, Skip RNN, SkipNet, event/self-triggered control, sparse recurrent graph pruning, directed equilibrium propagation, and observed multiscale cortical dynamics in `model/v0.4.0/related_work_matrix.md`.
- No code for the architecture, no task experiment, no ICEF numerical gate, and no biological connectome analysis were run in Phase I.

## Interpretation

The core research object is now framed as a learned, task-conditioned policy over event-indexed local recurrent updates. Dynamic computation, recurrent skipping, event-triggered execution, sparse graphs, and multiscale biological dynamics each have prior work. The potential distinction is their combination with graph-local asynchronous execution and a task-loss influence estimator under actual compute budgets. This distinction is only a hypothesis; the literature pass is not exhaustive and does not establish originality.

This choice is a real change from v0.3.4i, which approximates a frozen continuous-time ODE. The v0.3.4i WP1 result remains a correctness result for that solver contract; WP2 remains INCONCLUSIVE for its frozen identity-readout influence contract. Neither result is repurposed as evidence for or against v0.4.

## Critical issues carried into Phase II

1. **No metric escape hatch:** task quality and stability remain required outcomes. ICEF adds measurements of computation, but cannot replace external task outcomes when conventional accuracy or task loss is poor.
2. **Counterfactual target availability:** the initial C2 label uses future supervised targets offline. It cannot be presumed available for unlabeled or online tasks; those require a separate estimand.
3. **Cost completeness:** update counts or FLOPs alone cannot establish savings. The policy, event queue, messages, synchronization, readout, and hardware execution must be measured.
4. **Influence interactions:** single-node counterfactual effects need not add across nodes. Joint interventions and oracle selection must be separated from deployable estimators.
5. **Construct validity:** six metrics may overlap. Phase II must give each metric discriminating controls and known-answer sanity checks; no composite ICEF score or arbitrary success thresholds are accepted.
6. **Biological interpretation:** multiscale activity observations motivate hypotheses but do not demonstrate compute allocation or causal biological equivalence. Connectome claims require Phase IV controls.

## Open questions

The asynchronous update semantics are defined, but Phase II still needs to freeze exact policy features, model widths, optimization, update-order controls, query cadence, budgets, diagnostic task parameters, statistical units, and task-level pass/fail rules. The initial paper search also needs expansion around asynchronous non-spiking recurrent networks and learned per-node self-triggering before any novelty claim.

## Decision and next stage

Phase I design output is complete enough to build an ICEF prototype. Continue to Phase II only to operationalize and sanity-test the evaluation framework; do not begin capability or connectome claims. Phase II is a measurement-system development stage, not evidence that IRCN outperforms existing models.

## Sources

Primary research sources and their scope boundaries are recorded in `model/v0.4.0/related_work_matrix.md`. Key sources include [Adaptive Computation Time](https://arxiv.org/abs/1603.08983), [Skip RNN](https://openreview.net/pdf?id=HkwVAXyCW), [SkipNet](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html), [event-triggered and self-triggered control](https://research.tue.nl/en/publications/an-introduction-to-event-triggered-and-self-triggered-control/), and the cited primary studies on directed equilibrium propagation and cortical theta-gamma dynamics.
