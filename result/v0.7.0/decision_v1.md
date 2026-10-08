# IRCN v0.7.0 decision

## Decision

**READY for the v0.7 software-semantics milestone.**

**Do not enter E2 or make an efficacy claim from this result.** The v0.7 gate establishes only that the frozen v0.6 cell can be executed through the tick-synchronous bridge with parity on the preregistered fixtures and that the reviewed runtime implementation satisfies its software contract.

## Facts

- 36 v0.7 tests passed; 47 repository tests passed.
- All 42 final parity attempts passed. Maximum state error was `2.8311e-15`; maximum readout error was `1.3323e-14`, below `1e-10` limits.
- Independent contract review and final implementation defect review returned ACCEPT after all blocker/major findings were addressed.
- The evidence and source/input/output hashes are retained under `runs/final_candidate_review_01/`.

## Reasoning

Parity and code review justify continuing engineering on a separate preregistered research question. They do not show that the learned influence mechanism improves task quality or compute allocation. The v0.6 task is a single chain, so topology can explain much of the influence ranking. Connecting this predictor to the runtime now would therefore overstate what the evidence identifies.

## Next-stage boundary

The next stage may design and independently review a new capability-evaluation contract with same-topology, state-dependent action effects; explicit runtime counterfactual estimand; runtime feature availability; matched-budget controls; independent task/model initialization units; and prospective uncertainty rules. That design work must pass its own review before its implementation or confirmatory runs. No E2/E3/E4 execution, connectome claim, or manuscript integration is approved by this decision.
