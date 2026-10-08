# IRCN v0.4 Phase I — Architecture and computational semantics v0.1

**Status:** design contract for review and prototype construction. This is not an implemented or scientifically validated architecture.

## 1. Research object

IRCN v0.4 is proposed as a task-trained, event-indexed recurrent state machine on a directed graph. Its scheduler decides when and where to spend state-transition computation. The research object is the joint system of (a) local state transition, (b) scheduler, (c) versioned messages, and (d) task readout. A sparse graph or a small number of updates alone is not a successful result.

This version deliberately does **not** approximate a continuous-time ODE. Physical time is an input/event timestamp; node state is piecewise constant between committed local updates. Continuous-time dynamics may be explored as a separate extension with its own contract. This choice makes “skip” mean that the local state-transition function is not executed, rather than that a numerical solver approximates the skipped trajectory.

Let G=(V,E) be a directed graph with N nodes. Node i has state h_i in R^(d_i); the model has parameters theta, receives timestamped input x(t), and is trained against task target y*_t. When node i is selected to update at logical event time t, it computes:

```text
h_i_new = F_i(h_i_old, Agg_i({m_ji : j in Pred(i)}), x_i(t), delta_t_i; theta_i)
y_t     = R({h_i(t) : i in V_out}, x(t); theta_r)
```

Here m_ji is the latest delivered predecessor message and delta_t_i is elapsed physical time since node i's previous committed update. A task readout at a declared query time uses states currently committed by the asynchronous event process. A query does not silently refresh all nodes. If an application requires exact full-state readout at every query, that readout cost is measured and included.

The first prototype will use a shared local cell with a fixed hidden width d. For each directed edge j->i, the message is m_ji=P_m h_j; c_i is the incoming weighted sum of the latest delivered m_ji, divided by sqrt(max(1, in_degree(i))). Let d_i=log(1+delta_t_i). Then:

```text
g_i     = sigmoid(W_g [h_i; c_i; x_i(t); d_i] + b_g)
candidate_i = tanh(W_h h_i + W_c c_i + W_x x_i(t) + W_d d_i + b_h)
F_i     = (1 - g_i) elementwise-multiply h_i + g_i elementwise-multiply candidate_i
```

The local cell is evaluated only after UPDATE/REFINE has been selected. A separate policy score S_psi uses cached features (elapsed time, input-change summary, received-message deltas, and prior update residual); it must not evaluate F_i to decide whether to evaluate F_i. The exact feature list, hidden widths, graph normalization, parameter sharing, initialization, and training losses must be frozen in the Phase-II model contract before task experiments.

## 2. Runtime state and action semantics

For each node i, runtime stores the versioned record:

```text
z_i = (h_i, t_i, v_i, {v_ji : j in Pred(i)}, q_i, mode_i)
```

where h_i is the last committed state, t_i its timestamp, v_i its version, v_ji the last consumed predecessor version, and q_i a cached urgency summary. A message contains (source, destination, version, timestamp, state, summary). Stale versions are rejected; equal-time events use a frozen stable order.

At each scheduler decision, only locally available cached information may be used. The policy chooses among:

- **HOLD:** perform no local state-transition evaluation. Keep the last committed state until another action changes it.
- **UPDATE:** evaluate F_i once at the current event time from the last committed local state, elapsed time, and latest delivered predecessor messages; commit a new version and enqueue messages only for outgoing neighbors.
- **REFINE(k):** apply F_i for a prespecified k>1 local iterations using declared cached-input semantics, then commit one externally visible version and enqueue its messages. Intermediate states are not visible to other nodes.

A scheduler may include additional actions only through a versioned contract amendment. If the implementation evaluates F_i for every node before the gate decision, it is not selective computation and cannot count as a compute saving. Candidate and priority maintenance should be incremental from input/message events; any global scan must be disclosed and charged.

The local transition, priority computation, event queue, message handling, output materialization, and synchronization are all charged. Phase II will distinguish transition work, scheduler overhead, and total end-to-end cost.

## 3. Causality and event order

A node can respond only to its own input event, an incoming committed message, an explicitly scheduled update/refine deadline, or a declared readout request. Events are totally ordered by (logical_time, stable_sequence_id). A transition may consume only predecessor message versions delivered by that time. Updating a node creates a monotonically increasing version and invalidates older queued actions for that node. Simultaneous events are resolved deterministically; this order is part of the model semantics.

No global hidden-state scan is allowed in the selection path. Global reductions required by a task readout are charged as readout work. Input events and output queries use a declared timestamp stream, but node updates are asynchronous. A readout does not force an all-node update at every query.

## 4. Training and scheduling objective

For trajectory tau=(x_0:T, y*_0:T), define task loss L_task(tau; theta, pi), measured total resource vector C(tau; theta, pi), and stability/response constraints S_k <= 0. Training and policy selection are a constrained multi-objective problem:

```text
min_(theta, pi) E[L_task]
subject to E[C_m] <= B_m for each resource m
           Pr(S_k > 0) <= delta_k for each stability/response constraint k
```

A Lagrangian may be used as an optimizer, but reported conclusions must show the quality-cost-stability frontier and cannot reduce ICEF to one scalar score. Allowed resource axes and constraints are fixed before evaluation. Training compute is reported separately from deployment compute.

## 5. Task-conditioned update influence (algorithm hypothesis)

For node i, time t, forecast horizon H, and a fixed future task input/target path, define paired potential outcomes from identical pre-intervention runtime state:

```text
Delta_i(t,H) = L_task(y_skip[t:t+H], y_star[t:t+H])
             - L_task(y_update[t:t+H], y_star[t:t+H])
```

Positive Delta means updating helps; negative Delta means it harms task loss. For the initial supervised diagnostics, the target path is known offline and a paired counterfactual can produce labels. This definition does not directly apply where future task targets or rewards are unavailable; unsupervised and online-reward variants require a separate estimand. At deployment, Delta is unknown; an estimator Delta_hat_i must use only information available at decision time and include its own compute in the budget. Ranking absolute Delta can identify impactful decisions, but the scheduler must distinguish beneficial updates from harmful or unnecessary ones and account for action costs.

This is **algorithmic intervention impact on a declared task loss**, not biological causality. Interactions between multiple skipped nodes violate simple additivity; Phase II must test single-node labels separately from joint interventions and report oracle set selection as an upper bound only.

## 6. Required invariants

Any implementation must satisfy these executable properties before a capability experiment:

1. **Causal locality:** an event at node j cannot change a non-descendant until a declared path/message reaches it.
2. **No hidden full pass:** policy decisions cannot be preceded by computing every candidate node's transition or exact task influence.
3. **Version consistency:** stale messages and superseded deadlines cannot roll back state.
4. **Determinism under frozen tie-breaking:** the same inputs, graph, parameters, seed, and budget reproduce event/action traces.
5. **Reference-mode equivalence:** update-all mode follows the same F_i transition contract and declared event order. Any comparison with a separate dense-time reference is a distinct numerical result, not the definition of this architecture.
6. **Boundedness and liveness:** state and event queue remain bounded under declared input classes; critical-event tests include a maximum response delay.
7. **Cost completeness:** every transition, policy, message, queue, cache, and output operation is charged or measured.

These are design obligations, not claims already established by v0.3.4i or WP2.

## 7. Separation from prior IRCN results

The v0.3.4i scheduler approximates a frozen continuous-time ODE. WP2 tested an influence proxy against an identity full-state trajectory and was INCONCLUSIVE because one of 528 cases exceeded its frozen state-accuracy bound. That result remains valid for its old contract. It neither passes nor falsifies this proposed event-indexed architecture. Any v0.4 model, objective, metric, or task is a separate lineage and needs its own pre-registration and evidence.

## 8. Candidate contribution and claim boundary

Prior work already includes adaptive computation for RNNs, learned skipped state updates, dynamic layer routing, event/self-triggered control, and sparse recurrent computational graphs. None of those broad ideas alone is claimed as IRCN novelty. The narrower testable hypothesis is that a directed recurrent state architecture can jointly learn task-conditioned asynchronous local update decisions and graph-constrained event propagation under measured compute budgets, while preserving declared task quality and stability.

Whether that combination is novel is **unresolved** pending systematic review. Biological motivation and connectome-shaped edges do not establish biological equivalence or a biological contribution.
