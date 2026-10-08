# IRCN v0.6.0 — trainable cell and influence-estimator bridge v0.1

**Status:** pre-implementation contract. Exploratory synthetic mechanism study only. This stage does not deploy a learned policy in the event runtime and does not establish task generalization, speedup, connectome contribution, biological validity, publication novelty, or publication readiness. v0.5.0 and all older version artifacts remain immutable.

## 1. Research question and stage boundary

Can a shared local recurrent cell learn a declared graph task, and can a predictor trained on paired single-node update/hold counterfactuals estimate signed finite-horizon task-loss benefit from decision-time local context on held-out trajectory seeds?

This is the trainability and label-validity bridge before scheduler/runtime integration. It evaluates C2 prediction quality only. It does not infer the value of a joint set of simultaneous node updates, because single-node effects may interact. It does not execute E2/E3/E4, use a real-world or connectome dataset, alter NeuroConverge, or edit the NMI manuscript.

## 2. Frozen task and data generation

- Graph: 32 nodes. The only nonzero directed edges form the chain `0→1→…→7`, each weight 0.8. Nodes 8–31 have no edges. No self edges.
- Each top-level seed deterministically generates 32 independent length-16 trajectories using NumPy `Generator(PCG64(SeedSequence([top_seed, trajectory_index])))`. Per trajectory: draw source pulse count with `rng.integers(1,3)`; sample unique source ticks with `sort(rng.choice(9,size=count,replace=False))`; draw one amplitude per sorted tick using `rng.uniform(0.25,1.25)` and add it to node 0. Draw distractor pulse count with `rng.integers(0,3)`; for each distractor pulse, draw node with `rng.integers(8,32)`, tick with `rng.integers(0,16)`, and amplitude with `rng.uniform(-1,1)`; add it to that node/tick. All unmentioned inputs are zero. If multiple pulses collide, amplitudes add. Draw order is exactly source count, source tick sample then sort, source amplitudes in sorted tick order, distractor count, then each distractor's node/tick/amplitude in draw order.
- The teacher starts at `h[0]=0` and applies, for `t=0..15`, `h[t+1]=0.5*h[t]+W@h[t]+x[t]`. The target at decision tick t is `y*[t]=h_teacher[t+1,7]`. No noise is added. Test instances are not generated or examined until model/predictor hyperparameters are frozen.
- Split by whole top-level seed, never by time point or node opportunity: train `[100..109]`, validation `[200..204]`, test `[300..309]`. Each seed has exactly 32 trajectories. Initial model seed is 6060; cell data-order seed 6061; estimator initialization seed 6062; estimator data-order seed 6063.

## 3. Trainable cell and reference rollout

Use CPU `float64`, one Torch intra-op and inter-op thread, MPS/CUDA disabled, deterministic algorithms enabled. A shared width-8 cell is applied to all 32 nodes. For scalar task input:

```text
q_i = concat(h_i, a_i, x_i, log(2))       # dt=1 for every reference tick
z_i = sigmoid(W_z q_i + b_z)
c_i = tanh(W_c q_i + b_c)
h'_i = (1-z_i)⊙h_i + z_i⊙c_i
```

`a_i = sum_j W_ij h_j / sqrt(max(1, in_degree(i)))`. Messages use identity projection in v0.6.0; this is an explicit versioned choice and does not assert equivalence with the projection proposed in v0.4. Readout is an affine scalar head over node 7's width-8 state. The synchronous reference performs simultaneous all-node updates from the previous state at each tick; the readout target is evaluated from the resulting state after that tick.

Train the cell and readout on the training trajectories with Adam (`lr=0.001`, batch size 32, max 200 epochs), weighted MSE over all 16 outputs. Validation loss is the mean per-trajectory MSE. Check validation each epoch; `min_delta=1e-6`, patience 15; restore the lowest-validation-loss checkpoint. No hyperparameter search is permitted. Report test per-trajectory NRMSE, where denominator is the test target standard deviation for that trajectory; if it is zero, record NRMSE as undefined and retain the trajectory. Also report raw MSE and the per-trajectory constant-training-mean baseline. The software stage can complete even if learnability fails, but predictor findings then remain uninterpretable for model use.

Training implementation details are frozen as follows: construct records in ascending `(top_seed, trajectory_index, tick)` order; shuffle only trajectory indices each epoch with a CPU `torch.Generator` seeded 6061, and preserve within-trajectory tick order. A minibatch contains 32 trajectories; use the final smaller batch if needed. The cell and readout are initialized with Xavier-uniform weights from separate CPU generators seeded 6060 and 6064, and zero biases. Adam uses `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, and no gradient clipping or scheduler. The training loss is the arithmetic mean of squared errors over batch trajectories and all 16 ticks. On a validation-loss tie within `1e-12`, retain the earlier epoch. The constant baseline predicts the scalar mean of all training targets. Report both macro mean trajectory NRMSE and baseline NRMSE; the learnability descriptor is `BEATS_BASELINE` only if the macro NRMSE is strictly lower, finite for all nonconstant test trajectories, and the test set contains at least one nonconstant target. This is a descriptive held-out criterion, not a confirmatory claim.

## 4. Decision-time predictor context

For each reference state `h[t]`, derive local features without using `x[t+1:]` or future targets:

- own local state `h_i[t]` (8 values);
- local aggregate `a_i[t]` (8 values);
- current input `x_i[t]` (1 value);
- input-change magnitude `abs(x_i[t]-x_i[t-1])` (zero at t=0);
- weighted message-change summary `sum_j abs(W_ij)*norm(h_j[t]-h_j[t-1])` (zero at t=0);
- prior local residual `norm(h_i[t]-h_i[t-1])` (zero at t=0);
- elapsed time fixed at 1; consecutive holds fixed at 0; deadline flag fixed at 0 for this synchronous diagnostic;
- static local structure: in-degree, out-degree, shortest directed path distance from node i to readout node 7 (capped at 33 if unreachable), and readout reachability flag.

Feature dimension is 27 (=8 own-state + 8 aggregate + 1 current input + 6 event-summary scalars + 4 structural values). Structural values are derived only from the fixed graph/readout and are charged as preprocessing in any future deployment; this prototype reports them but does not claim end-to-end savings. Standardization statistics are fit on training records only; zero standard deviations are replaced by 1. Target Delta is standardized using training labels only and inverted for all reported metrics. No test-derived preprocessing or thresholds are allowed.

The static distance feature is required because identical dynamic local context at two positions in the chain can have different finite-horizon output influence. A pre-training collision test must verify that the complete feature vector distinguishes a hand-built pair with unequal signed effects. If complete features still collide with different label values beyond `1e-12`, record the collision, stop predictor training, and revise the feature contract before any test-set evaluation.

Feature order is exactly: state components 0–7; aggregate components 0–7; current input; input-change magnitude; weighted message-change; prior residual; elapsed; consecutive holds; deadline; in-degree; out-degree; capped directed distance; reachability. With edges interpreted source→target, `in_degree(i)` counts incoming edges and `out_degree(i)` counts outgoing edges. Distance follows edge direction from node i to readout node 7; unreachable nodes have distance 33 and reachability 0. At tick zero, previous input and previous state are zero. The graph is represented as `W[target, source]`, so `a=W@h`.

The collision adversary is a specific toy-teacher fixture, not a general identifiability proof: compare the same unit state placed at node 5 versus node 6, with all other state, prior state, and inputs zero. In the width-8 feature representation, the unit value is state component 0 and components 1–7 are zero. Their dynamic local features are equal, while directed distances to readout are 2 and 1; the frozen expected single-node deltas are 0.0832 and 0.1025. The test verifies this declared pair is distinguished, and makes no claim that the 27 features uniquely identify all possible counterfactual effects.

## 5. Paired single-node influence labels

Generate labels from the frozen trained-cell synchronous reference trajectory. For each trajectory, decision tick `t=0..11` and node `i=0..31`, use the same pre-intervention state `h[t]`, current input `x[t]`, and full future input sequence. At tick t, the UPDATE branch applies the trained cell to all nodes. The HOLD branch applies the same cell to all nodes except node i, whose state is copied unchanged. Both branches then use all-node updates for ticks `t+1..t+3`. Other nodes, inputs, and initial state are identical; branch state arrays never alias. The readout is evaluated after each of those four ticks.

```text
Delta_i(t) = mean_{k=t..t+3}(loss_HOLD[k] - loss_UPDATE[k])
```

`loss[k]=(y[k]-y*[k])^2`. Positive Delta means updating this node improves future-window task loss. All labels have a complete four-tick horizon; later ticks are omitted by the frozen `t≤11` rule, not truncated. This is an algorithmic `do(HOLD/UPDATE)` intervention effect under a synchronous trained-model reference, not biological causality and not a direct estimate of an event-runtime effect.

Keep all 32×12 labels for every trajectory. Train the predictor with equal total weight for path nodes 0–7 and isolated nodes 8–31, then equal weight per top-level train seed. Report path and isolated strata separately so the many zero-effect isolated nodes cannot dominate the headline metric. Store trajectory seed/id, node/tick, the pre-decision full reference snapshot `h[t]` (shape `(32,8)`, float64) hash and feature hash, both losses, signed Delta, trained-cell hash, source/config hashes, and failure status for each row. Snapshot and feature arrays use the frozen canonical ndarray serialization.

The hand-computed label fixture uses the linear teacher transition solely to test the branch/label implementation: the declared 32-node chain, `h[0,4]=1` and all other initial components zero, no inputs, decision `(t=0,i=4)`, horizon four, and readout node 7. UPDATE follows the teacher recurrence in both branches except the HOLD branch copies node 4 at tick 0. Readout differences after the four scored ticks are `[0,0,0,0.256]`; the UPDATE branch has zero loss against its teacher target, so expected `Delta=(0+0+0+0.256^2)/4=0.016384`. This fixture is a label-semantics test and is not a learned-cell result.

## 6. Predictor and evaluation

The predictor is a one-hidden-layer MLP, width 16, ReLU, scalar output, trained on signed Delta with weighted MSE. Adam `lr=0.001`, batch size 256, maximum 100 epochs, validation weighted MSE, `min_delta=1e-6`, patience 10, estimator seed 6062; restore best validation checkpoint. Its input is exactly the 27-value context above.

Predictor implementation details: one hidden affine layer and one scalar output layer, Xavier-uniform weights from a CPU generator seeded 6062 and zero biases; Adam uses `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, with no clipping or scheduler. Shuffle training records each epoch using CPU generator seed 6063; batches have at most 256 rows. Row weights give each top-level seed equal total weight, then half of that seed's weight to path opportunities and half to isolated opportunities, uniform within each stratum. The weighted MSE is the sum of `weight * squared_error` divided by the sum of weights in the batch. Validation uses the same frozen weighting rule; ties within `1e-12` retain the earlier epoch. Train-only feature and label mean/population-standard-deviation statistics use the same row weights; zero standard deviations map to 1.0.

Primary test metrics, computed per trajectory then summarized by top-level seed, are signed-Delta MSE, Spearman rank correlation, sign accuracy for labels with `abs(Delta)>1e-12`, and precision among the top 20% predicted-positive-benefit opportunities. Undefined Spearman/sign statistics remain NaN. Compare the predictor to: zero-Delta regressor (MSE baseline), absolute current input-change ranking, prior-residual ranking, out-degree ranking, shortest-distance-to-readout ranking, and fixed-seed random ranking. The exact labels are called the **single-intervention label oracle**; they are not a joint-set budget oracle or a deployable policy.

For ranking metrics, select `ceil(0.20 * 384)` opportunities per trajectory by descending score; break ties by ascending `(node,tick)`. Precision is the fraction with true `Delta>0`. Distance score is negative capped distance, so nearer reachable nodes rank higher. The random score is generated separately per test trajectory using `PCG64(SeedSequence([7070, top_seed, trajectory_index]))`. Spearman uses average ranks for ties. Bootstrap intervals are percentile intervals over 10,000 paired resamples of the 10 top-level test seeds with replacement, using PCG64 seed 8686; report predictor-minus-zero-baseline difference for MSE and predictor-minus-each-ranking-control differences for precision. Spearman intervals are descriptive per-method summaries. These intervals remain descriptive.

MSE is a regression metric and is reported only for the predictor and zero-Delta regression baseline; the activity, residual, degree, distance, and random controls are rankings and are evaluated by Spearman and top-20% precision only. Sign accuracy is descriptive for the predictor only. Compute MSE per trajectory over all 384 opportunities, then average trajectories within each top-level seed. Bootstrap paired predictor-minus-zero-baseline differences for MSE and predictor-minus-each-ranking-control differences for precision; Spearman receives descriptive per-method intervals. Use paired bootstrap over the 10 test top-level seeds for descriptive 95% intervals. These intervals are exploratory; no confirmatory significance or efficacy claim is made. No budget sweep, runtime scheduler comparison, wall-clock speed claim, or joint intervention optimum is in scope. Predictor inference timing may be profiled descriptively but may not be compared with saved cell time as a speed claim.

## 7. Development gates and failure handling

1. **Protocol/provenance:** generator, seed mapping, environment, source hashes, and golden input/teacher/feature hashes are frozen before training or test-label generation.
2. **Gradient/reproducibility:** cell and readout gradients are finite/nonzero on a known batch; fixed seeds reproduce parameter/loss hashes under the frozen CPU backend.
3. **Reference correctness:** hand-computed teacher trajectory and graph edge orientation tests pass; full-update state rollouts are deterministic.
4. **Counterfactual purity:** branch initial states are identical; mutation in one branch cannot affect the other; only node i's tick-t action differs; a change to future target values changes labels but leaves decision-time features byte-identical.
5. **Known answers:** isolated node labels are zero within `1e-12`; a frozen path-node fixture has the hand-computed Delta in a golden test; complete features pass the collision test.
6. **Cell learnability:** held-out NRMSE must beat the constant-training-mean baseline and be finite for all nonconstant-target test trajectories. If not, preserve the negative result and do not interpret learned-influence results as model-validity evidence.
7. **Estimator evaluation:** report all frozen baselines and all test seeds, including failed/undefined rows. Underperformance is a valid negative result and cannot trigger test-driven feature or threshold changes.

Software READY, cell learnability, and influence-estimator performance are separate outcomes. No single READY status is allowed to imply all three.

## 8. Environment and claim boundary

The version-specific environment is Python 3.12.13, macOS arm64, NumPy 2.4.4, PyTorch 2.11.0, CPU backend, one intra-op/inter-op thread; exact direct/transitive versions are pinned in `requirements.lock`. This pin is chosen because it is installed and verified in the current environment; it is not claimed to be the latest PyTorch release. All source, tests, generator, contracts, training config, and generated input/label manifests receive SHA-256 hashes. Keep every failure and raw log.

Adaptive computation and skipped recurrent updates are established ideas (ACT; Skip RNN); selective sparse recurrent modules (RIMs) and event-triggered recurrent dynamics also predate IRCN. No broad novelty claim is made. This stage only tests whether the defined signed counterfactual target is learnable from the chosen local features in one synthetic task family. Novelty and scientific utility remain unresolved.
