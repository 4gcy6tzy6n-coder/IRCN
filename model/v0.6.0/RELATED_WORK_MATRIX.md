# IRCN v0.6.0 focused related-work matrix

Focused overlap check only; not a systematic review or originality assessment.

| Work | Established mechanism | Relation | Claim boundary |
|---|---|---|---|
| Graves, 2016, Adaptive Computation Time ([arXiv](https://arxiv.org/abs/1603.08983)) | A recurrent model learns variable computation per input using a halting mechanism. | Overlaps learned allocation of recurrent compute. | Adaptive amount of recurrent computation is not novel by itself. |
| Campos et al., ICLR 2018, Skip RNN ([OpenReview](https://openreview.net/forum?id=HkwVAXyCW)) | Learns when to skip recurrent state updates and can regularize update count. | Directly overlaps HOLD/UPDATE decisions and learned update sparsity. | Skipping recurrent state updates or reducing update count is not a standalone novelty claim. |
| Goyal et al., 2019, Recurrent Independent Mechanisms ([arXiv](https://arxiv.org/abs/1909.10893)) | Multiple recurrent modules update selectively and communicate sparsely. | Overlaps modular selective recurrent computation and sparse interaction. | Sparse modular recurrence alone is established. |
| Lu et al., 2015, Event-triggered synaptic feedback ([arXiv](https://arxiv.org/abs/1504.08081)) | Event-triggered neighbor feedback for recurrent neural dynamics, with stability analysis. | Overlaps event-triggered communication/update behavior on recurrent networks. | Event-triggered recurrent neural updates alone are established. |

## Candidate question

v0.6.0 tests whether a supervised, signed, finite-horizon task-loss effect for an individual node update can be predicted from local state/input/aggregate plus static readout-distance context. Paired counterfactuals define the label; they do not establish biological causation. The predictor is not yet deployed, and the stage makes no efficiency claim. Novelty and utility remain unresolved. A future systematic review must expand to value-of-computation, influence prediction, self-triggered control, node-level asynchronous RNNs, and budgeted conditional computation before any manuscript novelty claim.
