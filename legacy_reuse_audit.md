# Legacy AI repository reuse audit

**Audit boundary:** the source cited by proposal v0.2 is `https://github.com/4gcy6tzy6n-coder/AI`, branch `main`. That checkout is not present in the writable workspace. Direct web indexing did not return the named files and network restrictions prevent a trustworthy clone/read of the source in this turn. The proposal itself contains a previous read-only audit summary of the named files; this document preserves those leads as *unverified source descriptions*, not as independently confirmed code facts. No old code or result is imported.

| Referenced legacy artifact | Candidate engineering idea from the supplied v0.2 audit notes | IRCN disposition | Verification status / prohibited inference |
|---|---|---|---|
| `docs/02_engine_specs/thinking_engine_spec_v1.md` | Dependency graph / state graph bookkeeping | Reimplement a minimal event-dependency trace only if needed for logs | Not independently inspected in this turn. A reasoning-step graph is not a biological synaptic graph. |
| `docs/01_theory_frozen/tsla_actions.md` | Explicit action routing | Use only numerical actions `SKIP/UPDATE/REFINE/FALLBACK` if they match the solver contract | Not independently inspected. Confidence thresholds such as 0.8/0.3 are not numerical tolerances. |
| `docs/02_engine_specs/memory_scheduler_spec_v1.md` | State lifecycle/versioning and rollback interface | May inform run-state lifecycle; no memory hierarchy is imported | Not independently inspected. Storage layers do not establish biological memory mechanisms. |
| `phase19_native_backbone/native_backbone_tiny_v1.py` | Small GRU baseline | No model code reused in M0–M2; no E3 model is in scope | Not independently inspected. A GRU is not an asynchronous local ODE solver. |
| `phase19_native_backbone/stage4b_four_way_comparison.py` | Experiment scaffolding / parameter recording | Reimplement a deterministic run manifest; exact parameter and runtime accounting required | Not independently inspected. Equal hidden dimension does not imply equal parameter count or FLOPs; Python `hash()` is not a stable cross-process seed. |
| `phase20_self_construct/PROJECT_STATUS.md` | Freeze/guard/incident documentation conventions | Reuse only the practice of recording freezes, failures, and run status | Not independently inspected. A “stable window” is not evidence of dynamical stability or generalization. |

## Reuse decision

- **Reused:** none of the legacy implementation or experimental evidence.
- **Allowed after source review:** generic bookkeeping patterns only, independently reimplemented and tested.
- **Not allowed:** importing model/task assumptions, trusting PASS labels, inheriting weights/data/results, or treating legacy training performance as IRCN evidence.
- **Open audit item:** a source-level, file-hash-backed review of the legacy repository is still required before claiming M0 reuse audit completion. This limitation is recorded in `data_provenance.md` and blocks any claim that the legacy source was fully audited.
