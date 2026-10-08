# Data and provenance record

## Inputs in this phase

| Input | Source / version | License and provenance | Use |
|---|---|---|---|
| Research plan | User-supplied v0.2 Markdown at `/Users/yyl/Downloads/IRCN_%E7%94%9F%E7%89%A9%E8%BF%9E%E6%8E%A5%E7%BB%84%E9%A9%B1%E5%8A%A8%E7%9A%84%E4%B8%8D%E8%A7%84%E5%88%99%E7%A5%9E%E7%BB%8F%E8%AE%A1%E7%AE%97_%E9%A1%B9%E7%9B%AE%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88.md`; SHA-256 `2b6b423a954b34e97f91e8e30a33441b0d970d2add707d21c778a09b4f0c5fc6` | Author-supplied project specification; copied hash into preregistration. | Scope, scientific boundaries, E0/E1 and G0/G1 governance. |
| Graphs, weights, node time constants, input gains, pulse schedules, phases | Generated locally by `src/ircn/core.py` from pre-registered seeds | Fully synthetic; no biological identity, connectome data, or external labels | E0/E1 numerical and runtime tests only. |
| Legacy AI repository | `https://github.com/4gcy6tzy6n-coder/AI`, `main` (as named in supplied plan) | Source checkout unavailable in current workspace; no hashable files collected | Read-only audit lead only; no code/data/results imported. |
| NeuroConverge and `nmi_feedback_routing` | Existing adjacent workspaces | Read-only boundary; not used as IRCN inputs | Not modified; old experimental conclusions are not evidence for IRCN. |

Each E0/E1 run records configuration hash, source hashes, environment, seed, raw outputs, timing rows, failures, and event traces. A run is tied to the actual code hash stored in its metadata; later edits do not rewrite earlier run metadata.

## Limits

No real connectome is used. The graph-generation process is an engineering control, not biological evidence. The legacy reuse audit is incomplete at source-file level because the cited repository could not be fetched or found locally; therefore “M0 fully audited” must not be claimed until that access gap is closed.
