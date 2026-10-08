# IRCN — Irregular Recurrent Circuit Network

Independent M0–M2 research branch for the v0.2 Build–Experiment–Decide pilot.

This repository contains a synthetic-graph solver correctness and runtime study. It does not contain or establish connectome-specific evidence (E2), downstream learning transfer (E3), or hardware-comparison claims (E4). No NeuroConverge artifacts are imported as experimental inputs or changed by this project.

## Status

The preregistration is frozen before benchmark evaluation. Read `pre_registration_v1.yaml` before running experiments. `decision_v1.md` is the sole project-level decision record and must distinguish observed results, interpretation, and unresolved issues. Raw run directories and event logs are stored under `reports/`.

## Run

```bash
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m ircn.cli e0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m ircn.cli e1
```

E1 refuses to run unless the root `E0_report.md` is PASS. Each invocation creates a run directory under `reports/` with configuration, environment, input/code hashes, raw timing rows, event logs, and exceptions. Failed runs are retained.

## Scientific boundary

The tested graphs are synthetic directed Erdős–Rényi graphs. Results can support only numerical correctness and CPU runtime claims for this solver/task/environment. They cannot support biological specificity, AI improvement, or general hardware superiority.
