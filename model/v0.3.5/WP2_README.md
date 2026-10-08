# IRCN model v0.3.5 — WP2 influence evaluation

This package inherits the frozen v0.3.4i model implementation and adds the WP2 influence-label generator, prespecified analysis, recovery-capable runner, contract, and tests. The data and scientific conclusion are in `result/v0.3.5/`.

WP2 status is **INCONCLUSIVE** because one of 528 candidate trajectories exceeded the frozen WP1 state-accuracy bound. The predictor ranking and cost metrics are descriptive only. WP3 remains blocked.

Run from the repository root with the locked environment:

```sh
PYTHONPATH=.:src python experiments/WP2/run_experiment.py
```

The archived complete result is not overwritten by this command; the runner fails closed when analysis output already exists.
