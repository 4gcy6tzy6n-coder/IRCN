# CBCC v2 model and experiment code

This folder contains the source, tests, frozen v2 preregistration, and dependency snapshot for the CBCC circuit-budget pilot. Run from the repository root with:

```sh
python model/v0.8.0/cbcc/run_experiment.py pilot
```

The pilot is reproducible from the registered training, validation, and pilot seeds. It found no per-seed budget-matched primary comparator. Do not run `confirm` using the infeasible pilot freeze. No confirmatory evaluation is included in this release.
