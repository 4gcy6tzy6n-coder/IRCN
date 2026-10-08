# IRCN v0.3.4i model package

Reproduction bundle for WP1 correctness/trajectory accuracy calibration and untouched-seed confirmation. Requires Python 3.12 and the exact pinned packages in `requirements.lock`. Run from this directory with `PYTHONPATH=src python -m pytest -q tests`. Frozen contracts are in `contracts/`; runners are under `experiments/WP1_v034i*`. This package establishes accuracy for its frozen synthetic scope only; it contains no efficiency comparison or E1/G1 claim.
