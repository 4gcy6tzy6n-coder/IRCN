# Superseded pre-review parity attempt

These raw outputs are retained rather than deleted. They were generated before the independent implementation review identified Major defects in event trace ordering, candidate semantics, failure handling, cost accounting, clone intervention coverage, and run provenance. They are superseded by `defectfix_01/`, then `defectfix_02/`, and finally `final_candidate_review_01/`. None of these earlier runs are pooled into the final v0.7 evidence.

The final run (`final_candidate_review_01/`) uses the repaired implementation, an explicit golden fixture, independent reruns, unique non-overwriting output paths, and source/input/output manifests. The two `defectfix_*` runs are intermediate repair evidence, not the final run.
