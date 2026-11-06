# Reproduction guide

Python 3.11 is recommended. requirements.txt pins primary packages from the execution environment. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for CPU reproducibility. The portable reproduce.py entry point performs source-selection controls and full-source transfer weighting from private prepared frozen features.

Feature extraction needs separately obtained upstream CBraMod and pretrained weights. Code revision b9e961003214326972c567eff390e75b0287e32a. Checkpoint SHA256 0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178. The paper reports TUEG pretraining; a complete per-record ledger was unavailable. Use model.eval(), no gradients, and flatten final channel/time/token outputs. Cho adaptation emits 3x3x200 epochs. Verify channel mapping, unit interpretation, event counts, and window bounds on real inputs before inference.

Historical b_ scripts retain the original relative experiment paths and expect private metadata/trials.csv, metadata/splits.json, frozen feature caches, and preceding experiment outputs at the paths referenced in their loading blocks. b_source_intervention.py and b_cho_intervention.py precede b_lowcost.py and b_weight.py. Complementarity scripts include their summaries. Original private trial IDs are excluded from this release. Exact private feature regeneration requires independently preparing the locked banks and metadata.

This release preserves CPU method implementations and aggregate/subject-level evidence. Original EDF preprocessing and the GPU center-distillation dependency chain are outside this release. Center training is represented by archived evidence and its No-Go outcome. The portable feature-based entry point is the normal public starting point.

Validation includes Python compilation, an end-to-end synthetic private-input run outside Git, source-bank cardinality and target-exclusion assertions, plus tracked-file and complete new-history scans. See docs/SECURITY_AUDIT.json for the final scan.
