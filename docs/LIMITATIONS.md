# Evidence boundaries

Both datasets now have method-development history. Further checks require development status. Pools overlap and trial-bank uncertainty remains incompletely explored. Cho2017 retained all structurally valid subjects without performance-based exclusion. Target labels are used only for scoring and are separately accounted for.

Historical ranking consumes 720 unique source annotations; final-readout labels are a subset. Repeated fitting does not multiply unique annotation cost. Subject-center distillation, measured complementarity, twelve-label screening, and the fixed full-source weighting strategy received protocol-specific No-Go decisions. Finite samples and fixed representations limit generalization of each negative result.

A causal physiological mechanism and novelty over all existing source-weighting methods remain unestablished. The original independent PhysioNet test partition stayed sealed. EEG, weights, sample-level features and predictions, private metadata, credentials, and infrastructure configuration are excluded from the release.

Exact numerical reproducibility depends on locked windows, sample order, checkpoint, upstream revision, precision and library versions. The new extraction path has spot-check parity, not complete regenerated-dataset validation. CPU FP32 may differ from CUDA BF16. Class-balanced subsampling assumes known task labels; annotation counts do not model blind acquisition or label discovery.

The complementarity predictor fits source-internal combination outcomes and evaluates their transfer to the held target; it is not a nested predictor validated on independent combination groups. Its No-Go does not disprove all forms of source complementarity. Old center-distillation evaluations had different C-selection histories; the archived No-Go does not turn those evaluations into a universally uniform-C experiment.
