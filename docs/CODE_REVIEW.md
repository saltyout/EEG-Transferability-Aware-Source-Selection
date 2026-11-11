# Research code review (2026-11-11)

## Scope and findings

Reviewed README, protocol/reproduction/results/limitations documents, all original Python scripts, dependency list, and the archived JSON/CSV evidence. Original result and `b_*.py` files are preserved. This review improves engineering without changing the research hypothesis, seeds, C, selection proportion, evidence or No-Go decisions.

| Finding | Resolution / remaining boundary |
|---|---|
| No public raw-EEG-to-feature entry point | Recovered filter, CAR, sampling, windows, scale, frozen model, precision and checkpoint settings from historical implementation; modular preparation entry and exact PhysioNet window/hash manifests added |
| Original PhysioNet trial order absent | Minimal public manifest recovered from the actual train-ID provenance; 1,080 windows, 24 subjects, 72 EDF files; no test-bank or private IDs exported |
| Every portable arm recorded 720 ranking labels | Corrected per-arm ranking, fitting and union counts; README explicitly distinguishes 180 final labels from 720 total |
| Portable matrix retrained identical source models for every recipient | Fit once per source and score disjoint recipients; no protocol change |
| Bounds/assert-only validation insufficient | Added explicit errors for nonfinite/malformed features, duplicate/out-of-range banks, inadequate subjects/classes, invalid pools and fit/target overlap; new Cho preprocessing uses explicit validation |
| No lightweight tests / public demo | Synthetic generator, 19 CPU tests, Actions workflow, archived results verifier |
| No per-run environment/input/fit provenance | Added input checksum, runtime package versions, fit indices and source pool/quality manifests, paired-target summary |
| One heavyweight environment | Separate CPU, EEG and development requirements; retain original version list as historical reference |
| Historical scripts assume private file trees, run at import | Preserve forensic copies; document inputs, output paths and dependency order; portable path is the supported entry |
| Weak public reproducibility claims | Distinguish reconstruction, spot-check parity and complete rerun; CPU FP32 explicitly differs from historical BF16 mode |
| No license, no topics | Document licensing boundary and suggested About/topics; remote settings untouched |

## Leakage and protocol assessment

The inspected source-selection/weighting logic fits standardization on source data only and excludes each outer target from its 16-source pool. Selection consumes source-to-source BA, not held-target BA. Cached all-subject matrices are reused but target row/column is excluded. Classifier C=0.01 is fixed, with no target-dependent PCA or C tuning. Historical self-BA weighting uses stratified three-fold source-only CV. New tests perturb target labels and out-of-pool matrix entries and verify predictions/ranks remain unchanged.

The historical labels used in source ranking are part of the method's cost. Selected readout trials are already included in that bank. Stratified sampling is a retrospective labeled-bank design, not a demonstration of prospective label discovery. Evaluation labels are not included in source annotation budgets. All prior datasets now carry development history. No independent test signals were read.

Complementarity uses source-internal outcomes to fit a simple predictor for the same combinations before held-target scoring. This avoids target-label fitting but does not validate a universal combination predictor across disjoint groups. Earlier center-distillation C-selection differences remain a limitation; no new center experiments or tuning were performed.

## Validation and evidence preservation

- 19 pytest cases pass; full synthetic run covers 17 targets × 3 pools × 7 arms = 357 score rows, with source-only ranking, selection, fitting and paired summaries.
- `verify_results.py`: 42 archived CSV/JSON numerical consistency checks plus result-index paths. This checks reporting arithmetic, not original training correctness or all confidence intervals.
- One real Cho subject's preprocessing arrays match the old cache exactly. First 32 PhysioNet epochs and first 32 feature rows per dataset match exactly with historical CUDA BF16 and checkpoint.
- B and C720 for one target / pool 101 in each dataset reproduce archived BA within 7.1e-15 pp. Details: `NUMERICAL_VALIDATION.json`.
- Full raw-data CLI extraction, all folds, automatic download, and CPU FP32 parity have not been verified. No GPU training occurred. CI is configured; no claim is made that a remote Actions run has completed.
- Existing historical results and scripts are unchanged in Git. The new generalized entry does not reimplement every low-cost/complementarity/center dependency chain.

## Remaining resources and decisions

Users must obtain the original EDF/MAT files, the exact upstream model checkout and matching checkpoint. Full end-to-end numerical validation still requires regenerating both complete feature banks and comparing every archived fold. Private legacy metadata/intermediates are needed for historical directory-bound scripts; center training is outside the public release. A record-level pretraining ledger is unavailable, so overlap cannot be independently ruled out. An owner-approved repository license remains unresolved.

## Release safety

`audit_release.py` scans tracked working files and all reachable Git history blobs for forbidden data/weight extensions, files over 1 MB, credential patterns, infrastructure paths and IP addresses. `REVIEW_SECURITY.json` is the pre-commit scan snapshot; the scan was also run after committing. This is a best-effort pattern check, complemented by reviewing the file list. It cannot certify absence of every possible secret or legal issue. Original `SECURITY_AUDIT.json` remains the historical release audit.
