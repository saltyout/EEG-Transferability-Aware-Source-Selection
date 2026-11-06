# EEG Transferability-Aware Source Selection

A controlled research study of cross-subject EEG transferability, source selection, and annotation cost with frozen CBraMod features.

| Experiment | PhysioNet | Cho2017 | Decision |
|---|---:|---:|---|
| Historical selection vs random, 180 readout labels | +4.34 pp | +2.29 pp | Positive within evaluated protocols |
| Historical selection vs all-source same readout budget | +4.26 pp | +1.37 pp | Positive |
| Two-stage selection vs all-source same 324 total labels | -1.86 pp | -0.84 pp | No-Go |
| Complementarity predictor MAE improvement | -0.0025 pp | -0.0074 pp | No-Go |
| Full-source weighting vs uniform, 720 labels | +0.46 pp | +0.10 pp | No-Go |

Ranking requires 720 unique source labels even when the final readout uses 180. All-source training with 720 labels exceeds selected-source training with 180 on average. Subject-center distillation failed new-development replication and is archived alongside the source-selection findings.

## Reproduction

Use Python 3.11. Install requirements.txt in a fresh virtual environment. Run:

    python scripts/reproduce.py --features /path/to/private/features.npz --output outputs/run1

The private NPZ contains X, y, subjects, and bank_indices. X is a finite two-dimensional frozen-feature matrix, y contains integer 0/1 labels, subjects contains string IDs, and bank_indices contains unique integer row indices. Each subject contributes exactly 45 bank trials. Each outer target is evaluated on all its trials. At least 17 subjects are required.

See [reproduction](docs/REPRODUCIBILITY.md), [protocol](docs/PROTOCOL.md), [results](docs/RESULTS.md), and [limitations](docs/LIMITATIONS.md).

## Data and model provenance

- CBraMod https://github.com/wjq-learning/CBraMod
- PhysioNet EEGMMIDB https://physionet.org/content/eegmmidb/
- Cho2017 https://gigadb.org/dataset/100295
- Cho loader reference https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/gigadb.py

Data and weights must be obtained separately under their providers' terms. This release has no assigned open-source license. Obtain permission before redistributing original code. Upstream components retain their licenses.
