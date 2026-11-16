# EEG Transferability-Aware Source Selection

Can a source subject's ability to classify other people predict which EEG data are useful for an unseen person? This research repository studies that question with **transferability-based source selection, frozen and fine-tuned CBraMod evaluation, and explicit annotation accounting** on PhysioNet EEGMMIDB and Cho2017.

The contribution is an auditable empirical study, including unsuccessful hypotheses. It does not establish a new foundation model, a causal physiological mechanism, or state-of-the-art performance.

## Main finding: source selection remains useful after fine-tuning

**Training-source selection helps even when CBraMod is allowed to adapt.** With the same 180 final training labels and the same fine-tuning procedure, transfer-ranked selection outperforms random selection on both datasets. Source ranking uses only training-source relationships; the held target never participates in selection or fitting.

| Primary paired comparison after fine-tuning | PhysioNet (24 targets) | Cho2017 (52 targets) |
|---|---:|---:|
| Transfer-selected BA | **70.29%** | **62.39%** |
| Random-source BA | 65.76% | 60.23% |
| Selected − random | **+4.53 pp** | **+2.16 pp** |
| Descriptive paired-target 95% interval | **[+1.67, +7.54] pp** | **[+0.72, +3.83] pp** |
| Gain by training seed (17 / 29 / 43) | +2.41 / +6.34 / +4.84 pp | +2.46 / +1.84 / +2.17 pp |
| Targets improved, after averaging seeds | 16/24 | 33/52 |

Both datasets show a positive selected-minus-random difference in all three training seeds. Intervals average seeds within target before 20,000 paired-target bootstrap draws; they are conditional on the fixed source pool and trial banks. This supports the selection rule in both frozen and adapted settings, without establishing a physiological mechanism or a reduced annotation budget.

Sources: [PhysioNet complete summary](results/finetuning_physionet/summary.json), [Cho2017 complete summary](results/finetuning_cho2017/summary.json). **Ranking still requires 720 unique source labels; the final 180 fitting labels are a subset.**

Both datasets have completed the fixed-200-step follow-up: **576 PhysioNet fits and 1,248 Cho2017 fits**. Each data arm compares a frozen encoder with full encoder fine-tuning, using the same AdamW-trained linear head. This differs from the historical logistic-regression readout reported below.

| Data arm | PhysioNet frozen | PhysioNet tuned | Cho2017 frozen | Cho2017 tuned |
|---|---:|---:|---:|---:|
| Transfer-selected 180 | 63.37 | **70.29** | 60.07 | **62.39** |
| Random 180 | 59.34 | 65.76 | 56.69 | 60.23 |
| All-source 180 | 58.27 | 61.85 | 58.86 | 59.80 |
| All-source 720 | 63.80 | 64.84 | 61.09 | 56.36 |

Fine-tuned selection exceeds fine-tuned random selection by **4.53 pp** on PhysioNet and **2.16 pp** on Cho2017. Selected-source tuning improves its paired frozen baseline by 6.92 / 2.33 pp. This supports persistence of the selection advantage under adaptation, not an extra interaction benefit: the selection-minus-random gap changes by +0.49 / -1.21 pp, with both descriptive intervals spanning zero.

All720 tuning falls by **4.72 pp on Cho2017**; this negative result is retained. Fixed 200 updates expose each 180-label trial about 35.6 times and each 720-label trial about 8.9 times. No claim is made about convergence or superiority over fully trained all-source models. Only one source-pool seed and one random-selection seed were tested with three training seeds. Selection still costs **720 unique source labels**.

See [final fine-tuning protocol and commands](docs/FINETUNING.md) and [result index](docs/RESULTS.md). The older Cho2017 interim snapshot remains archived for traceability and is superseded by the complete results.

## Method

```mermaid
flowchart LR
    A[Public EEG + locked trial definitions] --> B[Own C3/Cz/C4 CAR and filtering]
    B --> C[Frozen CBraMod features]
    C --> D[Outer held-out target]
    C --> E[16 training sources: 45 labels each]
    E --> F[Source-only scaling and single-source classifiers]
    F --> G[Mean BA on the other 15 sources]
    G --> H[Top 4 sources: 180 readout labels]
    H --> I[Source-only scaling + logistic regression C=0.01]
    I --> J[Evaluate held-out target]
    H --> K[Selected raw EEG: fine-tune CBraMod + linear head]
    K --> J
    D --> J
```

No target data enter ranking, standardization or fitting. Historical frozen-feature experiments use three source-pool seeds and three random-selection seeds. The fine-tuning follow-up uses one source-pool seed, one random-selection seed and three training seeds, with paired frozen-encoder controls. BA is the mean of class recalls, aggregated equally across targets.

## Historical frozen-feature results and negative controls

Archived results, BA in percent; differences are percentage points (pp). These are **historical measurements**, not outputs of the synthetic example.

| Readout | PhysioNet (24 targets) | Cho2017 (52 targets) | Ranking labels | Readout labels | Unique source labels |
|---|---:|---:|---:|---:|---:|
| Random 4 sources (A) | 59.00 | 58.20 | 0 | 180 | 180 |
| Transfer-ranked 4 sources (B) | 63.35 | 60.49 | 720 | 180 | **720** |
| All 16, matched readout (C) | 59.09 | 59.12 | 0 | 180 | 180 |
| All 16, full bank (C720) | **64.91** | **61.74** | 0 | 720 | 720 |

B−A: **+4.34 pp** (descriptive paired target-bootstrap 95% interval [2.03, 6.68]) and **+2.29 pp** ([1.30, 3.38]). B−C: +4.26 / +1.37 pp. This supports selection under a fixed **final readout** budget; it does **not** establish lower total annotation cost. The 180 fitting labels are a subset of the 720 ranking labels, not an additional cost. Target evaluation labels are separate.

The source pools overlap; intervals condition on these populations and trial banks, and are not evidence from independent pool replications. Both datasets now have development history. The separate PhysioNet test partition remains sealed.

| Negative control / follow-up | PhysioNet | Cho2017 | Decision |
|---|---:|---:|---|
| Two-stage selection vs all-source, 324 total labels | −1.86 pp | −0.84 pp | No-Go |
| Complementarity predictor MAE reduction | −0.0025 pp | −0.0074 pp | No-Go |
| Full-source transfer weighting vs uniform, 720 labels | +0.46 pp | +0.10 pp | No-Go: insufficient consistent advantage over controls |
| Subject-center distillation | Failed new-development replication | Not tested | No-Go |

See [all results](docs/RESULTS.md), [locked protocol](docs/PROTOCOL.md), and [limitations](docs/LIMITATIONS.md). Negative results and original scripts are preserved.

## Run in minutes: synthetic CPU example

Python 3.11 recommended. No EEG downloads, model weights, GPU or account are needed:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_results.py
python scripts/make_demo.py --output outputs/synthetic.npz
python scripts/reproduce.py --features outputs/synthetic.npz --output outputs/demo
```

The example validates computation only; its scores are **not EEG evidence**. Outputs include the transfer matrix, per-target/per-pool scores, annotation budgets, fit-index manifests, and package/input provenance. Output directories must be new.

## Public EEG to evaluation

1. Obtain public EEG and the upstream CBraMod checkpoint separately.
2. Follow [the complete preparation and execution commands](docs/REPRODUCIBILITY.md). The new `prepare_features.py` implements recovered historical preprocessing and inference; the locked PhysioNet development window manifest is included.
3. Run `reproduce.py` on the generated NPZ and compare against the result index.

**Validation boundary:** real-data spot checks reproduced preprocessing and the first 32 frozen features in each dataset exactly, and selected/full-source BA in one target/pool per dataset to floating-point precision. The complete newly assembled extraction CLI and every historical fold have not been rerun. CPU FP32 inference is a supported reference mode, not asserted identical to historical CUDA BF16. See [numerical validation](docs/NUMERICAL_VALIDATION.json) and [code review](docs/CODE_REVIEW.md).

## Repository map

- `scripts/prepare_features.py`, `download_physionet.py`: raw-data preparation and frozen inference.
- `scripts/reproduce.py`, `make_demo.py`: portable CPU experiment and synthetic fixture.
- `scripts/finetune_sources.py`, `summarize_finetuning.py`: paired encoder fine-tuning and prediction/score audit; see the GPU validation limits in [FINETUNING.md](docs/FINETUNING.md).
- `scripts/verify_results.py`: independently recompute 42 archived CSV/JSON numeric checks.
- `scripts/b_*.py`: preserved historical implementations; [dependencies and execution order](docs/REPRODUCIBILITY.md#historical-scripts).
- `configs/`: public development windows and EDF hashes; no test partition, private trial IDs or EEG signals.
- `results/`: unmodified aggregate and subject-level evidence, including No-Go experiments.
- `tests/`, `.github/workflows/`: lightweight validation and CI.

## Data, models and licensing

[PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/), [Cho2017](https://gigadb.org/dataset/100295), [CBraMod](https://github.com/wjq-learning/CBraMod), and the [MOABB Cho loader](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/gigadb.py) are external resources. No EEG signals, weights or upstream model code are bundled. The reported pretraining corpus is TUEG; record-level overlap has not been independently excluded.

This repository currently has **no assigned open-source license**. Public visibility alone does not grant reuse rights. Upstream resources retain their own terms; see [third-party notes](docs/THIRD_PARTY.md). An owner-approved license is needed before advertising this as an open-source package.
