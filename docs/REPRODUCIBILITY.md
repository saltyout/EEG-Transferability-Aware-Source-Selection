# Reproduction guide

## Environments

Use Python 3.11. `requirements-cpu.txt` contains NumPy/SciPy/pandas/scikit-learn; `requirements-dev.txt` adds pytest; `requirements-eeg.txt` adds MNE, PyTorch and einops. `requirements.txt` selects the CPU environment. Version ranges are compatibility constraints, not a bitwise lock. `requirements-historical.txt` preserves the original recorded versions; these were not newly installed or validated on every platform. Local lightweight checks used Python 3.13.5 with NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, pytest 8.3.4. Generated protocol JSON records the actual runtime versions.

For CPU work set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before Python starts (`export` on POSIX; `$env:NAME="1"` in PowerShell). CUDA PyTorch installation depends on the local driver; use an appropriate upstream wheel. No model training is required.

## Obtain public inputs

PhysioNet: the repository contains 1,080 ordered training-bank windows from 24 development subjects and hashes for their 72 public EDF files. These were recovered from the original provenance, not resampled. They contain public dataset IDs, labels, resampled starts and relative file names, with no private trial IDs or test partition.

```bash
python scripts/download_physionet.py --output data/physionet
```

The downloader verifies SHA-256 against the recorded originals, downloads only manifest files, resumes by skipping verified files, and fails on mismatch. Its network path has not been exercised in this review. Alternatively download those files from the official EEGMMIDB 1.0.0 release, preserving `Sxxx/SxxxRxx.edf` layout.

Cho2017: obtain `s01.mat` through `s52.mat` from https://gigadb.org/dataset/100295 and place them directly in `data/cho2017/`. No mirror availability or automatic download is assumed. Use the original MATLAB distribution with `eeg.srate`, `imagery_event`, `imagery_left`, and `imagery_right`; not a differently preprocessed export.

```bash
git clone https://github.com/wjq-learning/CBraMod.git external/CBraMod
git -C external/CBraMod checkout b9e961003214326972c567eff390e75b0287e32a
python -m pip install -r requirements-eeg.txt
```

Obtain `pretrained_weights.pth` using the upstream project's distribution instructions and save it in `weights/`. Required SHA-256: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`. Extraction rejects a different checkpoint or modified/wrong upstream checkout. The checkpoint is loaded with `weights_only=True` and strict keys, then `proj_out=Identity`, `eval()`, no gradients. Batch size is 32; channel/time/embedding axes are flattened in their original order.

## Raw EEG → frozen features

Historical precision was CUDA BF16 autocast. Commands below use that mode; no GPU training occurs.

```bash
python scripts/prepare_features.py --dataset physionet --data data/physionet --manifest configs/physionet_train_windows.json --cbramod external/CBraMod --weights weights/pretrained_weights.pth --device cuda --precision bf16 --output outputs/physionet/features.npz
python scripts/prepare_features.py --dataset cho2017 --data data/cho2017 --cbramod external/CBraMod --weights weights/pretrained_weights.pth --device cuda --precision bf16 --output outputs/cho2017/features.npz
```

For CPU reference inference use `--device cpu --precision fp32`; this changes numerical precision and is **not a claim of exact historical feature equivalence**. Full extraction takes substantial memory and time. Inputs are processed continuously before slicing; there is no fitted cross-subject EEG transform.

| Setting | PhysioNet | Cho2017 |
|---|---|---|
| Input sampling | 160 Hz EDF, MNE volts converted to µV | 512 Hz MATLAB, recorded µV interpretation |
| Channels | standardized C3, Cz, C4 | acquisition indices 12, 47, 49 (zero based) |
| Reference | own three-channel mean, before filtering | same |
| Filter | SciPy Butterworth order 4, 0.3–40 Hz, `sosfiltfilt` continuously | same at 512 Hz |
| Resample | `resample_poly(5,4)` | `resample_poly(25,64)` |
| Window | 800 samples at manifest `output_start` | 600 samples at rounded onset × 200/512 |
| Events | manifest T1/T2 labels 0/1 | nonzero marker rising edges; left=0, right=1 |
| Model scale / shape | µV/100; 3×4×200 | µV/100; 3×3×200 |
| Feature width | 2400 | 1800 |

No notch, trial DC removal, per-trial RMS normalization, PCA, channel interpolation or per-target adaptation is introduced. Cho's shared marker is applied separately to left and right recordings. Bank sampling uses seed 20261105 plus sorted subject index; per-class counts 22/23 alternate the extra class. Preserve sample and subject order: it affects seeded sampling.

PhysioNet manifests are ordered JSON lists with `file` (relative EDF), `output_start` (integer 200-Hz sample index), `label` (0 or 1), `subject` (string). The supplied manifest is the historical bank; a custom manifest represents a new experiment, even if preprocessing matches.

## Features → evaluation

```bash
python scripts/reproduce.py --features outputs/physionet/features.npz --output outputs/physionet/run1
python scripts/reproduce.py --features outputs/cho2017/features.npz --output outputs/cho2017/run1
python scripts/verify_results.py
```

NPZ contract: `X` is finite numeric `[trials, features]`; `y` is integer 0/1; `subjects` is a string vector; `bank_indices` is a unique in-bounds integer vector. Each subject contributes exactly 45 bank trials, with at least six per class; at least 17 subjects are required. PhysioNet uses only the 1,080 locked bank trials for outer scoring; Cho evaluates all available trials of the held subject. They do not have identical target trial budgets.

The portable entry implements selected/random/all180/all720 and transfer weighting. It writes `transfer_matrix.csv`, `scores.csv`, `fit_manifest.json`, `summary.json` (paired target bootstrap), and `protocol.json` under a new output directory. Fit manifests and generated features remain gitignored. Scalers fit only final training trials (or the single source during ranking). A full reusable matrix may contain all subjects, but outer scoring/ranking uses **only its current source-pool submatrix**, never the held target's row/column. No C search occurs. Changes to held labels cannot affect fitted predictions; tests cover this boundary.

Readout annotations are subsets of source banks. Reported totals count the union per arm/per outer pool; they do not sum over repeated fitting. Class-stratified bank draws are retrospective benchmarks with known task labels, not proof of a prospective blind acquisition policy. Equal-subject loss weighting applies even to 11/12-trial all180 controls.

## Historical scripts

Historical `b_*.py` files are unchanged. Some execute immediately on import and write to fixed relative paths; run them as scripts in a separate research copy, never import them as a library. The new portable entry is the recommended interface.

| Scripts | Required inputs / preceding steps | Outputs |
|---|---|---|
| `b_cho2017_adapter.py INPUT OUTPUT` | one original MAT file; SciPy/NumPy | epochs `x,y,event` |
| `b_source_intervention.py` | private `metadata/trials.csv`, `metadata/splits.json`, calibrated-KD `provenance.json`, frozen `train_wide_car_features.npy`, diagnostic `transfer_matrix.csv` | source intervention scores, manifests, predictions, summary |
| `b_cho_intervention.py` | `results/b_cho_external_20261105/features.npz` | intervention matrix, scores, manifests, summary |
| `b_lowcost.py` → `b_lowcost_audit.py` | both feature banks and preceding intervention scores/matrices | two-stage scores, label union audit, stability, summaries |
| `b_weight.py` | both banks and intervention scores/matrices | weighting controls, self-CV scores, summaries |
| `b_complementarity.py` → `b_complementarity_summary.py` → `b_complementarity_audit.py` | both banks | combination diagnostics and summaries |

All historical classifier/diagnostic scripts use the CPU requirements; the adapter requires SciPy. Exact relative paths remain visible in each loading block. Private metadata, sample-level prediction/manifests and old diagnostic intermediates are not bundled. New features do not automatically synthesize those historical directory trees. Center-distillation training and its full dependency chain are not included: only archived evidence is public. This is an explicit scope limit, not a claim that every historical script runs from a clean checkout.

## Validation status

`pytest` and the synthetic workflow validate CPU computation without EEG/GPU. `verify_results.py` checks 42 published mean/contrast values and index links from public CSVs. `docs/NUMERICAL_VALIDATION.json` records real-data spot checks: one complete Cho preprocessing file, the first 32 PhysioNet epochs, the first 32 features per dataset, and B/C720 in the first target with pool seed 101. All matched (maximum BA difference about 7.1e-15 pp). This is **not a complete fold-by-fold rerun or validation of every CLI/network path**. Frozen inference smoke checks used a GPU; no GPU training was run. Independent test EEG was not loaded.
