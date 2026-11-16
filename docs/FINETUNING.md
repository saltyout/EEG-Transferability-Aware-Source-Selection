# Fixed-budget fine-tuning follow-up

PhysioNet: COMPLETE, 576 fits, 24 targets. Cho2017: INTERIM, 765/1248 fits, 11 complete targets. This release does not alter running jobs. Original results and No-Go controls remain unchanged.

## Protocol

Pool seed 101, random-source seed 401, training seeds 17/29/43. Rank using the frozen source-only matrix within each 16-source pool; never rerank using the held target. Four data arms (selected180, random180, all180, all720) each have frozen and fine-tuned variants. Ranking uses 720 labels for selected180; its final 180 are a subset.

Exactly 200 replacement-sampled batches of 32; full encoder fine-tuning; AdamW encoder LR 1e-5 / head LR 1e-3, weight decay .01, gradient clipping 1, float32, final-step evaluation only. Linear head over flat features. Fixed normalization uses mean/std of each arm's frozen training features (std floor 1e-6); equal-subject loss weights. The frozen encoder runs in eval mode, the adapted encoder in train mode including dropout. This tests the whole adaptation procedure, not an isolated parameter-update mechanism. No early stopping or target tuning. PhysioNet tests its original 45-trial target bank; Cho tests all held-target trials. Both train only on the historical 45-trial source banks.

## Run

Install requirements-eeg.txt. Follow REPRODUCIBILITY.md, adding `--epochs-output outputs/cho2017/epochs.npz` to prepare_features.py (use the corresponding PhysioNet path for that dataset). Epoch NPZ contains aligned x/y/subjects; feature NPZ contains X/y/subjects/bank_indices. Use historical CUDA BF16 feature preparation; fine-tuning uses FP32. Run reproduce.py first to obtain transfer_matrix.csv.

```bash
python scripts/finetune_sources.py --dataset cho2017 --epochs outputs/cho2017/epochs.npz --features outputs/cho2017/features.npz --matrix outputs/cho2017/run1/transfer_matrix.csv --weights weights/pretrained_weights.pth --cbramod external/CBraMod --output outputs/cho-finetune
python scripts/summarize_finetuning.py --run outputs/cho-finetune --output outputs/cho-summary
```

For PhysioNet change dataset and paths. For four GPUs launch separate processes with CUDA_VISIBLE_DEVICES and worker IDs `--worker 0 --workers 4` through `--worker 3 --workers 4`, sharing output. Do not duplicate a worker. Resume only with matching protocol/input/code hashes; completed jobs are skipped. No trained weights are exported. Local trial predictions, indices and losses are for audit and must not be committed.

The portable runner checks checkpoint/revision, shapes, epoch alignment and recomputes the provided ranking matrix before fitting. Complete portable GPU CLI parity has not been tested. Live runs used scripts/history copies, whose infrastructure root alone is replaced by EEG_RESEARCH_ROOT; they require private legacy directory trees. The portable runner retains the sampling and optimization logic but its input adapter is new. Public validation does not claim a full training rerun.

## Boundaries

200 steps give approximately 35.6 sampled exposures per trial for 180 labels, 8.9 for 720. All720 may be undertrained; no superiority over a converged full-source model is established. Only one source-pool/random-selection seed was used. Three training seeds do not replace source-pool replication. Bootstrap averages seeds within target first and is descriptive. Interim summaries use fully paired targets only and omit intervals; ordered early completion is not a representative population sample. This frozen head is not the historical sklearn logistic regression. E remains sealed. Cho already has development history.
