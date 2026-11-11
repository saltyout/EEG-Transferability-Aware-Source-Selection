# Third-party and release notes

- CBraMod model code is obtained separately from its upstream repository at the recorded revision. No model code or checkpoint is vendored here. Review upstream LICENSE and checkpoint terms before redistribution or use.
- PhysioNet EEGMMIDB and Cho2017 are external public datasets; public availability does not remove their attribution or usage conditions. Download from the provider pages linked in README. The small PhysioNet manifest describes public experiment windows and checksums, not signals or a new dataset license.
- The Cho acquisition channel order and loader structure were checked against the MOABB dataset loader and original historical implementation. No MOABB installation is required.
- NumPy, SciPy, pandas, scikit-learn, PyTorch, MNE, einops and pytest are installed dependencies with their own licenses. GitHub Actions references upstream checkout/setup-python actions; no credentials are required for tests.
- This repository has no LICENSE file. A project owner must choose an appropriate license after checking code ownership. No license or authorship was invented during this review.

## GitHub presentation

Suggested About: `Reproducible EEG source-selection research with frozen CBraMod: cross-subject transfer, annotation budgets, and transparent negative results.`

Suggested topics: `eeg`, `brain-computer-interface`, `motor-imagery`, `transfer-learning`, `source-selection`, `reproducible-research`, `cbramod`.

At review time the About text was generic, topics were empty, and GitHub reported no license. Remote settings were not modified. Select a license before claiming open-source reuse; consider a release tag only after complete extraction/fold parity is checked. A DOI or publication citation should be added only if one exists.
