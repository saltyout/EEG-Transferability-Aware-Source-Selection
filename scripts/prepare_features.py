"""Recovered preprocessing; PhysioNet requires explicit historical trial manifest."""
import argparse
import contextlib
import hashlib
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from eeg_preprocessing import cho_epochs

CHECKPOINT_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"

def physionet_epochs(root, manifest):
    import mne
    rows = json.loads(manifest.read_text())
    if not isinstance(rows,list) or not rows:
        raise ValueError("Manifest must be a nonempty list")
    output = []
    labels = []
    subjects = []
    cache = {}
    for row in rows:
        if not isinstance(row.get("output_start"),int) or row.get("label") not in (0,1) or not isinstance(row.get("subject"),str):
            raise ValueError("Invalid window, label or subject in manifest")
        relative = Path(row["file"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("EDF path must stay inside data directory")
        if path not in cache:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            if raw.info["sfreq"] != 160:
                raise ValueError("Expected 160 Hz PhysioNet EEG")
            raw.pick(["C3", "Cz", "C4"])
            signal = raw.get_data(units="uV")
            signal -= signal.mean(axis=0, keepdims=True)
            filtered = sosfiltfilt(butter(4,[.3,40],fs=160,btype="bandpass",output="sos"),signal,axis=-1)
            cache[path] = resample_poly(filtered,5,4,axis=-1)
        start = int(row["output_start"])
        trial = cache[path][:,start:start+800]
        if start < 0 or trial.shape != (3,800):
            raise ValueError("Invalid trial window")
        output.append((trial/100).astype("float32").reshape(3,4,200))
        labels.append(int(row["label"]))
        subjects.append(row["subject"])
    return np.array(output), np.array(labels), np.array(subjects)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["cho2017","physionet"],required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--cbramod",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--precision",choices=["fp32","bf16"],default="fp32")
    a = p.parse_args()
    if a.output.suffix != ".npz":
        p.error("--output must end in .npz")
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise FileExistsError(a.output)
    revision = subprocess.check_output(["git","-C",str(a.cbramod),"rev-parse","HEAD"],text=True).strip()
    if revision != "b9e961003214326972c567eff390e75b0287e32a":
        raise ValueError("CBraMod checkout differs from recorded historical revision")
    if subprocess.check_output(["git","-C",str(a.cbramod),"status","--porcelain"],text=True).strip():
        raise ValueError("CBraMod checkout must be clean")
    digest = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from recorded historical weights")
    if a.dataset == "physionet":
        if a.manifest is None:
            p.error("PhysioNet requires the historical ordered trial manifest")
        epochs,y,subjects = physionet_epochs(a.data,a.manifest)
        bank = np.arange(len(y))
    else:
        files = sorted(a.data.glob("s[0-9][0-9].mat"))
        if [f.stem for f in files] != [f"s{i:02d}" for i in range(1,53)]:
            raise ValueError("Locked Cho2017 protocol requires all 52 files")
        records = [cho_epochs(path) for path in files]
        epochs = np.concatenate([d["x"] for d in records])
        y = np.concatenate([d["y"] for d in records])
        subjects = np.concatenate([np.repeat(path.stem,len(d["y"])) for path,d in zip(files,records)])
        bank = []
        for i,subject in enumerate(np.unique(subjects)):
            ix = np.flatnonzero(subjects == subject)
            rng = np.random.default_rng(20261105+i)
            for label in (0,1):
                bank.extend(rng.permutation(ix[y[ix] == label])[:22+int(label == i%2)])
        bank = np.array(bank)
    if not np.isfinite(epochs).all():
        raise ValueError("Non-finite EEG")
    from reproduce import validate_features
    validate_features(epochs.reshape(len(y),-1),y,subjects,bank)
    import torch
    sys.path.insert(0,str(a.cbramod))
    from models.cbramod import CBraMod
    model = CBraMod()
    model.load_state_dict(torch.load(a.weights,map_location="cpu",weights_only=True),strict=True)
    model.proj_out = torch.nn.Identity()
    model.requires_grad_(False).eval().to(a.device)
    features = []
    with torch.inference_mode():
        for start in range(0,len(y),32):
            context = torch.autocast(device_type=torch.device(a.device).type,dtype=torch.bfloat16) if a.precision == "bf16" else contextlib.nullcontext()
            with context:
                value = model(torch.tensor(epochs[start:start+32],device=a.device))
            features.append(value.float().flatten(1).cpu().numpy())
    x = np.concatenate(features)
    expected = 1800 if a.dataset == "cho2017" else 2400
    if x.shape != (len(y),expected):
        raise ValueError("Unexpected CBraMod feature shape")
    from reproduce import validate_features
    validate_features(x,y,subjects,bank)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,X=x,y=y,subjects=subjects,bank_indices=bank)
    a.output.with_suffix(".json").write_text(json.dumps(dict(dataset=a.dataset,checkpoint_sha256=digest,code_revision=revision,precision=a.precision,device=a.device,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest() if a.manifest else None,subjects=len(np.unique(subjects)),trials=len(y),features=x.shape[1],historical_numeric_parity="not yet established"),indent=2))

if __name__ == "__main__":
    main()
