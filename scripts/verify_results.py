"""Check archived CSV/JSON consistency without EEG, features or model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(root):
    checks=0
    for dataset in ("physionet","cho2017"):
        path=root/f"source_selection_{dataset}"
        table=pd.read_csv(path/("paired_scores.csv" if dataset=="physionet" else "per_subject_scores.csv")).groupby("target").mean(numeric_only=True)
        summary=json.loads((path/"summary.json").read_text())
        for arm,value in summary["means"].items():
            np.testing.assert_allclose(table[arm].mean(),value,rtol=0,atol=1e-10);checks+=1
        for contrast,value in summary["contrasts"].items():
            left,right=contrast.split("-")
            np.testing.assert_allclose((table[left]-table[right]).mean(),value["mean_pp"],rtol=0,atol=1e-10);checks+=1
        for experiment in ("low_cost","full_source_weighting"):
            path=root/experiment
            table=pd.read_csv(path/f"{dataset}_subjects.csv")
            summary=json.loads((path/"summary.json").read_text())[dataset]
            for arm,value in summary["BA"].items():
                np.testing.assert_allclose(table[arm].mean(),value,rtol=0,atol=1e-10);checks+=1
    for paths in json.loads((root/"index.json").read_text()).values():
        for path in paths:
            if not (root.parent/path).is_file(): raise FileNotFoundError(path)
    return checks

if __name__=="__main__":
    print(f"PASS: {verify(Path(__file__).resolve().parents[1]/'results')} archived numeric checks; result index resolves")
