from pathlib import Path
import json,itertools
import pandas as pd,numpy as np
from scipy.stats import spearmanr
p=Path(__file__).resolve().parents[1]/'results/b_complementarity_20261105'
d=pd.read_csv(p/'combinations.csv')
result={}
for dataset,g in d.groupby('dataset'):
    rows=[]
    for _,f in g.groupby(['pool_seed','target']):
        rows.append({k:spearmanr(f[k],f.target_BA).statistic for k in ['quality','instability','representation_distance','error_disagreement','double_fault','excess_double_fault']})
    result[dataset]=dict(mean_metrics=g[['error_disagreement','double_fault','excess_double_fault']].mean().to_dict(),descriptive_within_pool_rho=pd.DataFrame(rows).mean().to_dict())
    assert len(g)==(24 if dataset=='physionet' else 52)*3*12
    assert all(target not in sources.split('|') for target,sources in zip(g.target,g.sources))
(p/'descriptive.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
