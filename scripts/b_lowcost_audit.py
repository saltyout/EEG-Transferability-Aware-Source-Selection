import json,itertools
from pathlib import Path
import pandas as pd,numpy as np
p=Path(__file__).resolve().parents[1]/'results/b_lowcost_20261106'
m=json.loads((p/'label_manifest.json').read_text())
for r in m:
    assert len(set(r['pilot_ids']))==192
    assert len(set(r['unique_ids']))==324
    assert set(r['fit_ids'])<=set(r['unique_ids'])
    assert set(r['pilot_ids'])<=set(r['unique_ids'])
    if r['arm'] in ('L','R'):
        assert len(set(r['fit_ids']))==180
        assert len(set(r['pilot_ids'])&set(r['fit_ids']))==48
    else:assert len(set(r['fit_ids']))==324
rank=pd.read_csv(p/'ranking_stability.csv')
out={}
for dataset,g in rank.groupby('dataset'):
    overlaps=[]
    for _,pool in g.groupby(['pool_seed','target']):
        sets=[set(x.split('|')) for x in pool.selected]
        overlaps.extend([len(a&b) for a,b in itertools.combinations(sets,2)])
    out[dataset]=dict(mean_top4_overlap_across_pilot_seeds=np.mean(overlaps),full_reference_rho_by_pilot=g.groupby('pilot_seed').rho.mean().to_dict(),full_reference_overlap_by_pilot=g.groupby('pilot_seed').overlap.mean().to_dict())
(p/'budget_audit.json').write_text(json.dumps(dict(status='PASS',records=len(m),unique_labels=324,pilot_labels=192,selected_overlap=48,stability=out),indent=2))
print(json.dumps(out,indent=2))
