import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
P=Path(__file__).resolve().parents[1]/'results/b_complementarity_20261105'
c=pd.read_csv(P/'combinations.csv');p=pd.read_csv(P/'predictions.csv');s=pd.read_csv(P/'source_stability.csv')
rows=[]
for keys,g in c.groupby(['dataset','pool_seed','target']):
    pairs=[]
    for i in range(len(g)):
        for j in range(i):
            a,b=g.iloc[i],g.iloc[j]
            if abs(a.quality-b.quality)<=.02 and a.error_disagreement!=b.error_disagreement:
                high,low=(a,b) if a.error_disagreement>b.error_disagreement else (b,a)
                pairs.append(dict(dataset=keys[0],pool_seed=keys[1],target=keys[2],higher_group=int(high.group),lower_group=int(low.group),quality_gap_pp=100*(high.quality-low.quality),target_gap_pp=100*(high.target_BA-low.target_BA),internal_gap_pp=100*(high.internal_BA-low.internal_BA)))
    rows.extend(pairs)
pd.DataFrame(rows).to_csv(P/'quality_matched_pairs.csv',index=False)
summary={}
for dataset in c.dataset.unique():
    q=p[p.dataset==dataset]
    err=q.groupby(['pool_seed','target','model']).absolute_error_pp.mean().unstack()
    diff=err.base-err.plus_complementarity
    subject=diff.groupby('target').mean()
    rng=np.random.default_rng(20261105)
    boot=subject.to_numpy()[rng.integers(0,len(subject),(20000,len(subject)))].mean(1)
    matched=pd.DataFrame(rows);matched=matched[matched.dataset==dataset]
    ms=matched.groupby('target')[['target_gap_pp','quality_gap_pp']].mean()
    bootm=ms.target_gap_pp.to_numpy()[rng.integers(0,len(ms),(20000,len(ms)))].mean(1)
    ranks=[]
    for keys,g in q.groupby(['pool_seed','target','model']):
        ranks.append(dict(pool_seed=keys[0],target=keys[1],model=keys[2],rho=spearmanr(g.predicted_BA,g.target_BA).statistic))
    ranks=pd.DataFrame(ranks)
    st=s[s.dataset==dataset].groupby(['pool_seed','target'])
    sr=[dict(q_sd=spearmanr(g.quality,g.sd).statistic,q_worst=spearmanr(g.quality,g.worst).statistic,q_hold=spearmanr(g.quality,g.held_BA).statistic) for _,g in st]
    summary[dataset]=dict(MAE_pp=err.mean().to_dict(),MAE_gain_adding_complementarity=float(subject.mean()),MAE_gain_CI=np.quantile(boot,[.025,.975]).tolist(),MAE_improved_subjects=int((subject>0).sum()),MAE_gain_by_pool=diff.groupby('pool_seed').mean().to_dict(),rank_rho=ranks.groupby('model').rho.mean().to_dict(),matched_pairs=len(matched),matched_target_gap_pp=ms.target_gap_pp.mean(),matched_gap_CI=np.quantile(bootm,[.025,.975]).tolist(),matched_quality_gap_pp=ms.quality_gap_pp.mean(),source_relations=pd.DataFrame(sr).mean().to_dict())
    diff.to_csv(P/f'{dataset}_prediction_gain.csv')
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
