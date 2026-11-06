"""Locked two-stage source labeling audit; no GPU or parameter search."""
import json,csv,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
R=Path(__file__).resolve().parents[1];O=R/'results/b_lowcost_20261106';O.mkdir(exist_ok=False)
protocol=dict(pool_seeds=[101,202,303],pilot_seeds=[701,802,903],random_source_seeds=[401,502,603],pilot_per_source=12,pilot_per_class=6,selected_sources=4,final_per_selected_source=45,total_unique=324,C=.01,primary='L-R and L-U324',reference='H original 720-label ranking with 180-label readout',E_accessed=False)
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
records=[];manifests=[];stability=[]
def classifier():return make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
for dataset in ('physionet','cho2017'):
    if dataset=='physionet':
        meta={r['trial_id']:r for r in csv.DictReader((R/'metadata/trials.csv').open())}
        ids=json.loads((R/'results/b_calibrated_hybrid_kd_20261102/provenance.json').read_text())['train_ids']
        fullX=np.load(R/'results/b_teacher_factorial_20261030/student_flat_control/cache/train_wide_car_features.npy')
        fully=np.array([int(meta[i]['label']) for i in ids]);fulls=np.array([meta[i]['subject'] for i in ids]);bank=np.arange(len(ids))
        old=R/'results/b_source_intervention_20261105'
        matrix=R/'results/b_transfer_diag_20261105/transfer_matrix.csv'
    else:
        f=np.load(R/'results/b_cho_external_20261105/features.npz');fullX,fully,fulls=f['X'],f['y'],f['subjects']
        bank=[]
        for i,subject in enumerate(np.unique(fulls)):
            rng=np.random.default_rng(20261105+i);ix=np.flatnonzero(fulls==subject)
            for label in (0,1):bank.extend(rng.permutation(ix[fully[ix]==label])[:22+int(label==i%2)])
        bank=np.array(bank);old=R/'results/b_cho_external_20261105/intervention';matrix=old/'transfer_matrix.csv'
    X,y,s=fullX[bank],fully[bank],fulls[bank];subjects=np.unique(s);N=len(subjects)
    indices=[np.flatnonzero(s==subject) for subject in subjects]
    M=pd.read_csv(matrix,index_col=0).loc[subjects,subjects].to_numpy()
    old_scores=pd.read_csv(old/'scores.csv')
    for pilot_seed in protocol['pilot_seeds']:
        pilot=[];rest=[]
        for i,ix in enumerate(indices):
            rng=np.random.default_rng(pilot_seed+i*1000)
            chosen=np.concatenate([rng.permutation(ix[y[ix]==c])[:6] for c in (0,1)])
            pilot.append(chosen);rest.append(np.setdiff1d(ix,chosen))
        lowM=np.full((N,N),np.nan)
        for i in range(N):
            model=classifier().fit(X[pilot[i]],y[pilot[i]])
            for j in range(N):
                if i!=j:lowM[i,j]=balanced_accuracy_score(y[pilot[j]],model.predict(X[pilot[j]]))
        for seed in protocol['pool_seeds']:
            for hold in range(N):
                pool=np.random.default_rng(seed+hold*1000).choice([i for i in range(N) if i!=hold],16,replace=False)
                quality=lambda mat:np.array([np.mean(mat[i,[j for j in pool if j!=i]]) for i in pool])
                low,high=quality(lowM),quality(M)
                selected=pool[np.argsort(-low,kind='stable')[:4]];reference=pool[np.argsort(-high,kind='stable')[:4]]
                stability.append(dict(dataset=dataset,pilot_seed=pilot_seed,pool_seed=seed,target=subjects[hold],rho=spearmanr(low,high).statistic,overlap=len(set(selected)&set(reference)),selected='|'.join(subjects[selected])))
                pilot_ids=np.concatenate([pilot[i] for i in pool])
                train=np.concatenate([indices[i] for i in selected])
                union=np.union1d(pilot_ids,train)
                assert len(pilot_ids)==192 and len(train)==180 and len(union)==324
                rng=np.random.default_rng(20261106+seed+pilot_seed+hold*1000)
                order=rng.permutation(pool);uniform=list(pilot_ids)
                for k,i in enumerate(order):
                    count=9 if k<4 else 8;extra=int(rng.integers(2))
                    for c in (0,1):
                        choices=rest[i][y[rest[i]]==c]
                        uniform.extend(rng.permutation(choices)[:count//2+int(count%2 and c==extra)])
                uniform=np.array(uniform);assert len(set(uniform))==324
                for arm,fit,total in [('L',train,union),('U324',uniform,uniform)]:
                    assert subjects[hold] not in set(s[total])
                    weights=np.array([1/np.sum(s[fit]==subject) for subject in s[fit]]);weights/=weights.mean()
                    model=classifier().fit(X[fit],y[fit],logisticregression__sample_weight=weights)
                    test=fulls==subjects[hold]
                    ba=100*balanced_accuracy_score(fully[test],model.predict(fullX[test]))
                    key=dict(dataset=dataset,pilot_seed=pilot_seed,pool_seed=seed,target=subjects[hold],arm=arm)
                    records.append(dict(**key,BA=ba,total_unique=len(total),fit_labels=len(fit)))
                    manifests.append(dict(**key,pilot_ids=bank[pilot_ids].tolist(),fit_ids=bank[fit].tolist(),unique_ids=bank[total].tolist()))
                prior=old_scores[(old_scores.pool_seed==seed)&(old_scores.target==subjects[hold])]
                for arm,original,cost in [('R','A',324),('H','B',720),('C720','C720',720)]:
                    scores=prior[prior.arm==original];assert len(scores)==(3 if arm=='R' else 1)
                    records.append(dict(dataset=dataset,pilot_seed=pilot_seed,pool_seed=seed,target=subjects[hold],arm=arm,BA=scores.BA.mean(),total_unique=cost,fit_labels=720 if arm=='C720' else 180))
                # Random fit ID unions are checked even though scores are reused.
                for random_seed in protocol['random_source_seeds']:
                    chosen=np.random.default_rng(random_seed+seed*10000+hold*1000000).choice(pool,4,replace=False)
                    ix=np.concatenate([indices[i] for i in chosen])
                    assert len(np.union1d(pilot_ids,ix))==324
                    manifests.append(dict(dataset=dataset,pilot_seed=pilot_seed,pool_seed=seed,target=subjects[hold],arm='R',random_seed=random_seed,pilot_ids=bank[pilot_ids].tolist(),fit_ids=bank[ix].tolist(),unique_ids=bank[np.union1d(pilot_ids,ix)].tolist()))
        print(dataset,pilot_seed,flush=True)
pd.DataFrame(records).to_csv(O/'scores.csv',index=False)
pd.DataFrame(stability).to_csv(O/'ranking_stability.csv',index=False)
(O/'label_manifest.json').write_text(json.dumps(manifests))
summary={}
df=pd.DataFrame(records)
for dataset,g in df.groupby('dataset'):
    table=g.groupby(['target','pool_seed','pilot_seed','arm']).BA.mean().unstack()
    differences=pd.DataFrame({a+'-'+b:table[a]-table[b] for a,b in [('L','R'),('L','U324'),('L','H'),('H','R'),('L','C720')]})
    differences.to_csv(O/f'{dataset}_paired.csv')
    stats={}
    for col in differences:
        v=differences[col].groupby('target').mean()
        rng=np.random.default_rng(20261106);boot=v.to_numpy()[rng.integers(0,len(v),(20000,len(v)))].mean(1)
        stats[col]=dict(mean_pp=v.mean(),improved=int((v>0).sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),by_pilot=differences[col].groupby('pilot_seed').mean().to_dict(),by_pool=differences[col].groupby('pool_seed').mean().to_dict())
    rank=pd.DataFrame(stability);rank=rank[rank.dataset==dataset]
    summary[dataset]=dict(BA=table.mean().to_dict(),contrasts=stats,rank_rho=rank.rho.mean(),top4_overlap=rank.overlap.mean())
    table.groupby('target').mean().to_csv(O/f'{dataset}_subjects.csv')
(O/'summary.json').write_text(json.dumps(summary,indent=2))
(O/'acceptance.json').write_text(json.dumps(dict(status='PASS',new_pooled_fits=2*3*3*76,E_accessed=False)))
print(json.dumps(summary,indent=2))
