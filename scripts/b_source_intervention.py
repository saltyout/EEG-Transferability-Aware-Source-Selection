"""Locked pooled-source intervention; CPU only, no hyperparameter search."""
import json,csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
R=Path(__file__).resolve().parents[1]
O=R/'results/b_source_intervention_20261105'
O.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
protocol=dict(pool_seeds=[101,202,303],random_source_seeds=[401,502,603],label_subset_seed=20261105,C=.01,primary='B-A',secondary='B-C',budget=180,C_full_auxiliary_budget=720,encoder='Frozen pretrained CBraMod raw flat2400; source-only StandardScaler',ranking='Historical single-source classifier mean BA on other 15 sources; no target information',test_accessed=False)
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
meta={r['trial_id']:r for r in csv.DictReader((R/'metadata/trials.csv').open())}
ids=read(R/'results/b_calibrated_hybrid_kd_20261102/provenance.json')['train_ids']
s=np.array([meta[i]['subject'] for i in ids]);y=np.array([int(meta[i]['label']) for i in ids])
X=np.load(R/'results/b_teacher_factorial_20261030/student_flat_control/cache/train_wide_car_features.npy').astype(float)
subjects=np.unique(s)
M=pd.read_csv(R/'results/b_transfer_diag_20261105/transfer_matrix.csv',index_col=0).loc[subjects,subjects].to_numpy()
split=read(R/'metadata/splits.json')['physionet']
assert set(subjects)==set(split['quick_train']) and not set(subjects)&set(split['test'])
results=[];manifest=[];predictions=[]
for pool_seed in protocol['pool_seeds']:
    for hold in range(24):
        pool=np.random.default_rng(pool_seed+hold*1000).choice([i for i in range(24) if i!=hold],16,replace=False)
        utility=np.array([np.mean(M[i,[j for j in pool if j!=i]]) for i in pool])
        best=pool[np.argsort(-utility,kind='stable')[:4]]
        # Same 180 label budget, 12 subjects receive 11 trials and 4 receive 12.
        rng=np.random.default_rng(20261105+pool_seed+hold*1000)
        order=rng.permutation(pool)
        balanced=[]
        for k,source in enumerate(order):
            n=12 if k<4 else 11
            ix=np.flatnonzero(s==subjects[source])
            # Randomize the extra class for odd n independently of data values.
            extra=int(rng.integers(2))
            for label in (0,1):
                count=n//2+int(n%2 and label==extra)
                balanced.extend(rng.permutation(ix[y[ix]==label])[:count])
        variants=[('B',None,best,None),('C',None,pool,np.array(balanced)),('C720',None,pool,None)]
        for random_seed in protocol['random_source_seeds']:
            chosen=np.random.default_rng(random_seed+pool_seed*10000+hold*1000000).choice(pool,4,replace=False)
            variants.append(('A',random_seed,chosen,None))
        test=np.flatnonzero(s==subjects[hold])
        for arm,random_seed,selected,ix in variants:
            if ix is None:ix=np.flatnonzero(np.isin(s,subjects[selected]))
            assert subjects[hold] not in set(s[ix])
            assert len(ix)==(720 if arm=='C720' else 180)
            weights=np.array([1/np.sum(s[ix]==subject) for subject in s[ix]])
            weights/=weights.mean()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
            model.fit(X[ix],y[ix],logisticregression__sample_weight=weights)
            pred=model.predict(X[test])
            record=dict(pool_seed=pool_seed,target=subjects[hold],arm=arm,random_seed=random_seed,labels=len(ix),BA=100*balanced_accuracy_score(y[test],pred))
            results.append(record)
            manifest.append(dict(**record,selected_sources=subjects[selected].tolist(),fit_ids=[ids[i] for i in ix],pool=subjects[pool].tolist(),utility=utility.tolist()))
            for i,p in zip(test,pred):predictions.append(dict(**record,trial_id=ids[i],truth=int(y[i]),prediction=int(p)))
        print(pool_seed,subjects[hold],flush=True)
df=pd.DataFrame(results)
df.to_csv(O/'scores.csv',index=False)
pd.DataFrame(predictions).to_csv(O/'predictions.csv',index=False)
(O/'fit_manifest.json').write_text(json.dumps(manifest))
paired=df.groupby(['pool_seed','target','arm']).BA.mean().unstack()
paired.to_csv(O/'paired_scores.csv')
diff=pd.DataFrame({a+'-'+b:paired[a]-paired[b] for a,b in [('B','A'),('B','C'),('B','C720'),('C','C720'),('A','C')]})
diff.to_csv(O/'paired_differences.csv')
summary=dict(means=paired.mean().to_dict(),by_pool=paired.groupby('pool_seed').mean().to_dict(),contrasts={})
for column in diff:
    per_subject=diff[column].groupby('target').mean()
    rng=np.random.default_rng(20261105)
    boot=per_subject.to_numpy()[rng.integers(0,24,(20000,24))].mean(1)
    summary['contrasts'][column]=dict(mean_pp=per_subject.mean(),median_pp=per_subject.median(),improved=int((per_subject>0).sum()),bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),by_pool=diff[column].groupby('pool_seed').mean().to_dict())
for random_seed in protocol['random_source_seeds']:
    aa=df[(df.arm=='A')&(df.random_seed==random_seed)].set_index(['pool_seed','target']).BA
    summary.setdefault('B_minus_A_by_random_seed',{})[random_seed]=float((paired.B-aa).mean())
(O/'summary.json').write_text(json.dumps(summary,indent=2))
(O/'acceptance.json').write_text(json.dumps(dict(status='PASS',fits=len(df),outer_subjects=24,E_accessed=False,GPU_training=False)))
print(json.dumps(summary,indent=2))
