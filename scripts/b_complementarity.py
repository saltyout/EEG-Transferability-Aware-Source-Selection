"""Fixed CPU diagnostic of combination complementarity, never select a winner."""
import json,csv,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
R=Path(__file__).resolve().parents[1];O=R/'results/b_complementarity_20261105';O.mkdir(exist_ok=False)
protocol=dict(combinations_per_pool=12,combination_seed=20261105,pool_seeds=[101,202,303],C=.01,ridge_alpha=1,matched_quality_tolerance=.02,features=['quality','instability','representation_distance','error_disagreement'],primary='Held-target MAE reduction adding disagreement to quality+instability+representation-distance; no group selection',source_labels=720,pooled_fit_labels=180,E_accessed=False,GPU_training=False)
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
allrows=[];predrows=[];source_rows=[]
def model():return make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=3000))
for dataset in ('physionet','cho2017'):
    if dataset=='physionet':
        meta={r['trial_id']:r for r in csv.DictReader((R/'metadata/trials.csv').open())}
        ids=json.loads((R/'results/b_calibrated_hybrid_kd_20261102/provenance.json').read_text())['train_ids']
        fullX=np.load(R/'results/b_teacher_factorial_20261030/student_flat_control/cache/train_wide_car_features.npy')
        fully=np.array([int(meta[i]['label']) for i in ids]);fulls=np.array([meta[i]['subject'] for i in ids])
        bank=np.arange(len(fully))
    else:
        f=np.load(R/'results/b_cho_external_20261105/features.npz')
        fullX,fully,fulls=f['X'],f['y'],f['subjects']
        bank=[]
        for i,subject in enumerate(np.unique(fulls)):
            rng=np.random.default_rng(20261105+i);ix=np.flatnonzero(fulls==subject)
            for label in (0,1):bank.extend(rng.permutation(ix[fully[ix]==label])[:22+int(label==i%2)])
        bank=np.array(bank)
    X,y,s=fullX[bank],fully[bank],fulls[bank];subjects=np.unique(s);N=len(subjects)
    indices=[np.flatnonzero(s==subject) for subject in subjects]
    # All single-source predictions are reusable: each fit sees its source only.
    predictions=np.array([model().fit(X[ix],y[ix]).predict(X) for ix in indices])
    M=np.full((N,N),np.nan)
    for i in range(N):
        for j in range(N):
            if i!=j:M[i,j]=balanced_accuracy_score(y[indices[j]],predictions[i,indices[j]])
    for seed in (101,202,303):
        for hold in range(N):
            pool=np.random.default_rng(seed+hold*1000).choice([i for i in range(N) if i!=hold],16,replace=False)
            pool_ix=np.concatenate([indices[i] for i in pool])
            z=StandardScaler().fit(X[pool_ix])
            centers=np.array([z.transform(X[indices[i]]).mean(0) for i in range(N)])
            rng=np.random.default_rng(20261105+seed+hold*1000)
            groups=set()
            while len(groups)<12:groups.add(tuple(sorted(rng.choice(pool,4,replace=False).tolist())))
            rows=[]
            for g,group in enumerate(sorted(groups)):
                others=[i for i in pool if i not in group]
                q=float(np.mean(M[np.ix_(group,others)]))
                instability=float(np.std(M[np.ix_(group,others)],axis=1).mean())
                disagreement=[];double_fault=[];excess=[]
                for target in others:
                    ix=indices[target];errors=predictions[np.ix_(group,ix)]!=y[ix]
                    for left,right in itertools.combinations(range(4),2):
                        # Equal class weight, then equal target and source-pair weights.
                        for label in (0,1):
                            e1,e2=errors[left,y[ix]==label],errors[right,y[ix]==label]
                            disagreement.append(np.mean(e1!=e2))
                            double_fault.append(np.mean(e1&e2))
                            excess.append(np.mean(e1&e2)-e1.mean()*e2.mean())
                distance=float(np.mean([np.linalg.norm(centers[i]-centers[j]) for i,j in itertools.combinations(group,2)]))
                train=np.concatenate([indices[i] for i in group])
                clf=model().fit(X[train],y[train])
                internal=np.mean([balanced_accuracy_score(y[indices[i]],clf.predict(X[indices[i]])) for i in others])
                test=fulls==subjects[hold]
                score=balanced_accuracy_score(fully[test],clf.predict(fullX[test]))
                record=dict(dataset=dataset,pool_seed=seed,target=subjects[hold],group=g,sources='|'.join(subjects[list(group)]),quality=q,instability=instability,representation_distance=distance,error_disagreement=np.mean(disagreement),double_fault=np.mean(double_fault),excess_double_fault=np.mean(excess),internal_BA=internal,target_BA=score)
                rows.append(record);allrows.append(record)
            frame=pd.DataFrame(rows)
            features=protocol['features']
            for name,cols in [('quality',['quality']),('base',features[:3]),('plus_complementarity',features)]:
                reg=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(frame[cols],frame.internal_BA)
                pred=reg.predict(frame[cols])
                for record,p in zip(rows,pred):
                    predrows.append(dict(dataset=dataset,pool_seed=seed,target=subjects[hold],group=record['group'],model=name,predicted_BA=p,target_BA=record['target_BA'],absolute_error_pp=100*abs(p-record['target_BA'])))
            for source in pool:
                recipients=[i for i in pool if i!=source]
                source_rows.append(dict(dataset=dataset,pool_seed=seed,target=subjects[hold],source=subjects[source],quality=float(np.mean(M[source,recipients])),sd=float(np.std(M[source,recipients])),worst=float(np.min(M[source,recipients])),held_BA=M[source,hold]))
        print(dataset,seed,flush=True)
pd.DataFrame(allrows).to_csv(O/'combinations.csv',index=False)
pd.DataFrame(predrows).to_csv(O/'predictions.csv',index=False)
pd.DataFrame(source_rows).to_csv(O/'source_stability.csv',index=False)
(O/'acceptance.json').write_text(json.dumps(dict(status='PASS',combinations=len(allrows),E_accessed=False)))
