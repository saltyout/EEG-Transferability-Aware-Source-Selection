"""Full-source weighting, locked CPU evaluation."""
import csv,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as ba
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[1];O=R/'results/b_weight_20261106_final';O.mkdir(exist_ok=False)
(O/'protocol.json').write_text(json.dumps(dict(weight='v=tanh((source_BA-pool_mean)/0.10); w=1+0.45*(v-mean(v))',C=.01,permutations=[401,502,603],self_BA='3-fold stratified within source using its same 45 trials, seed 17',E_accessed=False)))
def weights(a):
    v=np.tanh((a-a.mean())/.10)
    return 1+.45*(v-v.mean())
rows=[];audit=[]
for ds in ['physionet','cho2017']:
    if ds=='physionet':
        ids=json.loads((R/'results/b_calibrated_hybrid_kd_20261102/provenance.json').read_text())['train_ids']
        meta={r['trial_id']:r for r in csv.DictReader((R/'metadata/trials.csv').open()) if r['trial_id'] in set(ids)}
        X=np.load(R/'results/b_teacher_factorial_20261030/student_flat_control/cache/train_wide_car_features.npy').astype(float)
        s=np.array([meta[i]['subject'] for i in ids]);y=np.array([int(meta[i]['label']) for i in ids]);bank=np.arange(len(y))
        prior=R/'results/b_source_intervention_20261105';matrix=R/'results/b_transfer_diag_20261105/transfer_matrix.csv'
    else:
        f=np.load(R/'results/b_cho_external_20261105/features.npz');X,y,s=f['X'],f['y'],f['subjects']
        bank=[]
        for i,subject in enumerate(np.unique(s)):
            rng=np.random.default_rng(20261105+i);ix=np.flatnonzero(s==subject)
            for c in (0,1):bank.extend(rng.permutation(ix[y[ix]==c])[:22+int(c==i%2)])
        bank=np.array(bank);prior=R/'results/b_cho_external_20261105/intervention';matrix=prior/'transfer_matrix.csv'
    subs=np.unique(s);M=pd.read_csv(matrix,index_col=0).loc[subs,subs].to_numpy()
    old=pd.read_csv(prior/'scores.csv');own=[]
    for sub in subs:
        ix=bank[s[bank]==sub];pred=np.zeros(len(ix),int)
        for tr,te in StratifiedKFold(3,shuffle=True,random_state=17).split(ix,y[ix]):
            scaler=StandardScaler().fit(X[ix[tr]])
            model=LogisticRegression(C=.01,max_iter=3000).fit(scaler.transform(X[ix[tr]]),y[ix[tr]])
            pred[te]=model.predict(scaler.transform(X[ix[te]]))
        own.append(ba(y[ix],pred))
    for seed in (101,202,303):
        for h,sub in enumerate(subs):
            pool=np.random.default_rng(seed+h*1000).choice(np.delete(np.arange(len(subs)),h),16,replace=False)
            ix=bank[np.isin(s[bank],subs[pool])];test=s==sub
            assert len(ix)==720 and not test[ix].any()
            scaler=StandardScaler().fit(X[ix]);xt=scaler.transform(X[ix]);xv=scaler.transform(X[test])
            transfer=np.nanmean(M[np.ix_(pool,pool)],axis=1);w=weights(transfer)
            opts={'A':np.ones(16),'B':w,'C':weights(np.array(own)[pool])}
            for k in (401,502,603):opts['D'+str(k)]=np.random.default_rng(k+seed*10000+h*1000000).permutation(w)
            for arm,sw in opts.items():
                assert np.isclose(sw.mean(),1) and sw.min()>.1 and sw.max()<1.9
                model=LogisticRegression(C=.01,max_iter=3000).fit(xt,y[ix],sample_weight=np.array([dict(zip(subs[pool],sw))[subj] for subj in s[ix]])*(np.ones(720)/45/(np.ones(720)/45).mean()))
                score=100*ba(y[test],model.predict(xv))
                if arm=='A':
                    ref=old[(old.arm=='C720')&(old.target==sub)&(old.pool_seed==seed)].BA.item()
                    assert abs(score-ref)<1e-9,(ds,seed,sub,score,ref)
                rows.append(dict(dataset=ds,target=sub,pool=seed,arm=arm,BA=score))
                audit.append(dict(dataset=ds,target=sub,pool=seed,arm=arm,sources=subs[pool].tolist(),weights=sw.tolist(),transfer=transfer.tolist(),self_BA=np.array(own)[pool].tolist()))
df=pd.DataFrame(rows);df.to_csv(O/'scores.csv',index=False)
(O/'weights.json').write_text(json.dumps(audit))
out={}
for ds,g in df.groupby('dataset'):
    t=g.pivot(index=['target','pool'],columns='arm',values='BA');t['D']=t[['D401','D502','D603']].mean(axis=1)
    out[ds]={'BA':t.mean().to_dict()}
    for arm in ('A','C','D'):
        delta=t.B-t[arm];v=delta.groupby('target').mean();rng=np.random.default_rng(20261106)
        boot=rng.choice(v,(20000,len(v)),replace=True).mean(1)
        out[ds]['B-'+arm]=dict(mean=v.mean(),positive=int((v>1e-10).sum()),ci=np.quantile(boot,[.025,.975]).tolist(),pools=delta.groupby('pool').mean().tolist())
        t['B-'+arm]=delta
    t.groupby('target').mean().to_csv(O/(ds+'_subjects.csv'))
(O/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

