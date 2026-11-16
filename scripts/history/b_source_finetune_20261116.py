import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,csv,hashlib,time,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import balanced_accuracy_score
R=Path(os.environ['EEG_RESEARCH_ROOT']);O=R/'results/b_source_finetune_20261116';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'external/CBraMod'))
from models.cbramod import CBraMod
worker=int(sys.argv[1]);workers=4
torch.set_num_threads(2)
ids=json.loads((R/'results/b_calibrated_hybrid_kd_20261102/provenance.json').read_text())['train_ids']
lookup={r['trial_id']:r for r in csv.DictReader((R/'metadata/trials.csv').open()) if r['trial_id'] in set(ids)}
y=np.array([int(lookup[i]['label']) for i in ids]);s=np.array([lookup[i]['subject'] for i in ids]);names=np.unique(s)
split=json.loads((R/'metadata/splits.json').read_text())['physionet'];assert set(names)==set(split['quick_train']) and not set(names)&set(split['test'])
x=np.load(R/'results/b_teacher_factorial_20261030/cache/train_wide_car_low.npy');f=np.load(R/'results/b_teacher_factorial_20261030/student_flat_control/cache/train_wide_car_features.npy')
assert x.shape==(1080,3,4,200) and f.shape==(1080,2400) and len(names)==24
M=pd.read_csv(R/'results/b_transfer_diag_20261105/transfer_matrix.csv',index_col=0).loc[names,names].to_numpy()
raw=torch.tensor(x,device='cuda');labels=torch.tensor(y,device='cuda');checkpoint=torch.load(R/'weights/pretrained_weights.pth',map_location='cpu',weights_only=True)
protocol=dict(dataset='physionet',subjects=24,pool_seed=101,model_seeds=[17,29,43],random_source_seed=401,steps=200,batch_size=32,encoder_lr=1e-5,head_lr=1e-3,optimizer='AdamW',weight_decay=.01,clip=1,precision='float32',trainable='all encoder parameters except replaced proj_out',head='Linear2400->2; fixed source-only frozen-feature standardization',selection='historical frozen source-pool matrix; no reranking',evaluation='final step only, no early stopping',arms=['selected180','random180','all180','all720'],paired_controls=['frozen','finetuned'],scope='exploratory single-pool development experiment; not final confirmation',E_accessed=False)
if worker==0:(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
jobs=[]
for hold in range(24):
 pool=np.random.default_rng(101+hold*1000).choice(np.delete(np.arange(24),hold),16,replace=False)
 q=np.nanmean(M[np.ix_(pool,pool)],axis=1);best=pool[np.argsort(-q,kind='stable')[:4]]
 chosen=np.random.default_rng(401+101*10000+hold*1000000).choice(pool,4,replace=False)
 rng=np.random.default_rng(20261105+101+hold*1000);uniform=[]
 for k,source in enumerate(rng.permutation(pool)):
  count=12 if k<4 else 11;extra=int(rng.integers(2));ix=np.flatnonzero(s==names[source])
  for label in (0,1):uniform.extend(rng.permutation(ix[y[ix]==label])[:count//2+int(count%2 and label==extra)])
 arms={'selected180':np.flatnonzero(np.isin(s,names[best])),'random180':np.flatnonzero(np.isin(s,names[chosen])),'all180':np.array(uniform),'all720':np.flatnonzero(np.isin(s,names[pool]))}
 for seed in (17,29,43):
  for arm,tr in arms.items():
   for mode in ('frozen','finetuned'):jobs.append((hold,seed,arm,tr,mode,pool))
for ji,job in enumerate(jobs):
 if ji%workers!=worker:continue
 hold,seed,arm,tr,mode,pool=job;name=f'{hold:02d}_{seed}_{arm}_{mode}';out=O/(name+'.json')
 if out.exists():continue
 te=np.flatnonzero(s==names[hold]);assert not set(s[tr])&set(s[te]);assert len(tr)==(720 if arm=='all720' else 180)
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
 net=CBraMod();net.load_state_dict(checkpoint,strict=True);net.proj_out=torch.nn.Identity();net.to('cuda');net.requires_grad_(mode=='finetuned')
 head=torch.nn.Linear(2400,2).cuda()
 mean=torch.tensor(f[tr].mean(0),device='cuda');sd=torch.tensor(f[tr].std(0).clip(1e-6),device='cuda')
 optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':1e-3}]+([{'params':net.parameters(),'lr':1e-5}] if mode=='finetuned' else []),weight_decay=.01)
 w=np.array([len(tr)/len(np.unique(s[tr]))/np.sum(s[tr]==v) for v in s[tr]]);rng=np.random.default_rng(seed);losses=[];start=time.time()
 initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
 for step in range(200):
  net.train(mode=='finetuned');head.train();batch=rng.integers(0,len(tr),32);ix=tr[batch]
  optimizer.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(mode=='finetuned'):z=net(raw[ix]).flatten(1)
  logits=head((z-mean)/sd);loss=(torch.nn.functional.cross_entropy(logits,labels[ix],reduction='none')*torch.tensor(w[batch],device='cuda')).mean()
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1,error_if_nonfinite=True);optimizer.step()
  if step%20==0:losses.append([step,float(loss.detach())])
 net.eval();head.eval();pred=[]
 with torch.no_grad():
  for k in range(0,len(te),32):pred.extend(head((net(raw[te[k:k+32]]).flatten(1)-mean)/sd).argmax(1).cpu().tolist())
 changed=initial!=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest();assert changed==(mode=='finetuned')
 record=dict(target=names[hold],seed=seed,arm=arm,mode=mode,BA=100*balanced_accuracy_score(y[te],pred),fit_indices=tr.tolist(),test_indices=te.tolist(),prediction=pred,truth=y[te].tolist(),sources=names[pool].tolist(),losses=losses,encoder_changed=changed,seconds=time.time()-start,ranking_labels=720 if arm=='selected180' else 0,fit_labels=len(tr),total_unique_labels=720 if arm=='selected180' else len(tr))
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record));tmp.replace(out)
 print(name,record['BA'],record['seconds'],flush=True)
 del net,head,optimizer;torch.cuda.empty_cache()
(O/f'worker{worker}_complete.json').write_text(json.dumps(dict(status='complete')))
