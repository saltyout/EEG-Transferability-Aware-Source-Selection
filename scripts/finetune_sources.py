import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,csv,hashlib,time,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import balanced_accuracy_score
import argparse
from reproduce import validate_features, transfer_matrix
p=argparse.ArgumentParser(description="Fixed-budget paired source-selection fine-tuning")
p.add_argument('--dataset',choices=['physionet','cho2017'],required=True)
p.add_argument('--epochs',type=Path,required=True)
p.add_argument('--features',type=Path,required=True)
p.add_argument('--matrix',type=Path,required=True,help='Historical source-only transfer_matrix.csv')
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--cbramod',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--worker',type=int,default=0)
p.add_argument('--workers',type=int,default=1)
args=p.parse_args();worker=args.worker;workers=args.workers
if not 0<=worker<workers: p.error('Invalid worker partition')
O=args.output;O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.cbramod))
from models.cbramod import CBraMod
import subprocess
revision=subprocess.check_output(['git','-C',str(args.cbramod),'rev-parse','HEAD'],text=True).strip()
if revision!='b9e961003214326972c567eff390e75b0287e32a':raise ValueError('Wrong upstream revision')
if hashlib.sha256(args.weights.read_bytes()).hexdigest()!='0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178':raise ValueError('Wrong checkpoint')
torch.set_num_threads(2)
with np.load(args.features,allow_pickle=False) as z:
 f,y,s,bank=z['X'],z['y'],z['subjects'],z['bank_indices']
validate_features(f,y,s,bank)
with np.load(args.epochs,allow_pickle=False) as z:
 x=z['x']
 if not np.array_equal(z['y'],y) or not np.array_equal(z['subjects'],s):raise ValueError('Epoch/feature row mismatch')
names=np.unique(s);n=24 if args.dataset=='physionet' else 52;t=4 if args.dataset=='physionet' else 3
if len(names)!=n or x.shape!=(len(y),3,t,200) or f.shape!=(len(y),3*t*200) or not np.isfinite(x).all():raise ValueError('Unexpected input shape or values')
M=pd.read_csv(args.matrix,index_col=0).loc[names,names].to_numpy()
# Verify supplied ranking matrix using the supplied features, not target outcomes.
computed=transfer_matrix(f,y,[bank[s[bank]==sub] for sub in names])
if not np.allclose(M,computed,atol=1e-12,equal_nan=True):raise ValueError('Transfer matrix does not match supplied feature bank')
raw=torch.tensor(x,device='cuda');labels=torch.tensor(y,device='cuda')
checkpoint=torch.load(args.weights,map_location='cpu',weights_only=True)
protocol=dict(dataset=args.dataset,subjects=n,pool_seed=101,model_seeds=[17,29,43],random_source_seed=401,steps=200,batch_size=32,encoder_lr=1e-5,head_lr=1e-3,optimizer='AdamW',weight_decay=.01,clip=1,precision='float32',trainable='all encoder parameters except replaced proj_out',head=f'Linear{f.shape[1]}->2; fixed source-only frozen-feature standardization',selection='historical frozen source-pool matrix; no reranking',evaluation='final step only, no early stopping',arms=['selected180','random180','all180','all720'],paired_controls=['frozen','finetuned'],scope='exploratory single-pool development experiment; not final confirmation',E_accessed=False)
protocol['input_sha256']={k:hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in ['epochs','features','matrix','weights']}
protocol['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
protocol_file=O/'protocol.json'
try:
 with protocol_file.open('x') as dst:json.dump(protocol,dst,indent=2)
except FileExistsError:
 for attempt in range(50):
  try:existing=json.loads(protocol_file.read_text());break
  except json.JSONDecodeError:time.sleep(.1)
 if existing!=protocol:raise ValueError('Output directory belongs to a different run')
jobs=[]
for hold in range(n):
 pool=np.random.default_rng(101+hold*1000).choice(np.delete(np.arange(n),hold),16,replace=False)
 q=np.nanmean(M[np.ix_(pool,pool)],axis=1);best=pool[np.argsort(-q,kind='stable')[:4]]
 chosen=np.random.default_rng(401+101*10000+hold*1000000).choice(pool,4,replace=False)
 rng=np.random.default_rng(20261105+101+hold*1000);uniform=[]
 for k,source in enumerate(rng.permutation(pool)):
  count=12 if k<4 else 11;extra=int(rng.integers(2));ix=bank[s[bank]==names[source]]
  for label in (0,1):uniform.extend(rng.permutation(ix[y[ix]==label])[:count//2+int(count%2 and label==extra)])
 arms={'selected180':bank[np.isin(s[bank],names[best])],'random180':bank[np.isin(s[bank],names[chosen])],'all180':np.array(uniform),'all720':bank[np.isin(s[bank],names[pool])]}
 for seed in (17,29,43):
  for arm,tr in arms.items():
   for mode in ('frozen','finetuned'):jobs.append((hold,seed,arm,tr,mode,pool))
for ji,job in enumerate(jobs):
 if ji%workers!=worker:continue
 hold,seed,arm,tr,mode,pool=job;name=f'{hold:02d}_{seed}_{arm}_{mode}';out=O/(name+'.json')
 if out.exists():continue
 te=np.flatnonzero(s==names[hold]);assert not set(s[tr])&set(s[te]);assert len(tr)==(720 if arm=='all720' else 180);assert set(tr)<=set(bank)
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
 net=CBraMod();net.load_state_dict(checkpoint,strict=True);net.proj_out=torch.nn.Identity();net.to('cuda');net.requires_grad_(mode=='finetuned')
 head=torch.nn.Linear(f.shape[1],2).cuda()
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
