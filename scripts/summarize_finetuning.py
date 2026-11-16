"""Audit saved predictions and summarize complete paired targets only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

def summarize(run,output):
 protocol=json.loads((run/'protocol.json').read_text());expected=protocol['subjects']*24
 rows=[]
 for path in sorted(run.glob('[0-9][0-9]_*.json')):
  r=json.loads(path.read_text())
  if not np.isclose(r['BA'],100*balanced_accuracy_score(r['truth'],r['prediction']),atol=1e-10,rtol=0):raise ValueError('BA mismatch')
  if set(r['fit_indices'])&set(r['test_indices']) or r['target'] in r['sources']:raise ValueError('Target overlap')
  if len(set(r['fit_indices']))!=r['fit_labels']:raise ValueError('Duplicate training trial')
  if r['encoder_changed']!=(r['mode']=='finetuned') or not all(np.isfinite(v) for _,v in r['losses']):raise ValueError('Training audit failed')
  rows.append(r)
 d=pd.DataFrame(rows)
 if d.empty:raise ValueError('No completed fits')
 if d.duplicated(['target','seed','arm','mode']).any():raise ValueError('Duplicate fit')
 for _,g in d.groupby(['target','seed','arm']):
  if len(g)==2 and (g.iloc[0].fit_indices!=g.iloc[1].fit_indices or g.iloc[0].test_indices!=g.iloc[1].test_indices):raise ValueError('Unpaired data')
 complete=d.groupby('target').size();complete=complete[complete==24].index
 v=d[d.target.isin(complete)]
 if v.empty:raise ValueError('No complete paired targets yet')
 z=v.pivot(index=['target','seed'],columns=['mode','arm'],values='BA');diff={}
 for arm in ['selected180','random180','all180','all720']:diff[arm+'_finetune_gain']=z['finetuned',arm]-z['frozen',arm]
 for mode in ['frozen','finetuned']:
  for arm in ['random180','all180','all720']:diff[mode+'_selected_minus_'+arm]=z[mode,'selected180']-z[mode,arm]
 diff['selection_gain_interaction']=diff['finetuned_selected_minus_random180']-diff['frozen_selected_minus_random180']
 stats={}
 for key,delta in diff.items():
  a=delta.groupby('target').mean();rng=np.random.default_rng(20261116);boot=rng.choice(a.to_numpy(),(20000,len(a)),replace=True).mean(1)
  stats[key]=dict(mean_pp=float(a.mean()),improved=int((a>0).sum()),by_seed=delta.groupby('seed').mean().to_dict())
  if len(d)==expected:stats[key]['descriptive95_pp']=np.quantile(boot,[.025,.975]).tolist()
 summary=dict(status='COMPLETE' if len(d)==expected else 'INTERIM',completed=len(d),expected=expected,complete_targets=len(complete),by_mode=d.groupby('mode').size().to_dict(),means=v.groupby(['arm','mode']).BA.mean().unstack().to_dict(),contrasts=stats,audit='PASS',scope='Complete paired targets only; interim subset is not a final population estimate')
 output.mkdir(parents=True,exist_ok=False)
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
 (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 v[['target','seed','arm','mode','BA','ranking_labels','fit_labels','total_unique_labels']].to_csv(output/'scores.csv',index=False)
 pd.DataFrame(diff).groupby('target').mean().to_csv(output/'subject_differences.csv')
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 print(json.dumps(summarize(a.run,a.output),indent=2))
