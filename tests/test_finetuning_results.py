import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

@pytest.mark.parametrize('dataset',['physionet','cho2017','cho2017_interim'])
def test_published_finetuning_summary(dataset):
 p=Path(__file__).resolve().parents[1]/'results'/('finetuning_'+dataset)
 s=json.loads((p/'summary.json').read_text());d=pd.read_csv(p/'scores.csv')
 assert not d.duplicated(['target','seed','arm','mode']).any()
 assert d.groupby('target').size().eq(24).all()
 for mode,arms in s['means'].items():
  for arm,expected in arms.items():
   actual=d[(d['mode']==mode)&(d.arm==arm)].BA.mean()
   assert np.isclose(actual,expected,atol=1e-10,rtol=0)
 assert (d[d.arm=='selected180'].total_unique_labels==720).all()
 assert (d[d.arm=='selected180'].fit_labels==180).all()
 if 'interim' in dataset:
  assert s['status']=='INTERIM' and s['completed']<s['expected']
  assert all('descriptive95_pp' not in v for v in s['contrasts'].values())
 else:assert s['status']=='COMPLETE' and s['completed']==(576 if dataset=='physionet' else 1248)

 assert d.target.nunique()==s['complete_targets']
 z=d.pivot(index=['target','seed'],columns=['mode','arm'],values='BA')
 for arm in ['selected180','random180','all180','all720']:
  delta=(z['finetuned',arm]-z['frozen',arm]).groupby('target').mean()
  recorded=s['contrasts'][arm+'_finetune_gain']
  assert np.isclose(delta.mean(),recorded['mean_pp'],atol=1e-10,rtol=0)
  assert int((delta>0).sum())==recorded['improved']
