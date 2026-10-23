import json,hashlib
from collections import defaultdict,Counter
import numpy as np
import torch
from scipy.stats import spearmanr
import train_single_region_risk_expanded as e
torch.set_num_threads(4)
names=['tau','geometry','trajectory_half','trajectory_onehalf','flow_zero','feature_transport','rgb_transport','rgb_local']
sets={'full':[0,2,3,4,5,6,7],'no_dynamic_alternatives':[0,4,7],'tau_only':[0],
      'no_trajectory_scales':[0,4,5,6,7],'no_transport':[0,2,3,4,7]}
paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']
records=defaultdict(list);pairs=defaultdict(list)
for split in ['train','val']:
 for path in paths[split]:
  key=hashlib.sha256(path.encode()).hexdigest()[:24]
  d=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False)
  raw=d['raw'].numpy()*1e-4/256;mask=d['mask'].numpy().astype(bool)
  replay=d['features'][:,34:42].reshape(4,2,8,8,8).mean(1).numpy()
  r={'path':path,'group':tuple(e.tr.prev.group(path)),'oracle':{},'candidate':{}}
  for name,ks in sets.items():
   v=np.concatenate([raw[:,k][mask[:,k]] for k in ks]);r['oracle'][name]=max(0.,float(v.max())) if len(v) else 0.
  allvals=raw.copy();allvals[~mask]=-np.inf;allvals[:,1]=-np.inf
  if np.isfinite(allvals).any():
   idx=np.unravel_index(allvals.argmax(),allvals.shape);r['winner']=names[idx[1]] if allvals[idx]>1e-8 else 'abstain_or_negligible'
  else:r['winner']='abstain_or_negligible'
  for k in [0,2,3,4,5,6,7]:
   v=raw[:,k][mask[:,k]]
   if len(v):r['candidate'][names[k]]={'mean':float(v.mean()),'positive':float((v>1e-8).mean()),'oracle':max(0.,float(v.max()))}
   for t in range(4):
    a=replay[t,k][mask[t,k]];b=raw[t,k][mask[t,k]]
    pairs[(split,k,t)].extend(zip(a.tolist(),b.tolist()))
  records[split].append(r)

def video_mean(rs,fn):
 g=defaultdict(list)
 for r in rs:
  v=fn(r)
  if v is not None:g[r['group']].append(v)
 return np.array([np.mean(v) for v in g.values()])

def summary(rs):
 means={k:float(video_mean(rs,lambda r:r['oracle'][k]).mean()) for k in sets}
 delta=video_mean(rs,lambda r:r['oracle']['full']-r['oracle']['no_dynamic_alternatives'])
 rng=np.random.default_rng(42);ci=np.quantile(delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1),[.025,.975]).tolist()
 wins=Counter(r['winner'] for r in rs)
 cand={}
 for name in names:
  if name=='geometry':continue
  cand[name]={key:float(video_mean(rs,lambda r:r['candidate'].get(name,{}).get(key)).mean()) for key in ['mean','positive','oracle']}
  cand[name]['oracle_winner_clip_count']=wins[name]
  cand[name]['oracle_winner_video_mean_fraction']=float(video_mean(rs,lambda r:float(r['winner']==name)).mean())
 return {'clips':len(rs),'videos':len({r['group'] for r in rs}),'oracle_video_means':means,
  'no_dynamic_retained_fraction':means['no_dynamic_alternatives']/means['full'] if means['full'] else None,
  'dynamic_marginal_gain_ci95':ci,'candidate_video_equal':cand,'winner_counts':dict(wins)}

result={'scope':'single two-frame hard replacement in max16 cached cells or abstain; reference remains geometry; removing alternatives is NOT a geometry-free baseline',
 'no_dynamic_definition':'remove trajectory_half, trajectory_onehalf, feature_transport, rgb_transport; retain tau/flow_zero/rgb_local alternatives. flow_zero keeps geometry tokens, rgb_local shares geometry-trained weights and support',
 'test_used':False,'splits':{sp:summary(rs) for sp,rs in records.items()},'val_by_source':{},'replay':{}}
for source in sorted({r['group'][0] for r in records['val']}):result['val_by_source'][source]=summary([r for r in records['val'] if r['group'][0]==source])
for sp in ['train','val']:
 result['replay'][sp]={}
 for k in [0,2,3,4,5,6,7]:
  horizons=[]
  for t in range(4):
   a=np.array(pairs[(sp,k,t)])
   rho=float(spearmanr(a[:,0],a[:,1]).statistic) if a[:,0].std()>1e-12 and a[:,1].std()>1e-12 else None
   horizons.append({'frames':[2*t+1,2*t+2],'count':len(a),'spearman':rho})
  a=np.array(sum([pairs[(sp,k,t)] for t in range(4)],[]))
  result['replay'][sp][names[k]]={'pooled_spearman':float(spearmanr(a[:,0],a[:,1]).statistic),'horizons':horizons}
e.tr.prev.dump(e.OUT/'candidate_structure_audit.json',result)
print(json.dumps(result,indent=2))
