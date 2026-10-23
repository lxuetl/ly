import json,hashlib,random
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr
import train_single_region_risk_expanded as e
torch.set_num_threads(4)
proto=json.loads((e.OUT/'protocol.json').read_text());parts={k:{tuple(g) for g in v} for k,v in proto['validation_groups'].items()}
paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']
m=e.tr.TemporalSelector().cuda().eval();ck=torch.load(e.OUT/'last.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
allrows=defaultdict(list);candidates=defaultdict(list);ranks=defaultdict(list)
with torch.no_grad():
 for split in ['train','val']:
  for path in paths[split]:
   key=hashlib.sha256(path.encode()).hexdigest();d=torch.load(e.OUT/'labels'/(key[:24]+'.pt'),map_location='cpu',weights_only=False)
   z=e.s.scores(m,d['features'][None].cuda())[0].cpu().numpy();raw=d['raw'].numpy()
   group=tuple(e.tr.prev.group(path));part='train' if split=='train' else next(k for k,v in parts.items() if group in v)
   # Shuffle the 16 already labeled cells without looking at predictions or labels.
   cells=torch.nonzero(d['mask'][:,0]).tolist();random.Random(int(key[:16],16)+31).shuffle(cells)
   for n in [4,8,16]:
    pairs=[(float(z[t,k,y,x]),float(raw[t,k,y,x])) for t,y,x in cells[:n] for k in range(8) if k!=1]
    if not pairs:continue
    arr=np.array(pairs);best=int(arr[:,0].argmax());selected=arr[best];actual=selected[1]*1e-4/256
    allrows[(part,n)].append({'path':path,'group':group,'pred':selected[0],'gain':actual,'oracle':max(0,arr[:,1].max())*1e-4/256,
       'random':float(arr[:,1].mean())*1e-4/256,'active':bool(selected[0]>.1)})
    if n==16:
     candidates[part].extend(pairs)
     if arr[:,0].std()>1e-10 and arr[:,1].std()>1e-10:ranks[part].append(float(spearmanr(arr[:,0],arr[:,1]).statistic))

def summarize(rows):
 gains=np.array([r['gain'] if r['active'] else 0 for r in rows]);groups=defaultdict(list)
 for r,g in zip(rows,gains):groups[r['group']].append(g)
 a=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(42)
 ci=np.quantile(a[rng.integers(0,len(a),(5000,len(a)))].mean(1),[.025,.975]).tolist()
 return {'clips':len(rows),'videos':len(groups),'predicted_top_mean':float(np.mean([r['pred'] for r in rows])),
 'raw_top_actual_gain':float(np.mean([r['gain'] for r in rows])), 'accepted_gain_clip':float(gains.mean()),'accepted_gain_video':float(a.mean()),'video_ci95':ci,
 'coverage':float(np.mean([r['active'] for r in rows])), 'harm_fraction_accepted':float(np.mean([r['gain'] < -1e-8 for r in rows if r['active']])) if any(r['active'] for r in rows) else None,
 'oracle_gain':float(np.mean([r['oracle'] for r in rows])),'uniform_candidate_gain':float(np.mean([r['random'] for r in rows]))}
result={'checkpoint_epoch':ck['epoch'],'threshold_fixed':.1,'test_used':False,'regions':'nested, seeded shuffle of cached observed-support cells; no GT selection','ranking':{},'search':{}}
for part,values in candidates.items():
 a=np.array(values);order=np.argsort(a[:,0]);bins=[]
 for ids in np.array_split(order,5):
  bins.append({'count':len(ids),'prediction':float(a[ids,0].mean()),'actual':float(a[ids,1].mean()),'harm_fraction':float(np.mean(a[ids,1]<-.0256))})
 result['ranking'][part]={'candidate_count':len(a),'pooled_spearman':float(spearmanr(a[:,0],a[:,1]).statistic),'mean_within_clip_spearman':float(np.mean(ranks[part])),'quintiles':bins}
for (part,n),rs in allrows.items():result['search'][part+'_'+str(n)]=summarize(rs)
e.tr.prev.dump(e.OUT/'ranking_audit.json',result)
print(json.dumps(result,indent=2))
