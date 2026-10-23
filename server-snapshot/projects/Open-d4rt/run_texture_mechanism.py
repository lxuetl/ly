"""Validation-only factorial mechanism experiment. No test evaluation."""
import json,time,os
from collections import defaultdict
import torch
import torch.nn.functional as F
import lpips,numpy as np
import run_history_ablations as a
OUT=a.t.ROOT/'texture_mechanism_val_20261015'
def save(name,obj):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(q,p)
def warp(x,fl,mode):
 yy,xx=torch.meshgrid(torch.arange(128,device=x.device),torch.arange(128,device=x.device),indexing='ij')
 grid=torch.stack([xx+fl[:,0],yy+fl[:,1]],-1)/127*2-1
 return F.grid_sample(x,grid,mode=mode,padding_mode='border',align_corners=True).clamp(0,1)
@torch.no_grad()
def alternatives(ctx,tau,d):
 sources={k:[] for k in ['linear4','cubic4','linear1','cubic1','zero1']}
 for step in range(8):
  ls=[];cs=[];ws=[]
  for k in range(12,16):
   fl,valid=a.h.maps(a.h.flow(tau[step],ctx[k]),a.h.flow(ctx[k],tau[step]))
   lin=a.h.warp(ctx[k:k+1],fl);cub=warp(ctx[k:k+1],fl,'bicubic')
   w=valid*torch.exp(-(a.h.low(lin)-a.h.low(tau[step:step+1])).abs().mean(1,keepdim=True)/.06)
   ls.append(lin);cs.append(cub);ws.append(w)
  ws=torch.stack(ws);den=ws.sum(0).clamp_min(1e-6)
  sources['linear4'].append((torch.stack(ls)*ws).sum(0)/den)
  sources['cubic4'].append((torch.stack(cs)*ws).sum(0)/den)
  sources['linear1'].append(ls[-1]);sources['cubic1'].append(cs[-1]);sources['zero1'].append(ctx[-1:])
 pred={};spec={}
 for k,seq in sources.items():
  src=torch.cat(seq);delta=.25*d['gate']*((src-a.h.low(src))-(tau-a.h.low(tau)))
  pred[k]=a.h.bound(delta,tau).half().float()
  spec[k]={'source_highfreq_energy':float((src-a.h.low(src)).abs().mean())}
 return pred,spec
@torch.no_grad()
def main():
 a.setup();OUT.mkdir(exist_ok=False);start=time.time()
 paths=json.loads((a.OUT/'manifest.json').read_text())['val']
 save('protocol.json',{'split':'validation only','clips':len(paths),'test_access':False,'frozen':'TAU cached outputs; no training; no D4RT/SDXL','grid':'linear/cubic x last1/last4; zero4 and zero1; motion blend thresholds 0.25,0.5,1,2,4','controls':'all variants share ORIGINAL linear-last4 confidence gate; four-frame weights also fixed from linear warps; same .25 alpha,.03 pixel cap,.001 meanabs cap; realized budgets reported','bicubic':'bicubic grid_sample clamped to [0,1], so intervention includes overshoot clipping','selection':'minimum video-equal LPIPS subject to retaining >=80% original MSE improvement vs TAU; temporal <= original*1.01 and <=TAU; original fallback','limits':'last1 vs last4 changes temporal source and averaging jointly, not pure blur isolation; zero-motion retains flow-derived gate; repeated validation reuse exploratory','script_sha256':a.e.sha(__file__)})
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[];parity=0
 for i,path in enumerate(paths):
  tick=time.time();d={k:v.cuda() for k,v in torch.load(a.OUT/'cache'/f'{i:04d}.pt',weights_only=False).items()}
  rgb,_=a.e.rgb24(path);ctx=rgb[:16].cuda().half().float()
  pred,spec=alternatives(ctx,d['tau'],d)
  baseline=a.predict(d,'A',0);err=float((pred['linear4']-baseline).abs().max());parity=max(parity,err);assert err<1e-6,(i,err)
  pred['tau']=d['tau'];pred['zero4']=a.h.bound(d['z'],d['tau']).half().float()
  for threshold in [.25,.5,1.,2.,4.]:pred[f'blend{threshold}']=a.predict(d,'A',threshold)
  metrics={m:a.metrics(p,d['target'],d['tau'],lp) for m,p in pred.items()}
  for m,p in pred.items():
   metrics[m]['highfreq_energy']=float((p-a.h.low(p)).abs().mean())
   metrics[m]['highfreq_mse']=float(((p-a.h.low(p))-(d['target']-a.h.low(d['target']))).square().mean())
  row={'path':path,'group':list(a.t.prev.group(path)),'metrics':metrics,'source_spectral':spec};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%10==0:
   progress={'done':i+1,'total':len(paths),'seconds':time.time()-start,'last_clip_seconds':time.time()-tick,'remaining_seconds':(time.time()-start)/(i+1)*(len(paths)-i-1)};save('progress.json',progress);print(json.dumps(progress),flush=True)
 videos,means=a.summarize(rows);gain=means['tau']['mse']-means['linear4']['mse'];assert gain>0
 limit=means['tau']['mse']-.8*gain
 eligible=[m for m,v in means.items() if v['mse']<=limit and v['temporal']<=min(means['tau']['temporal'],means['linear4']['temporal']*1.01)]
 selected=min(eligible,key=lambda m:means[m]['lpips'])
 sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 pairs=[(m,'linear4') for m in means if m not in ['tau','linear4']]+[('cubic1','linear1'),('cubic4','linear4'),('linear1','linear4'),('cubic1','cubic4')]
 ci={x+'-vs-'+y:{k:a.h.ci([v['metrics'][x][k]-v['metrics'][y][k] for v in videos]) for k in ['mse','lpips','temporal','highfreq_energy']} for x,y in pairs}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'per_source':{s:a.aggregate(v) for s,v in sources.items()},'paired_ci':ci,'baseline_parity_max':parity,'mse_limit':limit,'eligible':eligible,'selected':selected,'retained_mse_gain':{m:(means['tau']['mse']-v['mse'])/gain for m,v in means.items()}})
 save('COMPLETE.json',{'seconds':time.time()-start,'test_evaluated':False});print(json.dumps({'complete':True,'selected':selected,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
