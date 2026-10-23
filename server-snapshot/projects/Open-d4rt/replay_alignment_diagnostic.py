"""Causal 8->8 replay within validation contexts. Diagnostic only."""
import json,time
from collections import defaultdict
import numpy as np,torch,lpips
from scipy.stats import rankdata
import run_texture_mechanism as m
a=m.a
OUT=a.t.ROOT/'history_replay_alignment_20261015'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def pool(x):return torch.nn.functional.avg_pool2d(x,32,32).flatten().cpu().numpy()
def auc(y,x):
 n=int(y.sum());z=len(y)-n
 return None if n==0 or z==0 else float((rankdata(x)[y].sum()-n*(n+1)/2)/(n*z))
@torch.no_grad()
def main():
 a.setup();OUT.mkdir(exist_ok=False);start=time.time();paths=json.loads((a.OUT/'manifest.json').read_text())['val'];net=a.network();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[];regions=[]
 save('protocol.json',{'split':'validation only','clips':len(paths),'replay':'observed frames0:8; withheld8:16. TAU context: repeat frame0 eight times then observed8. Neither target nor later frames enters inference. Padding distribution shift: diagnostic only, not standard16->8 benchmark.','propagation':['TAU-guided last-frame backward flow','Eulerian constant velocity: backward flow last->penultimate scaled by horizon, approximate fixed-grid extrapolation'],'shared':'bicubic clamp[0,1]; raw RGB propagation, no repair training','harm_label':'regional source MSE minus TAU MSE >1e-6; nonharm <=1e-6','regions':'nonoverlapping32x32,8horizons; reject risk higher is worse','signals':['cycle_or_bounds_risk','appearance_mismatch','motion_magnitude','branch_disagreement'],'threshold':'leave-one-original-video-out: 90th percentile risk among nonharm regions in OTHER videos; heldout false rejection and harmful recall reported, no test data','limits':'pooled region AUROC descriptive, correlated observations; threshold calibration not confidence guarantee; no future visibility GT','script_sha256':a.e.sha(__file__)})
 for i,path in enumerate(paths):
  rgb,_=a.e.rgb24(path);obs=rgb[:8].cuda();ctx=torch.cat([obs[:1].expand(8,-1,-1,-1),obs]);_,hid,skip=a.t.core.extract_frozen_features(net,ctx[None]);tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float();obs=obs.half().float();last=obs[-1:];gt=rgb[8:16].cuda().half().float()
  base,basevalid=a.h.maps(a.h.flow(obs[-1],obs[-2]),a.h.flow(obs[-2],obs[-1]));preds={'tau':tau,'zero':last.expand(8,-1,-1,-1)};seqs={'align':[],'velocity':[]};group=list(a.t.prev.group(path))
  for t in range(8):
   fl,valid=a.h.maps(a.h.flow(tau[t],last[0]),a.h.flow(last[0],tau[t]));vf=base*(t+1)
   yy,xx=torch.meshgrid(torch.arange(128,device='cuda'),torch.arange(128,device='cuda'),indexing='ij');inside=((xx+vf[:,0]>=0)&(xx+vf[:,0]<=127)&(yy+vf[:,1]>=0)&(yy+vf[:,1]<=127))[:,None]
   srcs={'align':m.warp(last,fl,'bicubic'),'velocity':m.warp(last,vf,'bicubic')};dis=(srcs['align']-srcs['velocity']).abs().mean(1,keepdim=True);baseerr=(tau[t:t+1]-gt[t:t+1]).square().mean(1,keepdim=True)
   for b,s in srcs.items():
    seqs[b].append(s);flow=fl if b=='align' else vf;v=valid if b=='align' else basevalid*inside
    sig={'cycle_or_bounds_risk':pool(1-v),'appearance_mismatch':pool((a.h.low(s)-a.h.low(tau[t:t+1])).abs().mean(1,keepdim=True)),'motion_magnitude':pool(flow.square().sum(1,keepdim=True).sqrt()),'branch_disagreement':pool(dis)}
    gain=pool((s-gt[t:t+1]).square().mean(1,keepdim=True)-baseerr)
    for j in range(16):regions.append({'group':group,'clip':i,'horizon':t+1,'region':j,'branch':b,'delta_mse':float(gain[j]),**{k:float(vv[j]) for k,vv in sig.items()}})
  preds.update({k:torch.cat(v) for k,v in seqs.items()});row={'path':path,'group':group,'metrics':{k:a.metrics(p,gt,tau,lp) for k,p in preds.items()}};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%20==0:
   elapsed=time.time()-start;save('progress.json',{'done':i+1,'total':len(paths),'seconds':elapsed,'remaining_seconds':elapsed/(i+1)*(len(paths)-i-1)});print(json.dumps({'done':i+1,'seconds':elapsed}),flush=True)
 videos,means=a.summarize(rows);reports={}
 for b in seqs:
  rr=[r for r in regions if r['branch']==b];y=np.array([r['delta_mse']>1e-6 for r in rr]);g=np.array(['/'.join(r['group']) for r in rr]);reports[b]={'harmful_fraction':float(y.mean()),'signals':{}}
  for key in ['cycle_or_bounds_risk','appearance_mismatch','motion_magnitude','branch_disagreement']:
   x=np.array([r[key] for r in rr]);reject=np.zeros(len(rr),dtype=bool);fold=[]
   for group in sorted(set(g)):
    test=g==group;cal=(~test)&(~y)
    threshold=float(np.quantile(x[cal],.9));reject[test]=x[test]>threshold;fold.append({'group':group,'threshold':threshold,'harmful_recall':float(reject[test&y].mean()) if (test&y).any() else None,'nonharm_rejection':float(reject[test&~y].mean()) if (test&~y).any() else None})
   reports[b]['signals'][key]={'pooled_auroc':auc(y,x),'heldout_harmful_recall':float(reject[y].mean()),'heldout_nonharm_rejection':float(reject[~y].mean()),'folds':fold}
 save('risk_signals.json',reports);save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'paired_ci':{b:{k:a.h.ci([v['metrics'][b][k]-v['metrics']['tau'][k] for v in videos]) for k in ['mse','lpips','temporal']} for b in ['align','velocity','zero']}})
 with (OUT/'regions.jsonl').open('w') as f:
  for r in regions:f.write(json.dumps(r)+'\n')
 save('COMPLETE.json',{'seconds':time.time()-start,'test_used':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
