"""Three preregistered texture configurations, test only, no tuning."""
import json,time,hashlib
from collections import defaultdict
import torch,lpips,numpy as np
import run_texture_mechanism as m
a=m.a
OUT=a.t.ROOT/'texture_three_locked_test_20261015'
MODES=['linear4','linear1','cubic1']
PAIRS=[('linear1','linear4'),('cubic1','linear4'),('cubic1','linear1')]
def save(n,x):
 p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def intervals(videos):
 if len(videos)<2:return {'available':False,'reason':'Only one original video; between-video uncertainty not estimable.'}
 result={}
 for x,y in PAIRS:
  result[x+'-vs-'+y]={}
  for k in videos[0]['metrics'][x]:
   diff=np.asarray([v['metrics'][x][k] for v in videos])-np.asarray([v['metrics'][y][k] for v in videos])
   result[x+'-vs-'+y][k]=a.h.ci(diff) if diff.ndim==1 else [a.h.ci(diff[:,i]) for i in range(diff.shape[1])]
 return result
@torch.no_grad()
def main():
 a.setup();OUT.mkdir(exist_ok=False);start=time.time()
 splits=json.loads((a.OUT/'manifest.json').read_text());paths=splits['test'];assert len(set(paths))==len(paths)
 groups={k:{tuple(a.t.prev.group(p)) for p in v} for k,v in splits.items()}
 assert not groups['test']&(groups['train']|groups['val'])
 save('LOCKED_PROTOCOL.json',{'modes':MODES,'split':'test','clips':len(paths),'videos':len(groups['test']),'tuning':False,'configs':'original linear4; last1 linear; last1 cubic clamped [0,1]. All retain original four-frame confidence gate, alpha .25, pixelcap .03, meanabs cap .001','checkpoint_sha256':a.e.sha(a.t.ROOT/'tau128_full_fair_fast_20261009/best.pt'),'script_sha256':a.e.sha(__file__),'mechanism_script_sha256':a.e.sha(m.__file__),'manifest_sha256':hashlib.sha256(json.dumps(paths).encode()).hexdigest(),'bootstrap':'10000 paired resamples of original-video means; seed42; pointwise 95% percentile intervals, no multiple-comparison correction; no per-source CI when only1video','prior_exposure':'Existing test previously evaluated with other configurations. Not a never-observed dataset. No parameter selection in this run.','horizons':'1..8 for frame metrics; temporal indices correspond to transitions 1->2 through7->8'})
 save('manifest.json',{'paths':paths});net=a.network();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[];parity=0
 for i,path in enumerate(paths):
  tick=time.time();rgb,_=a.e.rgb24(path);ctx=rgb[:16].cuda()
  _,hid,skip=a.t.core.extract_frozen_features(net,ctx[None]);tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float();target=rgb[16:24].cuda().half().float()
  ctx=ctx.half().float();d=a.features(ctx,tau);pred,_=m.alternatives(ctx,tau,d)
  err=float((pred['linear4']-a.predict(d,'A',0)).abs().max());parity=max(parity,err);assert err<1e-6
  metrics={}
  for mode in MODES:
   p=pred[mode];metrics[mode]=a.metrics(p,target,tau,lp)
   metrics[mode]['horizon_mae']=(p-target).abs().mean((1,2,3)).cpu().tolist()
   metrics[mode]['horizon_psnr']=(-10*(p-target).square().mean((1,2,3)).clamp_min(1e-12).log10()).cpu().tolist()
   metrics[mode]['horizon_temporal']=((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean((1,2,3)).cpu().tolist()
  row={'path':path,'group':list(a.t.prev.group(path)),'metrics':metrics};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%10==0:
   progress={'done':i+1,'total':len(paths),'seconds':time.time()-start,'last_clip_seconds':time.time()-tick,'remaining_seconds':(time.time()-start)/(i+1)*(len(paths)-i-1)};save('progress.json',progress);print(json.dumps(progress),flush=True)
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 save('per_video.json',videos)
 per_source={s:{'videos':len(v),'clips':sum(x['clips'] for x in v),'metrics':a.aggregate(v),'paired_ci':intervals(v)} for s,v in sources.items()}
 save('per_source.json',per_source);save('paired_ci.json',intervals(videos))
 save('per_horizon.json',{mode:{k:val for k,val in stats.items() if k.startswith('horizon_')} for mode,stats in means.items()})
 save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'clip_equal':a.aggregate(rows),'baseline_parity_max':parity,'paired_ci':intervals(videos),'per_source':per_source})
 save('COMPLETE.json',{'seconds':time.time()-start,'configs_changed':False});print(json.dumps({'complete':True,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
