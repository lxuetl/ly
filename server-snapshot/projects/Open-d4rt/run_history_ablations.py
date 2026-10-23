"""Validation-only independent texture ablations; lock all choices before test."""
import os,json,time,sys,subprocess,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch,cv2,lpips
import evaluate_fixed_history_expanded as e
import history_texture_candidates as h
import train_propagation_fusion as t
from simvp_geometry_model import build_video_predictor_from_checkpoint
OUT=t.ROOT/'history_ablation_val_locked_20261015'
GRID={'A':[0,.25,.5,1.,2.], 'B':[1.,.75,.5,.25,0.], 'C':[1.,.75,.5,.25,0.]}
def save(name,x):
 p=OUT/name;p.with_suffix('.tmp').write_text(json.dumps(x,indent=2));os.replace(p.with_suffix('.tmp'),p)
def load(name):return json.loads((OUT/name).read_text())
def log(**kw):print(json.dumps(kw),flush=True)
@torch.no_grad()
def features(ctx,tau):
 aligned=[];zero=[];gates=[];motion=[]
 for step in range(8):
  cors=[];zs=[];weights=[];magnitudes=[]
  for k in range(12,16):
   back=h.flow(tau[step],ctx[k]);fw=h.flow(ctx[k],tau[step]);fl,valid=h.maps(back,fw)
   cor=h.warp(ctx[k:k+1],fl);appearance=(h.low(cor)-h.low(tau[step:step+1])).abs().mean(1,keepdim=True)
   weights.append(valid*torch.exp(-appearance/.06));cors.append(cor);zs.append(h.warp(ctx[k:k+1],fl*0));magnitudes.append(fl.square().sum(1,keepdim=True).sqrt())
  ws=torch.stack(weights);den=ws.sum(0).clamp_min(1e-6)
  src=(torch.stack(cors)*ws).sum(0)/den;z=(torch.stack(zs)*ws).sum(0)/den
  disagree=((torch.stack(cors)-src).abs().mean(2,keepdim=True)*ws).sum(0)/den
  aligned.append(src);zero.append(z);gates.append(ws.mean(0)*torch.exp(-disagree/.04));motion.append((torch.stack(magnitudes)*ws).sum(0)/den)
 gate=torch.cat(gates);a=torch.cat(aligned);z=torch.cat(zero)
 return {'tau':tau,'a':.25*gate*((a-h.low(a))-(tau-h.low(tau))), 'z':.25*gate*((z-h.low(z))-(tau-h.low(tau))), 'motion':torch.cat(motion),'gate':gate}
def predict(d,arm,v):
 if arm=='tau':return d['tau']
 if arm=='A':
  w=torch.ones_like(d['motion']) if v==0 else (d['motion']/v).clamp(0,1)
  delta=w*d['a']+(1-w)*d['z']
 else:
  # Filter the ungated texture difference, then apply the ORIGINAL gate once.
  raw=d['a']/d['gate'].clamp_min(1e-8)
  coarse=h.low(raw);fine=raw-coarse
  beta=v if arm=='B' else torch.linspace(1,v,8,device=raw.device).view(8,1,1,1)
  delta=d['gate']*(coarse+beta*fine)
  if v==1:delta=d['a'] # exact fixed baseline, avoiding roundoff
 return h.bound(delta,d['tau']).half().float()
def metrics(p,y,tau,lp):
 err=p-y;mse=err.square().mean((1,2,3));per=lp(p*2-1,y*2-1).flatten();tmp=((p[1:]-p[:-1])-(y[1:]-y[:-1])).abs().mean((1,2,3))
 return {'mse':float(mse.mean()),'mae':float(err.abs().mean()),'psnr':float((-10*mse.clamp_min(1e-12).log10()).mean()),'lpips':float(per.mean()),'temporal':float(tmp.mean()),'budget':float((p-tau).abs().mean()),'horizon_mse':mse.cpu().tolist(),'horizon_lpips':per.cpu().tolist()}
def aggregate(rows):
 return {m:{k:np.mean([r['metrics'][m][k] for r in rows],axis=0).tolist() for k in rows[0]['metrics'][m]} for m in rows[0]['metrics']}
def summarize(rows):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r['group'])].append(r)
 videos=[{'group':list(g),'clips':len(rr),'metrics':aggregate(rr)} for g,rr in sorted(groups.items())]
 return videos,aggregate(videos)
def setup():
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
def network():
 ck=torch.load(t.ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
 net=build_video_predictor_from_checkpoint(ck).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model']);return net
@torch.no_grad()
def make(path,net):
 rgb,_=e.rgb24(path);ctx=rgb[:16].cuda();_,hid,skip=t.core.extract_frozen_features(net,ctx[None]);tau=t.core.decode_frozen(net,hid,skip)[0].half().float()
 d=features(ctx.half().float(),tau);d['target']=rgb[16:24].cuda().half().float();return d
@torch.no_grad()
def validate(arm):
 setup();start=time.time();paths=load('manifest.json')['val'];lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for i,path in enumerate(paths):
  d={k:v.cuda() for k,v in torch.load(OUT/'cache'/f'{i:04d}.pt',weights_only=False).items()}
  ms={'tau':metrics(d['tau'],d['target'],d['tau'],lp)}
  for v in GRID[arm]:ms[str(v)]=metrics(predict(d,arm,v),d['target'],d['tau'],lp)
  row={'path':path,'group':list(t.prev.group(path)),'metrics':ms};rows.append(row)
  with (OUT/f'val_{arm}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%20==0:log(phase='validation',arm=arm,done=i+1,total=len(paths),seconds=time.time()-start)
 videos,means=summarize(rows);base=str(GRID[arm][0]);eligible=[]
 for v in GRID[arm]:
  m=means[str(v)]
  if v==GRID[arm][0] or all(m[k]<=means[ref][k]*1.01 for k in ['mse','temporal'] for ref in ['tau',base]):eligible.append(v)
 chosen=min(eligible,key=lambda v:means[str(v)]['lpips'])
 save(f'val_{arm}_summary.json',{'means':means,'eligible':eligible,'chosen':chosen,'seconds':time.time()-start,'videos':len(videos)})
 save(f'val_{arm}_videos.json',videos);log(phase='validation_complete',arm=arm,chosen=chosen,seconds=time.time()-start)
@torch.no_grad()
def run():
 setup();OUT.mkdir(exist_ok=False);(OUT/'cache').mkdir();start=time.time()
 splits=json.loads((t.OUT/'split_paths.json').read_text())['paths'];groups={k:{tuple(t.prev.group(p)) for p in v} for k,v in splits.items()}
 assert not groups['val']&groups['test'];assert not groups['train']&(groups['val']|groups['test'])
 save('manifest.json',splits)
 save('protocol.json',{'grid':GRID,'selection':'original-video equal LPIPS minimum, MSE and temporal <= 1.01 of both fixed and TAU; fixed baseline always available as fallback','A':'blend aligned and zero-motion texture by clamp(confidence-weighted estimated displacement / threshold,0,1); zero threshold means fixed','B':'5x5 split of ungated texture residual; constant fine weight','C':'same split, fine weight linear 1 to endpoint across 8 horizons','shared':'frozen TAU, history last4, alpha .25, original confidence gate once, pixelcap .03, meanabs cap .001,128x128','test_policy':'all three selected configs locked before test; no test-driven tuning; existing test has prior historical evaluation exposure','script_sha256':e.sha(__file__),'checkpoint_sha256':e.sha(t.ROOT/'tau128_full_fair_fast_20261009/best.pt'),'counts':{k:{'clips':len(v),'videos':len(groups[k])} for k,v in splits.items()}})
 # Verify unchanged baseline, before preparing any validation data.
 x=torch.load(h.SRC/'heldout_0_input.pt',weights_only=False);d=features(x['context'].cuda().float(),x['tau'].cuda().float());p,_=e.fixed(x['context'].cuda().float(),x['tau'].cuda().float())
 parity={arm:float((predict(d,arm,GRID[arm][0])-p['fixed_history']).abs().max()) for arm in GRID};assert max(parity.values())<1e-6;save('parity.json',parity);del x,d,p
 net=network()
 for i,path in enumerate(splits['val']):
  d=make(path,net);torch.save({k:v.cpu() for k,v in d.items()},OUT/'cache'/f'{i:04d}.pt')
  if (i+1)%20==0:
   progress={'phase':'cache','done':i+1,'total':len(splits['val']),'seconds':time.time()-start};save('progress.json',progress);log(**progress)
 del net,d;torch.cuda.empty_cache()
 procs=[];handles=[]
 for arm in GRID:
  f=(OUT/f'val_{arm}.log').open('w');handles.append(f);procs.append(subprocess.Popen([sys.executable,'-u',__file__,arm],stdout=f,stderr=subprocess.STDOUT))
 log(phase='parallel_validation_started',pids=[p.pid for p in procs]);save('progress.json',{'phase':'parallel_validation','pids':[p.pid for p in procs]})
 codes=[p.wait() for p in procs]
 for f in handles:f.close()
 assert codes==[0,0,0],codes
 selected={arm:load(f'val_{arm}_summary.json')['chosen'] for arm in GRID}
 save('LOCKED_CONFIGS.json',{'selected':selected,'locked_unix':time.time(),'source':'validation only','protocol_sha256':e.sha(OUT/'protocol.json')})
 log(phase='configs_locked',selected=selected)
 net=network();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[];test_start=time.time()
 for i,path in enumerate(splits['test']):
  d=make(path,net);pred={'tau':d['tau'],'fixed':predict(d,'A',0)};pred.update({arm:predict(d,arm,v) for arm,v in selected.items()})
  row={'path':path,'group':list(t.prev.group(path)),'metrics':{m:metrics(p,d['target'],d['tau'],lp) for m,p in pred.items()}};rows.append(row)
  with (OUT/'test_per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%20==0:
   progress={'phase':'locked_test','done':i+1,'total':len(splits['test']),'seconds':time.time()-test_start};save('progress.json',progress);log(**progress)
 videos,means=summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 ci={}
 for arm in GRID:
  for ref in ['tau','fixed']:
   ci[arm+'-vs-'+ref]={k:h.ci([v['metrics'][arm][k]-v['metrics'][ref][k] for v in videos]) for k in ['mse','mae','lpips','temporal','psnr']}
 save('test_per_video.json',videos);save('summary.json',{'selected':selected,'clips':len(rows),'videos':len(videos),'video_equal':means,'clip_equal':aggregate(rows),'per_source':{s:{'videos':len(v),'metrics':aggregate(v)} for s,v in sources.items()},'paired_ci':ci})
 save('COMPLETE.json',{'seconds':time.time()-start,'test_seconds':time.time()-test_start});log(phase='complete',seconds=time.time()-start)
if __name__=='__main__':
 if len(sys.argv)>1:validate(sys.argv[1])
 else:run()
