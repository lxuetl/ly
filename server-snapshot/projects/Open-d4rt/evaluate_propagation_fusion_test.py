"""Locked-checkpoint test; original-video clustered paired bootstrap."""
import json,time,hashlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import train_propagation_fusion as t
from audit_propagation_unit import region_mask
OUT=t.OUT/'test_locked'

@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4);t.prev.prior.seed_all(42);start=time.time()
 ps=json.loads((t.OUT/'split_paths.json').read_text())['paths'];paths=ps['test']
 groups={k:{tuple(t.prev.group(p)) for p in v} for k,v in ps.items()}
 assert not groups['test']&(groups['train']|groups['val'])
 models={};locked={}
 for mode in t.MODES:
  file=t.OUT/mode/'best.pt';ck=torch.load(file,map_location='cpu',weights_only=False)
  m=t.Fusion().cuda().eval().requires_grad_(False);m.load_state_dict(ck['model']);models[mode]=m
  locked[mode]={'epoch':ck['epoch'],'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
 t.prev.dump(OUT/'protocol.json',{'checkpoints':locked,'clips':len(paths),'videos':len(groups['test']),'primary':'video-equal composite cost','bootstrap':'10000 resamples of original-video means, paired','mask':'GT visible trajectories with >2px displacement; radius3 projection; includes camera motion, not object segmentation','selection':'no test tuning; checkpoint selected by previous validation','gt':'future trajectory+visibility diagnostic only','shared_gate':'predicted support in all modes, zero not geometry-free','budget':'common upper bound, not equal actual magnitude'})
 root=t.ROOT;cfg=SimpleNamespace(backbone_checkpoint=str(root/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(root/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(root/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
 backbone,joint,_,_=t.prev.old.load_frozen(cfg,torch.device('cuda'));backbone.eval().requires_grad_(False);joint.eval().requires_grad_(False)
 raster=t.core.SparseDenseFlow(9).cuda();ds=t.prev.VideoForecastDataset('','test',128,paths=paths)
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
 rows=[]
 for i,path in enumerate(paths):
  x=ds[i];obs={k:x[k][None].cuda() for k in t.prev.PAST};obs['context_rgb']=x['context_rgb'][None].cuda()
  src,hid,skip=t.core.extract_frozen_features(backbone,obs['context_rgb']);tau=t.core.decode_frozen(backbone,hid,skip)
  _,uv,rel,anchor=t.core.geometry_bundle('correct',joint,obs,8);last=obs['past_valid'][:,:,-1].bool()
  flow,support=raster(anchor,uv,rel*last[:,:,None],*hid.shape[-2:])
  gate=F.interpolate((support>1e-4).float().flatten(0,1),(128,128),mode='bilinear',align_corners=False).reshape(1,8,1,128,128)
  def warp(f):
   f=F.interpolate(f.flatten(0,1),(128,128),mode='bilinear',align_corners=True).reshape(1,8,2,128,128)
   return t.core.backward_warp(obs['context_rgb'][:,-1],f)
  warps={'pred':warp(flow),'wrong':warp(-flow),'zero':obs['context_rgb'][:,-1,None].expand_as(tau)}
  z=np.load(path);meta=json.loads(str(z['metadata_json']));s=t.prev.load_worldtrack_sequence(Path(meta['source_path']),24);ids=np.asarray(z['query_ids'],dtype=np.int64)
  guv=np.asarray(s['tracks_uv'][:,ids],np.float32).transpose(1,0,2).copy();hh,ww=s['video_rgb'].shape[1:3];guv/=np.array([ww-1,hh-1],np.float32)
  valid=np.asarray(s['visibility'][:,ids]).T.astype(bool)&np.isfinite(guv).all(-1)&(guv>=0).all(-1)&(guv<=1).all(-1)
  guv=torch.tensor(np.nan_to_num(guv),device='cuda')[None];valid=torch.tensor(valid,device='cuda')[None]
  pair=valid[:,:,15,None]&valid[:,:,16:24];gf,_=raster(guv[:,:,15],guv[:,:,16:24],(pair&last[:,:,None]).float(),*hid.shape[-2:]);warps['gt']=warp(gf)
  moving=(guv[:,:,16:24]-guv[:,:,15,None]).norm(dim=-1)*127>2;mask=region_mask(guv[:,:,16:24],pair&moving)
  # Match fp16 cache quantization used for training, then compute metrics in fp32.
  tau=tau.half().float();gate=gate.half().float();target=x['future_rgb'].cuda().half().float()
  preds={'tau':tau[0]}
  for mode,m in models.items():preds[mode]=m(tau,warps[mode].half().float(),gate)[0]
  met={}
  for mode,p in preds.items():
   er=(p-target).square();mse=float(er.mean());per=float(lp(p*2-1,target*2-1).mean());temporal=float(((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean())
   met[mode]={'mse':mse,'mae':float((p-target).abs().mean()),'psnr':float((-10*torch.log10(er.mean((1,2,3)).clamp_min(1e-12))).mean()),'lpips':per,'temporal':temporal,'cost':mse+.03*per+.1*temporal,'budget':float((p-tau[0]).abs().mean()),'moving_mse':float(er.mean(1,keepdim=True)[mask].mean()) if mask.any() else None,'moving_mae':float((p-target).abs().mean(1,keepdim=True)[mask].mean()) if mask.any() else None}
  row={'path':path,'group':list(t.prev.group(path)),'moving_pixels':int(mask.sum()),'metrics':met};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%20==0:print(json.dumps({'clips':i+1,'total':len(paths),'seconds':time.time()-start}),flush=True)
 vg=defaultdict(list)
 for r in rows:vg[tuple(r['group'])].append(r)
 def aggregate(rr):
  return {mode:{metric:float(np.mean([r['metrics'][mode][metric] for r in rr if r['metrics'][mode][metric] is not None])) if any(r['metrics'][mode][metric] is not None for r in rr) else None for metric in rr[0]['metrics'][mode]} for mode in preds}
 videos=[{'group':list(g),'clips':len(rr),'metrics':aggregate(rr)} for g,rr in sorted(vg.items())]
 sources=defaultdict(list)
 for r in videos:sources[str(r['group'][0])].append(r)
 summary={'clips':len(rows),'videos':len(videos),'clip_equal':aggregate(rows),'video_equal':aggregate(videos),'per_source_video_equal':{g:aggregate(rr) for g,rr in sources.items()},'paired_ci':{}}
 for a,b in [('gt','zero'),('pred','zero'),('pred','wrong'),('gt','tau'),('pred','tau'),('zero','tau')]:
  out={}
  for metric in ['cost','mse','lpips','temporal','moving_mse','moving_mae','budget']:
   d=np.array([r['metrics'][a][metric]-r['metrics'][b][metric] for r in videos if r['metrics'][a][metric] is not None]);rng=np.random.default_rng(42)
   out[metric]={'n':len(d),'delta':float(d.mean()),'ci95':np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975]).tolist()} if len(d) else None
  summary['paired_ci'][a+' minus '+b]=out
 t.prev.dump(OUT/'per_video.json',videos);t.prev.dump(OUT/'summary.json',summary);t.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
