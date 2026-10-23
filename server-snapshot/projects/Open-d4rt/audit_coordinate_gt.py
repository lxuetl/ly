import json,time,hashlib,random
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import verify_transport_direction as v
e=v.e;tr=v.tr;prev=v.prev;core=v.core
OUT=e.p.run.fit.ROOT/'coordinate_gt_audit_20261013'
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(4);prev.prior.seed_all(42);start=time.time()
 paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']['val'];groups=defaultdict(list)
 for path in paths:groups[tuple(prev.group(path))].append(path)
 rng=random.Random(42);paths=[rng.choice(sorted(ps)) for _,ps in sorted(groups.items())]
 root=e.p.run.fit.ROOT;cfg=SimpleNamespace(backbone_checkpoint=str(root/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(root/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(root/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
 backbone,joint,adapter,_=prev.old.load_frozen(cfg,torch.device('cuda'));ds=prev.VideoForecastDataset('','val',128,paths=paths)
 # Known affine correspondence: destination y=A*x+b, sampled destination flow must return x.
 coord=torch.linspace(.15,.85,33,device='cuda');yy,xx=torch.meshgrid(coord,coord,indexing='ij');dst=torch.stack([xx,yy],-1).reshape(1,-1,1,2)
 synthetic={}
 for name,scale,offset in [('translation',1.,.05),('affine_scale',1.15,-.075)]:
  anchor=((dst-offset)/scale)[:,:,0];f,sup=adapter.raster(anchor,dst,torch.ones(dst.shape[:-1],device='cuda'),64,64)
  sampled=F.grid_sample(f[:,0],dst[:,:,0,None]*2-1,align_corners=True).squeeze(-1).transpose(1,2)
  errors=(dst[:,:,0]-sampled-anchor).norm(dim=-1)*63
  synthetic[name]={'mean_feature_pixel_error':float(errors.mean()),'max_feature_pixel_error':float(errors.max())}
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
 def cost(a,b):return (a-b).square().mean()+.03*lp(a*2-1,b*2-1).mean()+.1*((a[1:]-a[:-1])-(b[1:]-b[:-1])).abs().mean()
 rows=[]
 for i,path in enumerate(paths):
  x=ds[i];obs={k:x[k][None].cuda() for k in prev.PAST};obs['context_rgb']=x['context_rgb'][None].cuda()
  source,hidden,skip=core.extract_frozen_features(backbone,obs['context_rgb']);tau=core.decode_frozen(backbone,hidden,skip)
  bundle=core.geometry_bundle('correct',joint,obs,8);adapter.mode='correct';corrected,_,_=adapter(source,hidden,bundle,bundle)
  predicted=adapter.last['flow'].clone();support=adapter.last['support'].clone();permission=((F.interpolate(support.flatten(0,1),(128,128),mode='bilinear',align_corners=False).reshape(1,8,1,128,128)-.01)/.05).clamp(0,1)
  geo=(tau+(prev.prior.match_pixel_budget(core.decode_frozen(backbone,corrected,skip),tau,.002,'fixed',.08)-tau)*permission).clamp(0,1);budget=(geo-tau).abs().mean()
  z=np.load(path);meta=json.loads(str(z['metadata_json']));sample=prev.load_worldtrack_sequence(Path(meta['source_path']),24);ids=np.asarray(z['query_ids'],dtype=np.int64)
  uv=np.asarray(sample['tracks_uv'][:,ids],np.float32).transpose(1,0,2).copy();hh,ww=sample['video_rgb'].shape[1:3];uv/=np.array([ww-1,hh-1],np.float32)
  valid=np.asarray(sample['visibility'][:,ids]).T.copy()&np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
  uv=torch.tensor(np.nan_to_num(uv),device='cuda')[None];vv=torch.tensor(valid,device='cuda')[None]
  gta=uv[:,:,15];gtf=uv[:,:,16:24];both=vv[:,:,15,None]&vv[:,:,16:24]
  raw,puv,rel,anchor=bundle
  gtweight=adapter._permission(gtf,rel*both.float(),gta)
  gtflow,_=adapter.raster(gta,gtf,gtweight,*hidden.shape[-2:])
  hist=(source[:,None]-hidden)*support;motion=(core.backward_warp(source,predicted)-source[:,None])*support;gm=(core.backward_warp(source,gtflow)-source[:,None])*support
  comp={'zero_history':hist,'pred_full':hist+motion,'gt_full':hist+gm,'pred_motion':motion,'gt_motion':gm,'tau_return':torch.zeros_like(hist)}
  featuregain=.006/comp['pred_full'].abs().mean().clamp_min(1e-7);deltas={}
  for name,r in comp.items():deltas[name]=(core.decode_frozen(backbone,hidden+(r*featuregain).clamp(-.12,.12),skip)-tau)*permission
  pg=budget/deltas['pred_full'].abs().mean().clamp_min(1e-9);preds={}
  for name,d in deltas.items():
   pred=(tau+(d*pg).clamp(-.08,.08)).clamp(0,1);dd=pred-tau;preds[name]=(tau+dd*(budget/dd.abs().mean().clamp_min(1e-9)).clamp(max=1))[0]
  target=x['future_rgb'].cuda();ref=cost(geo[0],target);key=hashlib.sha256(path.encode()).hexdigest()[:24];label=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False);cells=torch.nonzero(label['mask'][:,0]).tolist();gains={n:[] for n in comp}
  for t,y,xc in cells:
   for name,pred in preds.items():
    bank=torch.stack([tau[0],geo[0],pred])[None];w=torch.zeros(1,8,3,8,8,device='cuda');w[:,:,1]=1;w[:,2*t:2*t+2,:,y,xc]=0;w[:,2*t:2*t+2,2,y,xc]=1
    gains[name].append(float(ref-cost(tr.synthesize(bank,w,budget[None])[0],target)))
  observed_valid=obs['past_valid'].bool();eligible=observed_valid.any(2)
  endpoint=(puv-gtf).norm(dim=-1)*127;ancherr=(anchor-gta).norm(dim=-1)*127
  # Check GT flow returns GT anchors at their destination coordinates after smoothing.
  errs=[]
  for t in range(8):
   df=F.grid_sample(gtflow[:,t],gtf[:,:,t,None]*2-1,align_corners=True).squeeze(-1).transpose(1,2)
   er=(gtf[:,:,t]-df-gta).norm(dim=-1)*127;ok=gtweight[:,:,t]>0
   if ok.any():errs.extend(er[ok].tolist())
  row={'path':path,'gt_provenance':meta['source_path'],'group':list(prev.group(path)),
   'valid_gt_pairs':int(both.sum()),'stale_anchor_fraction':float((eligible&~observed_valid[:,:,-1]).sum()/eligible.sum().clamp_min(1)),
   'pred_uv_error_px':float(endpoint[both].mean()) if both.any() else None,'anchor_error_px':float(ancherr[vv[:,:,15]].mean()) if vv[:,:,15].any() else None,
   'gt_dense_backmap_error_px':float(np.mean(errs)) if errs else None,
   'oracle':{n:max([0.]+a) for n,a in gains.items()},'uniform':{n:float(np.mean(a)) if a else 0 for n,a in gains.items()},'gains':gains}
  rows.append(row)
  with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({'clips':i+1,'seconds':time.time()-start}),flush=True)
 summary={'clips':len(rows),'synthetic':synthetic,'gt_scope':'dataset XYZ projection using original query_ids, last observed frame15 to frames16:24; NOT D4RT teacher; original predicted support/permission and reliability retained; not dense GT upper bound',
  'oracle':{n:float(np.mean([r['oracle'][n] for r in rows])) for n in comp},'uniform':{n:float(np.mean([r['uniform'][n] for r in rows])) for n in comp},'diagnostics':{},'paired_ci':{}}
 for k in ['stale_anchor_fraction','pred_uv_error_px','anchor_error_px','gt_dense_backmap_error_px']:summary['diagnostics'][k]=float(np.mean([r[k] for r in rows if r[k] is not None]))
 rng=np.random.default_rng(42)
 for a,b in [('gt_full','pred_full'),('gt_full','zero_history'),('gt_motion','pred_motion'),('gt_motion','tau_return')]:
  d=np.array([r['oracle'][a]-r['oracle'][b] for r in rows]);summary['paired_ci'][a+'-'+b]={'gain':float(d.mean()),'ci95':np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975]).tolist()}
 prev.dump(OUT/'summary.json',summary);prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
