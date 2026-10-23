"""Read-only diagnostic: GT is used only for evaluation masks, never corrections."""
import json,time,os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import cv2,lpips
import evaluate_fixed_history_expanded as e
from simvp_geometry_model import build_video_predictor_from_checkpoint
OUT=e.t.ROOT/'history_regions_diagnostic_20261015'
MODES=['tau','fixed_history','zero_motion']
def dump(n,d):
 p=OUT/n;p.with_suffix('.tmp').write_text(json.dumps(d,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def raster(uv,valid):
 out=np.zeros((128,128),np.uint8);xy=np.rint(uv[valid]).astype(int)
 for x,y in xy:
  if 0<=x<128 and 0<=y<128:cv2.circle(out,(x,y),3,1,-1)
 return torch.tensor(out,device='cuda').bool()
def masks(rgb,source):
 # Dense motion proxies measured on real RGB, not generated outputs.
 dense=[]
 for k in range(16,24):
  back=e.h.flow(rgb[k],rgb[15]);fw=e.h.flow(rgb[15],rgb[k]);fl,conf=e.h.maps(back,fw)
  # conf exp(-cycle^2/4.5) combines bounds; .607 corresponds to cycle error1.5px.
  uncertain=conf[0,0]<np.exp(-.5);mag=fl[0].norm(dim=0)
  dense.append({'all':torch.ones(128,128,device='cuda',dtype=torch.bool),'rgb_motion_small':(~uncertain)&(mag<.5),'rgb_motion_medium':(~uncertain)&(mag>=.5)&(mag<2),'rgb_motion_large':(~uncertain)&(mag>=2),'visibility_uncertain_proxy':uncertain})
 # Sparse annotated trajectories; full image dimensions are needed for projection.
 s=e.t.prev.load_worldtrack_sequence(Path(source),24);uv=np.asarray(s['tracks_uv'],np.float64);vis=np.asarray(s['visibility'],bool)
 hh,ww=s['video_rgb'].shape[1:3];uv=uv/np.array([ww-1,hh-1])*127
 valid=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=127).all(-1);vis=vis&valid
 for f,k in enumerate(range(16,24)):
  delta=np.linalg.norm(uv[k]-uv[15],axis=-1);both=vis[k]&vis[15]
  dense[f].update({'track_moving':raster(uv[k],both&(delta>2)),'track_static_image':raster(uv[k],both&(delta<.5)),'track_newly_visible':raster(uv[k],vis[k]&~vis[15]),'track_persistent':raster(uv[k],both)})
 return dense
def mean_or_none(a):return float(np.mean(a)) if a else None
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);start=time.time()
 old=[json.loads(x) for x in (e.OUT/'per_clip.jsonl').read_text().splitlines()];oldmap={x['path']:x for x in old};paths=[x['path'] for x in old]
 dump('protocol.json',{'scope':'all347 clips46 videos, same locked parameters; no tuning','comparison':'fixed minus zero, negative is better for error metrics','regions':'dense real-RGB Farneback motion .5/2px, partitioned with FB/bounds-uncertain proxy; sparse GT-track radius3 neighborhoods for visibility labels','occlusion_limit':'newly visible GT points define only sparse neighborhoods; dense cycle failure is NOT true occlusion segmentation; image motion includes camera motion','regional_lpips':'spatial Alex LPIPS map averaged within masks, NOT cropped-image LPIPS; receptive fields cross region boundaries; map mean may differ from scalar LPIPS','aggregation':'frame means within clip then clip means per original video; missing regions omitted; retain counts; dense contribution weighted area sums to spatialLPIPS gap','failure':'same LPIPS gain>1e-4 and MSE/temporal<=1.01TAU rule; posthoc failure comparisons descriptive only'})
 ck=torch.load(e.t.ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False);net=build_video_predictor_from_checkpoint(ck).cuda().eval();net.load_state_dict(ck['model']);del ck
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();sp=lpips.LPIPS(net='alex',spatial=True,verbose=False).cuda().eval()
 rows=[];done=set()
 if (OUT/'per_clip.jsonl').exists():
  rows=[json.loads(x) for x in (OUT/'per_clip.jsonl').read_text().splitlines()];done={x['path'] for x in rows}
 for i,path in enumerate(paths):
  if path in done:continue
  rgb,source=e.rgb24(path);ctx=rgb[:16].cuda();_,hid,skip=e.t.core.extract_frozen_features(net,ctx[None]);tau=e.t.core.decode_frozen(net,hid,skip)[0].half().float();pred,_=e.fixed(ctx.half().float(),tau)
  target=rgb[16:24].cuda().half().float();mm=masks(rgb,source);record={'path':path,'group':oldmap[path]['group'],'failed':not e.h.safe(oldmap[path]['metrics']['fixed_history'],oldmap[path]['metrics']['tau']),'regions':{},'horizons':[],'spectral':{},'layers':{},'reproduction':{}}
  maps_by={};errors={}
  for mode in MODES:
   p=pred[mode];globalval,layers=lp(p*2-1,target*2-1,retPerLayer=True)
   record['reproduction'][mode]=abs(float(globalval.mean())-oldmap[path]['metrics'][mode]['lpips'])
   assert record['reproduction'][mode]<1e-5
   record['layers'][mode]=[float(x.mean()) for x in layers];maps_by[mode]=sp(p*2-1,target*2-1)[:,0];errors[mode]=(p-target).square().mean(1)
   hp=p-e.h.low(p);hpgt=target-e.h.low(target)
   record['spectral'][mode]={'hp_abs_energy':float(hp.abs().mean()),'hp_mse':float((hp-hpgt).square().mean()),'low_mse':float((e.h.low(p)-e.h.low(target)).square().mean()),'edit_hp_energy':float(((p-tau)-e.h.low(p-tau)).abs().mean())}
  record['spectral']['target']={'hp_abs_energy':float((target-e.h.low(target)).abs().mean())}
  for region in mm[0]:
   frames=[]
   for f in range(8):
    mask=mm[f][region];count=int(mask.sum())
    if not count:continue
    frame={'horizon':f+1,'pixels':count,'fraction':count/16384,'mse_diff':float((errors['fixed_history'][f]-errors['zero_motion'][f])[mask].mean()),'spatial_lpips_diff':float((maps_by['fixed_history'][f]-maps_by['zero_motion'][f])[mask].mean()),'lpips_contribution':float(((maps_by['fixed_history'][f]-maps_by['zero_motion'][f])*mask).mean()),'mse_contribution':float(((errors['fixed_history'][f]-errors['zero_motion'][f])*mask).mean())}
    frames.append(frame)
   if frames:record['regions'][region]={'frames':frames,'fraction':sum(x['fraction'] for x in frames)/8,**{k:float(np.mean([x[k] for x in frames])) for k in ['mse_diff','spatial_lpips_diff']},**{k:sum(x[k] for x in frames)/8 for k in ['lpips_contribution','mse_contribution']}}
  for f in range(8):record['horizons'].append({'mse_diff':float((errors['fixed_history'][f]-errors['zero_motion'][f]).mean()),'spatial_lpips_diff':float((maps_by['fixed_history'][f]-maps_by['zero_motion'][f]).mean())})
  partition=['rgb_motion_small','rgb_motion_medium','rgb_motion_large','visibility_uncertain_proxy']
  assert abs(sum(record['regions'].get(k,{}).get('lpips_contribution',0) for k in partition)-record['regions']['all']['lpips_contribution'])<1e-6
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  rows.append(record)
  if (i+1)%20==0:dump('progress.json',{'clips':len(rows),'total':len(paths),'seconds':time.time()-start});print(json.dumps({'clips':len(rows),'seconds':time.time()-start}),flush=True)
 vg=defaultdict(list)
 for r in rows:vg[tuple(r['group'])].append(r)
 videos=[]
 for g,rr in sorted(vg.items()):
  v={'group':list(g),'clips':len(rr),'regions':{},'layers':{m:np.mean([x['layers'][m] for x in rr],axis=0).tolist() for m in MODES},'spectral':{m:{k:float(np.mean([x['spectral'][m][k] for x in rr])) for k in rr[0]['spectral'][m]} for m in rr[0]['spectral']}}
  for reg in mm[0]:
   valid=[r['regions'][reg] for r in rr if reg in r['regions']]
   if valid:v['regions'][reg]={k:float(np.mean([x[k] for x in valid])) for k in ['fraction','mse_diff','spatial_lpips_diff']}
   if reg in v['regions']:
    for k in ['lpips_contribution','mse_contribution']:v['regions'][reg][k]=float(np.mean([r['regions'].get(reg,{}).get(k,0) for r in rr]))
  videos.append(v)
 summary={'clips':len(rows),'videos':len(videos),'failed_clips':sum(r['failed'] for r in rows),'region_video_equal':{},'per_source':{},'failure_vs_success':{},'layers_video_equal':{m:np.mean([v['layers'][m] for v in videos],axis=0).tolist() for m in MODES},'spectral_video_equal':{m:{k:float(np.mean([v['spectral'][m][k] for v in videos])) for k in videos[0]['spectral'][m]} for m in videos[0]['spectral']},'horizons_video_equal':[ {k:float(np.mean([np.mean([x['horizons'][f][k] for x in rr]) for rr in vg.values()])) for k in ['mse_diff','spatial_lpips_diff']} for f in range(8)]}
 for reg in mm[0]:
  valid=[v['regions'][reg] for v in videos if reg in v['regions']]
  summary['region_video_equal'][reg]={'videos':len(valid),**{k:e.h.ci([x[k] for x in valid]) for k in valid[0]}} if valid else {'videos':0}
 for src in sorted({v['group'][0] for v in videos}):
  vs=[v for v in videos if v['group'][0]==src];summary['per_source'][src]={reg:{'videos':sum(reg in v['regions'] for v in vs),**{k:mean_or_none([v['regions'][reg][k] for v in vs if reg in v['regions']]) for k in ['fraction','mse_diff','spatial_lpips_diff','lpips_contribution']}} for reg in mm[0]}
 for flag in [False,True]:
  rr=[x for x in rows if x['failed']==flag];summary['failure_vs_success'][str(flag)]={'clips':len(rr),'regions':{reg:{k:mean_or_none([x['regions'][reg][k] for x in rr if reg in x['regions']]) for k in ['fraction','mse_diff','spatial_lpips_diff']} for reg in mm[0]}}
 dump('per_video.json',videos);dump('summary.json',summary);dump('failures.json',[{'path':r['path'],'group':r['group'],'regions':{k:{a:b for a,b in v.items() if a!='frames'} for k,v in r['regions'].items()}} for r in rows if r['failed']]);dump('COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
