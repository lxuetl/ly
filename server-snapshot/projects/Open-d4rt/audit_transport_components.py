"""No-training decomposition using common gains, not independent renormalization."""
import json,time,hashlib,random
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import verify_transport_direction as v
e=v.e;tr=v.tr;prev=v.prev;core=v.core
OUT=e.p.run.fit.ROOT/'transport_components_control_20261013'
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);prev.prior.seed_all(42);start=time.time()
 paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']['val']
 groups=defaultdict(list)
 for path in paths:groups[tuple(prev.group(path))].append(path)
 rng=random.Random(42);paths=[rng.choice(sorted(ps)) for _,ps in sorted(groups.items())]
 base=e.p.run.fit.ROOT
 cfg=SimpleNamespace(backbone_checkpoint=str(base/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(base/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(base/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
 backbone,joint,adapter,_=prev.old.load_frozen(cfg,torch.device('cuda'));ds=prev.VideoForecastDataset('','val',128,paths=paths)
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
 def cost(a,b):return (a-b).square().mean()+.03*lp(a*2-1,b*2-1).mean()+.1*((a[1:]-a[:-1])-(b[1:]-b[:-1])).abs().mean()
 # grid_sample align_corners=True: unit flow corresponds to W-1/H-1 pixels.
 src=torch.zeros(1,1,9,9,device='cuda');src[0,0,4,3]=1
 f=torch.zeros(1,1,2,9,9,device='cuda');f[:,:,0]=1/8
 warped=core.backward_warp(src,f);peak=np.unravel_index(int(warped.argmax()),tuple(warped.shape))[-2:]
 synthetic={'expected_peak':[4,4],'actual_peak':[int(a) for a in peak],'passed':list(peak)==[4,4],'meaning':'positive x flow moves content right; normalized unit x spans W-1 pixels'}
 assert synthetic['passed']
 file=OUT/'rows.jsonl';rows=[json.loads(a) for a in file.read_text().splitlines()] if file.exists() else []
 done={r['path'] for r in rows};components={k:None for k in ['history','motion','full','motion_reverse','tau_return']}
 stats=rows[-1]['stats'] if rows else {}
 for i,path in enumerate(paths):
  if path in done:continue
  item=ds[i];obs={k:item[k][None].cuda() for k in prev.PAST};obs['context_rgb']=item['context_rgb'][None].cuda()
  source,hidden,skip=core.extract_frozen_features(backbone,obs['context_rgb']);tau=core.decode_frozen(backbone,hidden,skip)
  bundle=core.geometry_bundle('correct',joint,obs,8);adapter.mode='correct';corrected,_,_=adapter(source,hidden,bundle,bundle)
  flow=adapter.last['flow'].clone();support=adapter.last['support'].clone()
  permission=((F.interpolate(support.flatten(0,1),(128,128),mode='bilinear',align_corners=False).reshape(1,8,1,128,128)-.01)/.05).clamp(0,1)
  geo=(tau+(prev.prior.match_pixel_budget(core.decode_frozen(backbone,corrected,skip),tau,.002,'fixed',.08)-tau)*permission).clamp(0,1);budget=(geo-tau).abs().mean()
  hist=(source[:,None]-hidden)*support
  motion=(core.backward_warp(source,flow)-source[:,None])*support
  reverse=(core.backward_warp(source,-flow)-source[:,None])*support
  full=hist+motion;gain=.006/full.abs().mean().clamp_min(1e-7)
  components={'history':hist,'motion':motion,'full':full,'motion_reverse':reverse,'tau_return':torch.zeros_like(full)}
  uncapped={};stats={}
  for name,r in components.items():
   scaled=r*gain;clipped=scaled.clamp(-.12,.12)
   decoded=core.decode_frozen(backbone,hidden+clipped,skip)
   uncapped[name]=(decoded-tau)*permission
   stats[name]={'raw_l1':float(r.abs().mean()),'scaled_l1':float(scaled.abs().mean()),'clipped_l1':float(clipped.abs().mean()),'clipped_fraction':float((scaled.abs()>.12).float().mean()),'decoded_delta_l1':float(uncapped[name].abs().mean())}
  # Same gain as original full branch; only cap excessive branches, never fill their budget.
  pixel_gain=budget/uncapped['full'].abs().mean().clamp_min(1e-9)
  predictions={}
  for name,d in uncapped.items():
   pred=(tau+(d*pixel_gain).clamp(-.08,.08)).clamp(0,1)
   delta=pred-tau;cap=(budget/delta.abs().mean().clamp_min(1e-9)).clamp(max=1)
   predictions[name]=(tau+delta*cap)[0];stats[name]['pixel_l1']=float((predictions[name]-tau[0]).abs().mean())
  mag=torch.linalg.vector_norm(flow,dim=2);act=support[:,:,0]>.01
  h,w=flow.shape[-2:];pixels=torch.linalg.vector_norm(flow*torch.tensor([w-1,h-1],device='cuda')[None,None,:,None,None],dim=2)
  imagepixels=torch.linalg.vector_norm(flow*127,dim=2)
  stats['flow']={'feature_hw':[h,w],'active_feature_pixel_quantiles':torch.quantile(pixels[act],torch.tensor([.5,.9,.99],device='cuda')).tolist() if act.any() else [],'active_image_pixel_quantiles':torch.quantile(imagepixels[act],torch.tensor([.5,.9,.99],device='cuda')).tolist() if act.any() else [],'motion_history_l1_ratio':float(motion.abs().mean()/hist.abs().mean().clamp_min(1e-12)),'pixel_motion_full_ratio':stats['motion']['pixel_l1']/max(stats['full']['pixel_l1'],1e-12)}
  stats['direction_difference']={'raw_l1':float((motion-reverse).abs().mean()),'feature_after_cap_l1':float(((motion*gain).clamp(-.12,.12)-(reverse*gain).clamp(-.12,.12)).abs().mean()),'final_pixel_l1':float((predictions['motion']-predictions['motion_reverse']).abs().mean())}
  target=item['future_rgb'].cuda();refcost=cost(geo[0],target)
  key=hashlib.sha256(path.encode()).hexdigest()[:24];lab=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False);cells=torch.nonzero(lab['mask'][:,0]).tolist()
  gains={name:[] for name in components}
  for t,y,x in cells:
   for name,pred in predictions.items():
    bank=torch.stack([tau[0],geo[0],pred])[None];weights=torch.zeros(1,8,3,8,8,device='cuda');weights[:,:,1]=1;weights[:,2*t:2*t+2,:,y,x]=0;weights[:,2*t:2*t+2,2,y,x]=1
    final=tr.synthesize(bank,weights,budget[None])[0];gains[name].append(float(refcost-cost(final,target)))
  row={'path':path,'group':list(prev.group(path)),'stats':stats,'oracle':{k:max([0.]+a) for k,a in gains.items()},'uniform':{k:float(np.mean(a)) if a else 0 for k,a in gains.items()},'gains':gains};rows.append(row)
  with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({'clips':i+1,'total':len(paths),'seconds':time.time()-start}),flush=True)
 summary={'clips':len(rows),'synthetic':synthetic,'test_used':False,'sampling':'one seeded clip per original validation video,40 videos','common_gains':True,'oracle':{},'uniform':{},'component_mean':{},'flow':{}}
 for name in components:
  summary['oracle'][name]=float(np.mean([r['oracle'][name] for r in rows]));summary['uniform'][name]=float(np.mean([r['uniform'][name] for r in rows]))
  summary['component_mean'][name]={k:float(np.mean([r['stats'][name][k] for r in rows])) for k in stats[name]}
 for k in ['motion_history_l1_ratio','pixel_motion_full_ratio']:summary['flow'][k]=float(np.median([r['stats']['flow'][k] for r in rows]))
 for k in ['active_feature_pixel_quantiles','active_image_pixel_quantiles']:summary['flow'][k]=np.median([r['stats']['flow'][k] for r in rows if r['stats']['flow'][k]],axis=0).tolist()
 summary['paired_ci']={};rng=np.random.default_rng(42)
 for a,b in [('full','history'),('motion','motion_reverse'),('motion','tau_return')]:
  d=np.array([r['oracle'][a]-r['oracle'][b] for r in rows]);summary['paired_ci'][a+'-'+b]={'difference':float(d.mean()),'ci95':np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975]).tolist()}
 prev.dump(OUT/'summary.json',summary);prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
