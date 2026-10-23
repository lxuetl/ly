"""Locked historical texture compensation on the existing held-out-video test split."""
import os,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import cv2,lpips
import train_propagation_fusion as t
import history_texture_candidates as h
from simvp_geometry_model import build_video_predictor_from_checkpoint
OUT=t.ROOT/'fixed_history_expanded_20261015'
MODES=['tau','fixed_history','zero_motion','reverse_motion']
def dump(n,d):
 p=OUT/n;p.with_suffix('.tmp').write_text(json.dumps(d,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def sha(p):
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
@torch.no_grad()
def fixed(ctx,tau):
 aligned={m:[] for m in MODES[1:]};gates=[]
 for step in range(8):
  warped={m:[] for m in aligned};weights=[]
  for k in range(12,16):
   b=h.flow(tau[step],ctx[k]);fw=h.flow(ctx[k],tau[step]);fl,valid=h.maps(b,fw)
   cor=h.warp(ctx[k:k+1],fl);appearance=(h.low(cor)-h.low(tau[step:step+1])).abs().mean(1,keepdim=True)
   conf=valid*torch.exp(-appearance/.06);weights.append(conf)
   for mode,sign in [('fixed_history',1),('zero_motion',0),('reverse_motion',-1)]:warped[mode].append(h.warp(ctx[k:k+1],fl*sign))
  ws=torch.stack(weights);den=ws.sum(0).clamp_min(1e-6);fused={m:(torch.stack(v)*ws).sum(0)/den for m,v in warped.items()}
  disagree=((torch.stack(warped['fixed_history'])-fused['fixed_history']).abs().mean(2,keepdim=True)*ws).sum(0)/den
  gates.append(ws.mean(0)*torch.exp(-disagree/.04))
  for m in aligned:aligned[m].append(fused[m])
 gate=torch.cat(gates);pred={'tau':tau}
 for m,seq in aligned.items():
  src=torch.cat(seq);res=(src-h.low(src))-(tau-h.low(tau));pred[m]=h.bound(.25*gate*res,tau).half().float()
 return pred,{'confidence_mean':float(gate.mean()),'coverage_gt_.1':float((gate>.1).float().mean())}
def rgb24(path):
 with np.load(path) as z:meta=json.loads(str(z['metadata_json']))
 with np.load(meta['source_path'],allow_pickle=True) as pack:
  frames=pack['images_jpeg_bytes'][:24];assert len(frames)==24
  imgs=[]
  for b in frames:
   a=np.frombuffer(b,dtype=np.uint8) if isinstance(b,(bytes,bytearray)) else np.asarray(b,dtype=np.uint8)
   im=cv2.imdecode(a,cv2.IMREAD_COLOR);assert im is not None
   imgs.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_AREA))
 return torch.tensor(np.stack(imgs)).permute(0,3,1,2).float()/255,meta['source_path']
def aggregate(rows):
 return {m:{k:float(np.mean([r['metrics'][m][k] for r in rows])) for k in rows[0]['metrics'][m]} for m in MODES}
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1);t.prev.prior.seed_all(42);start=time.time()
 splits=json.loads((t.OUT/'split_paths.json').read_text())['paths'];groups={k:{tuple(t.prev.group(p)) for p in v} for k,v in splits.items()}
 dev=json.loads((t.ROOT/'sdxl_refiner_pilot_20261014/selection.json').read_text())['items'];devgroups={tuple(d['group']) for d in dev}
 assert not groups['test']&(groups['train']|groups['val']|devgroups)
 checkpoint=t.ROOT/'tau128_full_fair_fast_20261009/best.pt';backbone_split=json.loads((checkpoint.parent/'split_paths.json').read_text());bs=backbone_split.get('paths',backbone_split.get('selected',backbone_split))
 assert 'train' in bs and 'test' in bs
 assert not groups['test']&{tuple(t.prev.group(p)) for p in bs['train']}
 paths=splits['test'];assert len(paths)==len(set(paths))
 lock={'checkpoint_sha256':sha(checkpoint),'script_sha256':sha(__file__),'helper_sha256':sha(h.__file__),'manifest_sha256':hashlib.sha256(json.dumps(paths).encode()).hexdigest()}
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())['lock']==lock,'Changed implementation or manifest; do not mix runs'
 protocol={'lock':lock,'clips':len(paths),'videos':len(groups['test']),'excluded_development_groups':sorted(devgroups),'train_val_test_group_overlap':0,'backbone_training_group_overlap':0,'parameters':'last4 histories, Farneback, FB1.5, appearance .06, disagreement .04, highpass5x5, alpha .25, pixelcap .03, globalbudget .001','resolution':128,'context':16,'forecast':8,'primary':'video-equal LPIPS difference fixed_history minus TAU','secondary':['MSE','MAE','PSNR','temporal','safe improvement fraction','per-source and horizon metrics'],'bootstrap':'10000 paired resamples of original-video means','controls':'same correct-flow-derived gate and caps; zero/reverse only alter sampling displacement, not equal realized budget','prior_exposure':'existing test split previously evaluated for other architectures; NOT a never-observed benchmark; this method has not been tuned on these videos','future_gt':'metrics only, no flow/gates/config selection','geometry':False,'selection':'fixed config selected before this test; no retraining or tuning'}
 dump('protocol.json',protocol);dump('manifest.json',{'paths':paths,'groups':sorted(groups['test'])})
 # Bit-level check against original pilot implementation, without tuning.
 a=torch.load(h.SRC/'heldout_0_input.pt',weights_only=False);old=torch.load(h.OUT/'heldout_0.pt',weights_only=False)
 pred,_=fixed(a['context'].cuda().float(),a['tau'].cuda().float());parity={}
 for m,n in [('fixed_history','align_texture_correct_a0.25'),('zero_motion','align_texture_zero_a0.25'),('reverse_motion','align_texture_reverse_a0.25')]:
  parity[m]=float((pred[m].cpu()-old[n].float()).abs().max());assert parity[m]<1e-6
 dump('parity.json',parity);del pred,old,a
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);net=build_video_predictor_from_checkpoint(ck).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
 rows=[];done=set()
 if (OUT/'per_clip.jsonl').exists():
  for line in (OUT/'per_clip.jsonl').read_text().splitlines():
   row=json.loads(line);rows.append(row);done.add(row['path'])
 for i,path in enumerate(paths):
  if path in done:continue
  tick=time.time();rgb,source=rgb24(path);context=rgb[:16].cuda()
  _,hid,skip=t.core.extract_frozen_features(net,context[None]);tau=t.core.decode_frozen(net,hid,skip)[0].half().float()
  pred,diag=fixed(context.half().float(),tau)
  target=rgb[16:24].cuda().half().float();metrics={};horizons={}
  for m,p in pred.items():
   err=p-target;per=lp(p*2-1,target*2-1).flatten();mse=err.square().mean((1,2,3));mae=err.abs().mean((1,2,3));tmp=((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean((1,2,3))
   metrics[m]={'mse':float(mse.mean()),'mae':float(mae.mean()),'psnr':float((-10*mse.clamp_min(1e-12).log10()).mean()),'lpips':float(per.mean()),'temporal':float(tmp.mean()),'budget':float((p-tau).abs().mean())}
   horizons[m]={'mse':mse.cpu().tolist(),'lpips':per.cpu().tolist(),'temporal':tmp.cpu().tolist()}
  row={'path':path,'source_path':source,'group':list(t.prev.group(path)),'metrics':metrics,'horizons':horizons,'diagnostics':diag,'seconds':time.time()-tick}
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  rows.append(row)
  if (i+1)%20==0 or i+1==len(paths):
   elapsed=time.time()-start;progress={'clips':len(rows),'total':len(paths),'seconds':elapsed,'remaining_seconds_estimate':float(np.mean([r['seconds'] for r in rows[-20:]])*(len(paths)-len(rows)))};dump('progress.json',progress);print(json.dumps(progress),flush=True)
 vg=defaultdict(list)
 for row in rows:vg[tuple(row['group'])].append(row)
 videos=[{'group':list(g),'clips':len(rr),'metrics':aggregate(rr),'horizons':{m:{k:np.mean([r['horizons'][m][k] for r in rr],axis=0).tolist() for k in rr[0]['horizons'][m]} for m in MODES}} for g,rr in sorted(vg.items())]
 sources=defaultdict(list)
 for row in videos:sources[row['group'][0]].append(row)
 summary={'clips':len(rows),'videos':len(videos),'video_equal':aggregate(videos),'clip_equal':aggregate(rows),'per_source':{s:{'videos':len(v),'clips':sum(x['clips'] for x in v),'metrics':aggregate(v)} for s,v in sources.items()},'paired_ci':{},'fractions':{},'horizon_video_equal':{m:{k:np.mean([r['horizons'][m][k] for r in videos],axis=0).tolist() for k in videos[0]['horizons'][m]} for m in MODES}}
 for a,b in [('fixed_history','tau'),('zero_motion','tau'),('reverse_motion','tau'),('fixed_history','zero_motion'),('fixed_history','reverse_motion')]:
  summary['paired_ci'][a+'-vs-'+b]={k:h.ci([r['metrics'][a][k]-r['metrics'][b][k] for r in videos]) for k in videos[0]['metrics'][a]}
 for m in MODES[1:]:
  summary['fractions'][m]={'safe_videos':sum(h.safe(r['metrics'][m],r['metrics']['tau']) for r in videos),'lpips_improved_videos':sum(r['metrics'][m]['lpips']<r['metrics']['tau']['lpips'] for r in videos),'mse_worse_gt1pct_videos':sum(r['metrics'][m]['mse']>r['metrics']['tau']['mse']*1.01 for r in videos),'temporal_worse_gt1pct_videos':sum(r['metrics'][m]['temporal']>r['metrics']['tau']['temporal']*1.01 for r in videos),'safe_clips':sum(h.safe(r['metrics'][m],r['metrics']['tau']) for r in rows)}
 dump('per_video.json',videos);dump('summary.json',summary);dump('COMPLETE.json',{'seconds':time.time()-start,'clips':len(rows),'videos':len(videos)});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
