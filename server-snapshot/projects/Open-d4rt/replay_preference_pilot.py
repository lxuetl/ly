"""Causal replay calibration pilot; no geometry and no backbone/editor training."""
import os,json,time,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
import torch.nn.functional as F
import cv2
ROOT=Path('/root/autodl-fs/d4rt_training')
SRC=ROOT/'sdxl_refiner_pilot_20261014';OUT=ROOT/'replay_preference_20261014'
MODEL='/root/autodl-fs/models/sdxl-refiner-1.0'
CONFIGS=[(.1,42),(.1,43),(.2,42),(.2,43)]
def dump(n,x):
 p=OUT/n;p.with_suffix('.tmp').write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def resize(x,size):return F.interpolate(x.float(),(size,size),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
def save(n,d):torch.save({k:v.cpu().half() if torch.is_tensor(v) else v for k,v in d.items()},OUT/n)
@torch.no_grad()
def prepare():
 import train_propagation_fusion as t
 from simvp_geometry_model import build_video_predictor_from_checkpoint
 ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
 net=build_video_predictor_from_checkpoint(ck).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model'])
 items=torch.load(SRC/'inputs.pt',weights_only=False);selected=[];excluded=[]
 for i,d in enumerate(items):
  z=np.load(d['path']);meta=json.loads(str(z['metadata_json']));pack=np.load(meta['source_path'],allow_pickle=True)
  if len(pack['images_jpeg_bytes'])<32:excluded.append({'group':d['group'],'reason':'fewer than32 frames'});continue
  # Read RGB alone; no geometry, camera, trajectory or visibility fields used.
  frames=[]
  for b in pack['images_jpeg_bytes'][:32]:
   a=np.frombuffer(b,dtype=np.uint8) if isinstance(b,(bytes,bytearray)) else np.asarray(b,dtype=np.uint8)
   im=cv2.imdecode(a,cv2.IMREAD_COLOR);assert im is not None
   frames.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_AREA))
  rgb=torch.tensor(np.stack(frames)).permute(0,3,1,2).float()/255
  j=len(selected);selected.append({'index':j,'old_index':i,'group':d['group'],'path':d['path']})
  for phase,offset in [('replay',0),('heldout',8)]:
   context=rgb[offset:offset+16].cuda();_,hid,skip=t.core.extract_frozen_features(net,context[None]);tau=t.core.decode_frozen(net,hid,skip)[0]
   if phase=='replay':
    old=torch.load(SRC/f'clip_{i:02d}.pt',weights_only=False)
    err=float((tau.cpu()-old['tau'].float()).abs().max())
    assert err<.01,('cached replay mismatched',err)
    # Precisely reuse the old baseline underlying existing edited candidates.
    tau=old['tau'].cuda().float()
   save(f'{phase}_{j}_input.pt',{'context':context,'tau':tau})
   # Label store is separate and only accessed in the terminal calibration/evaluation stage.
   save(f'{phase}_{j}_label.pt',{'target':rgb[offset+16:offset+24]})
 dump('selection.json',{'selected':selected,'excluded':excluded})
@torch.no_grad()
def candidates():
 from diffusers import StableDiffusionXLImg2ImgPipeline
 sel=json.loads((OUT/'selection.json').read_text())['selected']
 pipe=StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL,torch_dtype=torch.float16,variant='fp16',local_files_only=True,add_watermarker=False).to('cuda')
 pipe.enable_vae_slicing();pipe.set_progress_bar_config(disable=True)
 for comp in [pipe.unet,pipe.vae,pipe.text_encoder_2]:comp.requires_grad_(False).eval()
 start=time.time()
 for d in sel:
  j=d['index'];old=torch.load(SRC/f"clip_{d['old_index']:02d}.pt",weights_only=False)
  names=['tau']+[f's{s}_seed{k}' for s,k in CONFIGS]+[f'r0.25_s{s}_seed{k}' for s,k in CONFIGS]
  save(f'replay_{j}_candidates.pt',{n:old[n] for n in names})
  if (OUT/f'heldout_{j}_candidates.pt').exists():continue
  inp=torch.load(OUT/f'heldout_{j}_input.pt',weights_only=False);tau=inp['tau'].cuda().float();up=resize(tau,512)
  preds={'tau':tau};roundtrips={}
  pipe.vae.to(dtype=torch.float32)
  for seed in [42,43]:
   rec=[]
   for frame in up:
    latent=pipe.vae.encode(frame[None]*2-1).latent_dist.sample(generator=torch.Generator('cuda').manual_seed(seed))
    rec.append(resize((pipe.vae.decode(latent).sample/2+.5).clamp(0,1),128)[0])
   roundtrips[seed]=torch.stack(rec)
  pipe.vae.to(dtype=torch.float16)
  for strength,seed in CONFIGS:
   frames=[]
   for frame in up:
    y=pipe(prompt='a clear natural image, realistic colors and details',negative_prompt='blur, artifacts, distorted shapes, oversharpening, text, watermark',image=frame[None],strength=strength,num_inference_steps=50,guidance_scale=3.,generator=torch.Generator('cuda').manual_seed(seed),output_type='pt').images
    frames.append(resize(y,128)[0])
   n=f's{strength}_seed{seed}';preds[n]=torch.stack(frames);preds['r0.25_'+n]=(tau+.25*(preds[n]-roundtrips[seed])).clamp(0,1)
   print(json.dumps({'stage':'candidates','video':j,'config':n,'elapsed':time.time()-start}),flush=True)
  save(f'heldout_{j}_candidates.pt',preds)
 del pipe;torch.cuda.empty_cache()
def flow(a,b):
 a=(a.permute(1,2,0).numpy()*255).astype('uint8');b=(b.permute(1,2,0).numpy()*255).astype('uint8')
 return cv2.calcOpticalFlowFarneback(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None,.5,3,15,3,5,1.2,0)
def warper(a,b):
 # Output destination-grid backward sampling coordinates; flow derived from TAU only.
 back=flow(b,a);forward=flow(a,b);yy,xx=np.mgrid[:128,:128].astype('float32')
 mx=xx+back[...,0];my=yy+back[...,1]
 f=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 mask=((np.linalg.norm(back+f,axis=-1)<1.5)&(mx>=0)&(mx<=127)&(my>=0)&(my<=127)).astype('float32')
 grid=torch.tensor(np.stack([mx/127*2-1,my/127*2-1],-1))[None].cuda()
 return grid,torch.tensor(mask)[None,None].cuda()
def masked(x,m):return (x.abs()*m).sum()/(m.sum()*x.shape[1]).clamp_min(1)
def stress(tau):
 smooth=F.avg_pool2d(tau,3,1,1);gen=torch.Generator('cuda').manual_seed(20261014)
 noise=torch.randn(tau.shape,device='cuda',generator=gen)*.025
 texture=torch.randn((1,3,128,128),device='cuda',generator=gen)*.025
 return {'stress_blur':smooth,'stress_sharpen':(tau+1.5*(tau-smooth)).clamp(0,1),'stress_flicker':(tau+noise).clamp(0,1),'stress_texture':(tau+texture).clamp(0,1)}
@torch.no_grad()
def features():
 import pyiqa,lpips
 quality=pyiqa.create_metric('musiq',device='cuda').eval();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
 sel=json.loads((OUT/'selection.json').read_text())['selected'];records=[]
 for d in sel:
  for phase in ['replay','heldout']:
   j=d['index'];inp=torch.load(OUT/f'{phase}_{j}_input.pt',weights_only=False);tau=inp['tau'].float().cuda();last=inp['context'][-1:].float().cuda()
   pred=torch.load(OUT/f'{phase}_{j}_candidates.pt',weights_only=False);pred={n:p.float().cuda() for n,p in pred.items()};pred.update(stress(tau))
   # Fixed grids/masks for all candidates; no target-derived flow or masks.
   full=torch.cat([last,tau]).cpu();grids=[warper(full[f],full[f+1]) for f in range(8)]
   candidate_rows=[]
   for name,p in pred.items():
    edit=p-tau;low=(F.avg_pool2d(p,8)-F.avg_pool2d(tau,8)).abs().mean()
    tmp=[];excess=[];seq=torch.cat([last,p]);base=torch.cat([last,tau])
    for f,(g,m) in enumerate(grids):
     a=F.grid_sample(seq[f:f+1],g,align_corners=True);b=F.grid_sample(base[f:f+1],g,align_corners=True)
     tmp.append(masked(seq[f+1:f+2]-a,m));excess.append(masked((seq[f+1:f+2]-a)-(base[f+1:f+2]-b),m))
    # Unsupported detail relative to flow-aligned last observed frame, excluding unreliable flow.
    g,m=warper(last[0].cpu(),tau[-1].cpu());last_w=F.grid_sample(last,g,align_corners=True)
    hp=lambda x:x-F.avg_pool2d(x,5,1,2)
    detail=masked(hp(p[-1:])-hp(last_w),m)
    x=[float(quality(p).mean()),float(lp(p*2-1,tau*2-1).mean()),float(low),float(detail),float(torch.stack(tmp).mean()),float(torch.stack(excess).mean()),float((edit[1:]-edit[:-1]).abs().mean())]
    candidate_rows.append({'name':name,'x':x})
   records.append({'video':j,'group':d['group'],'phase':phase,'candidates':candidate_rows})
   # Stress controls saved for subsequent GT evaluation, not treated as naturally occurring repairs.
   save(f'{phase}_{j}_stress.pt',stress(tau));print(json.dumps({'stage':'features','video':j,'phase':phase}),flush=True)
 dump('features_no_labels.json',{'names':['musiq','lpips_to_tau','lowfreq_change','unsupported_detail','warp_residual','warp_residual_change','edit_flicker'],'records':records})
 del quality,lp;torch.cuda.empty_cache()
def bootstrap(a):
 a=np.asarray(a);r=np.random.default_rng(42);return {'mean':float(a.mean()),'ci95':np.quantile(a[r.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()}
@torch.no_grad()
def evaluate():
 import lpips
 from scipy.optimize import nnls
 from scipy.stats import spearmanr
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();data=json.loads((OUT/'features_no_labels.json').read_text());rows=data['records']
 for r in rows:
  j=r['video'];phase=r['phase'];target=torch.load(OUT/f'{phase}_{j}_label.pt',weights_only=False)['target'].cuda().float()
  pred=torch.load(OUT/f'{phase}_{j}_candidates.pt',weights_only=False);pred.update(torch.load(OUT/f'{phase}_{j}_stress.pt',weights_only=False))
  for c in r['candidates']:
   p=pred[c['name']].cuda().float();d=p-target
   c['mse']=float(d.square().mean());c['lpips']=float(lp(p*2-1,target*2-1).mean());c['temporal']=float(((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean())
  b=next(c for c in r['candidates'] if c['name']=='tau')
  for c in r['candidates']:
   c['benefit']=b['lpips']-c['lpips']
   c['safe_repair']=bool(c['lpips']<b['lpips']-.0001 and c['mse']<=b['mse']*1.01 and c['temporal']<=b['temporal']*1.01)
 # Monotonic calibration: quality positive, content/temporal penalties nonnegative.
 # Each heldout video excluded from both coefficient fitting and feature scaling.
 modes={'quality':[0],'content':[0,1,2,3],'temporal':[0,4,5,6],'combined':list(range(7))}
 tests=[];folds=[]
 for test in [r for r in rows if r['phase']=='heldout']:
  training=[r for r in rows if r['phase']=='replay' and r['video']!=test['video']]
  natural=lambda r:[c for c in r['candidates'] if not c['name'].startswith('stress')]
  for mode,cols in modes.items():
   if mode=='quality':w=np.ones(1);scale=np.ones(1)
   else:
    xs=[];ys=[]
    for r in training:
     cs=natural(r)
     for i in range(len(cs)):
      for k in range(i):
       x=(np.array(cs[i]['x'])-np.array(cs[k]['x']))*np.array([1,-1,-1,-1,-1,-1,-1]);xs.append(x[cols]);ys.append(cs[i]['benefit']-cs[k]['benefit'])
    xs=np.asarray(xs);ys=np.asarray(ys);scale=xs.std(0).clip(1e-6);x=xs/scale
    w=nnls(np.vstack([x,np.eye(len(cols))*.1]),np.r_[ys,np.zeros(len(cols))])[0]
   base=next(c for c in test['candidates'] if c['name']=='tau')
   scores={c['name']:float((((np.array(c['x'])-np.array(base['x']))*np.array([1,-1,-1,-1,-1,-1,-1]))[cols]/scale)@w) for c in test['candidates']}
   cs=natural(test);best=max(cs,key=lambda c:scores[c['name']]);best=best if scores[best['name']]>1e-8 else base
   rho=spearmanr([scores[c['name']] for c in cs],[c['benefit'] for c in cs]).statistic
   pairs=[(scores[a['name']]-scores[b['name']])*(a['benefit']-b['benefit']) for i,a in enumerate(cs) for b in cs[:i] if abs(a['benefit']-b['benefit'])>1e-5]
   tests.append({'video':test['video'],'group':test['group'],'mode':mode,'selected':best['name'],'scores':scores,'spearman':float(rho) if np.isfinite(rho) else None,'pair_accuracy':float(np.mean([.5 if v==0 else float(v>0) for v in pairs])) if pairs else None,'mse_delta':best['mse']-base['mse'],'lpips_delta':best['lpips']-base['lpips'],'temporal_delta':best['temporal']-base['temporal'],'safe_repair_selected':best['safe_repair'],'harmful_selected':best['lpips']>base['lpips']+.0001 or best['mse']>base['mse']*1.01 or best['temporal']>base['temporal']*1.01,'safe_repairs_available':sum(c['safe_repair'] for c in cs),'stress_rejected':{c['name']:scores[c['name']]<=0 for c in test['candidates'] if c['name'].startswith('stress')}})
   folds.append({'video':test['video'],'mode':mode,'training_videos':[r['video'] for r in training],'weights':w.tolist(),'scale':scale.tolist(),'columns':cols})
 report={'videos':len(tests)//len(modes),'protocol':'leave-original-video-out, fit on other videos replay only; heldout temporal windows never used to fit; no test split use','modes':{},'replay_safe_repairs':sum(c['safe_repair'] for r in rows if r['phase']=='replay' for c in r['candidates'] if not c['name'].startswith('stress')),'heldout_safe_repairs':sum(c['safe_repair'] for r in rows if r['phase']=='heldout' for c in r['candidates'] if not c['name'].startswith('stress'))}
 for mode in modes:
  rs=[r for r in tests if r['mode']==mode];report['modes'][mode]={'accepted':sum(r['selected']!='tau' for r in rs),'safe_selected':sum(r['safe_repair_selected'] for r in rs),'harmful_selected':sum(r['harmful_selected'] for r in rs),'mean_spearman':float(np.mean([r['spearman'] for r in rs if r['spearman'] is not None])) if any(r['spearman'] is not None for r in rs) else None,'mean_pair_accuracy':float(np.mean([r['pair_accuracy'] for r in rs if r['pair_accuracy'] is not None])),'deltas':{m:bootstrap([r[m+'_delta'] for r in rs]) for m in ['mse','lpips','temporal']},'stress_rejected':{n:sum(r['stress_rejected'][n] for r in rs) for n in rs[0]['stress_rejected']}}
 dump('labeled_records.json',rows);dump('fold_parameters.json',folds);dump('heldout_decisions.json',tests);dump('summary.json',report);print(json.dumps(report),flush=True)
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);start=time.time()
 dump('protocol.json',{'purpose':'low-cost preference ranking, not diffusion optimization or training','time_indices_zero_based':{'replay_input':[0,16],'replay_labels':[16,24],'heldout_input':[8,24],'heldout_labels':[24,32]},'calibration':'leave-original-video-out; fit only other videos replay labels; no heldout label fitting','features':'MUSIQ, LPIPS to TAU, low-frequency content change, unsupported detail to last observed frame, fixed TAU-flow warp residual and edit residual/flicker','flow':'Farneback RGB only, same for all candidates, forward-backward mask','safe_repair':'LPIPS gain>0.0001 and MSE/temporal<=1.01 baseline','selection':'max calibrated score relative TAU; reject if <=0; no target gating','stress':'blur/sharpen/independent noise/persistent texture; secondary controls, excluded from fit and natural selection','limits':'7 validation videos, partial sources, no claim of generalization; inability to measure positive recall if no safe repairs','models_frozen':True,'geometry_used':False})
 if not (OUT/'selection.json').exists():prepare()
 candidates()
 if not (OUT/'features_no_labels.json').exists():features()
 evaluate();dump('COMPLETE.json',{'seconds':time.time()-start})
if __name__=='__main__':main()
