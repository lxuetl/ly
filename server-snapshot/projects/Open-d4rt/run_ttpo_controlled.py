"""Small SDXL adaptation of TTPO; frozen models, no future-GT preference labels."""
import os,json,time,types,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput
import pyiqa,lpips
SRC=Path('/root/autodl-fs/d4rt_training/sdxl_refiner_pilot_20261014')
OUT=SRC.parent/'ttpo_controlled_20261014'
MODEL='/root/autodl-fs/models/sdxl-refiner-1.0'
ARMS={'plain':(0.,False,False),'structure':(1000.,False,False),'pref100':(100.,True,False),'pref1000':(1000.,True,False),'reverse1000':(1000.,True,True)}
def dump(name,x):
 p=OUT/name;p.with_suffix('.tmp').write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def up(x):return F.interpolate(x.float(),(512,512),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
def down(x):return F.interpolate(x.float(),(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
def bands(z):
 fy=torch.fft.fftfreq(z.shape[-2],device=z.device)*2;fx=torch.fft.fftfreq(z.shape[-1],device=z.device)*2
 mask=torch.exp(-(fy[:,None]**2+fx[None,:]**2)/(2*.9**2))
 low=torch.fft.ifft2(torch.fft.fft2(z,norm='ortho')*mask,norm='ortho').real
 return low,z-low
def ci(x):
 a=np.asarray(x);r=np.random.default_rng(42);return {'mean':float(a.mean()),'ci95':np.quantile(a[r.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()}
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);start=time.time()
 dump('protocol.json',{'kind':'TTPO-inspired SDXL adaptation, NOT exact FLUX paper reproduction','candidate_pool':['tau','s0.1_seed42','s0.1_seed43','s0.2_seed42','s0.2_seed43'],'preference':'MUSIQ only, framewise argmax/argmin; fixed before generation; no target access','input':'same 9 validation videos as prior pilot','arms':ARMS,'scheduler':'EulerDiscrete epsilon, use scheduler predicted original sample and stop-gradient denoiser','steps':50,'strength':.2,'effective_steps':10,'seed':42,'guidance_scale':3,'beta':1,'alpha':.5,'frequency':'Gaussian real-space low/high decomposition via FFT, normalized Nyquist cutoff .9','stage':'first30% structure only; middle60% both; final10% preference only; normalized active steps (adaptation)','guidance':'subtract g * gradient of mean losses from scheduler prev_sample; no gradient normalization','postprocessing':'native plus fixed .25 VAE-roundtrip-subtracted residual for EVERY arm','primary_comparison':'pref1000 vs plain; pref100 is prespecified sensitivity; report both without test selection','target_usage':'evaluation only after all outputs saved','geometry':False,'temporal_guidance':False,'limitations':'single NR-IQA; cached128 candidate re-encoding; framewise editing; exploratory small validation set'})
 items=torch.load(SRC/'inputs.pt',weights_only=False,map_location='cpu')
 quality=pyiqa.create_metric('musiq',device='cuda').eval()
 names=['tau','s0.1_seed42','s0.1_seed43','s0.2_seed42','s0.2_seed43'];prefs=[]
 for i,item in enumerate(items):
  cache=torch.load(SRC/f'clip_{i:02d}.pt',weights_only=False,map_location='cpu')
  scores=torch.stack([quality(cache[n].cuda().float()).flatten() for n in names])
  assert scores.shape==(5,8) and torch.isfinite(scores).all()
  prefs.append({'scores':scores.cpu().tolist(),'win':scores.argmax(0).cpu().tolist(),'lose':scores.argmin(0).cpu().tolist()})
 dump('preferences.json',{'names':names,'clips':prefs});del quality;torch.cuda.empty_cache()
 pipe=StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL,torch_dtype=torch.float16,variant='fp16',local_files_only=True,add_watermarker=False).to('cuda')
 pipe.set_progress_bar_config(disable=True)
 for component in [pipe.unet,pipe.vae,pipe.text_encoder_2]:component.eval().requires_grad_(False)
 assert pipe.scheduler.config.prediction_type=='epsilon'
 original_step=pipe.scheduler.step;state={};trace=[]
 def guided_step(self,model_output,timestep,sample,**kwargs):
  # Run the actual Euler scheduler first; its x0 estimate supplies scheduler-correct scaling.
  result=original_step(model_output,timestep,sample,return_dict=True,**{k:v for k,v in kwargs.items() if k!='return_dict'})
  gain,use_pref,reverse=ARMS[state['arm']];j=state['j'];state['j']+=1
  with torch.enable_grad():
   z=sample.detach().float().requires_grad_(True)
   clean=z+(result.pred_original_sample.float()-sample.float()).detach()
   low,high=bands(clean);base_low,_=bands(state['base'])
   _,win_high=bands(state['lose'] if reverse else state['win']);_,lose_high=bands(state['win'] if reverse else state['lose'])
   preference=F.softplus((high-win_high).abs().mean()-(high-lose_high).abs().mean())
   structure=(low-base_low).square().mean()
   # Same structure term in forward/reverse arms; structure-only control lacks preference.
   loss=(structure if j<9 else structure*0)+(.5*preference if use_pref and j>=3 else preference*0)
   grad=torch.autograd.grad(loss,z)[0];delta=-gain*grad
  assert torch.isfinite(delta).all()
  updated=result.prev_sample if gain==0 else (result.prev_sample.float()+delta).to(result.prev_sample.dtype)
  trace.append({'clip':state['clip'],'frame':state['frame'],'arm':state['arm'],'step':j,'preference':float(preference),'structure':float(structure),'gradient_rms':float(grad.square().mean().sqrt()),'update_rms':float(delta.square().mean().sqrt())})
  if kwargs.get('return_dict',True):return EulerDiscreteSchedulerOutput(prev_sample=updated,pred_original_sample=result.pred_original_sample)
  return (updated,result.pred_original_sample)
 pipe.scheduler.step=types.MethodType(guided_step,pipe.scheduler)
 for i,item in enumerate(items):
  if (OUT/f'clip_{i:02d}.pt').exists():continue
  tau=item['tau'].cuda().float();cache=torch.load(SRC/f'clip_{i:02d}.pt',weights_only=False,map_location='cpu')
  pipe.vae.to(dtype=torch.float32);encoded={}
  for n in names:encoded[n]=pipe.vae.encode(up(cache[n].cuda())*2-1).latent_dist.mode()*pipe.vae.config.scaling_factor
  pipe.vae.to(dtype=torch.float16);preds={'tau':tau};full=up(tau)
  pipe.scheduler.step=original_step
  direct=pipe(prompt='a clear natural image, realistic colors and details',negative_prompt='blur, artifacts, distorted shapes, oversharpening, text, watermark',image=full[:1],strength=.2,num_inference_steps=50,guidance_scale=3.,generator=torch.Generator('cuda').manual_seed(42),output_type='pt').images
  direct=down(direct)
  pipe.scheduler.step=types.MethodType(guided_step,pipe.scheduler)
  for arm in ARMS:
   frames=[];tick=time.time()
   for f in range(8):
    state.update(arm=arm,j=0,clip=i,frame=f,base=encoded['tau'][f:f+1],win=encoded[names[prefs[i]['win'][f]]][f:f+1],lose=encoded[names[prefs[i]['lose'][f]]][f:f+1])
    y=pipe(prompt='a clear natural image, realistic colors and details',negative_prompt='blur, artifacts, distorted shapes, oversharpening, text, watermark',image=full[f:f+1],strength=.2,num_inference_steps=50,guidance_scale=3.,generator=torch.Generator('cuda').manual_seed(42),output_type='pt').images
    assert state['j']==10 and torch.isfinite(y).all();frames.append(down(y)[0])
   preds[arm]=torch.stack(frames)
   preds[arm+'_residual']=(tau+.25*(preds[arm]-cache['vae_sample42'].cuda().float())).clamp(0,1)
   print(json.dumps({'clip':i,'arm':arm,'seconds':time.time()-tick,'elapsed':time.time()-start}),flush=True)
  # Cache is float16, allowing <= half-precision quantization error for the g=0 identity check.
  parity=float((preds['plain']-cache['s0.2_seed42'].cuda().float()).abs().max())
  live_parity=float((preds['plain'][:1]-direct).abs().max())
  assert live_parity<1e-5,('live zero-guidance parity failed',live_parity)
  torch.save({n:v.cpu().half() for n,v in preds.items()},OUT/f'clip_{i:02d}.pt')
  dump(f'trace_{i:02d}.json',{'plain_max_abs_vs_prior':parity,'plain_max_abs_vs_live_pipeline':live_parity,'steps':[t for t in trace if t['clip']==i]})
  dump('progress.json',{'done':i+1,'total':len(items),'seconds':time.time()-start})
 del pipe;torch.cuda.empty_cache()
 # Separate terminal phase: only now use future targets to calculate metrics.
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for i,item in enumerate(items):
  preds=torch.load(OUT/f'clip_{i:02d}.pt',weights_only=False);target=item['target'].cuda().float();metrics={}
  for n,p in preds.items():
   p=p.cuda().float();d=p-target
   metrics[n]={'mse':float(d.square().mean()),'mae':float(d.abs().mean()),'lpips':float(lp(p*2-1,target*2-1).mean()),'temporal':float(((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean())}
  rows.append({'path':item['path'],'group':item['group'],'metrics':metrics})
 dump('per_video.json',rows);summary={'n':len(rows),'seconds':time.time()-start,'means':{},'paired':{}}
 for n in rows[0]['metrics']:
  summary['means'][n]={m:float(np.mean([r['metrics'][n][m] for r in rows])) for m in metrics[n]}
  for baseline in ['tau','plain' if not n.endswith('_residual') else 'plain_residual']:
   summary['paired'][n+'-vs-'+baseline]={m:ci([r['metrics'][n][m]-r['metrics'][baseline][m] for r in rows]) for m in metrics[n]}
 dump('summary.json',summary);dump('COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
