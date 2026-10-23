"""SDXL candidate upper-bound pilot. No future labels in generation or prompts."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from diffusers import StableDiffusionXLImg2ImgPipeline
OUT=Path('/root/autodl-fs/d4rt_training/sdxl_refiner_pilot_20261014')
MODEL='/root/autodl-fs/models/sdxl-refiner-1.0'
CONFIGS=[(.1,42),(.1,43),(.2,42),(.2,43)]
def dump(name,data):
 p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,p)
def down(x):return F.interpolate(x,size=(128,128),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
def image(x):return Image.fromarray((x.detach().float().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))
def ci(a):
 a=np.array(a);rng=np.random.default_rng(42)
 return {'mean':float(a.mean()),'ci95':np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()}

@torch.no_grad()
def main():
 torch.set_num_threads(4);start=time.time();items=torch.load(OUT/'inputs.pt',map_location='cpu',weights_only=False)
 dump('protocol.json',{'model':MODEL,'weights_frozen':True,'resolution':512,'evaluation_resolution':128,'steps_nominal':50,'effective_steps':[5,10],'guidance_scale':3.,'configs':CONFIGS,'candidate_unit':'8-frame sequence, same configuration and same reset seed per frame; no temporal model','controls':['tau','resize_only','vae_mode','vae_sample42','vae_sample43'],'guardrail':'oracle diagnostic requires each clip MSE and temporal MAE <=1.01 times TAU; no test-time GT selection claimed','target_usage':'evaluation only','geometry':False,'prompt':'a clear natural image, realistic colors and details','negative_prompt':'blur, artifacts, distorted shapes, oversharpening, text, watermark','selection':'validation, max2 videos per source; not representative full-dataset benchmark'})
 protocol=json.loads((OUT/'protocol.json').read_text())
 protocol['exploratory_extension']={'reason':'added after validation-only VAE roundtrip audit, before diffusion candidate evaluation','formula':'tau + alpha * (edited - matched-seed VAE roundtrip)','alphas':[.25,.5],'no_test_data':True}
 dump('protocol.json',protocol)
 pipe=StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL,torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True,add_watermarker=False).to('cuda')
 pipe.set_progress_bar_config(disable=True)
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
 rows=[]
 for index,item in enumerate(items):
  output=OUT/f'clip_{index:02d}.pt';rowfile=OUT/f'clip_{index:02d}.json'
  if rowfile.exists():rows.append(json.loads(rowfile.read_text()));continue
  tau=item['tau'].cuda().float();up=F.interpolate(tau,(512,512),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
  predictions={'tau':tau,'resize_only':down(up)}
  pipe.vae.to(dtype=torch.float32)
  controls={n:[] for n in ['vae_mode','vae_sample42','vae_sample43']}
  for frame in up:
   dist=pipe.vae.encode(frame[None]*2-1).latent_dist
   for name in controls:
    latent=dist.mode() if name=='vae_mode' else dist.sample(generator=torch.Generator('cuda').manual_seed(int(name[-2:])))
    reconstructed=(pipe.vae.decode(latent).sample/2+.5).clamp(0,1)
    controls[name].append(down(reconstructed)[0])
  predictions.update({n:torch.stack(v) for n,v in controls.items()});pipe.vae.to(dtype=torch.float16)
  for strength,seed in CONFIGS:
   name=f's{strength}_seed{seed}';frames=[];tick=time.time()
   for frame in up:
    edited=pipe(prompt='a clear natural image, realistic colors and details',negative_prompt='blur, artifacts, distorted shapes, oversharpening, text, watermark',image=frame[None],strength=strength,num_inference_steps=50,guidance_scale=3.,generator=torch.Generator('cuda').manual_seed(seed),output_type='pt').images
    assert torch.isfinite(edited).all(),name
    frames.append(down(edited.float())[0])
   predictions[name]=torch.stack(frames)
   print(json.dumps({'clip':index+1,'candidate':name,'seconds':time.time()-tick,'elapsed':time.time()-start}),flush=True)
  # Exploratory validation-only arm, added after the separate VAE control audit.
  # Subtract matched-seed VAE roundtrip before transferring edit residual to TAU.
  for strength,seed in CONFIGS:
   name=f's{strength}_seed{seed}'
   for alpha in [.25,.5]:
    predictions[f'r{alpha}_{name}']=(tau+alpha*(predictions[name]-predictions[f'vae_sample{seed}'])).clamp(0,1)
  target=item['target'].cuda().float();metrics={}
  for name,pred in predictions.items():
   diff=pred-target;metrics[name]={'mse':float(diff.square().mean()),'mae':float(diff.abs().mean()),'lpips':float(lp(pred*2-1,target*2-1).mean()),'temporal':float(((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()),'delta_from_tau':float((pred-tau).abs().mean())}
  base=metrics['tau'];candidates=[n for n in predictions if n.startswith(('s0.','r0.'))]
  eligible=[n for n in candidates if metrics[n]['mse']<=base['mse']*1.01 and metrics[n]['temporal']<=base['temporal']*1.01]
  oracle=min(['tau']+eligible,key=lambda n:metrics[n]['lpips'])
  row={'index':index,'path':item['path'],'group':item['group'],'metrics':metrics,'oracle_guarded':oracle,'oracle_unconstrained':min(['tau']+candidates,key=lambda n:metrics[n]['lpips']),'eligible':eligible}
  torch.save({n:p.cpu().half() for n,p in predictions.items()},output);dump(f'clip_{index:02d}.json',row);rows.append(row)
  # Fixed visualization: all candidates + controls, frames 1/4/8, not metric-selected.
  names=['target']+[n for n in predictions if not n.startswith('r0.')];canvas=Image.new('RGB',(3*256+180,len(names)*256),'white');draw=ImageDraw.Draw(canvas)
  for r,name in enumerate(names):
   draw.text((5,r*256+10),name,fill='black');seq=target if name=='target' else predictions[name]
   for c,j in enumerate([0,3,7]):canvas.paste(image(seq[j]).resize((256,256)),(180+c*256,r*256))
  canvas.save(OUT/f'contact_{index:02d}.jpg',quality=92)
  dump('progress.json',{'clips':index+1,'total':len(items),'seconds':time.time()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30})
 summary={'clips':len(rows),'means':{},'delta_ci':{},'guarded_oracle':{},'seconds':time.time()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30}
 for name in rows[0]['metrics']:
  summary['means'][name]={m:float(np.mean([r['metrics'][name][m] for r in rows])) for m in ['mse','lpips','temporal','mae','delta_from_tau']}
  summary['delta_ci'][name]={m:ci([r['metrics'][name][m]-r['metrics']['tau'][m] for r in rows]) for m in ['mse','lpips','temporal']}
 summary['guarded_oracle']={'improved_clips':sum(r['oracle_guarded']!='tau' for r in rows),'unconstrained_improved_clips':sum(r['oracle_unconstrained']!='tau' for r in rows),'chosen':[r['oracle_guarded'] for r in rows],'means':{m:float(np.mean([r['metrics'][r['oracle_guarded']][m] for r in rows])) for m in ['mse','lpips','temporal']},'warning':'uses future GT only to measure candidate ceiling; NOT deployable selector'}
 dump('summary.json',summary);dump('COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
