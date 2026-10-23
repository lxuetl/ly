import json,time
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL
from evaluate_sdxl_refiner_pilot import OUT,MODEL,down,dump
@torch.no_grad()
def main():
 torch.set_num_threads(4);start=time.time()
 vae=AutoencoderKL.from_pretrained(MODEL,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
 import lpips
 lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval()
 rows=[]
 for item in torch.load(OUT/'inputs.pt',map_location='cpu',weights_only=False):
  tau=item['tau'].cuda().float();target=item['target'].cuda().float();up=F.interpolate(tau,(512,512),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
  frames=[]
  for x in up:frames.append(down((vae.decode(vae.encode(x[None]*2-1).latent_dist.mode()).sample/2+.5).clamp(0,1))[0])
  metrics={}
  for name,pred in {'tau':tau,'resize':down(up),'vae':torch.stack(frames)}.items():
   metrics[name]={'mse':float((pred-target).square().mean()),'lpips':float(lp(pred*2-1,target*2-1).mean()),'temporal':float(((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean())}
  rows.append({'path':item['path'],'metrics':metrics})
 summary={'clips':len(rows),'seconds':time.time()-start,'means':{n:{k:float(np.mean([r['metrics'][n][k] for r in rows])) for k in ['mse','lpips','temporal']} for n in ['tau','resize','vae']}}
 dump('vae_check.json',{'summary':summary,'rows':rows});print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
