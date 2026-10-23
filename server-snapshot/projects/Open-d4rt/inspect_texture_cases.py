import json,time
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw
import test_texture_locked as q
a=q.a;m=q.m
OUT=a.t.ROOT/'texture_visual_audit_20261015'
def delta(r,k):return r['metrics']['cubic1'][k]-r['metrics']['linear4'][k]
def main():
 a.setup();OUT.mkdir(exist_ok=False)
 rows=[json.loads(s) for s in (q.OUT/'per_clip.jsonl').read_text().splitlines()]
 chosen=[];seen=set()
 for label,rr,count in [('worst_mse',sorted(rows,key=lambda r:delta(r,'mse'),reverse=True),2),('adt',sorted([r for r in rows if r['group'][0]=='adt_mini'],key=lambda r:delta(r,'mse'),reverse=True),2),('lpips_regression',sorted(rows,key=lambda r:delta(r,'lpips'),reverse=True),1),('best_lpips',sorted(rows,key=lambda r:delta(r,'lpips')),1)]:
  n=0
  for r in rr:
   if tuple(r['group']) in seen:continue
   chosen.append((label,r));seen.add(tuple(r['group']));n+=1
   if n==count:break
 stats={'clips':len(rows),'mse_regression_clips':sum(delta(r,'mse')>0 for r in rows),'lpips_regression_clips':sum(delta(r,'lpips')>0 for r in rows),'temporal_regression_clips':sum(delta(r,'temporal')>0 for r in rows),'selection':'Posthoc diagnostic extremes, unique original videos, not representative estimates; no parameter updates','cases':[]}
 net=a.network()
 def pil(x):return Image.fromarray((x.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8'))
 with torch.no_grad():
  for idx,(label,r) in enumerate(chosen):
   rgb,_=a.e.rgb24(r['path']);ctx=rgb[:16].cuda();_,hid,skip=a.t.core.extract_frozen_features(net,ctx[None]);tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float();ctx=ctx.half().float();d=a.features(ctx,tau);p,_=m.alternatives(ctx,tau,d);gt=rgb[16:24].cuda().half().float()
   canvas=Image.new('RGB',(1536,880),'white');draw=ImageDraw.Draw(canvas)
   draw.text((8,5),f'{idx} {label} {r["group"]} | deltaMSE={delta(r,"mse"):.8f} deltaLPIPS={delta(r,"lpips"):.6f}',fill='black')
   for j,h in enumerate([0,3,7]):
    diff=(p['cubic1'][h]-gt[h]).square().mean(0)-(p['linear4'][h]-gt[h]).square().mean(0)
    # Fixed symmetric range for all cases. Red=worse; blue=better.
    z=(diff/.002).clamp(-1,1);heat=torch.stack([1+z.clamp(max=0),1-z.abs(),1-z.clamp(min=0)])
    ims=[ctx[-1],gt[h],tau[h],p['linear4'][h],p['cubic1'][h],heat]
    names=['Last observed','Future GT','TAU','Original linear4','Fixed cubic1','MSE change +/-0.002']
    y=35+j*280
    for col,(im,name) in enumerate(zip(ims,names)):
     draw.text((col*256+4,y),f'h={h+1} {name}',fill='black');canvas.paste(pil(im).resize((256,256),Image.Resampling.NEAREST),(col*256,y+20))
   canvas.save(OUT/f'case_{idx}.png')
   # Crop highest mean absolute correction difference, selected for diagnosis only.
   h=7;change=(p['cubic1'][h]-p['linear4'][h]).abs().mean(0)
   score=torch.nn.functional.avg_pool2d(change[None,None],32,1)[0,0];pos=int(score.argmax());yy=pos//score.shape[1];xx=pos%score.shape[1]
   zoom=Image.new('RGB',(1280,300),'white');dr=ImageDraw.Draw(zoom)
   for col,(im,name) in enumerate([(gt[h],'GT'),(tau[h],'TAU'),(p['linear4'][h],'Original'),(p['cubic1'][h],'Fixed'),((p['cubic1'][h]-p['linear4'][h])*20+.5,'Change x20 +0.5')]):
    dr.text((col*256+4,5),name,fill='black');zoom.paste(pil(im).crop((xx,yy,xx+32,yy+32)).resize((256,256),Image.Resampling.NEAREST),(col*256,30))
   zoom.save(OUT/f'zoom_{idx}.png');stats['cases'].append({'id':idx,'label':label,'group':r['group'],'path':r['path'],'metrics':r['metrics'],'crop_xy':[xx,yy],'horizon':8})
   print(json.dumps({'case':idx,'group':r['group']}),flush=True)
 (OUT/'summary.json').write_text(json.dumps(stats,indent=2))
if __name__=='__main__':main()
