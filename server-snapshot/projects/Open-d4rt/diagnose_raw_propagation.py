"""Raw propagation source audit on validation only; targets used for metrics/plots."""
import json,time
from collections import defaultdict
import torch,numpy as np,lpips
from PIL import Image,ImageDraw
import run_texture_mechanism as m
a=m.a
OUT=a.t.ROOT/'raw_propagation_val_20261015'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
@torch.no_grad()
def inputs(i,path):
 d={k:v.cuda() for k,v in torch.load(a.OUT/'cache'/f'{i:04d}.pt',weights_only=False).items()}
 rgb,_=a.e.rgb24(path);last=rgb[15:16].cuda().half().float();src=[];valids=[]
 for t in range(8):
  fl,valid=a.h.maps(a.h.flow(d['tau'][t],last[0]),a.h.flow(last[0],d['tau'][t]));src.append(m.warp(last,fl,'bicubic'));valids.append(valid)
 return d,torch.cat(src),last.expand(8,-1,-1,-1),torch.cat(valids)
def draw_case(i,path,name):
 d,src,last,valid=inputs(i,path);ims={'Last':last,'TAU':d['tau'],'Raw propagated':src,'Future GT':d['target'],'Gate':d['gate'].expand(-1,3,-1,-1)}
 canvas=Image.new('RGB',(1280,865),'white');dr=ImageDraw.Draw(canvas)
 for j,t in enumerate([0,3,7]):
  for c,(label,seq) in enumerate(ims.items()):
   dr.text((c*256+4,j*285+3),f'h={t+1} {label}',fill='black')
   ar=(seq[t].cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8');canvas.paste(Image.fromarray(ar).resize((256,256),Image.Resampling.NEAREST),(c*256,j*285+24))
 canvas.save(OUT/(name+'.png'))
@torch.no_grad()
def main():
 a.setup();OUT.mkdir(exist_ok=False);start=time.time();paths=json.loads((a.OUT/'manifest.json').read_text())['val'];lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
 save('protocol.json',{'split':'validation only','clips':len(paths),'source':'last observed frame warped toward TAU forecast via unchanged Farneback and bicubic clamp; no GT flow','comparators':['TAU','raw source','last observed zero-motion'],'regions':'whole image, gate-weighted, gate>0.1 intersect in-bounds/cycle confidence>=exp(-0.5); threshold fixed diagnostic, not tuned','metrics':'MSE/MAE/lowfrequency MSE, full-image LPIPS, per-horizon; regional LPIPS not computed to avoid receptive-field ambiguity','caveats':'gate uses original four-frame confidence; image motion includes camera motion; no GT visibility labels; no deployment tuning; best/worst figures selected posthoc','script_sha256':a.e.sha(__file__)})
 rows=[]
 for i,path in enumerate(paths):
  d,src,last,valid=inputs(i,path);y=d['target'];mask=((d['gate']>.1)&(valid>=np.exp(-.5))).float();weights={'all':torch.ones_like(mask),'gate_weighted':d['gate'],'reliable':mask};metrics={}
  for name,p in [('tau',d['tau']),('source',src),('zero',last)]:
   err=(p-y);lowerr=(a.h.low(p)-a.h.low(y)).square().mean(1,keepdim=True)
   mm={'lpips':float(lp(p*2-1,y*2-1).mean())}
   for region,w in weights.items():
    den=w.sum((1,2,3));ms=(err.square().mean(1,keepdim=True)*w).sum((1,2,3))/den.clamp_min(1e-8)
    mm[region+'_mse']=float(ms.mean());mm[region+'_horizon_mse']=ms.cpu().tolist();mm[region+'_mae']=float(((err.abs().mean(1,keepdim=True)*w).sum((1,2,3))/den.clamp_min(1e-8)).mean());mm[region+'_low_mse']=float(((lowerr*w).sum((1,2,3))/den.clamp_min(1e-8)).mean())
   metrics[name]=mm
  row={'index':i,'path':path,'group':list(a.t.prev.group(path)),'metrics':metrics,'coverage':float(mask.mean()),'empty_reliable_frames':int((mask.sum((1,2,3))==0).sum()),'source_pixel_win_fraction':float(((src-y).square().mean(1)<(d['tau']-y).square().mean(1)).float().mean())};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%40==0:print(json.dumps({'done':i+1,'total':len(paths),'seconds':time.time()-start}),flush=True)
 assert sum(r['empty_reliable_frames'] for r in rows)==0,'Empty regional masks: do not interpret zero errors; inspect per_clip'
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 ci={k:a.h.ci([v['metrics']['source'][k]-v['metrics']['tau'][k] for v in videos]) for k in ['all_mse','reliable_mse','gate_weighted_mse','all_low_mse','reliable_low_mse','lpips']}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'paired_ci_source_minus_tau':ci,'per_source':{s:a.aggregate(v) for s,v in sources.items()},'coverage_clip_mean':float(np.mean([r['coverage'] for r in rows])),'video_wins':{k:sum(v['metrics']['source'][k]<v['metrics']['tau'][k] for v in videos) for k in ['all_mse','reliable_mse','lpips']}})
 ranked=sorted(rows,key=lambda r:r['metrics']['source']['all_mse']-r['metrics']['tau']['all_mse']);chosen=[('best_source',ranked[0]),('worst_source',ranked[-1])];adt=[r for r in rows if r['group'][0]=='adt_mini']
 if adt:chosen.append(('adt',max(adt,key=lambda r:r['metrics']['source']['all_mse']-r['metrics']['tau']['all_mse'])))
 for name,r in chosen:draw_case(r['index'],r['path'],name)
 save('visual_cases.json',{name:r for name,r in chosen});save('COMPLETE.json',{'seconds':time.time()-start});print('COMPLETE',flush=True)
if __name__=='__main__':main()
