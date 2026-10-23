"""Frozen TAU + historical texture transport. Candidate ceiling, not GT-guided inference."""
import json,time,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2,lpips
SRC=Path('/root/autodl-fs/d4rt_training/replay_preference_20261014')
OUT=SRC.parent/'history_texture_candidates_20261015'
ALPHAS=[.03,.1,.25]
def dump(n,d):
 p=OUT/n;p.with_suffix('.tmp').write_text(json.dumps(d,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def flow(a,b):
 def gray(x):return cv2.cvtColor((x.permute(1,2,0).cpu().numpy().clip(0,1)*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
 return cv2.calcOpticalFlowFarneback(gray(a),gray(b),None,.5,3,15,3,5,1.2,0)
def maps(back,forward):
 yy,xx=np.mgrid[:128,:128].astype('float32');mx=xx+back[...,0];my=yy+back[...,1]
 ff=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 err=np.linalg.norm(back+ff,axis=-1)
 good=((mx>=0)&(mx<=127)&(my>=0)&(my<=127)).astype('float32')*np.exp(-err**2/(2*1.5**2))
 return torch.tensor(back).permute(2,0,1)[None].cuda(),torch.tensor(good)[None,None].cuda()
def warp(x,fl):
 yy,xx=torch.meshgrid(torch.arange(128,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
 grid=torch.stack([xx+fl[:,0],yy+fl[:,1]],-1)/127*2-1
 return F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=True)
def low(x):return F.avg_pool2d(F.pad(x,(2,2,2,2),mode='reflect'),5,1)
def bound(x,tau):
 delta=x.clamp(-.03,.03);scale=(.001/delta.abs().mean().clamp_min(1e-8)).clamp(max=1)
 return (tau+delta*scale).clamp(0,1)
def ci(x):
 x=np.asarray(x);r=np.random.default_rng(42);return {'mean':float(x.mean()),'ci95':np.quantile(x[r.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist()}
def safe(c,b):return c['lpips']<b['lpips']-.0001 and c['mse']<=b['mse']*1.01 and c['temporal']<=b['temporal']*1.01
@torch.no_grad()
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);start=time.time()
 dump('protocol.json',{'frozen':'TAU; no editor training; no D4RT','input':'same7 videos, same replay and heldout windows as replay_preference_20261014','branches':['TAU-guided dense alignment of last4 historical frames','constant-velocity extrapolation from last2 historical frames'], 'alignment':'Farneback RGB; no target flow','confidence':'forward/backward consistency + bounds + lowfrequency appearance consistency + historical texture disagreement','correction':['historical high-frequency residual','historical RGB residual'],'alphas':ALPHAS,'cap':.03,'mean_abs_budget_max':.001,'controls':'zero and reversed displacement, same source, coefficients, gate and caps; realized budget reported, not forced equal','target_usage':'only terminal metrics; oracle diagnostic labeled; deployment config leave-video-out selected on other videos replay','safe':'LPIPS gain>0.0001; MSE and temporal at most1% worse','limits':'7 videos4sources; inferred dense motion not ground truth; no visibility claimed for disoccluded areas; no full benchmark'})
 # Translation unit check: positive backward x flow samples x+1.
 ramp=torch.arange(128,device='cuda')[None,None,None,:].float().expand(1,1,128,128)
 f=torch.zeros(1,2,128,128,device='cuda');f[:,0]=1
 assert (warp(ramp,f)[...,:-1]-ramp[...,1:]).abs().max()<1e-4
 selected=json.loads((SRC/'selection.json').read_text())['selected']
 for item in selected:
  j=item['index']
  for phase in ['replay','heldout']:
   out=OUT/f'{phase}_{j}.pt'
   if out.exists():continue
   inp=torch.load(SRC/f'{phase}_{j}_input.pt',weights_only=False);ctx=inp['context'].cuda().float();tau=inp['tau'].cuda().float()
   fields={};gates={};sources={};diag={}
   # Direct correspondence anchors geometry to the baseline's predicted layout.
   aligned={m:[] for m in ['correct','zero','reverse']};allgate=[]
   for h in range(8):
    warped={m:[] for m in aligned};weights=[]
    for k in range(12,16):
     b=flow(tau[h],ctx[k]);fw=flow(ctx[k],tau[h]);fl,valid=maps(b,fw)
     cor=warp(ctx[k:k+1],fl)
     appearance=(low(cor)-low(tau[h:h+1])).abs().mean(1,keepdim=True)
     conf=valid*torch.exp(-appearance/.06)
     weights.append(conf)
     for mode,sign in [('correct',1),('zero',0),('reverse',-1)]:warped[mode].append(warp(ctx[k:k+1],fl*sign))
    ws=torch.stack(weights);den=ws.sum(0).clamp_min(1e-6)
    fused={m:(torch.stack(v)*ws).sum(0)/den for m,v in warped.items()}
    disagreement=((torch.stack(warped['correct'])-fused['correct']).abs().mean(2,keepdim=True)*ws).sum(0)/den
    gate=ws.mean(0)*torch.exp(-disagreement/.04)
    allgate.append(gate)
    for mode in aligned:aligned[mode].append(fused[mode])
   sources['align']={m:torch.cat(v) for m,v in aligned.items()};gates['align']=torch.cat(allgate)
   # Historical velocity field uses no predicted or actual future frames to estimate motion.
   back=flow(ctx[-1],ctx[-2]);fw=flow(ctx[-2],ctx[-1]);basefl,basevalid=maps(back,fw)
   motion={m:[] for m in aligned};masks=[]
   for h in range(8):
    fl=basefl*(h+1);cor=warp(ctx[-1:],fl)
    yy,xx=torch.meshgrid(torch.arange(128,device='cuda'),torch.arange(128,device='cuda'),indexing='ij')
    inside=((xx+fl[:,0]>=0)&(xx+fl[:,0]<=127)&(yy+fl[:,1]>=0)&(yy+fl[:,1]<=127))[:,None]
    conf=basevalid*inside*torch.exp(-(low(cor)-low(tau[h:h+1])).abs().mean(1,keepdim=True)/.06)
    conf=conf/(1+.15*h);masks.append(conf)
    for mode,sign in [('correct',1),('zero',0),('reverse',-1)]:motion[mode].append(warp(ctx[-1:],fl*sign))
   sources['velocity']={m:torch.cat(v) for m,v in motion.items()};gates['velocity']=torch.cat(masks)
   pred={'tau':tau}
   for branch in sources:
    diag[branch]={'mean_confidence':float(gates[branch].mean()),'coverage_above_.1':float((gates[branch]>.1).float().mean())}
    for mode,src in sources[branch].items():
     for kind in ['texture','rgb']:
      residual=(src-low(src))-(tau-low(tau)) if kind=='texture' else src-tau
      for alpha in ALPHAS:pred[f'{branch}_{kind}_{mode}_a{alpha}']=bound(alpha*gates[branch]*residual,tau)
   assert all(torch.isfinite(p).all() for p in pred.values())
   torch.save({n:p.cpu().half() for n,p in pred.items()},out);dump(f'{phase}_{j}_diagnostics.json',diag)
   print(json.dumps({'stage':'generation','video':j,'phase':phase,'seconds':time.time()-start}),flush=True)
 # All candidates are finalized before labels are read.
 lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False);rows=[]
 for item in selected:
  for phase in ['replay','heldout']:
   j=item['index'];pred=torch.load(OUT/f'{phase}_{j}.pt',weights_only=False)
   target=torch.load(SRC/f'{phase}_{j}_label.pt',weights_only=False)['target'].cuda().float();tau=pred['tau'].cuda().float();metrics={}
   for n,p in pred.items():
    p=p.cuda().float();metrics[n]={'mse':float((p-target).square().mean()),'lpips':float(lp(p*2-1,target*2-1).mean()),'temporal':float(((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean()),'budget':float((p-tau).abs().mean())}
   b=metrics['tau'];correct=[n for n in metrics if '_correct_' in n]
   positives=[n for n in correct if safe(metrics[n],b)]
   rows.append({'video':j,'group':item['group'],'phase':phase,'metrics':metrics,'safe_correct_candidates':positives,'oracle_safe':min(['tau']+positives,key=lambda n:metrics[n]['lpips'])})
 names=list(rows[0]['metrics']);tests=[r for r in rows if r['phase']=='heldout'];summary={'seconds':time.time()-start,'videos':len(tests),'phase_counts':{},'heldout_configs':{},'deployable_crossfit':[],'paired_controls':{}}
 for phase in ['replay','heldout']:
  rs=[r for r in rows if r['phase']==phase]
  summary['phase_counts'][phase]={'videos_with_safe_correct':sum(bool(r['safe_correct_candidates']) for r in rs),'safe_correct_count':sum(len(r['safe_correct_candidates']) for r in rs),'correct_candidate_count':len(rs)*12,'videos_with_safe_zero':sum(any(safe(v,r['metrics']['tau']) for n,v in r['metrics'].items() if '_zero_' in n) for r in rs),'oracle_choices':[r['oracle_safe'] for r in rs],'oracle_deltas':{m:ci([r['metrics'][r['oracle_safe']][m]-r['metrics']['tau'][m] for r in rs]) for m in ['mse','lpips','temporal']},'oracle_warning':'uses labels to measure ceiling, NOT deployable selection'}
 for n in names:
  summary['heldout_configs'][n]={'safe_videos':sum(safe(r['metrics'][n],r['metrics']['tau']) for r in tests),'mean_budget':float(np.mean([r['metrics'][n]['budget'] for r in tests])),'delta':{m:ci([r['metrics'][n][m]-r['metrics']['tau'][m] for r in tests]) for m in ['mse','lpips','temporal']}}
  if '_correct_' in n:
   for control in ['zero','reverse']:
    other=n.replace('_correct_',f'_{control}_');summary['paired_controls'][n+'-vs-'+control]={m:ci([r['metrics'][n][m]-r['metrics'][other][m] for r in tests]) for m in ['mse','lpips','temporal']}
 for r in tests:
  train=[a for a in rows if a['phase']=='replay' and a['video']!=r['video']]
  eligible=['tau']
  for n in names:
   if '_correct_' not in n:continue
   b={m:float(np.mean([x['metrics']['tau'][m] for x in train])) for m in ['mse','lpips','temporal']};c={m:float(np.mean([x['metrics'][n][m] for x in train])) for m in b}
   if safe(c,b):eligible.append(n)
  chosen=min(eligible,key=lambda n:np.mean([x['metrics'][n]['lpips'] for x in train]));b=r['metrics']['tau'];c=r['metrics'][chosen]
  summary['deployable_crossfit'].append({'video':r['video'],'group':r['group'],'chosen':chosen,'safe':safe(c,b),'delta':{m:c[m]-b[m] for m in ['mse','lpips','temporal']}})
 dump('per_video.json',rows);dump('summary.json',summary);dump('COMPLETE.json',{'seconds':time.time()-start});print(json.dumps({'complete':True,'seconds':time.time()-start,'counts':summary['phase_counts'],'crossfit':summary['deployable_crossfit']}),flush=True)
if __name__=='__main__':main()
