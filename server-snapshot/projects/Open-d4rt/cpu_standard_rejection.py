"""CPU-only standard16->8 evaluation with frozen replay thresholds and cached TAU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,sys
from collections import defaultdict
import numpy as np,torch,cv2,lpips
import run_history_ablations as a
OUT=a.t.ROOT/'cpu_standard16_frozen_rejection_20261016'
RISK=a.t.ROOT/'history_replay_alignment_20261015/risk_signals.json'
def save(n,x):
 p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def flowwarp(last,flow):
 fl=torch.from_numpy(flow).permute(2,0,1)[None];yy,xx=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij');grid=torch.stack([xx+fl[:,0],yy+fl[:,1]],-1)/127*2-1
 return torch.nn.functional.grid_sample(last,grid,mode='bicubic',padding_mode='border',align_corners=True).clamp(0,1)
@torch.no_grad()
def main():
 torch.set_num_threads(4);cv2.setNumThreads(1);OUT.mkdir(exist_ok=True);start=time.time();paths=json.loads((a.OUT/'manifest.json').read_text())['val'];risk=json.loads(RISK.read_text());thresholds={b:{v['group']:v['threshold'] for v in risk[b]['signals']['branch_disagreement']['folds']} for b in ['align','velocity']}
 protocol={'device':'cpu','context':16,'forecast':8,'split':'validation only','clips':len(paths),'thresholds':thresholds,'risk_sha256':a.e.sha(RISK),'script_sha256':a.e.sha(__file__),'cache_manifest_sha256':a.e.sha(a.OUT/'manifest.json'),'policy':'Unchanged prior leave-video-out replay thresholds. No recalibration for standard context.32x32 hard fallback to TAU. Raw propagation, not small-budget texture correction.','limitation':'Thresholds transferred from padded8->8 replay; same validation videos reused, exploratory not independent generalization. Cached standard TAU outputs, no TAU execution.'}
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 else:save('protocol.json',protocol)
 lp=lpips.LPIPS(net='alex',verbose=False).cpu().eval();rows=[]
 if (OUT/'per_clip.jsonl').exists():rows=[json.loads(x) for x in (OUT/'per_clip.jsonl').read_text().splitlines()]
 done={r['path'] for r in rows};limit=int(sys.argv[1]) if len(sys.argv)>1 else len(paths);elapsed_clips=0
 for i,path in enumerate(paths):
  if path in done:continue
  if len(rows)>=limit:break
  tick=time.time();d=torch.load(a.OUT/'cache'/f'{i:04d}.pt',map_location='cpu',weights_only=False);tau=d['tau'].float();rgb,_=a.e.rgb24(path);obs=rgb[:16].half().float();gt=rgb[16:24].half().float();assert torch.equal(gt,d['target'].float()),'cache path/target mismatch';last=obs[-1:];group=list(a.t.prev.group(path));g='/'.join(group)
  base=a.h.flow(obs[-1],obs[-2]);seq={'align':[],'velocity':[]}
  for t in range(8):
   seq['align'].append(flowwarp(last,a.h.flow(tau[t],last[0])));seq['velocity'].append(flowwarp(last,base*(t+1)))
  pred={b:torch.cat(x) for b,x in seq.items()};pred['tau']=tau;pred['zero']=last.expand(8,-1,-1,-1)
  riskmap=torch.nn.functional.avg_pool2d((pred['align']-pred['velocity']).abs().mean(1,keepdim=True),32,32);rates={}
  for b in seq:
   reject=(riskmap>thresholds[b][g]).repeat_interleave(32,-2).repeat_interleave(32,-1);pred[b+'_reject']=torch.where(reject,tau,pred[b]).half().float();rates[b]=float(reject.float().mean())
  metrics={}
  for b,p in pred.items():
   mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['tile_boundary_gradient_error']=float(torch.stack(edges).mean());metrics[b]=mm
  row={'path':path,'group':group,'metrics':metrics,'rejected_fraction':rates,'seconds':time.time()-tick};rows.append(row);elapsed_clips+=1
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if len(rows)%10==0:
   speed=(time.time()-start)/elapsed_clips;progress={'done':len(rows),'total':len(paths),'seconds_per_clip':speed,'remaining_seconds':speed*(len(paths)-len(rows))};save('progress.json',progress);print(json.dumps(progress),flush=True)
 if len(rows)!=len(paths):print('SMOKE_COMPLETE',flush=True);return
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 ci={}
 for b in ['align','velocity']:
  for ref in [b,'tau']:
   ci[b+'_reject-vs-'+ref]={k:a.h.ci([v['metrics'][b+'_reject'][k]-v['metrics'][ref][k] for v in videos]) for k in ['mse','lpips','temporal','tile_boundary_gradient_error']}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'per_source':{s:a.aggregate(v) for s,v in sources.items()},'paired_ci':ci,'rejected_fraction_clip_mean':{b:float(np.mean([r['rejected_fraction'][b] for r in rows])) for b in seq}});save('COMPLETE.json',{'total_clip_seconds':sum(r['seconds'] for r in rows),'device':'cpu'});print('COMPLETE',flush=True)
if __name__=='__main__':main()
