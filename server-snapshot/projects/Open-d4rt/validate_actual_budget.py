"""Validation-only control: spatially uniform fusion at gate's actual L1 budget."""
import cpu_rejection_smoothing as s
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np,json,time
from collections import defaultdict
a=s.a
OUT=a.t.ROOT/'actual_budget_control_val_20261017'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def blend(tau,src,alpha):return ((1-alpha)*tau+alpha*src).half().float()
@torch.no_grad()
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False);start=time.time()
 paths=json.loads((a.OUT/'manifest.json').read_text())['val'];thresholds=json.loads((s.OUT/'protocol.json').read_text())['thresholds']
 save('protocol.json',{'split':'original validation only; no TUM/test','context':16,'forecast':8,'resolution':128,'training':False,'thresholds':'unchanged leave-original-video-out thresholds from parent','parent_sha256':a.e.sha(s.OUT/'protocol.json'),'script_sha256':a.e.sha(__file__),'gate_smoothing':'causal temporal 0.5 then spatial 9x9 replicate unchanged','control':'one uniform propagation coefficient per clip, constant across all 8 horizons; constant mask invariant under same smoothing operators','budget':'mean absolute final half-rounded output minus TAU over T,C,H,W','matching':'24-step bisection in [0,1], select nearest endpoint; target is gate budget, NEVER future ground truth','caveat':'budget-matched diagnostic control, not a globally fixed deployment coefficient; no quality-based parameter selection'})
 lp=s.c.lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for i,path in enumerate(paths):
  d={k:v.cuda() for k,v in torch.load(s.OUT/'cache'/f'{i:04d}.pt',map_location='cpu',weights_only=False).items()};tau=d['tau'];gt=d['target'];group=list(a.t.prev.group(path));g='/'.join(group)
  risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32);pred={'tau':tau};diag={}
  for b in ['align','velocity']:
   hard=(risk>thresholds[b][g]).float().repeat_interleave(32,-2).repeat_interleave(32,-1);w=s.space(s.temporal(hard));p=(w*tau+(1-w)*d[b]).half().float();budget=float((p-tau).abs().mean());lo=0.;hi=1.
   for _ in range(24):
    mid=(lo+hi)/2;mb=float((blend(tau,d[b],mid)-tau).abs().mean())
    if mb<budget:lo=mid
    else:hi=mid
   alpha=min([lo,hi],key=lambda z:abs(float((blend(tau,d[b],z)-tau).abs().mean())-budget));fixed=blend(tau,d[b],alpha);actual=float((fixed-tau).abs().mean());err=abs(actual-budget)
   assert err<2e-6,(i,b,budget,actual)
   pred[b+'_gate']=p;pred[b+'_budget_fixed']=fixed
   diag[b]={'alpha':alpha,'gate_budget':budget,'fixed_budget':actual,'absolute_budget_error':err,'gate_horizon_budget':(p-tau).abs().mean((1,2,3)).tolist(),'fixed_horizon_budget':(fixed-tau).abs().mean((1,2,3)).tolist()}
  metrics={}
  for name,p in pred.items():
   mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['boundary_error']=float(torch.stack(edges).mean());metrics[name]=mm
  row={'path':path,'group':group,'metrics':metrics,'diagnostics':diag};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%10==0:
   elapsed=time.time()-start;progress={'done':i+1,'total':len(paths),'seconds':elapsed,'remaining_seconds':elapsed/(i+1)*(len(paths)-i-1)};save('progress.json',progress);print(json.dumps(progress),flush=True)
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 ci={}
 for b in ['align','velocity']:
  for x,y in [(b+'_gate',b+'_budget_fixed'),(b+'_gate','tau'),(b+'_budget_fixed','tau')]:
   ci[x+'-vs-'+y]={k:a.h.ci([v['metrics'][x][k]-v['metrics'][y][k] for v in videos]) for k in ['mse','mae','lpips','temporal','boundary_error']}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'paired_ci':ci,'per_source':{k:a.aggregate(v) for k,v in sources.items()},'max_budget_error':max(r['diagnostics'][b]['absolute_budget_error'] for r in rows for b in ['align','velocity'])});save('COMPLETE.json',{'seconds':time.time()-start,'training':False,'test_used':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
