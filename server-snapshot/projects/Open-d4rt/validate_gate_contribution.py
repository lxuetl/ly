"""Validation-only gate contribution controls on immutable cached sources."""
import cpu_rejection_smoothing as s
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np,json,time
from collections import defaultdict
a=s.a
OUT=a.t.ROOT/'gate_contribution_val_20261017'
GRID=[0.,.25,.5,.75,1.]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
@torch.no_grad()
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False);start=time.time();paths=json.loads((a.OUT/'manifest.json').read_text())['val'];thresholds=json.loads((s.OUT/'protocol.json').read_text())['thresholds']
 save('protocol.json',{'split':'validation only','branches':['align','velocity'],'thresholds':'unchanged prior leave-original-video-out','fixed_propagation_weights':GRID,'constant_selection':'leave-one-original-video-out minimum video-equal MSE; separate descriptive minimum LPIPS, no test evaluation','controls':['correct risk gate','spatial horizontal/vertical/both reflection of SAME smoothed gate','per-horizon spatial mean of SAME gate','fixed blend grid'],'invariants':'reflections preserve exact per-frame distribution and spatial/temporal total variation, no wrap seams; matched_mean preserves per-frame propagation amount but discards location','limitations':'reflection can accidentally preserve meaningful structure; reused validation exploratory; constants selected outside heldout video; not a test result','script_sha256':a.e.sha(__file__)})
 lp=s.c.lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for i,path in enumerate(paths):
  d={k:v.cuda() for k,v in torch.load(s.OUT/'cache'/f'{i:04d}.pt',map_location='cpu',weights_only=False).items()};tau=d['tau'];gt=d['target'];group=list(a.t.prev.group(path));g='/'.join(group);risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32)
  pred={'tau':tau};diag={}
  for b in ['align','velocity']:
   hard=(risk>thresholds[b][g]).float().repeat_interleave(32,-2).repeat_interleave(32,-1);w=s.space(s.temporal(hard));weights={'correct':w,'flip_x':w.flip(-1),'flip_y':w.flip(-2),'flip_xy':w.flip((-2,-1)),'matched_mean':w.mean((-2,-1),keepdim=True).expand_as(w)}
   for label,rw in weights.items():
    assert (rw.mean((-2,-1))-w.mean((-2,-1))).abs().max()<1e-6
    pred[b+'_'+label]=(rw*tau+(1-rw)*d[b]).half().float()
   for alpha in GRID:pred[b+'_constant_'+str(alpha)]=((1-alpha)*tau+alpha*d[b]).half().float()
   diag[b]=float((1-w).mean())
  metrics={}
  for name,p in pred.items():
   mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['boundary_error']=float(torch.stack(edges).mean());metrics[name]=mm
  row={'path':path,'group':group,'metrics':metrics,'propagation_weight':diag};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%20==0:print(json.dumps({'done':i+1,'total':len(paths),'seconds':time.time()-start,'remaining_seconds':(time.time()-start)/(i+1)*(len(paths)-i-1)}),flush=True)
 videos,_=a.summarize(rows);choices=[]
 for j,v in enumerate(videos):
  others=[z for k,z in enumerate(videos) if k!=j];record={'group':v['group']}
  for b in ['align','velocity']:
   for criterion in ['mse','lpips']:
    alpha=min(GRID,key=lambda x:np.mean([z['metrics'][b+'_constant_'+str(x)][criterion] for z in others]));v['metrics'][b+'_constant_loo_'+criterion]=v['metrics'][b+'_constant_'+str(alpha)];record[b+'_'+criterion]=alpha
   v['metrics'][b+'_mismatch_avg']={k:np.mean([v['metrics'][b+'_'+q][k] for q in ['flip_x','flip_y','flip_xy']],axis=0).tolist() for k in v['metrics'][b+'_correct']}
  choices.append(record)
 ci={}
 for b in ['align','velocity']:
  for ref in ['matched_mean','mismatch_avg','constant_loo_mse','constant_loo_lpips']:
   ci[b+'_correct-vs-'+ref]={k:a.h.ci([v['metrics'][b+'_correct'][k]-v['metrics'][b+'_'+ref][k] for v in videos]) for k in ['mse','lpips','temporal','boundary_error']}
 sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 save('per_video.json',videos);save('loo_choices.json',choices);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':a.aggregate(videos),'paired_ci':ci,'per_source':{k:a.aggregate(v) for k,v in sources.items()}});save('COMPLETE.json',{'seconds':time.time()-start,'test_used':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
