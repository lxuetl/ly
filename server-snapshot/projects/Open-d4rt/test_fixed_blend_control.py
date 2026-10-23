import cpu_rejection_smoothing as s
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np,json,time
from collections import defaultdict
a=s.a
OUT=a.t.ROOT/'fixed_blend_control_test_20261017'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
@torch.no_grad()
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False);start=time.time()
 validation=a.t.ROOT/'gate_contribution_val_20261017/summary.json';vm=json.loads(validation.read_text())['video_equal'];grid=[0.,.25,.5,.75,1.];chosen={b:min(grid,key=lambda x:vm[b+'_constant_'+str(x)]['mse']) for b in ['align','velocity']}
 parent=a.t.ROOT/'gpu_locked_smoothing_test_20261016';lock=json.loads((parent/'LOCKED_PROTOCOL.json').read_text());paths=json.loads((a.OUT/'manifest.json').read_text())['test']
 save('LOCKED_PROTOCOL.json',{'alpha':chosen,'selection':'video-equal validation MSE only, original five-value grid','validation_sha256':a.e.sha(validation),'parent_sha256':a.e.sha(parent/'LOCKED_PROTOCOL.json'),'script_sha256':a.e.sha(__file__),'alpha_meaning':'propagation fraction; prediction=(1-alpha)*TAU+alpha*raw propagated RGB','test_tuning':False,'prior_exposure':'existing repeatedly evaluated test split; not fresh benchmark'})
 lp=s.c.lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 old={r['path']:r for r in [json.loads(x) for x in (parent/'per_clip.jsonl').read_text().splitlines()]}
 for i,path in enumerate(paths):
  cache=parent/'cache'/f'{i:04d}.pt'
  if not cache.exists():cache=a.t.ROOT/'cpu_locked_smoothing_test_20261016/cache'/f'{i:04d}.pt'
  d={k:v.cuda() for k,v in torch.load(cache,map_location='cpu',weights_only=False).items()};tau=d['tau'];gt=d['target'];pred={'tau':tau};risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32)
  for b in ['align','velocity']:
   alpha=chosen[b];pred[b+'_fixed']=((1-alpha)*tau+alpha*d[b]).half().float();hard=(risk>lock['thresholds'][b]).float().repeat_interleave(32,-2).repeat_interleave(32,-1);w=s.space(s.temporal(hard));pred[b+'_gate']=(w*tau+(1-w)*d[b]).half().float()
  metrics={}
  for name,p in pred.items():
   mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['boundary_error']=float(torch.stack(edges).mean());metrics[name]=mm
  for b in ['align','velocity']:assert abs(metrics[b+'_gate']['mse']-old[path]['metrics'][b+'_both']['mse'])<1e-7
  row={'path':path,'group':list(a.t.prev.group(path)),'metrics':metrics};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%50==0:print(json.dumps({'done':i+1,'total':len(paths),'seconds':time.time()-start}),flush=True)
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 pairs=[(b+'_fixed',ref) for b in ['align','velocity'] for ref in ['tau',b+'_gate']]
 def cis(v):return {x+'-vs-'+y:{k:a.h.ci([z['metrics'][x][k]-z['metrics'][y][k] for z in v]) for k in ['mse','mae','psnr','lpips','temporal','boundary_error']} for x,y in pairs} if len(v)>1 else {'unavailable':'one video'}
 save('per_video.json',videos);save('summary.json',{'alpha':chosen,'clips':len(rows),'videos':len(videos),'video_equal':means,'paired_ci':cis(videos),'per_source':{k:{'videos':len(v),'metrics':a.aggregate(v),'paired_ci':cis(v)} for k,v in sources.items()}});save('COMPLETE.json',{'seconds':time.time()-start});print('COMPLETE',flush=True)
if __name__=='__main__':main()
