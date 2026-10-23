"""Fixed spatial/temporal rejection ablations, CPU, validation only."""
import cpu_standard_rejection as c
import torch,numpy as np,json,time,sys
from collections import defaultdict
a=c.a
OUT=a.t.ROOT/'cpu_rejection_smoothing_20261016'
def save(n,x):
 p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def space(w):return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(w,(4,4,4,4),mode='replicate'),9,1)
def temporal(w):
 seq=[w[:1]]
 for t in range(1,8):seq.append(.5*w[t:t+1]+.5*seq[-1])
 return torch.cat(seq)
@torch.no_grad()
def main():
 torch.set_num_threads(1);c.cv2.setNumThreads(1);OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True);start=time.time()
 paths=json.loads((a.OUT/'manifest.json').read_text())['val'];old=json.loads((c.OUT/'protocol.json').read_text());thresholds=old['thresholds']
 protocol={'thresholds':thresholds,'parent_protocol_sha256':a.e.sha(c.OUT/'protocol.json'),'script_sha256':a.e.sha(__file__),'device':'CPU','split':'standard16->8 validation only','spatial':'9x9 mean, replicate padding','temporal':'causal horizon recursion w[t]=.5*hard[t]+.5*w[t-1]; initial=hard[0]','both':'temporal then spatial','scope':'both align and velocity; no parameter fitting. Soft weights may restore some propagation in hard-rejected regions; report actual weights. No future targets in risk/masks.','cache':'float32 raw sources, TAU,target; perclip consolidated file'}
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
 else:save('protocol.json',protocol)
 baseline={r['path']:r for r in [json.loads(s) for s in (c.OUT/'per_clip.jsonl').read_text().splitlines()]}
 lp=c.lpips.LPIPS(net='alex',verbose=False).cpu().eval();rows=[]
 if (OUT/'per_clip.jsonl').exists():rows=[json.loads(s) for s in (OUT/'per_clip.jsonl').read_text().splitlines()]
 done={r['path'] for r in rows};new=0;limit=int(sys.argv[1]) if len(sys.argv)>1 else len(paths)
 for i,path in enumerate(paths):
  if path in done:continue
  if len(rows)>=limit:break
  tick=time.time();cache=OUT/'cache'/f'{i:04d}.pt'
  if cache.exists():d=torch.load(cache,map_location='cpu',weights_only=False)
  else:
   d0=torch.load(a.OUT/'cache'/f'{i:04d}.pt',map_location='cpu',weights_only=False);tau=d0['tau'].float();rgb,_=a.e.rgb24(path);obs=rgb[:16].half().float();gt=rgb[16:24].half().float();assert torch.equal(gt,d0['target'].float());last=obs[-1:];base=a.h.flow(obs[-1],obs[-2]);src={'align':[],'velocity':[]}
   for t in range(8):
    src['align'].append(c.flowwarp(last,a.h.flow(tau[t],last[0])));src['velocity'].append(c.flowwarp(last,base*(t+1)))
   d={b:torch.cat(v) for b,v in src.items()};d.update(tau=tau,target=gt);torch.save(d,cache)
  tau=d['tau'];gt=d['target'];group=list(a.t.prev.group(path));g='/'.join(group);risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32)
  pred={'tau':tau};diagnostics={}
  for b in ['align','velocity']:
   hard=(risk>thresholds[b][g]).float().repeat_interleave(32,-2).repeat_interleave(32,-1)
   weights={'hard':hard,'spatial':space(hard),'temporal':temporal(hard),'both':space(temporal(hard))}
   for mode,w in weights.items():
    name=b+'_'+mode;pred[name]=(w*tau+(1-w)*d[b]).half().float();diagnostics[name]={'propagation_weight':float((1-w).mean()),'weight_temporal_change':float((w[1:]-w[:-1]).abs().mean()),'propagation_in_hard_rejected':float(((1-w)*hard).sum()/hard.sum().clamp_min(1))}
  metrics={}
  for name,p in pred.items():
   mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['tile_boundary_gradient_error']=float(torch.stack(edges).mean());metrics[name]=mm
  parity={b:abs(metrics[b+'_hard']['mse']-baseline[path]['metrics'][b+'_reject']['mse']) for b in ['align','velocity']};assert max(parity.values())<1e-7,parity
  row={'path':path,'group':group,'metrics':metrics,'diagnostics':diagnostics,'hard_mse_parity':parity,'seconds':time.time()-tick};rows.append(row);new+=1
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if len(rows)%10==0:
   speed=(time.time()-start)/new;progress={'done':len(rows),'total':len(paths),'seconds_per_clip':speed,'remaining_seconds':speed*(len(paths)-len(rows))};save('progress.json',progress);print(json.dumps(progress),flush=True)
 if len(rows)<len(paths):print('SMOKE_COMPLETE',flush=True);return
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 ci={}
 for b in ['align','velocity']:
  for mode in ['spatial','temporal','both']:
   for ref in [b+'_hard','tau']:
    name=b+'_'+mode;ci[name+'-vs-'+ref]={k:a.h.ci([v['metrics'][name][k]-v['metrics'][ref][k] for v in videos]) for k in ['mse','lpips','temporal','tile_boundary_gradient_error']}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'per_source':{s:a.aggregate(v) for s,v in sources.items()},'paired_ci':ci,'diagnostics_clip_mean':{n:{k:float(np.mean([r['diagnostics'][n][k] for r in rows])) for k in rows[0]['diagnostics'][n]} for n in rows[0]['diagnostics']}});save('COMPLETE.json',{'device':'CPU','clip_seconds':sum(r['seconds'] for r in rows)});print('COMPLETE',flush=True)
if __name__=='__main__':main()
