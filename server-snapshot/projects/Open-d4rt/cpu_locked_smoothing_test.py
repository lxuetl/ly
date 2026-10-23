"""Locked four-way standard16->8 test. CPU. No test-driven selection."""
import cpu_rejection_smoothing as s
import os
GPU_RUN=os.environ.get('LOCKED_TEST_GPU')=='1'
if GPU_RUN:os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np,json,time,sys
from collections import defaultdict
a=s.a;c=s.c
CPU_OUT=a.t.ROOT/'cpu_locked_smoothing_test_20261016'
OUT=a.t.ROOT/('gpu_locked_smoothing_test_20261016' if GPU_RUN else 'cpu_locked_smoothing_test_20261016')
def save(n,x):
 p=OUT/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
@torch.no_grad()
def main():
 torch.set_num_threads(1);c.cv2.setNumThreads(1);OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True);start=time.time()
 splits=json.loads((a.OUT/'manifest.json').read_text());paths=splits['test'];vg={tuple(a.t.prev.group(p)) for p in splits['val']};tg={tuple(a.t.prev.group(p)) for p in paths};assert not vg&tg
 riskfile=a.t.ROOT/'history_replay_alignment_20261015/regions.jsonl';values={'align':[],'velocity':[]}
 with riskfile.open() as f:
  for line in f:
   r=json.loads(line);assert tuple(r['group']) in vg
   if r['delta_mse']<=1e-6:values[r['branch']].append(r['branch_disagreement'])
 thresholds={b:float(np.quantile(v,.9)) for b,v in values.items()}
 protocol={'thresholds':thresholds,'calibration':'all validation replay nonharm regions (deltaMSE<=1e-6), fixed90th percentile of branch disagreement; no optimization','calibration_sha256':a.e.sha(riskfile),'script_sha256':a.e.sha(__file__),'modes':['tau','velocity_hard','velocity_both','align_both'],'spatial':9,'temporal':.5,'test_clips':len(paths),'test_videos':len(tg),'threshold_freeze_unix':None,'prior_exposure':'test previously used in other historical experiments, not unseen benchmark','device':'CPU','checkpoint_sha256':a.e.sha(a.t.ROOT/'tau128_full_fair_fast_20261009/best.pt')}
 if (OUT/'LOCKED_PROTOCOL.json').exists():
  old=json.loads((OUT/'LOCKED_PROTOCOL.json').read_text());protocol['threshold_freeze_unix']=old['threshold_freeze_unix'];assert old==protocol
 else:protocol['threshold_freeze_unix']=time.time();save('LOCKED_PROTOCOL.json',protocol)
 oldrows={r['path']:r for r in [json.loads(x) for x in (a.t.ROOT/'fixed_history_expanded_20261015/per_clip.jsonl').read_text().splitlines()]}
 device=torch.device('cuda' if GPU_RUN else 'cpu')
 save('execution.json',{'device':str(device),'previous_cpu_results_preserved':True,'uniform_metric_device':True,'reuse_cpu_source_cache':GPU_RUN})
 ck=torch.load(a.t.ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False);net=a.build_video_predictor_from_checkpoint(ck).to(device).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck
 lp=c.lpips.LPIPS(net='alex',verbose=False).to(device).eval();rows=[]
 if (OUT/'per_clip.jsonl').exists():rows=[json.loads(x) for x in (OUT/'per_clip.jsonl').read_text().splitlines()]
 done={r['path'] for r in rows};limit=int(sys.argv[1]) if len(sys.argv)>1 else len(paths);new=0
 for i,path in enumerate(paths):
  if path in done:continue
  if len(rows)>=limit:break
  tick=time.time();cache=OUT/'cache'/f'{i:04d}.pt'
  if GPU_RUN and not cache.exists() and (CPU_OUT/'cache'/f'{i:04d}.pt').exists():cache=CPU_OUT/'cache'/f'{i:04d}.pt'
  if cache.exists():d=torch.load(cache,map_location='cpu',weights_only=False)
  else:
   rgb,_=a.e.rgb24(path);ctx=rgb[:16];_,hid,skip=a.t.core.extract_frozen_features(net,ctx[None].to(device));tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float().cpu();obs=ctx.half().float();last=obs[-1:];base=a.h.flow(obs[-1],obs[-2]);src={'align':[],'velocity':[]}
   for t in range(8):src['align'].append(c.flowwarp(last,a.h.flow(tau[t],last[0])));src['velocity'].append(c.flowwarp(last,base*(t+1)))
   d={b:torch.cat(v) for b,v in src.items()};d.update(tau=tau,target=rgb[16:24].half().float());torch.save(d,cache)
  tau=d['tau'];gt=d['target'];risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32);pred={'tau':tau};diag={}
  for b in ['align','velocity']:
   hard=(risk>thresholds[b]).float().repeat_interleave(32,-2).repeat_interleave(32,-1);w=s.space(s.temporal(hard));pred[b+'_both']=(w*tau+(1-w)*d[b]).half().float();diag[b]=float((1-w).mean())
   if b=='velocity':pred['velocity_hard']=(hard*tau+(1-hard)*d[b]).half().float()
  metrics={}
  for name,p in pred.items():
   mm=a.metrics(p.to(device),gt.to(device),tau.to(device),lp);er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['tile_boundary_gradient_error']=float(torch.stack(edges).mean());metrics[name]=mm
  parity=abs(metrics['tau']['mse']-oldrows[path]['metrics']['tau']['mse']);assert parity<max(1e-6,oldrows[path]['metrics']['tau']['mse']*.001),('CPU baseline differs',path,parity)
  row={'path':path,'group':list(a.t.prev.group(path)),'metrics':metrics,'propagation_weight':diag,'tau_mse_vs_prior_abs':parity,'seconds':time.time()-tick};rows.append(row);new+=1
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if len(rows)%5==0:
   speed=(time.time()-start)/new;p={'done':len(rows),'total':len(paths),'seconds_per_clip':speed,'remaining_seconds':speed*(len(paths)-len(rows))};save('progress.json',p);print(json.dumps(p),flush=True)
 if len(rows)<len(paths):print('SMOKE_COMPLETE',flush=True);return
 videos,means=a.summarize(rows);sources=defaultdict(list)
 for v in videos:sources[v['group'][0]].append(v)
 pairs=[('velocity_hard','tau'),('velocity_both','tau'),('align_both','tau'),('velocity_both','velocity_hard'),('velocity_both','align_both')]
 def cis(v):return {x+'-vs-'+y:{k:a.h.ci([z['metrics'][x][k]-z['metrics'][y][k] for z in v]) for k in ['mse','mae','psnr','lpips','temporal','tile_boundary_gradient_error']} for x,y in pairs} if len(v)>1 else {'unavailable':'only one original video'}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'clip_equal':a.aggregate(rows),'paired_ci':cis(videos),'per_source':{k:{'videos':len(v),'metrics':a.aggregate(v),'paired_ci':cis(v)} for k,v in sources.items()},'max_tau_mse_parity_error':max(r['tau_mse_vs_prior_abs'] for r in rows)});save('COMPLETE.json',{'clip_seconds':sum(r['seconds'] for r in rows),'tuning':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
