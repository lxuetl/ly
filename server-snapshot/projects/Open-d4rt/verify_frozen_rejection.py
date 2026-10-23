"""Apply previous leave-video-out thresholds verbatim, no new calibration."""
import json,time
import torch,numpy as np,lpips
import replay_alignment_diagnostic as r
a=r.a;m=r.m
OUT=a.t.ROOT/'frozen_disagreement_rejection_20261016'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
@torch.no_grad()
def main():
 a.setup();OUT.mkdir(exist_ok=False);start=time.time();paths=json.loads((a.OUT/'manifest.json').read_text())['val'];risk=json.loads((r.OUT/'risk_signals.json').read_text())
 thresholds={b:{v['group']:v['threshold'] for v in risk[b]['signals']['branch_disagreement']['folds']} for b in ['align','velocity']}
 save('LOCKED_PROTOCOL.json',{'thresholds':thresholds,'threshold_file_sha256':a.e.sha(r.OUT/'risk_signals.json'),'script_sha256':a.e.sha(__file__),'rule':'32x32 regional mean branch disagreement > frozen threshold => TAU, else raw source. No blending/tuning. Separate prior LOO thresholds for each branch.','scope':'same validation8->8 padded history replay; thresholds excluded own original video; exploratory reused validation, not independent test','metrics':'full assembled image LPIPS/MSE and full sequence temporal; seam error at internal32px tile borders; paired bootstrap over original videos','test_used':False})
 net=a.network();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for i,path in enumerate(paths):
  rgb,_=a.e.rgb24(path);obs=rgb[:8].cuda();ctx=torch.cat([obs[:1].expand(8,-1,-1,-1),obs]);_,hid,skip=a.t.core.extract_frozen_features(net,ctx[None]);tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float();obs=obs.half().float();last=obs[-1:];gt=rgb[8:16].cuda().half().float();group=list(a.t.prev.group(path));g='/'.join(group)
  base,_=a.h.maps(a.h.flow(obs[-1],obs[-2]),a.h.flow(obs[-2],obs[-1]));seq={b:[] for b in ['align','velocity']}
  for t in range(8):
   fl,_=a.h.maps(a.h.flow(tau[t],last[0]),a.h.flow(last[0],tau[t]));seq['align'].append(m.warp(last,fl,'bicubic'));seq['velocity'].append(m.warp(last,base*(t+1),'bicubic'))
  pred={b:torch.cat(s) for b,s in seq.items()};pred['tau']=tau
  diff=(pred['align']-pred['velocity']).abs().mean(1,keepdim=True);riskmap=torch.nn.functional.avg_pool2d(diff,32,32);rates={}
  for b in ['align','velocity']:
   reject=(riskmap>thresholds[b][g]).repeat_interleave(32,-2).repeat_interleave(32,-1);pred[b+'_reject']=torch.where(reject,tau,pred[b]).half().float();rates[b]=float(reject.float().mean())
  metrics={}
  for b,p in pred.items():
   mm=a.metrics(p,gt,tau,lp)
   # Error in image gradients crossing internal tile boundaries, relative to GT.
   er=p-gt;edges=[]
   for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
   mm['tile_boundary_gradient_error']=float(torch.stack(edges).mean());metrics[b]=mm
  row={'path':path,'group':group,'metrics':metrics,'rejected_fraction':rates};rows.append(row)
  with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if (i+1)%40==0:print(json.dumps({'done':i+1,'seconds':time.time()-start}),flush=True)
 videos,means=a.summarize(rows);ci={}
 for b in ['align','velocity']:
  for ref in [b,'tau']:
   ci[b+'_reject-vs-'+ref]={k:a.h.ci([v['metrics'][b+'_reject'][k]-v['metrics'][ref][k] for v in videos]) for k in ['mse','lpips','temporal','tile_boundary_gradient_error']}
 save('per_video.json',videos);save('summary.json',{'clips':len(rows),'videos':len(videos),'video_equal':means,'paired_ci':ci,'rejected_fraction_clip_mean':{b:float(np.mean([v['rejected_fraction'][b] for v in rows])) for b in ['align','velocity']}});save('COMPLETE.json',{'seconds':time.time()-start});print('COMPLETE',flush=True)
if __name__=='__main__':main()
