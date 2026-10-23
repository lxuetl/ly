import cpu_rejection_smoothing as s
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,numpy as np,json,time,cv2
from pathlib import Path
from collections import defaultdict
a=s.a
OUT=a.t.ROOT/'tum_locked_external_20261017'
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
@torch.no_grad()
def main():
 audit=a.t.ROOT/'tum_holdout_audit_20261017/summary.json';au=json.loads(audit.read_text());assert not au['history_name_hits'] and not au['decoded128_exact_matches'] and not au['dhash_equal_candidates'] and not au['errors'],'Audit requires review; evaluation not started'
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False);start=time.time();thresholds=json.loads((a.t.ROOT/'gpu_locked_smoothing_test_20261016/LOCKED_PROTOCOL.json').read_text())['thresholds'];alpha=json.loads((a.t.ROOT/'fixed_blend_control_test_20261017/LOCKED_PROTOCOL.json').read_text())['alpha'];paths=sorted(Path('data/worldtrack_release/tum').glob('*.npz'));assert len(paths)==8
 save('LOCKED_PROTOCOL.json',{'audit_sha256':a.e.sha(audit),'thresholds':thresholds,'alpha':alpha,'starts':[0,24,48],'context':16,'forecast':8,'resolution':128,'spatial':9,'temporal':.5,'tuning':False,'selection':'all8 audited sequences,3 nonoverlapping windows each; last18frames unused','limitations':'independence only within audited retained records; shared environment possible;8sequence bootstrap exploratory; native stored frame order, no FPS normalization','script_sha256':a.e.sha(__file__)})
 ck=torch.load(a.t.ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False);net=a.build_video_predictor_from_checkpoint(ck).cuda().eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;lp=s.c.lpips.LPIPS(net='alex',verbose=False).cuda().eval();rows=[]
 for path in paths:
  with np.load(path,allow_pickle=True) as z:frames=z['images_jpeg_bytes'];assert len(frames)>=72
  images=[]
  for b in frames[:72]:
   ar=np.frombuffer(b,dtype=np.uint8) if isinstance(b,(bytes,bytearray)) else np.asarray(b,dtype=np.uint8);im=cv2.imdecode(ar,cv2.IMREAD_COLOR);assert im is not None;images.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_AREA))
  rgb=torch.tensor(np.stack(images)).permute(0,3,1,2).float()/255
  for startframe in [0,24,48]:
   ctx=rgb[startframe:startframe+16].cuda();gt=rgb[startframe+16:startframe+24].cuda().half().float();_,hid,skip=a.t.core.extract_frozen_features(net,ctx[None]);tau=a.t.core.decode_frozen(net,hid,skip)[0].half().float();obs=ctx.cpu().half().float();last=obs[-1:];base=a.h.flow(obs[-1],obs[-2]);src={'align':[],'velocity':[]}
   for t in range(8):src['align'].append(s.c.flowwarp(last,a.h.flow(tau[t].cpu(),last[0])));src['velocity'].append(s.c.flowwarp(last,base*(t+1)))
   d={b:torch.cat(v).cuda() for b,v in src.items()};risk=torch.nn.functional.avg_pool2d((d['align']-d['velocity']).abs().mean(1,keepdim=True),32,32);pred={'tau':tau}
   for b in d:
    pred[b+'_fixed']=((1-alpha[b])*tau+alpha[b]*d[b]).half().float();hard=(risk>thresholds[b]).float().repeat_interleave(32,-2).repeat_interleave(32,-1);w=s.space(s.temporal(hard));pred[b+'_gate']=(w*tau+(1-w)*d[b]).half().float()
   metrics={}
   for name,p in pred.items():
    mm=a.metrics(p,gt,tau,lp);er=p-gt;edges=[]
    for k in [32,64,96]:edges.extend([(er[:,:,:,k]-er[:,:,:,k-1]).abs().mean(),(er[:,:,k,:]-er[:,:,k-1,:]).abs().mean()])
    mm['boundary_error']=float(torch.stack(edges).mean());metrics[name]=mm
   row={'path':str(path),'start':startframe,'group':['TUM',path.stem],'metrics':metrics};rows.append(row)
   with (OUT/'per_clip.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps({'done':len(rows),'total':24,'seconds':time.time()-start}),flush=True)
 videos,means=a.summarize(rows);pairs=[(b+'_'+mode,ref) for b in ['align','velocity'] for mode in ['fixed','gate'] for ref in (['tau'] if mode=='fixed' else ['tau',b+'_fixed'])];ci={x+'-vs-'+y:{k:a.h.ci([v['metrics'][x][k]-v['metrics'][y][k] for v in videos]) for k in ['mse','mae','lpips','temporal','boundary_error']} for x,y in pairs}
 save('per_video.json',videos);save('summary.json',{'clips':24,'videos':8,'video_equal':means,'paired_ci':ci,'subgroups':{label:a.aggregate([v for v in videos if label in v['group'][1]]) for label in ['sitting','walking']}});save('COMPLETE.json',{'seconds':time.time()-start});print('COMPLETE',flush=True)
if __name__=='__main__':main()
