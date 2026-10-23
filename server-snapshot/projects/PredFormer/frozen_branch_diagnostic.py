"""Spatial audit followed by fixed-budget branch-only fitting; no test set."""
import json,time,random,argparse,hashlib,math
from pathlib import Path
import torch,numpy as np,lpips
from torch.nn import functional as F
from subset_ablation import Model,Data,BASE,RUN as PREVIOUS
from stage1_baseline import save
from token_audit import metrics
RUN=Path('/root/autodl-tmp/predformer_frozen32_20261022')
def digest(model):
 h=hashlib.sha256()
 for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def spatial_stats(a):
 a=a.float();center=a-a.mean(-2,keepdim=True);u=F.normalize(a,dim=-1);cos=u@u.transpose(-1,-2);n=a.shape[-2]
 return {'spatial_rms':float(center.square().mean().sqrt()),'total_rms':float(a.square().mean().sqrt()),'spatial_energy_fraction':float(center.square().mean()/a.square().mean().clamp_min(1e-12)),'off_diagonal_cosine':float((cos.sum((-1,-2))-n).div(n*(n-1)).mean())}
def prepare():
 RUN.mkdir(exist_ok=False);s=json.loads((PREVIOUS/'subset.json').read_text())['paths'];paths=sorted(random.Random(42).sample(s['train'],32));save(RUN/'subset.json',{'train':paths,'test_used':False,'selection':'seed42 sample32 from previous128 training subset'})
 ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);m=Model(ck,'omega','pre').cuda().eval();w=torch.load(PREVIOUS/'pre_omega/best_trainable.pt',map_location='cpu',weights_only=False);m.load_state_dict(w['trainable'],strict=False)
 ds=Data(paths,BASE/'rgb_cache','omega');rows=[]
 with torch.no_grad():
  for x,y,path,g in ds:
   x=x[None].cuda();g=g[None].cuda();prefix=m.encode(x);raw=g[:,:,:16].float();projected=m.project(raw);pos=m.coord(m.spatial_pos)[None];mem=(projected+pos).reshape(1,256,256);q=m.norm(prefix[:,:8]+m.horizon.weight[None,:,None,:]);mem=mem[:,None].expand(-1,8,-1,-1).reshape(8,256,256)
   delta,attn=m.read(q.reshape(8,256,256),mem,mem,attn_mask=m.local_bias,need_weights=True,average_attn_weights=False)
   cap=m.history_cap(x);gate=torch.sigmoid(m.gate(q))*cap[:,None]*m.horizon_cap[None,:,None,None];delta=delta.reshape(1,8,256,256)*gate
   mask=(y.cuda()-x[0,-1]).abs().mean(1)>.05;support=F.interpolate(cap.transpose(1,2).reshape(1,1,16,16),size=(128,128),mode='nearest')[0,0]>0
   ent=-(attn.float()*attn.float().clamp_min(1e-12).log()).sum(-1)/math.log(64)
   rows.append({'path':path,'raw':spatial_stats(raw),'projected':spatial_stats(projected),'with_position':spatial_stats(projected+pos),'attention_entropy_over_log64':float(ent.mean()),'attention_peak':float(attn.max(-1).values.mean()),'historical_support_fraction':float((cap>0).float().mean()),'gate_mean':float(gate.mean()),'delta_to_query_rms':float(delta.square().mean().sqrt()/prefix[:,:8].square().mean().sqrt()),'future_change_coverage_eval_only':float((mask&support).sum()/mask.sum().clamp_min(1))})
 save(RUN/'audit_per_clip.json',rows)
 summary={k:{a:float(np.mean([r[k][a] for r in rows])) for a in rows[0][k]} if isinstance(rows[0][k],dict) else float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
 save(RUN/'audit.json',{'checkpoint':'previous pre_omega epoch2','clips':32,'precision':'float32','attention_uniform_reference':'64 allowed keys; entropy/log64=1 means uniform within allowed set; gaussian locality bias itself affects entropy','future_change':'evaluation only RGB threshold .05, not object motion truth','summary':summary});print(json.dumps(summary),flush=True)
def main(arm):
 torch.manual_seed(42);np.random.seed(42);random.seed(42)
 out=RUN/arm;out.mkdir(exist_ok=False);paths=json.loads((RUN/'subset.json').read_text())['train'];ds=Data(paths,BASE/'rgb_cache','omega' if arm.startswith('omega') else arm)
 ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);m=Model(ck,arm,'pre').cuda();m.base.requires_grad_(False);m.base.eval();basehash=digest(m.base)
 names={n for n,p in m.named_parameters() if p.requires_grad};params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False);scaler=torch.amp.GradScaler('cuda')
 save(out/'protocol.json',{'arm':arm,'architecture':'pre TSST3, unchanged locality/history cap/sqrt(h)','base':str(BASE/'best.pt'),'init':'fresh identical zero-output adapter seed42; NOT previous fitted adapters','base_fully_frozen':True,'base_eval_mode':True,'trainable_parameters':sum(p.numel() for p in params),'epochs':100,'steps':3200,'clips':32,'lr':1e-4,'loss':'MSE','selection':'fixed steps; no validation selection','report_epochs':[0,10,30,50,100],'test_used':False,'base_hash':basehash,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 def evaluate(epoch):
  m.eval();results={}
  for mode in ['correct','shuffle','zero','bypass']:
   rows=[]
   with torch.no_grad():
    for x,y,path,g in ds:
     x=x[None].cuda();y=y.cuda();g=g[None].cuda()
     with torch.autocast('cuda',dtype=torch.float16):p=m(x,g,mode=mode).float()[0]
     r=metrics(p,y,net);r.update(path=path,late_mse=float((p[4:]-y[4:]).square().mean()),late_lpips=metrics(p[4:],y[4:],net)['lpips']);rows.append(r)
   save(out/('epoch%d_%s.json'%(epoch,mode)),rows);results[mode]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
  assert digest(m.base)==basehash
  save(out/('epoch%d_summary.json'%epoch),results)
  torch.save({'trainable':{n:p.detach().cpu() for n,p in m.named_parameters() if n in names},'epoch':epoch,'arm':arm,'base_hash':basehash},out/('epoch%d.pt'%epoch))
  return results
 x,y,path,g=ds[0];m.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):err=float((m(x[None].cuda(),g[None].cuda())-m.base(x[None].cuda())[:,:8]).abs().max())
 assert err==0
 save(out/'checks.json',{'identity_error':err,'frozen_base_hash':basehash,'trainable_names':sorted(names)})
 start=time.time();evaluate(0);rng=random.Random(2026);history=[]
 for epoch in range(1,101):
  tick=time.time();m.train();m.base.eval();order=list(range(32));rng.shuffle(order);total=0;grad=0
  for i in order:
   x,y,path,g=ds[i];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):p=m(x[None].cuda(),g[None].cuda()).float()[0];loss=(p-y.cuda()).square().mean()
   assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
   assert all(p.grad is None for p in m.base.parameters());assert m.read.out_proj.weight.grad is not None
   grad+=float(m.read.out_proj.weight.grad.norm());torch.nn.utils.clip_grad_norm_(params,1);scaler.step(opt);scaler.update();total+=float(loss.detach());assert m.last_outside_max==0
  r={'epoch':epoch,'steps':epoch*32,'train_mse':total/32,'out_proj_grad_norm':grad/32,'train_seconds':time.time()-tick,'elapsed_seconds':time.time()-start}
  if epoch in [10,30,50,100]:r['evaluation']=evaluate(epoch)
  history.append(r);save(out/'history.json',history);save(out/'progress.json',r);print(json.dumps(r),flush=True)
 save(out/'COMPLETE.json',{'seconds':time.time()-start,'base_unchanged':digest(m.base)==basehash,'fixed_epoch':100,'metrics':history[-1]['evaluation']})
if __name__=='__main__':
 torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=['omega','rgb','null','omega_shuffled']);a=p.parse_args()
 if a.prepare:prepare()
 else:main(a.arm)
