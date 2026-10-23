"""Motion-aware trajectory losses and detailed UV metrics."""
from __future__ import annotations
import numpy as np,torch
import torch.nn.functional as F

def last_uv(batch):
 v=batch['past_valid'].bool();u=torch.nan_to_num(batch['past_uv'].float());idx=(v.long()*torch.arange(1,v.shape[2]+1,device=v.device)).argmax(2);return u.gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)

def motion_aware_losses(model,b,uv_weight=5.,motion_weight=2.):
 xyz,uv=model(b['past_xyz'].float(),b['past_uv'].float(),b['past_valid'],b['past_confidence'].float());tx=b['future_xyz_teacher'].float();tu=b['future_uv_teacher'].float();mask=b['future_valid'].bool()&torch.isfinite(tx).all(-1)&torch.isfinite(tu).all(-1);anchor=last_uv(b);amp=torch.linalg.vector_norm(torch.nan_to_num(tu)-anchor[:,:,None],dim=-1);conf=torch.nan_to_num(b['future_confidence'].float()).clamp(0,1);weight=1+motion_weight*(amp/0.05).clamp(0,2)*conf;lxyz=F.smooth_l1_loss(xyz,torch.nan_to_num(tx),reduction='none').mean(-1)[mask].mean();raw=F.smooth_l1_loss(uv,torch.nan_to_num(tu),reduction='none').mean(-1);luv=(raw[mask]*weight[mask]).sum()/weight[mask].sum().clamp_min(1);return lxyz+uv_weight*luv,lxyz,luv,xyz,uv,mask

@torch.no_grad()
def evaluate_detailed(model,loader,device,uv_weight=5.,motion_weight=2.):
 model.eval();losses=[];xyz_sum=uv_sum=n=fxyz=fuv=fn=0.;groups={k:[0.,0.,0.,0.] for k in ('dynamic','static','warp')};bins=[(0,.02),(.02,.05),(.05,.10),(.10,float('inf'))];bucket={f'{lo:.2f}-{hi if np.isfinite(hi) else "inf"}':[0.,0.,0.,0.] for lo,hi in bins}
 for b in loader:
  b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()};loss,_,_,xyz,uv,mask=motion_aware_losses(model,b,uv_weight,motion_weight);losses.append(loss.item());tx=torch.nan_to_num(b['future_xyz_teacher'].float());tu=torch.nan_to_num(b['future_uv_teacher'].float());ex=torch.linalg.vector_norm(xyz-tx,dim=-1);eu=torch.linalg.vector_norm(uv-tu,dim=-1);xyz_sum+=ex[mask].sum().item();uv_sum+=eu[mask].sum().item();n+=mask.sum().item();fm=mask[:,:,-1];fxyz+=ex[:,:,-1][fm].sum().item();fuv+=eu[:,:,-1][fm].sum().item();fn+=fm.sum().item();anchor=last_uv(b);gtamp=torch.linalg.vector_norm(tu[:,:,-1]-anchor,dim=-1);predamp=torch.linalg.vector_norm(uv[:,:,-1]-anchor,dim=-1);histconf=torch.nan_to_num(b['past_confidence'].float()).amax(2).clamp(0,1);pointok=b['past_valid'].bool().any(2)
  selectors={'dynamic':gtamp>=.05,'static':gtamp<.05}
  for name,sel in selectors.items():
   ade=mask&sel[:,:,None];fde=fm&sel;groups[name][0]+=eu[ade].sum().item();groups[name][1]+=ade.sum().item();groups[name][2]+=eu[:,:,-1][fde].sum().item();groups[name][3]+=fde.sum().item()
  # Match the actual soft rasterizer: every valid point contributes in
  # proportion to predicted displacement and historical confidence.
  strength=(predamp/.04).clamp(0,1)*histconf*pointok.float();wade=mask.float()*strength[:,:,None];wfde=fm.float()*strength
  groups['warp'][0]+=(eu*wade).sum().item();groups['warp'][1]+=wade.sum().item();groups['warp'][2]+=(eu[:,:,-1]*wfde).sum().item();groups['warp'][3]+=wfde.sum().item()
  for (lo,hi),name in zip(bins,bucket):
   sel=(gtamp>=lo)&(gtamp<hi);ade=mask&sel[:,:,None];fde=fm&sel;bucket[name][0]+=eu[ade].sum().item();bucket[name][1]+=ade.sum().item();bucket[name][2]+=eu[:,:,-1][fde].sum().item();bucket[name][3]+=fde.sum().item()
 out={'loss':float(np.mean(losses)),'XYZ_ADE':xyz_sum/max(n,1),'XYZ_FDE':fxyz/max(fn,1),'UV_ADE_norm':uv_sum/max(n,1),'UV_FDE_norm':fuv/max(fn,1)}
 for name,(a,an,f,fnn) in groups.items():out[f'{name}_UV_ADE_norm']=a/max(an,1e-8);out[f'{name}_UV_FDE_norm']=f/max(fnn,1e-8);out[f'{name}_points' if name!='warp' else 'warp_effective_points']=int(fnn) if name!='warp' else float(fnn)
 out['motion_bins']={name:{'UV_ADE_norm':v[0]/max(v[1],1),'UV_FDE_norm':v[2]/max(v[3],1),'points':int(v[3])} for name,v in bucket.items()};return out
