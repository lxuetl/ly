"""Train the joint future XYZ/UV trajectory predictor."""

from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from trajectory_model import JointTrajectoryForecaster
from video_forecast_dataset import VideoForecastDataset,video_pad_collate

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cache-root',default='data/d4rt_forecasting_cache'); p.add_argument('--output-dir',default='outputs/joint_trajectory_forecaster'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--hidden-dim',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--uv-weight',type=float,default=5.0); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda'); return p.parse_args()

def move(b,d): return {k:(v.to(d) if torch.is_tensor(v) else v) for k,v in b.items()}
def losses(model,b,uv_weight):
 xyz,uv=model(b['past_xyz'].float(),b['past_uv'].float(),b['past_valid'],b['past_confidence'].float())
 target_xyz=b['future_xyz_teacher'].float(); target_uv=b['future_uv_teacher'].float()
 mask=b['future_valid'].bool()&torch.isfinite(target_xyz).all(-1)&torch.isfinite(target_uv).all(-1)
 lxyz=F.smooth_l1_loss(xyz,torch.nan_to_num(target_xyz),reduction='none').mean(-1)[mask].mean()
 luv=F.smooth_l1_loss(uv,torch.nan_to_num(target_uv),reduction='none').mean(-1)[mask].mean()
 return lxyz+uv_weight*luv,lxyz,luv,xyz,uv,mask
@torch.no_grad()
def evaluate(model,loader,d,uv_weight):
 model.eval(); vals=[]; xyz_sum=uv_sum=n=fxyz=fuv=fn=0.0
 for b in loader:
  b=move(b,d); loss,_,_,xyz,uv,mask=losses(model,b,uv_weight); vals.append(loss.item())
  ex=torch.linalg.vector_norm(xyz-torch.nan_to_num(b['future_xyz_teacher'].float()),dim=-1); eu=torch.linalg.vector_norm(uv-torch.nan_to_num(b['future_uv_teacher'].float()),dim=-1)
  xyz_sum+=ex[mask].sum().item(); uv_sum+=eu[mask].sum().item(); n+=mask.sum().item(); fm=mask[:,:,-1]; fxyz+=ex[:,:,-1][fm].sum().item(); fuv+=eu[:,:,-1][fm].sum().item(); fn+=fm.sum().item()
 return {'loss':float(np.mean(vals)),'XYZ_ADE':xyz_sum/max(n,1),'XYZ_FDE':fxyz/max(fn,1),'UV_ADE_norm':uv_sum/max(n,1),'UV_FDE_norm':fuv/max(fn,1)}
def main():
 a=args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); d=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 tr=VideoForecastDataset(a.cache_root,'train',64); va=VideoForecastDataset(a.cache_root,'val',64); te=VideoForecastDataset(a.cache_root,'test',64)
 train=DataLoader(tr,a.batch_size,shuffle=True,collate_fn=video_pad_collate,num_workers=0); val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate,num_workers=0); test=DataLoader(te,1,collate_fn=video_pad_collate,num_workers=0)
 k=tr[0]['future_xyz_teacher'].shape[1]; model=JointTrajectoryForecaster(k,a.hidden_dim).to(d); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); best=float('inf'); hist=[]
 print(f'data train={len(tr)} val={len(va)} test={len(te)} params={sum(p.numel() for p in model.parameters()):,}')
 for e in range(1,a.epochs+1):
  model.train(); total=0.
  for b in train:
   b=move(b,d); loss,*_=losses(model,b,a.uv_weight); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()
  vm=evaluate(model,val,d,a.uv_weight); row={'epoch':e,'train_loss':total/len(train),**vm}; hist.append(row)
  if vm['loss']<best: best=vm['loss']; torch.save({'model':model.state_dict(),'future_frames':k,'hidden_dim':a.hidden_dim,'uv_weight':a.uv_weight},out/'best.pt')
  if e==1 or e%10==0: print(f"epoch={e:03d} train={row['train_loss']:.6f} val={vm['loss']:.6f} xyz={vm['XYZ_ADE']:.5f} uv={vm['UV_ADE_norm']:.5f}",flush=True)
 (out/'history.json').write_text(json.dumps(hist,indent=2)); ck=torch.load(out/'best.pt',map_location=d); model.load_state_dict(ck['model']); metrics=evaluate(model,test,d,a.uv_weight); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
