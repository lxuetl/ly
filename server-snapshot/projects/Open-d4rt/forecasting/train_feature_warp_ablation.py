"""Progressive ablation for geometry-guided SimVP decoder feature warping."""
from __future__ import annotations
import argparse,json,math,random,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from simvp_feature_warp import GeometryFeatureWarpSimVP
from simvp_geometry_model import GeometrySimVP
from trajectory_model import GeometryConditionEncoder,JointTrajectoryForecaster
from train_mixed_adt_ds import make_splits
from train_simvp_ablation import move,seed_all
from video_forecast_dataset import VideoForecastDataset,video_pad_collate

VARIANTS={'feature_warp':(0,0,0),'feature_warp_consistency':(1,0,0),'feature_warp_visibility':(1,1,0),'feature_warp_dense':(1,1,1)}
def args():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',default='data/d4rt_forecasting_cache');p.add_argument('--mixed-root',default='outputs/mixed_adt_ds');p.add_argument('--output-dir',default='outputs/mixed_adt_ds/video/feature_warp_ablation');p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');return p.parse_args()
def geom(joint,b):
 with torch.no_grad(): xyz,uv=joint(b['past_xyz'].float(),b['past_uv'].float(),b['past_valid'].bool(),b['past_confidence'].float())
 conf=torch.nan_to_num(b['past_confidence'].float()).amax(2)[:,:,None].expand(-1,-1,xyz.shape[2]);return xyz,uv,conf
@torch.no_grad()
def validate(model,joint,loader,d):
 model.eval();sq=cnt=da=dn=0.
 for b in loader:
  b=move(b,d);xyz,uv,conf=geom(joint,b);p,_=model(b['context_rgb'],uv,xyz,conf,b['past_uv'].float(),b['past_valid'].bool());y=b['future_rgb'];sq+=(p-y).square().sum().item();cnt+=y.numel();m=((y-b['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)>0.05).expand_as(y);da+=(p-y).abs()[m].sum().item();dn+=m.sum().item()
 return sq/cnt,da/max(dn,1)
@torch.no_grad()
def test(model,joint,loader,d):
 model.eval();a=sq=cnt=ss=sn=da=ds=dn=inter=union=pp=gp=gate=0.
 for b in loader:
  b=move(b,d);xyz,uv,conf=geom(joint,b);p,g=model(b['context_rgb'],uv,xyz,conf,b['past_uv'].float(),b['past_valid'].bool());gate+=g.item();y=b['future_rgb'];e=p-y;a+=e.abs().sum().item();sq+=e.square().sum().item();cnt+=y.numel();last=b['context_rgb'][:,-1:];gm=(y-last).abs().mean(2)>0.05;pm=(p-last).abs().mean(2)>0.05;m=gm[:,:,None].expand_as(y);da+=e.abs()[m].sum().item();ds+=e.square()[m].sum().item();dn+=m.sum().item();inter+=(gm&pm).sum().item();union+=(gm|pm).sum().item();pp+=pm.sum().item();gp+=gm.sum().item()
  pn=p.cpu().numpy();yn=y.cpu().numpy()
  for bi in range(len(pn)):
   for ti in range(pn.shape[1]):ss+=structural_similarity(yn[bi,ti].transpose(1,2,0),pn[bi,ti].transpose(1,2,0),channel_axis=-1,data_range=1.);sn+=1
 mse=sq/cnt;dm=ds/max(dn,1);pr=inter/max(pp,1);re=inter/max(gp,1);return {'MAE':a/cnt,'MSE':mse,'PSNR':-10*math.log10(mse),'SSIM':float(ss/sn),'dynamic_MAE':da/max(dn,1),'dynamic_PSNR':-10*math.log10(dm),'motion_mask_IoU':inter/max(union,1),'motion_mask_F1':2*pr*re/max(pr+re,1e-12),'motion_mask_precision':pr,'motion_mask_recall':re,'mean_feature_gate':gate/len(loader)}
def main():
 a=args();d=torch.device(a.device if torch.cuda.is_available() else 'cpu');splits=make_splits(a.cache_root,a.seed);ds=lambda s:VideoForecastDataset(a.cache_root,s,64,paths=splits[s],cache_items=True);tr,va,te=ds('train'),ds('val'),ds('test');vl=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);tl=DataLoader(te,1,collate_fn=video_pad_collate);root=Path(a.mixed_root)
 bck=torch.load(root/'video/baseline_no_geometry/best.pt',map_location='cpu');jck=torch.load(root/'joint_trajectory/best.pt',map_location='cpu');joint=JointTrajectoryForecaster(jck['future_frames'],jck['hidden_dim']);joint.load_state_dict(jck['model']);joint.to(d).eval();[p.requires_grad_(False) for p in joint.parameters()];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);summary={}
 for name,flags in VARIANTS.items():
  seed_all(a.seed);base=GeometrySimVP(bck['context_frames'],bck['future_frames'],bck['image_size'],injection_type=bck['injection']);base.load_state_dict(bck['model']);model=GeometryFeatureWarpSimVP(base,*map(bool,flags)).to(d);opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=1e-4);loader=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate);od=out/name;od.mkdir(parents=True,exist_ok=True);best=1e9;hist=[]
  for ep in range(1,a.epochs+1):
   model.train();model.base.eval();total=0.
   for b in loader:
    b=move(b,d);xyz,uv,conf=geom(joint,b);p,g=model(b['context_rgb'],uv,xyz,conf,b['past_uv'].float(),b['past_valid'].bool());y=b['future_rgb'];motion=((y-b['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)>0.05).float();loss=(((p-y).abs()+.1*(p-y).square())*(1+3*motion)).mean()+.005*g;opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()
   vm,vd=validate(model,joint,vl,d);score=vm+.01*vd;hist.append({'epoch':ep,'train_loss':total/len(loader),'val_MSE':vm,'val_dynamic_MAE':vd,'score':score});print(name,ep,vm,vd,flush=True)
   if score<best:best=score;torch.save({'model':model.state_dict(),'flags':flags},od/'best.pt')
  (od/'history.json').write_text(json.dumps(hist,indent=2));model.load_state_dict(torch.load(od/'best.pt',map_location=d)['model']);summary[name]=test(model,joint,tl,d);(od/'metrics.json').write_text(json.dumps(summary[name],indent=2))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
