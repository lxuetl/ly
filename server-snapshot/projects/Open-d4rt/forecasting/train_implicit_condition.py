"""Train implicit scene-conditioned trajectories and evaluate downstream RGB compensation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from geometry_residual_model import GeometryMotionCompensator
from simvp_geometry_model import GeometrySimVP
from trajectory_model import ImplicitConditionedJointForecaster
from train_all_datasets import make_splits
from train_geometry_residual import evaluate as motion_evaluate,forward as motion_forward,loss_fn,validation_metrics
from train_joint_forecaster import evaluate,losses,move
from trajectory_metrics import evaluate_detailed,motion_aware_losses
from trajectory_sampling import clip_histograms,curriculum_loader,curriculum_losses
from train_simvp_ablation import seed_all
from video_forecast_dataset import VideoForecastDataset,video_pad_collate
def args():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',default='data/d4rt_forecasting_cache');p.add_argument('--base-root',default='outputs/all_datasets');p.add_argument('--output-dir',default='outputs/implicit_condition');p.add_argument('--epochs',type=int,default=80);p.add_argument('--motion-epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--motion-aware',action='store_true');p.add_argument('--uv-motion-weight',type=float,default=2.);p.add_argument('--curriculum-sampling',action='store_true',help='curriculum clip and point motion-bucket sampling');return p.parse_args()
def main():
 a=args();seed_all(a.seed);d=torch.device(a.device if torch.cuda.is_available() else 'cpu');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);splits,by=make_splits(a.cache_root,['adt_mini','ds_mini','po_mini','pstudio_mini'],a.seed);mk=lambda s:VideoForecastDataset(a.cache_root,s,64,paths=splits[s],cache_items=True);tr,va,te=mk('train'),mk('val'),mk('test');train=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate);clip_hist=clip_histograms(tr) if a.curriculum_sampling else None;val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);test=DataLoader(te,1,collate_fn=video_pad_collate);k=tr[0]['future_xyz_teacher'].shape[1];model=ImplicitConditionedJointForecaster(k,128).to(d);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);td=out/'joint_trajectory';td.mkdir(exist_ok=True);best=1e9;hist=[]
 for ep in range(1,a.epochs+1):
  model.train();total=0.
  epoch_train,global_mix=(curriculum_loader(tr,clip_hist,ep,a.epochs,a.batch_size,a.seed) if a.curriculum_sampling else (train,None));sampled=[0,0,0,0]
  for b in epoch_train:
   b=move(b,d)
   if a.curriculum_sampling:
    result=curriculum_losses(model,b,ep,a.epochs,5.);loss,*_,diag=result
    sampled=[x+y for x,y in zip(sampled,diag['sampled_counts'])]
   else:loss,*_=(motion_aware_losses(model,b,5.,a.uv_motion_weight) if a.motion_aware else losses(model,b,5.))
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()
  vm=evaluate_detailed(model,val,d,5.,a.uv_motion_weight);score=vm['loss']+.02*vm['dynamic_UV_ADE_norm']+.03*vm['motion_bins']['0.10-inf']['UV_ADE_norm']+.02*vm['warp_UV_ADE_norm'] if a.curriculum_sampling else vm['loss']+(.02*vm['dynamic_UV_ADE_norm'] if a.motion_aware else 0);hist.append({'epoch':ep,'train_loss':total/len(epoch_train),'score':score,'curriculum_sampled_counts':sampled,'train_natural_mix':global_mix,**vm});print('trajectory',ep,vm,'sampled',sampled,flush=True)
  if score<best:best=score;torch.save({'model':model.state_dict(),'future_frames':k,'hidden_dim':128,'curriculum_sampling':a.curriculum_sampling,'epoch':ep},td/'best.pt')
 (td/'history.json').write_text(json.dumps(hist,indent=2));model.load_state_dict(torch.load(td/'best.pt',map_location=d)['model']);tm=evaluate_detailed(model,test,d,5.,a.uv_motion_weight);(td/'metrics.json').write_text(json.dumps(tm,indent=2));model.eval();[p.requires_grad_(False) for p in model.parameters()]
 bck=torch.load(Path(a.base_root)/'video/baseline_no_geometry/best.pt',map_location='cpu');base=GeometrySimVP(bck['context_frames'],bck['future_frames'],bck['image_size'],injection_type=bck['injection']);base.load_state_dict(bck['model']);base.to(d).eval();[p.requires_grad_(False) for p in base.parameters()];corr=GeometryMotionCompensator(sigma=.03,dilation_kernel=3).to(d);opt=torch.optim.AdamW(corr.parameters(),lr=3e-4,weight_decay=1e-4);train2=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate);val2=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);md=out/'geometry_motion_comp';md.mkdir(exist_ok=True);best=1e9;mh=[]
 for ep in range(1,a.motion_epochs+1):
  corr.train();total=0.
  for b in train2:b=move(b,d);p,r,m=motion_forward(base,corr,model,b,k);loss=loss_fn(p,b['future_rgb'],b['context_rgb'][:,-1],r,m,3.,.01,.01);opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()
  vm=validation_metrics(base,corr,model,val2,d,k);score=vm['val_MSE']+.01*vm['val_dynamic_MAE'];mh.append({'epoch':ep,'train_loss':total/len(train2),**vm,'score':score});print('motion',ep,vm,flush=True)
  if score<best:best=score;torch.save({'model':corr.state_dict(),'epoch':ep},md/'best.pt')
 (md/'history.json').write_text(json.dumps(mh,indent=2));corr.load_state_dict(torch.load(md/'best.pt',map_location=d)['model']);allm=motion_evaluate(base,corr,model,test,d,k,md);per={}
 for subset,v in by.items():
  od=out/'per_domain'/subset;od.mkdir(parents=True,exist_ok=True);ds=VideoForecastDataset(a.cache_root,'test',64,paths=v['test'],cache_items=True);loader=DataLoader(ds,1,collate_fn=video_pad_collate);per[subset]={'trajectory':evaluate_detailed(model,loader,d,5.,a.uv_motion_weight),'motion':motion_evaluate(base,corr,model,loader,d,k,od)}
 summary={'trajectory':tm,'motion':allm,'per_domain':per};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
