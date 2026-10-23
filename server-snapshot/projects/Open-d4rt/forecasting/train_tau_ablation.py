"""Train fair TAU geometry ablations and local RGB motion compensation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from geometry_residual_model import GeometryMotionCompensator
from simvp_geometry_model import build_video_predictor_from_checkpoint
from train_all_datasets import make_splits,serializable
from train_geometry_residual import evaluate as motion_evaluate,forward as motion_forward,loss_fn,validation_metrics
from train_simvp_ablation import move,seed_all,train_mode
from trajectory_model import JointTrajectoryForecaster
from video_forecast_dataset import VideoForecastDataset,video_pad_collate


def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache-root',default='data/d4rt_forecasting_cache')
 p.add_argument('--joint-checkpoint',default='outputs/all_datasets/joint_trajectory/best.pt')
 p.add_argument('--output-dir',default='outputs/tau_ablation')
 p.add_argument('--subsets',default='adt_mini,ds_mini,po_mini,pstudio_mini,tum')
 p.add_argument('--epochs',type=int,default=25);p.add_argument('--motion-epochs',type=int,default=30)
 p.add_argument('--batch-size',type=int,default=2);p.add_argument('--image-size',type=int,default=64)
 p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--device',default='cuda');p.add_argument('--hid-s',type=int,default=64)
 p.add_argument('--hid-t',type=int,default=256);p.add_argument('--n-s',type=int,default=2)
 p.add_argument('--n-t',type=int,default=6);p.add_argument('--drop-path',type=float,default=.1)
 p.add_argument('--modes',default='no_geometry,history_geometry,joint_future_geometry')
 p.add_argument('--skip-motion',action='store_true')
 return p.parse_args()


def main():
 a=args();seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 subsets=[x.strip() for x in a.subsets.split(',') if x.strip()]
 splits,by=make_splits(a.cache_root,subsets,a.seed)
 (out/'split_manifest.json').write_text(json.dumps(serializable(by,a.cache_root),indent=2))
 mk=lambda s:VideoForecastDataset(a.cache_root,s,a.image_size,paths=splits[s],cache_items=True)
 tr,va,te=mk('train'),mk('val'),mk('test');sample=tr[0]
 c,k=sample['context_rgb'].shape[0],sample['future_rgb'].shape[0]
 jck=torch.load(a.joint_checkpoint,map_location='cpu')
 joint=JointTrajectoryForecaster(jck['future_frames'],jck['hidden_dim'])
 joint.load_state_dict(jck['model']);joint.to(device).eval();[p.requires_grad_(False) for p in joint.parameters()]
 cfg=SimpleNamespace(seed=a.seed,run_name='',output_dir=str(out),batch_size=a.batch_size,
     epochs=a.epochs,image_size=a.image_size,injection='film',lr=a.lr,dynamic_loss_weight=0.,
     backbone='tau',hid_s=a.hid_s,hid_t=a.hid_t,n_s=a.n_s,n_t=a.n_t,drop_path=a.drop_path)
 summary={}
 modes=[x.strip() for x in a.modes.split(',') if x.strip()]
 for mode in modes:
  summary[mode]=train_mode(mode,cfg,tr,va,te,None,joint,device,c,k)

 if a.skip_motion:
  summary['meta']={'backbone':'tau','context_frames':c,'future_frames':k,'train_clips':len(tr),
                   'val_clips':len(va),'test_clips':len(te)}
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return
 bck=torch.load(out/'no_geometry/best.pt',map_location='cpu')
 base=build_video_predictor_from_checkpoint(bck);base.load_state_dict(bck['model']);base.to(device).eval()
 [p.requires_grad_(False) for p in base.parameters()]
 corrector=GeometryMotionCompensator(sigma=.03,dilation_kernel=3).to(device)
 opt=torch.optim.AdamW(corrector.parameters(),lr=3e-4,weight_decay=1e-4)
 train=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate)
 val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);test=DataLoader(te,1,collate_fn=video_pad_collate)
 md=out/'rgb_local_motion_comp';md.mkdir(exist_ok=True);best=float('inf');history=[]
 for ep in range(1,a.motion_epochs+1):
  corrector.train();total=0.
  for batch in train:
   batch=move(batch,device);pred,residual,mask=motion_forward(base,corrector,joint,batch,k)
   loss=loss_fn(pred,batch['future_rgb'],batch['context_rgb'][:,-1],residual,mask,3.,.01,.01)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(corrector.parameters(),1);opt.step();total+=loss.item()
  vm=validation_metrics(base,corrector,joint,val,device,k);score=vm['val_MSE']+.01*vm['val_dynamic_MAE']
  history.append({'epoch':ep,'train_loss':total/len(train),**vm,'score':score})
  print('tau motion',ep,vm,flush=True)
  if score<best:best=score;torch.save({'model':corrector.state_dict(),'epoch':ep},md/'best.pt')
 (md/'history.json').write_text(json.dumps(history,indent=2))
 corrector.load_state_dict(torch.load(md/'best.pt',map_location=device)['model'])
 summary['rgb_local_motion_comp']=motion_evaluate(base,corrector,joint,test,device,k,md)
 (md/'metrics.json').write_text(json.dumps(summary['rgb_local_motion_comp'],indent=2))
 summary['meta']={'backbone':'tau','context_frames':c,'future_frames':k,'train_clips':len(tr),
                  'val_clips':len(va),'test_clips':len(te),'params':sum(p.numel() for p in base.parameters())}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
