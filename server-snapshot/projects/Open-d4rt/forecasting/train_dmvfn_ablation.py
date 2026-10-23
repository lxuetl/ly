"""Train DMVFN and test reliable D4RT-guided local motion compensation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from dmvfn_adapter import DMVFNVideoPredictor
from geometry_residual_model import GeometryMotionCompensator
from train_all_datasets import make_splits,serializable
from train_geometry_residual import evaluate as motion_evaluate,loss_fn,latest_uv
from train_simvp_ablation import make_geometry,move,seed_all,test_metrics
from trajectory_model import JointTrajectoryForecaster
from video_forecast_dataset import VideoForecastDataset,video_pad_collate


def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache-root',default='data/d4rt_forecasting_cache');p.add_argument('--output-dir',default='outputs/dmvfn_ablation')
 p.add_argument('--joint-checkpoint',default='outputs/all_datasets/joint_trajectory/best.pt')
 p.add_argument('--backbone-checkpoint',default='',help='Load a trained DMVFN checkpoint and skip backbone training')
 p.add_argument('--subsets',default='adt_mini,ds_mini,po_mini,pstudio_mini,tum')
 p.add_argument('--epochs',type=int,default=60);p.add_argument('--motion-epochs',type=int,default=60)
 p.add_argument('--batch-size',type=int,default=2);p.add_argument('--lr',type=float,default=3e-4)
 p.add_argument('--motion-loss-weight',type=float,default=0.,help='Maximum extra weight for large-motion pixels')
 p.add_argument('--motion-scale',type=float,default=.05,help='RGB temporal difference mapped to unit motion')
 p.add_argument('--dynamic-oversample',type=float,default=0.,help='Strength of clip-level dynamic oversampling')
 p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');return p.parse_args()


def motion_weighted_loss(pred,target,last,extra_weight,motion_scale):
 """Continuously emphasize moving pixels while keeping the loss scale stable."""
 error=(pred-target).abs()+.1*(pred-target).square()
 amplitude=(target-last[:,None]).abs().mean(2,keepdim=True)
 weight=1.+float(extra_weight)*(amplitude/float(motion_scale)).clamp(0.,1.)
 return (error*weight).sum()/(weight.sum()*error.shape[2]).clamp_min(1.)


def dynamic_sampler(dataset,strength,seed,motion_threshold=.05):
 """Mildly resample clips by their fraction of genuinely changing pixels."""
 ratios=[]
 for i in range(len(dataset)):
  item=dataset[i];last=item['context_rgb'][-1:]
  ratios.append(((item['future_rgb']-last).abs().mean(1)>motion_threshold).float().mean().item())
 ratios=torch.tensor(ratios,dtype=torch.double);relative=ratios/ratios.mean().clamp_min(1e-8)
 weights=(1.+float(strength)*(relative-1.)).clamp(.5,3.)
 sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=torch.Generator().manual_seed(seed))
 return sampler,{'min':weights.min().item(),'max':weights.max().item(),'mean':weights.mean().item(),'dynamic_fraction_mean':ratios.mean().item()}


@torch.no_grad()
def val_mse(model,loader,device,seed):
 torch.manual_seed(seed);model.eval();total=count=0.
 for b in loader:
  b=move(b,device);p=model(b['context_rgb']);total+=F.mse_loss(p,b['future_rgb'],reduction='sum').item();count+=b['future_rgb'].numel()
 return total/count


def reliable_forward(base,corrector,joint,batch,k):
 with torch.no_grad():
  base_pred=base(batch['context_rgb']);geometry,weight,uv=make_geometry('joint_future_geometry_reliable',None,joint,batch,k)
 return corrector(base_pred,batch['context_rgb'][:,-1],geometry,uv,weight,latest_uv(batch))


@torch.no_grad()
def motion_val(base,corrector,joint,loader,device,k):
 corrector.eval();total=count=dyn_abs=dyn_count=0.
 for b in loader:
  b=move(b,device);p,_,_=reliable_forward(base,corrector,joint,b,k);total+=F.mse_loss(p,b['future_rgb'],reduction='sum').item();count+=b['future_rgb'].numel()
  dyn=((b['future_rgb']-b['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)>.05).expand_as(p)
  dyn_abs+=(p-b['future_rgb']).abs()[dyn].sum().item();dyn_count+=dyn.sum().item()
 return {'val_MSE':total/count,'val_dynamic_MAE':dyn_abs/max(dyn_count,1)}


def main():
 a=args();seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else 'cpu');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 subsets=[x.strip() for x in a.subsets.split(',') if x.strip()];splits,by=make_splits(a.cache_root,subsets,a.seed)
 (out/'split_manifest.json').write_text(json.dumps(serializable(by,a.cache_root),indent=2))
 mk=lambda s:VideoForecastDataset(a.cache_root,s,64,paths=splits[s],cache_items=True);tr,va,te=mk('train'),mk('val'),mk('test')
 sampler,sampler_stats=(dynamic_sampler(tr,a.dynamic_oversample,a.seed) if a.dynamic_oversample>0 else (None,None))
 train=DataLoader(tr,a.batch_size,shuffle=sampler is None,sampler=sampler,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate)
 val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);test=DataLoader(te,1,collate_fn=video_pad_collate)
 c,k=tr[0]['context_rgb'].shape[0],tr[0]['future_rgb'].shape[0];model=DMVFNVideoPredictor(c,k).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);bd=out/'no_geometry';bd.mkdir(exist_ok=True);best=float('inf');history=[]
 if a.backbone_checkpoint:
  ck=torch.load(a.backbone_checkpoint,map_location=device);model.load_state_dict(ck['model'])
  (bd/'source_checkpoint.txt').write_text(str(Path(a.backbone_checkpoint).resolve()))
 else:
  for ep in range(1,a.epochs+1):
   model.train();total=0.
   for b in train:
    b=move(b,device);pred=model.teacher_forced_stages(b['context_rgb'],b['future_rgb'])[-1]
    loss=motion_weighted_loss(pred,b['future_rgb'],b['context_rgb'][:,-1],a.motion_loss_weight,a.motion_scale)
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()
   vm=val_mse(model,val,device,a.seed);history.append({'epoch':ep,'train_loss':total/len(train),'val_MSE':vm});print('dmvfn',ep,history[-1],flush=True)
   if vm<best:best=vm;torch.save({'model':model.state_dict(),'epoch':ep,'context_frames':c,'future_frames':k},bd/'best.pt')
  (bd/'history.json').write_text(json.dumps(history,indent=2));ck=torch.load(bd/'best.pt',map_location=device);model.load_state_dict(ck['model'])
 torch.manual_seed(a.seed);base_metrics=test_metrics(model,None,None,test,'no_geometry',device,k,bd);(bd/'metrics.json').write_text(json.dumps(base_metrics,indent=2))
 model.eval();[p.requires_grad_(False) for p in model.parameters()]
 jck=torch.load(a.joint_checkpoint,map_location='cpu');joint=JointTrajectoryForecaster(jck['future_frames'],jck['hidden_dim']);joint.load_state_dict(jck['model']);joint.to(device).eval();[p.requires_grad_(False) for p in joint.parameters()]
 corrector=GeometryMotionCompensator(sigma=.03,dilation_kernel=3).to(device);opt=torch.optim.AdamW(corrector.parameters(),lr=3e-4,weight_decay=1e-4)
 md=out/'reliable_geometry_motion_comp';md.mkdir(exist_ok=True);best=float('inf');mh=[]
 for ep in range(1,a.motion_epochs+1):
  corrector.train();total=0.
  for b in train:
   b=move(b,device);p,r,m=reliable_forward(model,corrector,joint,b,k);loss=loss_fn(p,b['future_rgb'],b['context_rgb'][:,-1],r,m,3.,.01,.01)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(corrector.parameters(),1);opt.step();total+=loss.item()
  vm=motion_val(model,corrector,joint,val,device,k);score=vm['val_MSE']+.01*vm['val_dynamic_MAE'];mh.append({'epoch':ep,'train_loss':total/len(train),**vm,'score':score});print('dmvfn motion',ep,vm,flush=True)
  if score<best:best=score;torch.save({'model':corrector.state_dict(),'epoch':ep},md/'best.pt')
 (md/'history.json').write_text(json.dumps(mh,indent=2));corrector.load_state_dict(torch.load(md/'best.pt',map_location=device)['model'])
 # Reuse evaluator by temporarily supplying the reliable forward contract locally.
 from train_geometry_residual import evaluate
 import train_geometry_residual as residual_module
 original=residual_module.forward;residual_module.forward=reliable_forward
 try: motion_metrics=evaluate(model,corrector,joint,test,device,k,md)
 finally: residual_module.forward=original
 (md/'metrics.json').write_text(json.dumps(motion_metrics,indent=2))
 summary={'no_geometry':base_metrics,'reliable_geometry_motion_comp':motion_metrics,'meta':{'backbone':'dmvfn','params':sum(p.numel() for p in model.parameters()),'best_epoch':ck['epoch'],'train_clips':len(tr),'val_clips':len(va),'test_clips':len(te),'motion_loss_weight':a.motion_loss_weight,'motion_scale':a.motion_scale,'dynamic_oversample':a.dynamic_oversample,'sampler_stats':sampler_stats}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
