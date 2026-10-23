"""Retrain joint trajectories, SimVP baseline and RGB motion compensation on all cached domains."""
from __future__ import annotations
import argparse,json,random,re,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from geometry_residual_model import GeometryMotionCompensator
from simvp_geometry_model import GeometrySimVP
from train_geometry_residual import evaluate as motion_evaluate,forward as motion_forward,loss_fn,validation_metrics
from train_mixed_adt_ds import train_joint
from train_simvp_ablation import move,seed_all,test_metrics,train_mode
from video_forecast_dataset import VideoForecastDataset,video_pad_collate

def args():
 p=argparse.ArgumentParser();p.add_argument('--cache-root',default='data/d4rt_forecasting_cache');p.add_argument('--output-dir',default='outputs/all_datasets');p.add_argument('--subsets',default='adt_mini,ds_mini,po_mini,pstudio_mini,tum');p.add_argument('--trajectory-epochs',type=int,default=80);p.add_argument('--video-epochs',type=int,default=25);p.add_argument('--motion-epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');return p.parse_args()
def group_name(subset,stem):
 if subset=='ds_mini':return stem.split('-')[0]
 return re.sub(r'_\d+$','',stem)
def make_splits(root,subsets,seed):
 result={x:[] for x in ('train','val','test')};by={};rng=random.Random(seed)
 for subset in subsets:
  paths=sorted((Path(root)/subset).glob('*.npz'))
  groups=sorted({group_name(subset,p.stem) for p in paths});rng.shuffle(groups)
  if len(groups)<3: print('skip subset with fewer than 3 groups',subset,len(groups));continue
  nt=max(1,round(.6*len(groups)));nv=max(1,round(.2*len(groups)));sets={'train':set(groups[:nt]),'val':set(groups[nt:nt+nv]),'test':set(groups[nt+nv:])}
  by[subset]={'groups':{k:sorted(v) for k,v in sets.items()}}
  for split,gs in sets.items():
   selected=[p for p in paths if group_name(subset,p.stem) in gs];result[split]+=selected;by[subset][split]=selected
 return result,by
def serializable(by,root):
 rr=Path(root).resolve();return {s:{k:([str(p.resolve().relative_to(rr)) for p in v] if k!='groups' else v) for k,v in d.items()} for s,d in by.items()}
def loaders_for(paths,root,batch=1):
 ds=VideoForecastDataset(root,'test',64,paths=paths,cache_items=True);return ds,DataLoader(ds,batch,collate_fn=video_pad_collate)
def main():
 a=args();seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else 'cpu');out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);subsets=[x.strip() for x in a.subsets.split(',') if x.strip()];splits,by=make_splits(a.cache_root,subsets,a.seed);(out/'split_manifest.json').write_text(json.dumps(serializable(by,a.cache_root),indent=2))
 tr=VideoForecastDataset(a.cache_root,'train',64,paths=splits['train'],cache_items=True);va=VideoForecastDataset(a.cache_root,'val',64,paths=splits['val'],cache_items=True);te=VideoForecastDataset(a.cache_root,'test',64,paths=splits['test'],cache_items=True);print('total',len(tr),len(va),len(te),{s:{k:len(v) if k!='groups' else {q:len(z) for q,z in v.items()} for k,v in d.items()} for s,d in by.items()},flush=True)
 joint,jmetrics=train_joint(a,tr,va,te,device,out);sample=tr[0];c=sample['context_rgb'].shape[0];k=sample['future_rgb'].shape[0]
 cfg=SimpleNamespace(seed=a.seed,run_name='baseline_no_geometry',output_dir=str(out/'video'),batch_size=a.batch_size,epochs=a.video_epochs,image_size=64,injection='film',lr=3e-4,dynamic_loss_weight=0.)
 baseline_metrics=train_mode('no_geometry',cfg,tr,va,te,None,joint,device,c,k)
 bck=torch.load(out/'video/baseline_no_geometry/best.pt',map_location='cpu');base=GeometrySimVP(c,k,64,injection_type=bck['injection']);base.load_state_dict(bck['model']);base.to(device).eval();[p.requires_grad_(False) for p in base.parameters()]
 corrector=GeometryMotionCompensator(sigma=.03,dilation_kernel=3).to(device);opt=torch.optim.AdamW(corrector.parameters(),lr=3e-4,weight_decay=1e-4);train=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate);val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate);test=DataLoader(te,1,collate_fn=video_pad_collate);md=out/'video/geometry_motion_comp';md.mkdir(parents=True,exist_ok=True);best=1e9;hist=[]
 for ep in range(1,a.motion_epochs+1):
  corrector.train();total=0.
  for b in train:
   b=move(b,device);p,r,m=motion_forward(base,corrector,joint,b,k);loss=loss_fn(p,b['future_rgb'],b['context_rgb'][:,-1],r,m,3.,.01,.01);opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=loss.item()
  vm=validation_metrics(base,corrector,joint,val,device,k);score=vm['val_MSE']+.01*vm['val_dynamic_MAE'];hist.append({'epoch':ep,'train_loss':total/len(train),**vm,'score':score});print('motion',ep,vm,flush=True)
  if score<best:best=score;torch.save({'model':corrector.state_dict(),'epoch':ep},md/'best.pt')
 (md/'history.json').write_text(json.dumps(hist,indent=2));corrector.load_state_dict(torch.load(md/'best.pt',map_location=device)['model']);motion_metrics=motion_evaluate(base,corrector,joint,test,device,k,md);(md/'metrics.json').write_text(json.dumps(motion_metrics,indent=2))
 per_domain={}
 for subset,dct in by.items():
  ddir=out/'per_domain'/subset;ddir.mkdir(parents=True,exist_ok=True);_,loader=loaders_for(dct['test'],a.cache_root)
  (ddir/'baseline').mkdir(exist_ok=True);(ddir/'motion').mkdir(exist_ok=True)
  bm=test_metrics(base,None,joint,loader,'no_geometry',device,k,ddir/'baseline')
  mm=motion_evaluate(base,corrector,joint,loader,device,k,ddir/'motion');per_domain[subset]={'baseline':bm,'motion':mm}
 summary={'joint':jmetrics,'baseline':baseline_metrics,'motion':motion_metrics,'per_domain':per_domain};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
