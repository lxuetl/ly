"""Evaluate ADT-trained SimVP variants directly on every DS-mini cache."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from simvp_geometry_model import GeometrySimVP
from trajectory_model import JointTrajectoryForecaster,LiteTrajectoryForecaster
from train_simvp_ablation import test_metrics
from video_forecast_dataset import VideoForecastDataset,video_pad_collate

def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--cache-root',default='data/d4rt_forecasting_cache'); p.add_argument('--subset',default='ds_mini'); p.add_argument('--output-dir',default='outputs/ds_mini_cross_domain'); p.add_argument('--device',default='cuda'); return p.parse_args()
def main():
 a=args(); d=torch.device(a.device if torch.cuda.is_available() else 'cpu'); ds=VideoForecastDataset(a.cache_root,'all',64,subset=a.subset); loader=DataLoader(ds,1,collate_fn=video_pad_collate,num_workers=0); sample=ds[0]; c=sample['context_rgb'].shape[0]; k=sample['future_rgb'].shape[0]
 lck=torch.load('outputs/trajectory_forecaster/best.pt',map_location='cpu'); lite=LiteTrajectoryForecaster(lck['future_frames'],lck['hidden_dim']); lite.load_state_dict(lck['model']); lite.to(d).eval()
 jck=torch.load('outputs/joint_trajectory_forecaster/best.pt',map_location='cpu'); joint=JointTrajectoryForecaster(jck['future_frames'],jck['hidden_dim']); joint.load_state_dict(jck['model']); joint.to(d).eval()
 specs={
  'no_geometry':('outputs/simvp_dynamic_ablation/no_geometry/best.pt','no_geometry'),
  'joint_add':('outputs/simvp_dynamic_ablation/joint_future_geometry/best.pt','joint_future_geometry'),
  'oracle_add':('outputs/simvp_dynamic_ablation/oracle_future_geometry/best.pt','oracle_future_geometry'),
  'joint_film':('outputs/simvp_film_ablation/film_joint/best.pt','joint_future_geometry'),
 }
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); summary={}
 for name,(path,mode) in specs.items():
  ck=torch.load(path,map_location='cpu'); model=GeometrySimVP(c,k,64,injection_type=ck.get('injection','add')); model.load_state_dict(ck['model'],strict=False); model.to(d).eval(); md=out/name; md.mkdir(exist_ok=True)
  metrics=test_metrics(model,lite,joint,loader,mode,d,k,md); metrics['num_sequences']=len(ds); metrics['trained_subset']='adt_mini'; metrics['test_subset']=a.subset; summary[name]=metrics; (md/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(name,json.dumps(metrics),flush=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)); print('summary',out/'summary.json')
if __name__=='__main__': main()
