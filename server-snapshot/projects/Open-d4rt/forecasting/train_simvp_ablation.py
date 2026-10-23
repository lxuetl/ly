"""Stages 5-6: train fair SimVP geometry ablations and evaluate them."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from simvp_geometry_model import build_video_predictor
from trajectory_model import GeometryConditionEncoder, JointTrajectoryForecaster, LiteTrajectoryForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


MODES = ("no_geometry", "history_geometry", "future_pred_geometry", "oracle_future_geometry", "joint_future_geometry", "joint_future_geometry_reliable")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--trajectory-checkpoint", default="outputs/trajectory_forecaster/best.pt")
    p.add_argument("--joint-trajectory-checkpoint", default="outputs/joint_trajectory_forecaster/best.pt")
    p.add_argument("--output-dir", default="outputs/simvp_ablation")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda")
    p.add_argument("--injection", choices=("add","film"), default="add")
    p.add_argument("--dynamic-loss-weight", type=float, default=0.0)
    p.add_argument("--run-name", default="", help="Output/summary key for controlled variants.")
    p.add_argument("--backbone", choices=("simvp","tau"), default="simvp")
    p.add_argument("--hid-s", type=int, default=None); p.add_argument("--hid-t", type=int, default=None)
    p.add_argument("--n-s", type=int, default=2); p.add_argument("--n-t", type=int, default=None)
    p.add_argument("--drop-path", type=float, default=None)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_geometry(mode, trajectory, joint_trajectory, batch, future_frames):
    if mode == "no_geometry": return None, None, None
    valid = batch["past_valid"].bool(); point_ok = valid.any(2)
    if mode == "future_pred_geometry":
        with torch.no_grad():
            xyz = trajectory(batch["past_xyz"].float(), batch["past_uv"].float(), valid,
                             batch["past_confidence"].float())
        conf = torch.nan_to_num(batch["past_confidence"].float()).amax(2)
        conf = conf[:, :, None].expand(-1, -1, future_frames)
        uv_hist = torch.nan_to_num(batch["past_uv"].float())
        pair = valid[:, :, 1:] & valid[:, :, :-1]
        velocity = uv_hist[:, :, 1:] - uv_hist[:, :, :-1]
        if pair.shape[2] > 4: pair[:, :, :-4] = False
        uv_vel = (velocity * pair[..., None]).sum(2) / pair.sum(2).clamp_min(1)[..., None]
        last_idx = (valid.long() * torch.arange(1, valid.shape[2]+1, device=valid.device)).argmax(2)
        last_uv = uv_hist.gather(2, last_idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        steps = torch.arange(1,future_frames+1,device=valid.device).view(1,1,-1,1)
        raster_uv = (last_uv[:,:,None] + uv_vel[:,:,None] * steps).clamp(0,1)
    elif mode in ("joint_future_geometry","joint_future_geometry_reliable"):
        with torch.no_grad():
            xyz,raster_uv=joint_trajectory(batch["past_xyz"].float(),batch["past_uv"].float(),valid,batch["past_confidence"].float())
        conf=torch.nan_to_num(batch["past_confidence"].float()).amax(2)[:,:,None].expand(-1,-1,future_frames)
        if mode == "joint_future_geometry_reliable":
            uv_hist=torch.nan_to_num(batch['past_uv'].float())
            last_idx=(valid.long()*torch.arange(1,valid.shape[2]+1,device=valid.device)).argmax(2)
            last_uv=uv_hist.gather(2,last_idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            displacement=torch.linalg.vector_norm(raster_uv-last_uv[:,:,None],dim=-1)
            # Ignore near-static jitter; smoothly grant authority between 0.015 and 0.05.
            dynamic=((displacement-.015)/.035).clamp(0,1)
            velocity=torch.cat((raster_uv[:,:,:1]-last_uv[:,:,None],raster_uv[:,:,1:]-raster_uv[:,:,:-1]),2)
            acceleration=torch.zeros_like(displacement)
            acceleration[:,:,1:]=torch.linalg.vector_norm(velocity[:,:,1:]-velocity[:,:,:-1],dim=-1)
            consistency=torch.exp(-acceleration/.05)
            point_ok=point_ok[:,:,None].float()*conf.clamp(0,1)*dynamic*consistency
    elif mode == "history_geometry":
        xyz = torch.nan_to_num(batch["past_xyz"].float()[:, :, -future_frames:])
        conf = torch.nan_to_num(batch["past_confidence"].float()[:, :, -future_frames:])
        raster_uv = torch.nan_to_num(batch["past_uv"].float()[:, :, -future_frames:]).clamp(0,1)
    elif mode == "oracle_future_geometry":
        # Upper-bound only: these labels were produced with access to future RGB
        # and must never be used at deployment time.
        xyz = torch.nan_to_num(batch["future_xyz_teacher"].float())
        conf = torch.nan_to_num(batch["future_confidence"].float())
        raster_uv = torch.nan_to_num(batch["future_uv_teacher"].float()).clamp(0,1)
        point_ok = point_ok & batch["future_valid"].bool().any(2) & torch.isfinite(batch["future_uv_teacher"]).all(-1).any(2)
    else: raise ValueError(mode)
    raw = GeometryConditionEncoder.raw_features(xyz, raster_uv, conf)
    return raw, point_ok, raster_uv


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def prediction_loss(pred,target,last_context,dynamic_weight):
    l1=(pred-target).abs(); mse=(pred-target).square()
    if dynamic_weight>0:
        motion=((target-last_context[:,None]).abs().mean(2)>0.05)[:,:,None]
        weight=1.0+float(dynamic_weight)*motion.float()
        return (l1*weight).mean()+0.1*(mse*weight).mean()
    return l1.mean()+0.1*mse.mean()


@torch.no_grad()
def validation_mse(model, trajectory, joint_trajectory, loader, mode, device, k):
    model.eval(); sq = count = 0.0
    for batch in loader:
        batch=move(batch,device); geom,ok,raster_uv=make_geometry(mode,trajectory,joint_trajectory,batch,k)
        pred=model(batch["context_rgb"],geom,raster_uv,ok)
        sq += F.mse_loss(pred,batch["future_rgb"],reduction="sum").item(); count += batch["future_rgb"].numel()
    return sq/count


@torch.no_grad()
def test_metrics(model, trajectory, joint_trajectory, loader, mode, device, k, mode_dir, motion_threshold=0.05):
    model.eval(); metrics_accumulator=VideoMetricAccumulator(device,k)
    dyn_abs=dyn_sq=dyn_count=0.0; inter=union=pred_pos=gt_pos=0.0
    first=None
    for batch in loader:
        batch=move(batch,device); geom,ok,raster_uv=make_geometry(mode,trajectory,joint_trajectory,batch,k)
        pred=model(batch["context_rgb"],geom,raster_uv,ok)
        target=batch["future_rgb"]; diff=pred-target
        last = batch["context_rgb"][:, -1:, :, :, :]
        metrics_accumulator.update(pred,target,last[:,0])
        gt_motion = (target-last).abs().mean(2) > motion_threshold
        pred_motion = (pred-last).abs().mean(2) > motion_threshold
        dyn3 = gt_motion[:, :, None].expand_as(diff)
        dyn_abs += diff.abs()[dyn3].sum().item(); dyn_sq += diff.square()[dyn3].sum().item(); dyn_count += dyn3.sum().item()
        inter += (gt_motion & pred_motion).sum().item(); union += (gt_motion | pred_motion).sum().item()
        pred_pos += pred_motion.sum().item(); gt_pos += gt_motion.sum().item()
        if first is None: first=(batch["context_rgb"][0].cpu(),target[0].cpu(),pred[0].cpu())
    dyn_mse=dyn_sq/max(dyn_count,1); precision=inter/max(pred_pos,1); recall=inter/max(gt_pos,1)
    metrics=metrics_accumulator.compute();metrics.update({"dynamic_MAE":float(dyn_abs/max(dyn_count,1)),
                              "dynamic_MSE":float(dyn_mse),
                              "dynamic_PSNR":float(-10*math.log10(max(dyn_mse,1e-12))),
                              "motion_mask_IoU":float(inter/max(union,1)),
                              "motion_mask_F1":float(2*precision*recall/max(precision+recall,1e-12)),
                              "motion_mask_precision":float(precision),"motion_mask_recall":float(recall),
                              "dynamic_pixel_fraction":float(gt_pos/(metrics_accumulator.count/3)),
                              "motion_threshold":float(motion_threshold)})
    context,target,pred=first; fig,axes=plt.subplots(2,5,figsize=(15,6))
    ids=[-1,0,1,k//2,k-1]; titles=["last context","t+1","t+2",f"t+{k//2+1}",f"t+{k}"]
    for col,idx in enumerate(ids):
        top=context[-1] if idx==-1 else target[idx]; bot=context[-1] if idx==-1 else pred[idx]
        for row,img in enumerate((top,bot)):
            axes[row,col].imshow(img.permute(1,2,0).clamp(0,1)); axes[row,col].axis("off")
        axes[0,col].set_title(titles[col]); axes[0,0].set_ylabel("target"); axes[1,0].set_ylabel("prediction")
    fig.suptitle(mode); fig.tight_layout(); fig.savefig(mode_dir/"test_montage.png",dpi=140); plt.close(fig)
    return metrics


def train_mode(mode,a,train_ds,val_ds,test_ds,trajectory,joint_trajectory,device,c,k):
    seed_all(a.seed); key=a.run_name or mode; mode_dir=Path(a.output_dir)/key; mode_dir.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(a.seed)
    train=DataLoader(train_ds,a.batch_size,shuffle=True,generator=generator,collate_fn=video_pad_collate,num_workers=0)
    val=DataLoader(val_ds,a.batch_size,collate_fn=video_pad_collate,num_workers=0)
    test=DataLoader(test_ds,1,collate_fn=video_pad_collate,num_workers=0)
    model=build_video_predictor(c,k,a.image_size,backbone=getattr(a,"backbone","simvp"),
        injection_type=a.injection,hid_s=getattr(a,"hid_s",None),hid_t=getattr(a,"hid_t",None),
        n_s=getattr(a,"n_s",2),n_t=getattr(a,"n_t",None),drop_path=getattr(a,"drop_path",None)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best=float("inf"); history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.
        for batch in train:
            batch=move(batch,device); geom,ok,raster_uv=make_geometry(mode,trajectory,joint_trajectory,batch,k)
            pred=model(batch["context_rgb"],geom,raster_uv,ok)
            loss=prediction_loss(pred,batch["future_rgb"],batch["context_rgb"][:,-1],a.dynamic_loss_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()
        vmse=validation_mse(model,trajectory,joint_trajectory,val,mode,device,k)
        row={"epoch":epoch,"train_loss":total/len(train),"val_MSE":vmse}; history.append(row)
        if vmse<best:
            best=vmse; torch.save({"model":model.state_dict(),"mode":mode,"context_frames":c,
                                  "future_frames":k,"image_size":a.image_size,"injection":a.injection,
                                  "dynamic_loss_weight":a.dynamic_loss_weight,
                                  "backbone":model.backbone_name,"backbone_config":model.backbone_config},mode_dir/"best.pt")
        print(f"[{mode}] epoch={epoch:03d} train={row['train_loss']:.5f} val_mse={vmse:.6f}",flush=True)
    (mode_dir/"history.json").write_text(json.dumps(history,indent=2))
    ck=torch.load(mode_dir/"best.pt",map_location=device); model.load_state_dict(ck["model"])
    metrics=test_metrics(model,trajectory,joint_trajectory,test,mode,device,k,mode_dir); (mode_dir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    metrics.update({"injection":a.injection,"dynamic_loss_weight":a.dynamic_loss_weight})
    if a.injection=="film":
        metrics["latent_gate"]=float(torch.sigmoid(model.latent_gate_logit)); metrics["skip_gate"]=float(torch.sigmoid(model.skip_gate_logit))
    (mode_dir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(f"[{mode}] test {json.dumps(metrics)}",flush=True); return metrics


def main():
    a=parse_args(); device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    train_ds=VideoForecastDataset(a.cache_root,"train",a.image_size); val_ds=VideoForecastDataset(a.cache_root,"val",a.image_size); test_ds=VideoForecastDataset(a.cache_root,"test",a.image_size)
    sample=train_ds[0]; c=sample["context_rgb"].shape[0]; k=sample["future_rgb"].shape[0]
    ck=torch.load(a.trajectory_checkpoint,map_location="cpu"); trajectory=LiteTrajectoryForecaster(ck["future_frames"],ck["hidden_dim"])
    trajectory.load_state_dict(ck["model"]); trajectory.to(device).eval()
    for p in trajectory.parameters(): p.requires_grad_(False)
    jck=torch.load(a.joint_trajectory_checkpoint,map_location="cpu"); joint_trajectory=JointTrajectoryForecaster(jck["future_frames"],jck["hidden_dim"])
    joint_trajectory.load_state_dict(jck["model"]); joint_trajectory.to(device).eval()
    for p in joint_trajectory.parameters(): p.requires_grad_(False)
    modes=[x.strip() for x in a.modes.split(",") if x.strip()]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summary_path=out/"summary.json"
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(f"data train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} modes={modes}")
    for mode in modes:
        key=a.run_name or mode
        summary[key]=train_mode(mode,a,train_ds,val_ds,test_ds,trajectory,joint_trajectory,device,c,k)
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
