"""Stage 5: train and validate geometry-conditioned video prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from trajectory_model import GeometryConditionEncoder, LiteTrajectoryForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_prediction_model import GeometryConditionedVideoPredictor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--trajectory-checkpoint", default="outputs/trajectory_forecaster/best.pt")
    p.add_argument("--output-dir", default="outputs/geometry_video")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def geometry_features(traj_model, b):
    with torch.no_grad():
        pred = traj_model(b["past_xyz"].float(), b["past_uv"].float(), b["past_valid"], b["past_confidence"].float())
        conf = b["past_confidence"].float().nan_to_num().amax(2)[:, :, None].expand(-1, -1, pred.shape[2])
        raw = GeometryConditionEncoder.raw_features(pred, b["query_uv"].float(), conf)
    return raw, b["past_valid"].any(2)


@torch.no_grad()
def validate(model, traj, loader, device, use_geometry):
    model.eval(); total = 0.
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
        geom, ok = geometry_features(traj, b)
        pred = model(b["context_rgb"], geom, b["query_uv"], ok, use_geometry)
        total += F.l1_loss(pred, b["future_rgb"]).item()
    return total / len(loader)


def save_montage(path, context, target, pred):
    # last context, first/last target, first/last prediction
    imgs = (context[-1], target[0], pred[0], target[-1], pred[-1])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ("last context", "target t+1", "pred t+1", "target t+K", "pred t+K")
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im.permute(1,2,0).clamp(0,1).cpu()); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    a=parse_args(); device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    train_ds=VideoForecastDataset(a.cache_root,"train",a.image_size); val_ds=VideoForecastDataset(a.cache_root,"val",a.image_size)
    train=DataLoader(train_ds,a.batch_size,shuffle=True,collate_fn=video_pad_collate,num_workers=0)
    val=DataLoader(val_ds,a.batch_size,collate_fn=video_pad_collate,num_workers=0)
    ck=torch.load(a.trajectory_checkpoint,map_location="cpu")
    traj=LiteTrajectoryForecaster(ck["future_frames"],ck["hidden_dim"]); traj.load_state_dict(ck["model"]); traj.to(device).eval()
    for p in traj.parameters(): p.requires_grad_(False)
    sample=train_ds[0]; c=sample["context_rgb"].shape[0]; k=sample["future_rgb"].shape[0]
    model=GeometryConditionedVideoPredictor(c,k).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); best=float("inf"); history=[]
    print(f"split train={len(train_ds)} val={len(val_ds)} video_params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.
        for b in train:
            b={key:(v.to(device) if torch.is_tensor(v) else v) for key,v in b.items()}
            geom,ok=geometry_features(traj,b); pred=model(b["context_rgb"],geom,b["query_uv"],ok,True)
            loss=F.l1_loss(pred,b["future_rgb"]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.item()
        vg=validate(model,traj,val,device,True); vn=validate(model,traj,val,device,False)
        row={"epoch":epoch,"train_l1":total/len(train),"val_l1_geometry":vg,"val_l1_no_geometry":vn}; history.append(row)
        if vg<best:
            best=vg; torch.save({"model":model.state_dict(),"context_frames":c,"future_frames":k,"image_size":a.image_size},out/"best.pt")
        print(f"epoch={epoch:03d} train={row['train_l1']:.5f} val_geom={vg:.5f} val_no_geom={vn:.5f}")
    (out/"history.json").write_text(json.dumps(history,indent=2))
    b=next(iter(val)); b={key:(v.to(device) if torch.is_tensor(v) else v) for key,v in b.items()}
    geom,ok=geometry_features(traj,b); model.eval()
    with torch.no_grad(): pred=model(b["context_rgb"],geom,b["query_uv"],ok,True)
    save_montage(out/"prediction_montage.png",b["context_rgb"][0],b["future_rgb"][0],pred[0])
    print(f"best={best:.6f} outputs={out}")


if __name__=="__main__": main()
