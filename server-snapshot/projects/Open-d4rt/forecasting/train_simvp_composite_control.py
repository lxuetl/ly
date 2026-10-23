#!/usr/bin/env python3
"""Fine-tune the strong SimVP with the geometry model's core composite losses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from simvp_geometry_model import build_video_predictor_from_checkpoint
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all, test_metrics
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True); p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--init-checkpoint", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--dynamic-weight", type=float, default=3.0)
    p.add_argument("--temporal-weight", type=float, default=.10); p.add_argument("--lpips-weight", type=float, default=.03)
    p.add_argument("--gradient-weight", type=float, default=.05); p.add_argument("--static-identity-weight", type=float, default=.10)
    p.add_argument("--correction-weight", type=float, default=.005)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_model(checkpoint, device):
    model = build_video_predictor_from_checkpoint(checkpoint)
    result = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {"latent_gate_logit", "skip_gate_logit", "latent_film.weight", "latent_film.bias",
               "skip_film.weight", "skip_film.bias"}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise RuntimeError((result.missing_keys, result.unexpected_keys))
    return model.to(device)


def components(prediction, target, last, reference, a, perceptual):
    dynamic = ((target - last).abs().mean(2, keepdim=True) > .05).float()
    error = (prediction - target).abs() + .1 * (prediction - target).square()
    reconstruction = (error * (1 + a.dynamic_weight * dynamic)).mean()
    pred_previous = torch.cat((last, prediction[:, :-1]), 1)
    target_previous = torch.cat((last, target[:, :-1]), 1)
    temporal = ((prediction - pred_previous) - (target - target_previous)).abs().mean()
    b, k, c, h, w = prediction.shape
    lpips_loss = perceptual(
        prediction.reshape(b*k,c,h,w)*2-1, target.reshape(b*k,c,h,w)*2-1
    ).mean()
    pred_x, target_x = prediction[...,1:] - prediction[...,:-1], target[...,1:] - target[...,:-1]
    pred_y, target_y = prediction[...,1:,:] - prediction[...,:-1,:], target[...,1:,:] - target[...,:-1,:]
    gradient = .5 * ((pred_x-target_x).abs().mean() + (pred_y-target_y).abs().mean())
    correction = prediction - reference
    static = 1 - dynamic
    static_identity = (correction.abs().mean(2,keepdim=True)*static).sum()/static.sum().clamp_min(1)
    correction_size = correction.abs().mean()
    total = (reconstruction + a.temporal_weight*temporal + a.lpips_weight*lpips_loss
             + a.gradient_weight*gradient + a.static_identity_weight*static_identity
             + a.correction_weight*correction_size)
    return total, {"reconstruction":reconstruction,"temporal":temporal,"lpips":lpips_loss,
                   "gradient":gradient,"static_identity":static_identity,"correction":correction_size}


@torch.no_grad()
def validate(model, reference, loader, device):
    model.eval(); sq=count=dyn_abs=dyn_count=temporal=batches=0.
    for batch in loader:
        batch=move(batch,device); prediction=model(batch["context_rgb"]); target=batch["future_rgb"]
        diff=prediction-target; sq+=diff.square().sum().item(); count+=target.numel()
        last=batch["context_rgb"][:,-1:]; dynamic=((target-last).abs().mean(2,keepdim=True)>.05).expand_as(target)
        dyn_abs+=diff.abs()[dynamic].sum().item();dyn_count+=dynamic.sum().item()
        pp=torch.cat((last,prediction[:,:-1]),1);tp=torch.cat((last,target[:,:-1]),1)
        temporal+=((prediction-pp)-(target-tp)).abs().mean().item();batches+=1
    return {"MSE":sq/count,"dynamic_MAE":dyn_abs/max(dyn_count,1),"temporal_MAE":temporal/batches}


def main():
    a=args();seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    splits,meta=combined_splits(SimpleNamespace(**vars(a)))
    make=lambda split:VideoForecastDataset(a.cache_root,split,64,paths=splits[split],cache_items=False)
    train_ds,val_ds,test_ds=make("train"),make("val"),make("test")
    train=DataLoader(train_ds,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate,num_workers=0)
    val=DataLoader(val_ds,a.batch_size,collate_fn=video_pad_collate,num_workers=0)
    test=DataLoader(test_ds,1,collate_fn=video_pad_collate,num_workers=0)
    initial=torch.load(a.init_checkpoint,map_location="cpu")
    model=make_model(initial,device);reference=make_model(initial,device).eval()
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    import lpips
    perceptual=lpips.LPIPS(net="alex",verbose=False).to(device).eval()
    for p in perceptual.parameters():p.requires_grad_(False)
    best=float("inf");history=[]
    for epoch in range(1,a.epochs+1):
        model.train();totals={k:0. for k in ("loss","reconstruction","temporal","lpips","gradient","static_identity","correction")}
        for batch in train:
            batch=move(batch,device)
            with torch.no_grad():reference_prediction=reference(batch["context_rgb"])
            prediction=model(batch["context_rgb"])
            loss,pieces=components(prediction,batch["future_rgb"],batch["context_rgb"][:,-1:],reference_prediction,a,perceptual)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            totals["loss"]+=loss.item()
            for key,value in pieces.items():totals[key]+=value.item()
        metrics=validate(model,reference,val,device)
        score=metrics["MSE"]+.01*metrics["dynamic_MAE"]+.05*metrics["temporal_MAE"]
        row={"epoch":epoch,**{f"train_{k}":v/len(train) for k,v in totals.items()},**{f"val_{k}":v for k,v in metrics.items()},"score":score}
        history.append(row);print(json.dumps(row),flush=True)
        checkpoint={"model":model.state_dict(),"epoch":epoch,"best":min(best,score),"mode":"no_geometry",
                    "context_frames":initial["context_frames"],"future_frames":initial["future_frames"],"image_size":64,
                    "injection":initial.get("injection","add"),"backbone":"simvp","backbone_config":getattr(model,"backbone_config",{}),
                    "optimizer":optimizer.state_dict(),"composite_loss":True}
        torch.save(checkpoint,out/"last.pt")
        if score<best:best=score;torch.save(checkpoint,out/"best.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2))
    checkpoint=torch.load(out/"best.pt",map_location=device);model.load_state_dict(checkpoint["model"])
    metrics=test_metrics(model,None,None,test,"no_geometry",device,initial["future_frames"],out)
    summary={"metrics":metrics,"meta":{"best_epoch":checkpoint["epoch"],"split":meta,"trainable_params":sum(p.numel() for p in model.parameters()),"init_checkpoint":a.init_checkpoint}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));(out/"COMPLETE").write_text("complete\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
