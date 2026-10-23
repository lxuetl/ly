#!/usr/bin/env python3
"""Train a zero-initialized VGGT-Omega residual and an RGB-only capacity control."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--geometry-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dynamic-weight", type=float, default=2.0)
    p.add_argument("--static-weight", type=float, default=0.1)
    p.add_argument("--temporal-weight", type=float, default=0.03)
    p.add_argument("--correction-weight", type=float, default=0.005)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class GeometryDataset(Dataset):
    def __init__(self, base, geometry_root):
        self.base = base
        self.geometry_root = Path(geometry_root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        path = Path(self.base.paths[index])
        geometry_path = self.geometry_root / path.parent.name / f"{path.stem}.npz"
        with np.load(geometry_path) as geometry:
            item["vggt_depth"] = torch.from_numpy(geometry["depth"].astype(np.float32))
            item["vggt_confidence"] = torch.from_numpy(geometry["depth_confidence"].astype(np.float32))
            item["vggt_pose"] = torch.from_numpy(geometry["pose_enc"].astype(np.float32))
        return item


def geometry_collate(items):
    from video_forecast_dataset import video_pad_collate
    output = video_pad_collate(items)
    for key in ("vggt_depth", "vggt_confidence", "vggt_pose"):
        output[key] = torch.stack([item[key] for item in items])
    return output


class VGGTResidual(nn.Module):
    def __init__(self, context_frames=16, future_frames=8, hidden=32):
        super().__init__()
        self.context_frames = context_frames
        self.future_frames = future_frames
        input_channels = context_frames * (2 + 9)
        self.geometry_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden + 7, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def standardize(value, dimensions):
        mean = value.mean(dimensions, keepdim=True)
        std = value.std(dimensions, keepdim=True).clamp_min(1e-4)
        return (value - mean) / std

    def geometry_features(self, depth, confidence, pose, use_geometry):
        b, t, h, w = depth.shape
        if use_geometry:
            depth = self.standardize(torch.log(depth.clamp_min(1e-4)), (1, 2, 3))
            confidence = self.standardize(torch.log1p(confidence.clamp_min(0)), (1, 2, 3))
            pose = pose - pose[:, :1]
            pose = pose / pose.square().mean(1, keepdim=True).sqrt().clamp_min(1e-3)
            pose = pose[:, :, :, None, None].expand(-1, -1, -1, h, w)
            raw = torch.cat((depth[:, :, None], confidence[:, :, None], pose), 2)
        else:
            raw = depth.new_zeros((b, t, 11, h, w))
        return self.geometry_encoder(raw.reshape(b, t * 11, h, w))

    def forward(self, baseline, last, depth, confidence, pose, use_geometry=True):
        latent = self.geometry_features(depth, confidence, pose, use_geometry)
        b, k, _, h, w = baseline.shape
        latent = latent[:, None].expand(-1, k, -1, -1, -1)
        last = last[:, None].expand(-1, k, -1, -1, -1)
        horizon = torch.linspace(0, 1, k, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, k, 1, 1, 1).expand(b, -1, -1, h, w)
        features = torch.cat((baseline, last, latent, horizon), 2)
        correction = self.residual_head(features.reshape(b * k, -1, h, w))
        correction = correction.reshape(b, k, 3, h, w)
        return (baseline + correction).clamp(0, 1), correction


def objective(prediction, baseline, correction, target, last, a):
    motion = (target - last[:, None]).abs().mean(2, keepdim=True)
    dynamic = (motion > 0.05).float()
    error = prediction - target
    reconstruction = (error.square() * (1 + a.dynamic_weight * dynamic)).mean()
    static = 1 - dynamic
    static_identity = (
        correction.abs().mean(2, keepdim=True) * static
    ).sum() / static.sum().clamp_min(1)
    temporal = (correction[:, 1:] - correction[:, :-1]).abs().mean()
    correction_size = correction.abs().mean()
    loss = (
        reconstruction + a.static_weight * static_identity
        + a.temporal_weight * temporal + a.correction_weight * correction_size
    )
    return loss


@torch.no_grad()
def validation(model, use_geometry, backbone, loader, device):
    model.eval()
    squared = count = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        baseline = backbone(batch["context_rgb"], None, None, None)
        prediction, _ = model(
            baseline, batch["context_rgb"][:, -1], batch["vggt_depth"],
            batch["vggt_confidence"], batch["vggt_pose"], use_geometry,
        )
        squared += (prediction - batch["future_rgb"]).square().sum().item()
        count += batch["future_rgb"].numel()
    return squared / max(count, 1)


@torch.no_grad()
def evaluate(models, backbone, loader, device, future_frames):
    from video_metrics import VideoMetricAccumulator
    accumulators = {
        "fair_simvp": VideoMetricAccumulator(device, future_frames),
        **{name: VideoMetricAccumulator(device, future_frames) for name in models},
    }
    dynamic = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in accumulators}
    motion_counts = {name: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for name in accumulators}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        baseline = backbone(batch["context_rgb"], None, None, None)
        predictions = {"fair_simvp": baseline}
        for name, (model, use_geometry) in models.items():
            predictions[name] = model(
                baseline, batch["context_rgb"][:, -1], batch["vggt_depth"],
                batch["vggt_confidence"], batch["vggt_pose"], use_geometry,
            )[0]
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        target_motion = (target - last).abs().mean(2, keepdim=True) > 0.05
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, last[:, 0])
            difference = prediction - target
            selected = target_motion.expand_as(difference)
            dynamic[name]["abs"] += difference.abs()[selected].sum().item()
            dynamic[name]["sq"] += difference.square()[selected].sum().item()
            dynamic[name]["count"] += selected.sum().item()
            predicted_motion = (prediction - last).abs().mean(2, keepdim=True) > 0.05
            motion_counts[name]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[name]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[name]["fn"] += (~predicted_motion & target_motion).sum().item()
    results = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute()
        count = max(dynamic[name]["count"], 1.0)
        dmse = dynamic[name]["sq"] / count
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics.update({
            "dynamic_MAE": dynamic[name]["abs"] / count,
            "dynamic_MSE": dmse,
            "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)),
            "Motion_IoU": tp / max(tp + fp + fn, 1),
            "Motion_F1": 2 * tp / max(2 * tp + fp + fn, 1),
        })
        results[name] = metrics
    return results


def main():
    a = parse_args()
    d4rt_root = Path(a.d4rt_root)
    sys.path.insert(0, str(d4rt_root / "forecasting"))
    sys.path.insert(0, str(d4rt_root))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from train_pointodyssey_dmvfn_injection import combined_splits
    from train_simvp_ablation import seed_all
    from video_forecast_dataset import VideoForecastDataset

    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(SimpleNamespace(**vars(a)))
    limits = {"train": a.max_train, "val": a.max_val, "test": a.max_test}
    datasets = {}
    for split in ("train", "val", "test"):
        paths = splits[split][: limits[split] or None]
        base_dataset = VideoForecastDataset(a.cache_root, split, 64, paths=paths, cache_items=False)
        datasets[split] = GeometryDataset(base_dataset, a.geometry_cache)
    loaders = {
        "train": DataLoader(
            datasets["train"], a.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(a.seed),
            collate_fn=geometry_collate, num_workers=0,
        ),
        "val": DataLoader(datasets["val"], a.batch_size, collate_fn=geometry_collate, num_workers=0),
        "test": DataLoader(datasets["test"], 1, collate_fn=geometry_collate, num_workers=0),
    }
    checkpoint = torch.load(a.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    context_frames = int(checkpoint["context_frames"])
    future_frames = int(checkpoint["future_frames"])
    models = {
        "vggt_omega_geometry": (VGGTResidual(context_frames, future_frames).to(device), True),
        "rgb_capacity_control": (VGGTResidual(context_frames, future_frames).to(device), False),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        for name, (model, _) in models.items()
    }
    best = {name: float("inf") for name in models}
    history = []
    for epoch in range(1, a.epochs + 1):
        started = time.time()
        totals = {name: 0.0 for name in models}
        for model, _ in models.values():
            model.train()
        for batch in loaders["train"]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                baseline = backbone(batch["context_rgb"], None, None, None)
            for name, (model, use_geometry) in models.items():
                prediction, correction = model(
                    baseline, batch["context_rgb"][:, -1], batch["vggt_depth"],
                    batch["vggt_confidence"], batch["vggt_pose"], use_geometry,
                )
                loss = objective(
                    prediction, baseline, correction, batch["future_rgb"],
                    batch["context_rgb"][:, -1], a,
                )
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name] += loss.item()
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, (model, use_geometry) in models.items():
            val_mse = validation(model, use_geometry, backbone, loaders["val"], device)
            row[f"{name}_train_loss"] = totals[name] / max(len(loaders["train"]), 1)
            row[f"{name}_val_MSE"] = val_mse
            saved = {
                "model": model.state_dict(), "epoch": epoch, "config": vars(a),
                "use_geometry": use_geometry,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
            torch.save(saved, output / f"{name}_last.pt")
            if val_mse < best[name]:
                best[name] = val_mse
                torch.save(saved, output / f"{name}_best.pt")
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    for name, (model, _) in models.items():
        selected = torch.load(output / f"{name}_best.pt", map_location=device)
        model.load_state_dict(selected["model"])
    metrics = evaluate(models, backbone, loaders["test"], device, future_frames)
    summary = {
        "metrics": metrics,
        "meta": {
            "split": split_meta,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "best_epochs": {
                name: torch.load(output / f"{name}_best.pt", map_location="cpu")["epoch"]
                for name in models
            },
            "backbone_checkpoint": a.backbone_checkpoint,
            "geometry_cache": a.geometry_cache,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
