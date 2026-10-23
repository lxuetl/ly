#!/usr/bin/env python3
"""Train an identity-initialized geometry-only residual on fair SimVP."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "forecasting"))
sys.path.insert(0, str(HERE))

from capsr_stable_router_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--geometry-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dynamic-weight", type=float, default=2.0)
    p.add_argument("--static-weight", type=float, default=0.2)
    p.add_argument("--temporal-weight", type=float, default=0.05)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SafeGeometryResidual(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(21, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, base, state):
        candidates = state["candidate_stack"]
        residuals = candidates[:, :, 1:] - base[:, :, None]
        availability = state["path_availability"].float()
        prior = state["prior_weights"].detach()
        b, k, _, _, h, w = residuals.shape
        horizon = torch.linspace(0, 1, k, device=base.device, dtype=base.dtype)
        horizon = horizon.view(1, k, 1, 1, 1).expand(b, -1, -1, h, w)
        features = torch.cat((
            base,
            residuals.reshape(b, k, 9, h, w),
            prior,
            availability,
            horizon,
        ), dim=2)
        weights = torch.tanh(
            self.head(features.reshape(b * k, 21, h, w)).reshape(b, k, 3, h, w)
        )
        permission = availability[:, :, 1:]
        correction = (
            weights[:, :, :, None] * permission[:, :, :, None] * residuals
        ).sum(2)
        prediction = (base + correction).clamp(0, 1)
        return prediction, correction, weights


def limited(dataset, maximum):
    return dataset if maximum <= 0 else Subset(dataset, range(min(maximum, len(dataset))))


@torch.no_grad()
def geometry_state(base_model, trajectory, geometry, batch):
    base, xyz, uv = frozen_prediction(base_model, trajectory, batch)
    _, state = geometry(
        base, batch["context_rgb"][:, -1], xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(),
        batch["past_valid"], batch["past_confidence"].float(), mode="structured",
    )
    return base, state


def objective(prediction, base, correction, target, last, a):
    motion = (target - last).abs().mean(2, keepdim=True)
    dynamic = (motion > 0.05).float()
    squared = (prediction - target).square()
    reconstruction = (squared * (1 + a.dynamic_weight * dynamic)).mean()
    static = 1 - dynamic
    static_identity = (
        correction.abs().mean(2, keepdim=True) * static
    ).sum() / static.sum().clamp_min(1)
    temporal = (correction[:, 1:] - correction[:, :-1]).abs().mean()
    size = correction.abs().mean()
    total = (
        reconstruction + a.static_weight * static_identity
        + a.temporal_weight * temporal + a.correction_weight * size
    )
    return total, {
        "reconstruction": reconstruction.item(),
        "static_identity": static_identity.item(),
        "temporal": temporal.item(), "correction": size.item(),
    }


@torch.no_grad()
def validate(head, base_model, trajectory, geometry, loader, device):
    head.eval()
    squared = absolute = count = dynamic_squared = dynamic_count = 0.0
    correction = weight = batches = 0.0
    for batch in loader:
        batch = move(batch, device)
        base, state = geometry_state(base_model, trajectory, geometry, batch)
        prediction, delta, weights = head(base, state)
        difference = prediction - batch["future_rgb"]
        squared += difference.square().sum().item()
        absolute += difference.abs().sum().item()
        count += difference.numel()
        last = batch["context_rgb"][:, -1:]
        mask = ((batch["future_rgb"] - last).abs().mean(2, keepdim=True) > 0.05).expand_as(difference)
        dynamic_squared += difference.square()[mask].sum().item()
        dynamic_count += mask.sum().item()
        correction += delta.abs().mean().item()
        weight += weights.abs().mean().item()
        batches += 1
    return {
        "MSE": squared / max(count, 1), "MAE": absolute / max(count, 1),
        "dynamic_MSE": dynamic_squared / max(dynamic_count, 1),
        "correction_abs": correction / max(batches, 1),
        "weight_abs": weight / max(batches, 1),
    }


@torch.no_grad()
def evaluate(head, base_model, trajectory, geometry, loader, device, k):
    head.eval()
    accumulators = {
        "fair_simvp": VideoMetricAccumulator(device, k),
        "safe_geometry_residual": VideoMetricAccumulator(device, k),
    }
    dynamic = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in accumulators}
    for batch in loader:
        batch = move(batch, device)
        base, state = geometry_state(base_model, trajectory, geometry, batch)
        prediction, _, _ = head(base, state)
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        mask = ((target - last).abs().mean(2, keepdim=True) > 0.05).expand_as(target)
        for name, value in (("fair_simvp", base), ("safe_geometry_residual", prediction)):
            accumulators[name].update(value, target, batch["context_rgb"][:, -1])
            difference = value - target
            dynamic[name]["abs"] += difference.abs()[mask].sum().item()
            dynamic[name]["sq"] += difference.square()[mask].sum().item()
            dynamic[name]["count"] += mask.sum().item()
    result = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute()
        count = max(dynamic[name]["count"], 1.0)
        dmse = dynamic[name]["sq"] / count
        metrics.update({
            "dynamic_MAE": dynamic[name]["abs"] / count,
            "dynamic_MSE": dmse,
            "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)),
        })
        result[name] = metrics
    return result


def main():
    a = arguments()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(SimpleNamespace(**vars(a)))
    datasets = {
        split: limited(VideoForecastDataset(
            a.cache_root, split, 64, paths=splits[split], cache_items=False
        ), getattr(a, f"max_{split}"))
        for split in ("train", "val", "test")
    }
    train_loader = DataLoader(
        datasets["train"], a.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(a.seed),
        collate_fn=video_pad_collate, num_workers=0,
    )
    val_loader = DataLoader(
        datasets["val"], a.batch_size, collate_fn=video_pad_collate, num_workers=0
    )
    test_loader = DataLoader(
        datasets["test"], 1, collate_fn=video_pad_collate, num_workers=0
    )
    base_model, trajectory, _, trajectory_checkpoint = load_frozen_models(a, device)
    checkpoint = torch.load(a.geometry_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    geometry = StructuredCaPSR(
        float(config.get("sigma", 0.035)), int(config.get("hidden", 48)),
        num_motion_clusters=int(config.get("num_motion_clusters", 4)),
    ).to(device)
    geometry.load_state_dict(checkpoint["model"])
    geometry.eval()
    for parameter in geometry.parameters():
        parameter.requires_grad_(False)
    head = SafeGeometryResidual().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    best = float("inf")
    history = []
    started = time.time()
    for epoch in range(1, a.epochs + 1):
        head.train()
        totals = {name: 0.0 for name in ("loss", "reconstruction", "static_identity", "temporal", "correction")}
        epoch_started = time.time()
        for batch in train_loader:
            batch = move(batch, device)
            base, state = geometry_state(base_model, trajectory, geometry, batch)
            prediction, delta, _ = head(base, state)
            loss, pieces = objective(
                prediction, base, delta, batch["future_rgb"],
                batch["context_rgb"][:, -1:], a,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            for name, value in pieces.items():
                totals[name] += value
        val = validate(head, base_model, trajectory, geometry, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / max(len(train_loader), 1) for name, value in totals.items()},
            **{f"val_{name}": value for name, value in val.items()},
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        saved = {
            "model": head.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "config": vars(a), "trainable_params": trainable,
        }
        torch.save(saved, output / "last.pt")
        if val["MSE"] < best:
            best = val["MSE"]
            torch.save(saved, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
    best_checkpoint = torch.load(output / "best.pt", map_location=device)
    head.load_state_dict(best_checkpoint["model"])
    metrics = evaluate(
        head, base_model, trajectory, geometry, test_loader, device,
        int(trajectory_checkpoint["future_frames"]),
    )
    summary = {
        "metrics": metrics,
        "meta": {
            "best_epoch": best_checkpoint["epoch"], "trainable_params": trainable,
            "used": {name: len(value) for name, value in datasets.items()},
            "split": split_meta, "identity_initialized": True,
            "backbone_frozen": True, "geometry_frozen": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
