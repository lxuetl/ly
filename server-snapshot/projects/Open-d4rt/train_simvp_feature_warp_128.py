#!/usr/bin/env python3
"""Low-cost, single-model v1 feature-warp experiment at configurable resolution."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_simvp_feature_warp as core


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--track-weight", type=float, default=0.25)
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--feature-correction-weight", type=float, default=0.002)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--flow-kernel", type=int, default=17)
    p.add_argument("--feature-scale", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate
    splits, metadata = core.select_paths(args)
    datasets = {
        split: VideoForecastDataset(args.cache_root, split, args.image_size,
                                    paths=paths, cache_items=False)
        for split, paths in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True,
                            generator=generator, collate_fn=video_pad_collate,
                            num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], args.batch_size,
                          collate_fn=video_pad_collate, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], args.batch_size,
                           collate_fn=video_pad_collate, num_workers=args.num_workers),
    }
    return loaders, datasets, metadata


@torch.no_grad()
def validate(model, loader, backbone, joint, device, future_frames):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        corrected, _, _ = model(source, hidden, *bundle)
        prediction = core.decode_frozen(backbone, corrected, skip)
        error = prediction - batch["future_rgb"]
        totals["MSE"] += error.square().mean().item()
        totals["MAE"] += error.abs().mean().item()
        totals["support"] += model.last["support"].mean().item()
        totals["gate"] += model.last["gate"].mean().item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(model, train_loader, val_loader, backbone, joint, lpips_model,
          device, args, output, future_frames):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(
                    backbone, batch["context_rgb"])
                baseline = core.decode_frozen(backbone, hidden, skip)
                bundle = core.geometry_bundle("correct", joint, batch, future_frames)
            corrected, app_delta, geo_delta = model(source, hidden, *bundle)
            prediction = core.decode_frozen(backbone, corrected, skip)
            loss, pieces = core.objective(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                baseline, app_delta, geo_delta, batch, lpips_model, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            batches += 1
        metrics = validate(model, val_loader, backbone, joint, device, future_frames)
        row = {
            "epoch": epoch, "epoch_seconds": time.time() - started,
            **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch,
                 "validation": metrics, "config": vars(args)}
        torch.save(state, output / "last.pt")
        if metrics["MSE"] < best:
            best = metrics["MSE"]
            torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "feature_warp_128", **row}), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device)["model"])


@torch.no_grad()
def evaluate(model, loader, backbone, joint, device, future_frames, seed, image_size,
             backbone_name):
    from video_metrics import VideoMetricAccumulator
    prefix = str(int(image_size))
    baseline_name = f"{backbone_name}_{prefix}"
    correct_name = f"{backbone_name}_v1_{prefix}_correct"
    zero_name = f"{backbone_name}_v1_{prefix}_zero"
    shuffled_name = f"{backbone_name}_v1_{prefix}_shuffled"
    variants = (baseline_name, correct_name, zero_name, shuffled_name)
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in variants}
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    model.eval()
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        predictions = {baseline_name: core.decode_frozen(backbone, hidden, skip)}
        for mode, name in (("correct", correct_name),
                           ("zero", zero_name),
                           ("shuffled", shuffled_name)):
            bundle = core.geometry_bundle(mode, joint, batch, future_frames)
            corrected, _, _ = model(source, hidden, *bundle)
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
        utilization["correct_vs_zero_l1"].append(
            (predictions[correct_name] - predictions[zero_name]).abs().mean().item())
        utilization["correct_vs_shuffled_l1"].append(
            (predictions[correct_name] - predictions[shuffled_name]).abs().mean().item())
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > 0.05
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[variant]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[variant]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[variant]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, name in enumerate(batch["name"]):
                records.append({
                    "name": name, "variant": variant,
                    **core.sequence_metrics(prediction[index:index + 1],
                                            target[index:index + 1], last[index:index + 1]),
                })
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    bootstrap = {
        reference: core.paired_bootstrap(records, reference, correct_name, seed)
        for reference in (baseline_name, zero_name, shuffled_name)
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(values)) for key, values in utilization.items()}


def main():
    args = arguments()
    seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import (ImplicitConditionedJointForecaster,
                                  VelocityResidualImplicitForecaster)
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = make_loaders(args)

    checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError(f"Backbone is {checkpoint['image_size']}px, requested {args.image_size}px")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])

    trajectory_checkpoint = torch.load(args.trajectory_checkpoint, map_location="cpu")
    cls = (VelocityResidualImplicitForecaster
           if trajectory_checkpoint.get("model_variant") == "velocity_residual"
           else ImplicitConditionedJointForecaster)
    joint = cls(trajectory_checkpoint["future_frames"],
                trajectory_checkpoint["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"])
    joint.eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)

    backbone_name = str(checkpoint.get("backbone", "simvp")).lower()
    feature_channels = int(getattr(backbone, "backbone_config", {}).get("hid_s", 16))
    model = core.DualPathFeatureWarp(
        feature_channels, args.flow_kernel, args.feature_scale).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    train(model, loaders["train"], loaders["val"], backbone, joint,
          lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = evaluate(
        model, loaders["test"], backbone, joint, device, future_frames,
        args.seed, args.image_size, backbone_name)
    summary = {
        "metrics": metrics, "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args), "split": split_metadata,
            "used": {key: len(value) for key, value in datasets.items()},
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "backbone": backbone_name,
            "feature_channels": feature_channels,
            "experiment": f"{args.image_size}px fair {backbone_name} plus single v1 correct model",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
