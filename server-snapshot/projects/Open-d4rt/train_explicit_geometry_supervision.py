#!/usr/bin/env python3
"""Train equal-capacity adapters with explicit VGGT-static + D4RT-dynamic losses."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--dense-geometry-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--static-weight", type=float, default=0.10)
    p.add_argument("--d4rt-weight", type=float, default=0.25)
    p.add_argument("--static-objective", choices=("warp", "protect"), default="warp")
    p.add_argument("--include-d4rt-only", action="store_true")
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--track-propagation-weight", type=float, default=0.10)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--dynamic-radius", type=int, default=3)
    p.add_argument("--max-correction", type=float, default=0.05)
    p.add_argument("--max-train", type=int, default=216)
    p.add_argument("--max-val", type=int, default=80)
    p.add_argument("--max-test", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class DenseGeometryDataset(Dataset):
    def __init__(self, base, root):
        self.base, self.root = base, Path(root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        path = Path(self.base.paths[index])
        cached = self.root / path.parent.name / f"{path.stem}.npz"
        with np.load(cached) as geometry:
            item["static_grid"] = torch.from_numpy(geometry["static_grid"].astype(np.float32))
            item["static_confidence"] = torch.from_numpy(geometry["static_confidence"].astype(np.float32))
        return item


def dense_collate(items):
    from video_forecast_dataset import video_pad_collate
    output = video_pad_collate(items)
    output["static_grid"] = torch.stack([item["static_grid"] for item in items])
    output["static_confidence"] = torch.stack([item["static_confidence"] for item in items])
    return output


class ResidualAdapter(nn.Module):
    def __init__(self, max_correction=0.05, hidden=32):
        super().__init__()
        self.max_correction = float(max_correction)
        self.network = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, baseline, last):
        b, k, _, h, w = baseline.shape
        last_video = last[:, None].expand(-1, k, -1, -1, -1)
        horizon = torch.linspace(0, 1, k, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, k, 1, 1, 1).expand(b, -1, -1, h, w)
        inputs = torch.cat((baseline, last_video, horizon), 2).reshape(b * k, 7, h, w)
        correction = self.max_correction * torch.tanh(self.network(inputs))
        correction = correction.reshape(b, k, 3, h, w)
        return (baseline + correction).clamp(0, 1), correction


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def last_observation(batch):
    valid = batch["past_valid"].bool()
    b, n, t = valid.shape
    indices = torch.arange(t, device=valid.device).view(1, 1, t).expand(b, n, -1)
    last_index = torch.where(valid, indices, torch.zeros_like(indices)).max(2).values
    uv = torch.nan_to_num(batch["past_uv"].float()).clamp(0, 1)
    anchor = uv.gather(2, last_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    return anchor, valid.any(2)


def trajectory_state(batch, threshold):
    raw_uv = batch["future_uv_teacher"].float()
    uv = torch.nan_to_num(raw_uv).clamp(0, 1)
    valid = batch["future_valid"].bool() & torch.isfinite(raw_uv).all(-1)
    confidence = torch.nan_to_num(batch["future_confidence"].float()).clamp(0, 1)
    anchor, past_valid = last_observation(batch)
    displacement = torch.where(valid, (uv - anchor[:, :, None]).norm(dim=-1), torch.zeros_like(confidence))
    dynamic = displacement.amax(2) > threshold
    weight = confidence * valid.float() * dynamic[:, :, None].float() * past_valid[:, :, None].float()
    return uv, anchor, weight, dynamic


def sample_video(video, uv):
    # video B,K,C,H,W; uv B,N,K,2 -> B,N,K,C
    outputs = []
    for step in range(video.shape[1]):
        grid = uv[:, :, step].mul(2).sub(1).unsqueeze(2)
        value = F.grid_sample(video[:, step], grid, mode="bilinear", padding_mode="border", align_corners=True)
        outputs.append(value[:, :, :, 0].transpose(1, 2))
    return torch.stack(outputs, 2)


def sample_image(image, uv):
    grid = uv.mul(2).sub(1).unsqueeze(2)
    value = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return value[:, :, :, 0].transpose(1, 2)


def dynamic_support(batch, height, width, threshold, radius):
    uv, _, weight, _ = trajectory_state(batch, threshold)
    b, n, k = weight.shape
    masks = torch.zeros((b, k, 1, height, width), device=uv.device, dtype=weight.dtype)
    x = (uv[..., 0] * (width - 1)).round().long().clamp(0, width - 1)
    y = (uv[..., 1] * (height - 1)).round().long().clamp(0, height - 1)
    for batch_index in range(b):
        for step in range(k):
            flat = masks[batch_index, step, 0].flatten()
            flat.scatter_add_(0, y[batch_index, :, step] * width + x[batch_index, :, step], weight[batch_index, :, step])
    kernel = max(1, 2 * int(radius) + 1)
    masks = F.max_pool2d(masks.flatten(0, 1), kernel, stride=1, padding=kernel // 2)
    return masks.reshape(b, k, 1, height, width).clamp(0, 1)


def static_dense_loss(prediction, last, batch, args):
    b, k, _, h, w = prediction.shape
    grid = batch["static_grid"].reshape(b * k, h, w, 2)
    source = last[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, 3, h, w)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.reshape(b, k, 3, h, w)
    dynamic = dynamic_support(batch, h, w, args.dynamic_threshold, args.dynamic_radius)
    weight = batch["static_confidence"][:, :, None].clamp(0, 1) * (1 - dynamic)
    error = torch.sqrt((prediction - warped).square() + 1e-6).mean(2, keepdim=True)
    return (error * weight).sum() / weight.sum().clamp_min(1), weight.mean(), warped


def static_protection_loss(correction, batch, args):
    """Prevent the residual adapter from editing reliable static background.

    VGGT supplies only the permission mask.  Unlike static_dense_loss, this
    objective never asks RGB predictions to imitate a VGGT-derived warp.
    """
    b, k, _, h, w = correction.shape
    dynamic = dynamic_support(batch, h, w, args.dynamic_threshold, args.dynamic_radius)
    weight = batch["static_confidence"][:, :, None].clamp(0, 1) * (1 - dynamic)
    edit = correction.abs().mean(2, keepdim=True)
    return (edit * weight).sum() / weight.sum().clamp_min(1), weight.mean()


def training_static_loss(prediction, last, correction, batch, args):
    if args.static_objective == "protect":
        return static_protection_loss(correction, batch, args)
    value, fraction, _ = static_dense_loss(prediction, last, batch, args)
    return value, fraction


def d4rt_track_loss(prediction, target, last, batch, args):
    uv, anchor_uv, weight, _ = trajectory_state(batch, args.dynamic_threshold)
    pred_samples = sample_video(prediction, uv)
    target_samples = sample_video(target, uv)
    anchor = sample_image(last, anchor_uv)
    point_error = (pred_samples - target_samples).abs().mean(-1)
    pred_sequence = torch.cat((anchor[:, :, None], pred_samples), 2)
    target_sequence = torch.cat((anchor[:, :, None], target_samples), 2)
    temporal_error = ((pred_sequence[:, :, 1:] - pred_sequence[:, :, :-1])
                      - (target_sequence[:, :, 1:] - target_sequence[:, :, :-1])).abs().mean(-1)
    propagation_error = (pred_samples - anchor[:, :, None]).abs().mean(-1)
    denominator = weight.sum().clamp_min(1)
    point = (point_error * weight).sum() / denominator
    temporal = (temporal_error * weight).sum() / denominator
    propagation = (propagation_error * weight).sum() / denominator
    total = point + args.track_temporal_weight * temporal + args.track_propagation_weight * propagation
    return total, {"track_point": point, "track_temporal": temporal, "track_propagation": propagation,
                   "dynamic_track_weight": weight.mean()}


def common_loss(prediction, target, last, correction, lpips_model, args):
    difference = prediction - target
    rgb = difference.abs().mean() + 0.1 * difference.square().mean()
    pred_previous = torch.cat((last[:, None], prediction[:, :-1]), 1)
    target_previous = torch.cat((last[:, None], target[:, :-1]), 1)
    temporal = ((prediction - pred_previous) - (target - target_previous)).abs().mean()
    b, k = prediction.shape[:2]
    perceptual = lpips_model(
        prediction.reshape(b * k, 3, *prediction.shape[-2:]).mul(2).sub(1),
        target.reshape(b * k, 3, *target.shape[-2:]).mul(2).sub(1),
    ).mean()
    correction_size = correction.abs().mean()
    total = rgb + args.lpips_weight * perceptual + args.temporal_weight * temporal + args.correction_weight * correction_size
    return total, {"rgb": rgb, "lpips": perceptual, "temporal": temporal, "correction": correction_size}


@torch.no_grad()
def validate(model, loader, backbone, device, args):
    model.eval(); totals = defaultdict(float); batches = 0
    for raw in loader:
        batch = move(raw, device)
        baseline = backbone(batch["context_rgb"], None, None, None)
        prediction, correction = model(baseline, batch["context_rgb"][:, -1])
        difference = prediction - batch["future_rgb"]
        static_reprojection, fraction, _ = static_dense_loss(
            prediction, batch["context_rgb"][:, -1], batch, args,
        )
        static_edit, _ = static_protection_loss(correction, batch, args)
        track, pieces = d4rt_track_loss(prediction, batch["future_rgb"], batch["context_rgb"][:, -1], batch, args)
        totals["MSE"] += difference.square().mean().item(); totals["MAE"] += difference.abs().mean().item()
        totals["static_reprojection"] += static_reprojection.item()
        totals["static_edit"] += static_edit.item(); totals["static_fraction"] += fraction.item()
        totals["d4rt_track"] += track.item()
        for key, value in pieces.items(): totals[key] += value.item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_models(models, train_loader, val_loader, backbone, lpips_model, device, args, output):
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) for name, model in models.items()}
    best, history = {name: float("inf") for name in models}, []
    for epoch in range(1, args.epochs + 1):
        for model in models.values(): model.train()
        totals = {name: defaultdict(float) for name in models}; batches = 0; started = time.time()
        for raw in train_loader:
            batch = move(raw, device)
            with torch.no_grad(): baseline = backbone(batch["context_rgb"], None, None, None)
            for name, model in models.items():
                prediction, correction = model(baseline, batch["context_rgb"][:, -1])
                loss, pieces = common_loss(
                    prediction, batch["future_rgb"], batch["context_rgb"][:, -1], correction, lpips_model, args,
                )
                static = prediction.new_zeros(()); track = prediction.new_zeros(())
                uses_static = name in ("explicit_geometry", "masked_dynamic_geometry")
                uses_d4rt = uses_static or name == "d4rt_only"
                if uses_static:
                    static, static_fraction = training_static_loss(
                        prediction, batch["context_rgb"][:, -1], correction, batch, args,
                    )
                    loss = loss + args.static_weight * static
                    pieces["static_fraction"] = static_fraction
                if uses_d4rt:
                    track, track_pieces = d4rt_track_loss(
                        prediction, batch["future_rgb"], batch["context_rgb"][:, -1], batch, args,
                    )
                    loss = loss + args.d4rt_weight * track
                    pieces.update(track_pieces)
                optimizers[name].zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizers[name].step()
                totals[name]["loss"] += loss.item(); totals[name]["static_dense"] += static.item()
                totals[name]["d4rt_track"] += track.item()
                for key, value in pieces.items(): totals[name][key] += float(value.detach())
            batches += 1
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, model in models.items():
            metrics = validate(model, val_loader, backbone, device, args)
            for key, value in metrics.items(): row[f"{name}_val_{key}"] = value
            for key, value in totals[name].items(): row[f"{name}_train_{key}"] = value / max(batches, 1)
            saved = {"model": model.state_dict(), "epoch": epoch, "config": vars(args),
                     "trainable_params": sum(p.numel() for p in model.parameters()), "validation": metrics}
            torch.save(saved, output / f"{name}_last.pt")
            # The same primary video-prediction criterion is used for both branches.
            if metrics["MSE"] < best[name]:
                best[name] = metrics["MSE"]; torch.save(saved, output / f"{name}_best.pt")
        history.append(row); (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "adapter", **row}), flush=True)
    for name, model in models.items():
        model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device)["model"])
    return history


@torch.no_grad()
def evaluate(models, loader, backbone, device, args, future_frames):
    from video_metrics import VideoMetricAccumulator
    names = ("frozen_simvp", *models.keys())
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in names}
    auxiliary = {name: defaultdict(list) for name in names}; records = []
    for raw in loader:
        batch = move(raw, device)
        baseline = backbone(batch["context_rgb"], None, None, None)
        predictions = {"frozen_simvp": baseline}
        predictions.update({name: model(baseline, batch["context_rgb"][:, -1])[0] for name, model in models.items()})
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2, keepdim=True) > 0.05
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, last)
            error = prediction - target; selected = target_motion.expand_as(error)
            dmse = error.square()[selected].mean().item() if selected.any() else float("nan")
            dmae = error.abs()[selected].mean().item() if selected.any() else float("nan")
            static_reprojection, static_fraction, _ = static_dense_loss(prediction, last, batch, args)
            if name == "frozen_simvp":
                correction = torch.zeros_like(prediction)
            else:
                correction = prediction - baseline
            static_edit, _ = static_protection_loss(correction, batch, args)
            track, pieces = d4rt_track_loss(prediction, target, last, batch, args)
            auxiliary[name]["dynamic_MAE"].append(dmae); auxiliary[name]["dynamic_MSE"].append(dmse)
            auxiliary[name]["static_reprojection_loss"].append(static_reprojection.item())
            auxiliary[name]["static_edit_magnitude"].append(static_edit.item())
            auxiliary[name]["static_fraction"].append(static_fraction.item())
            auxiliary[name]["d4rt_track_loss"].append(track.item())
            for key, value in pieces.items(): auxiliary[name][key].append(value.item())
            records.append({"name": batch["name"][0], "variant": name,
                            "MSE": error.square().mean().item(), "MAE": error.abs().mean().item(),
                            "dynamic_MAE": dmae, "dynamic_MSE": dmse,
                            "static_reprojection_loss": static_reprojection.item(),
                            "static_edit_magnitude": static_edit.item(),
                            "d4rt_track_loss": track.item()})
    results = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute()
        for key, values in auxiliary[name].items(): metrics[key] = float(np.nanmean(values))
        metrics["dynamic_PSNR"] = -10 * math.log10(max(metrics["dynamic_MSE"], 1e-12))
        results[name] = metrics
    return results, records


def main():
    args = parse_args(); seed_all(args.seed)
    root = Path(args.d4rt_root); sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from train_pointodyssey_dmvfn_injection import combined_splits
    from video_forecast_dataset import VideoForecastDataset
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.dense_geometry_cache) / "manifest.json").read_text())
    paths = {split: [Path(path) for path in manifest["selected"][split]] for split in ("train", "val", "test")}
    # Assert that this run uses exactly the predeclared/cache-complete sample set.
    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
    for split in paths:
        valid = {str(Path(path)) for path in splits[split]}
        missing = [str(path) for path in paths[split] if str(path) not in valid]
        if missing: raise RuntimeError(f"{split}: {len(missing)} cached paths are outside the current split")
    datasets = {}
    for split in ("train", "val", "test"):
        base = VideoForecastDataset(args.cache_root, split, 64, paths=paths[split], cache_items=False)
        datasets[split] = DenseGeometryDataset(base, args.dense_geometry_cache)
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed), collate_fn=dense_collate, num_workers=0),
        "val": DataLoader(datasets["val"], args.batch_size, collate_fn=dense_collate, num_workers=0),
        "test": DataLoader(datasets["test"], 1, collate_fn=dense_collate, num_workers=0),
    }
    checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device).eval()
    backbone.load_state_dict(checkpoint["model"])
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    context_frames, future_frames = int(checkpoint["context_frames"]), int(checkpoint["future_frames"])
    geometry = ResidualAdapter(args.max_correction).to(device)
    primary_name = "masked_dynamic_geometry" if args.static_objective == "protect" else "explicit_geometry"
    models = {primary_name: geometry}
    if args.include_d4rt_only:
        models["d4rt_only"] = copy.deepcopy(geometry)
    models["zero_geometry_control"] = copy.deepcopy(geometry)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters(): parameter.requires_grad_(False)
    history = train_models(models, loaders["train"], loaders["val"], backbone, lpips_model, device, args, output)
    metrics, records = evaluate(models, loaders["test"], backbone, device, args, future_frames)
    summary = {"metrics": metrics, "meta": {"config": vars(args), "used": {k: len(v) for k, v in datasets.items()},
                "split": split_meta, "context_frames": context_frames, "future_frames": future_frames,
                "fairness": "identical architecture/initialization; only loss supervision differs",
                "static_role": ("permission-mask correction penalty" if args.static_objective == "protect"
                                else "RGB warp target")}}
    (output / "per_sequence.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
