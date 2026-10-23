#!/usr/bin/env python3
"""Distill a VGGT score critic, then train dual-critic and fair control adapters."""

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
from torch.utils.data import DataLoader


GEOMETRY_KEYS = (
    "camera_translation_acceleration",
    "camera_rotation_acceleration",
    "depth_reprojection_worst3",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--geometry-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--old-geometry-checkpoint", required=True)
    p.add_argument("--old-control-checkpoint", required=True)
    p.add_argument("--teacher-train-jsonl", required=True)
    p.add_argument("--teacher-val-jsonl", required=True)
    p.add_argument("--teacher-test-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--critic-epochs", type=int, default=12)
    p.add_argument("--adapter-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--vggt-weight", type=float, default=0.02)
    p.add_argument("--d4rt-weight", type=float, default=0.25)
    p.add_argument("--max-correction", type=float, default=0.05)
    p.add_argument("--min-critic-spearman", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_labels(path):
    labels = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[row["name"]][row["variant"]] = row
    return dict(labels)


def paths_for_labels(paths, labels):
    selected = [Path(path) for path in paths if Path(path).stem in labels]
    missing = sorted(set(labels) - {path.stem for path in selected})
    if missing:
        raise RuntimeError(f"Could not map {len(missing)} labeled clips, e.g. {missing[:3]}")
    return selected


class LatentGeometryScoreCritic(nn.Module):
    def __init__(self, outputs=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 12, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(3, 12), nn.GELU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1),
            nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv3d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 48, 3, stride=2, padding=1),
            nn.GroupNorm(8, 48), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(48, 48), nn.GELU(), nn.Linear(48, outputs))

    def forward(self, video):
        # video: B,T,C,H,W
        latent = self.features(video.permute(0, 2, 1, 3, 4))
        return self.head(latent.mean((2, 3, 4)))


class ResidualAdapter(nn.Module):
    def __init__(self, max_correction=0.05, hidden=32):
        super().__init__()
        self.max_correction = float(max_correction)
        self.network = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, baseline, last):
        batch, steps, _, height, width = baseline.shape
        last = last[:, None].expand(-1, steps, -1, -1, -1)
        horizon = torch.linspace(0, 1, steps, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, steps, 1, 1, 1).expand(batch, -1, -1, height, width)
        raw = torch.cat((baseline, last, horizon), 2).reshape(batch * steps, 7, height, width)
        correction = self.max_correction * torch.tanh(self.network(raw))
        correction = correction.reshape(batch, steps, 3, height, width)
        return (baseline + correction).clamp(0, 1), correction


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def label_tensor(names, labels, variants, mean, std, device):
    values = []
    for name in names:
        for variant in variants:
            row = labels[name][variant]
            values.append([row[key] for key in GEOMETRY_KEYS])
    values = torch.tensor(values, dtype=torch.float32, device=device).log1p()
    return ((values - mean) / std).clamp(-6, 6)


def score_normalizer(labels, device):
    values = []
    for variants in labels.values():
        for row in variants.values():
            values.append([row[key] for key in GEOMETRY_KEYS])
    values = torch.tensor(values, dtype=torch.float32, device=device).log1p()
    return values.mean(0), values.std(0).clamp_min(1e-5)


@torch.no_grad()
def old_predictions(batch, backbone, old_geometry, old_control):
    baseline = backbone(batch["context_rgb"], None, None, None)
    geometry = old_geometry(
        baseline, batch["context_rgb"][:, -1], batch["vggt_depth"],
        batch["vggt_confidence"], batch["vggt_pose"], True,
    )[0]
    control = old_control(
        baseline, batch["context_rgb"][:, -1], batch["vggt_depth"],
        batch["vggt_confidence"], batch["vggt_pose"], False,
    )[0]
    return baseline, {
        "ground_truth": batch["future_rgb"], "fair_simvp": baseline,
        "vggt_omega_geometry": geometry, "rgb_capacity_control": control,
    }


def critic_video_batch(context, predictions, variants):
    videos = [torch.cat((context, predictions[variant]), 1) for variant in variants]
    return torch.stack(videos, 1).flatten(0, 1)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[valid]), rankdata(y[valid])
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else float("nan")


@torch.no_grad()
def validate_critic(critic, loader, labels, backbone, old_geometry, old_control, mean, std, device):
    critic.eval()
    variants = ("ground_truth", "fair_simvp", "vggt_omega_geometry", "rgb_capacity_control")
    squared = 0.0
    count = 0
    predicted, target = [], []
    for raw in loader:
        batch = move(raw, device)
        _, predictions = old_predictions(batch, backbone, old_geometry, old_control)
        video = critic_video_batch(batch["context_rgb"], predictions, variants)
        output = critic(video)
        truth = label_tensor(batch["name"], labels, variants, mean, std, device)
        squared += (output - truth).square().sum().item()
        count += truth.numel()
        predicted.append(output.cpu())
        target.append(truth.cpu())
    predicted = torch.cat(predicted).numpy()
    target = torch.cat(target).numpy()
    return {
        "standardized_MSE": squared / max(count, 1),
        **{f"spearman_{key}": spearman(predicted[:, index], target[:, index]) for index, key in enumerate(GEOMETRY_KEYS)},
        "mean_spearman": float(np.nanmean([spearman(predicted[:, i], target[:, i]) for i in range(len(GEOMETRY_KEYS))])),
    }


def train_critic(args, critic, train_loader, val_loader, train_labels, val_labels,
                 backbone, old_geometry, old_control, mean, std, device, output):
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
    variants = ("ground_truth", "fair_simvp", "vggt_omega_geometry", "rgb_capacity_control")
    history, best = [], float("inf")
    for epoch in range(1, args.critic_epochs + 1):
        critic.train(); total = 0.0; batches = 0; started = time.time()
        for raw in train_loader:
            batch = move(raw, device)
            _, predictions = old_predictions(batch, backbone, old_geometry, old_control)
            video = critic_video_batch(batch["context_rgb"], predictions, variants)
            truth = label_tensor(batch["name"], train_labels, variants, mean, std, device)
            predicted = critic(video)
            loss = F.smooth_l1_loss(predicted, truth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); batches += 1
        validation = validate_critic(
            critic, val_loader, val_labels, backbone, old_geometry, old_control,
            mean, std, device,
        )
        row = {"epoch": epoch, "epoch_seconds": time.time() - started,
               "train_loss": total / max(batches, 1), **validation}
        history.append(row)
        (output / "critic_history.json").write_text(json.dumps(history, indent=2))
        saved = {"model": critic.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                 "geometry_keys": GEOMETRY_KEYS, "epoch": epoch, "validation": validation}
        torch.save(saved, output / "critic_last.pt")
        if validation["standardized_MSE"] < best:
            best = validation["standardized_MSE"]
            torch.save(saved, output / "critic_best.pt")
        print(json.dumps({"stage": "critic", **row}), flush=True)
    selected = torch.load(output / "critic_best.pt", map_location=device)
    critic.load_state_dict(selected["model"])
    return history, selected["validation"]


def dynamic_track_loss(prediction, target, batch):
    uv_raw = batch["future_uv_teacher"].float()
    uv = torch.nan_to_num(uv_raw).clamp(0, 1)
    valid = batch["future_valid"].bool() & torch.isfinite(uv_raw).all(-1)
    confidence = torch.nan_to_num(batch["future_confidence"].float()).clamp(0, 1)
    past_uv = torch.nan_to_num(batch["past_uv"].float()).clamp(0, 1)
    past_valid = batch["past_valid"].bool()
    b, n, steps = uv.shape[:3]
    indices = torch.arange(past_valid.shape[2], device=prediction.device).view(1, 1, -1).expand(b, n, -1)
    last_index = torch.where(past_valid, indices, torch.zeros_like(indices)).max(2).values
    anchor = past_uv.gather(2, last_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    amplitude = torch.where(valid, (uv - anchor[:, :, None]).norm(dim=-1), torch.zeros_like(valid, dtype=prediction.dtype)).amax(2)
    dynamic = amplitude > 0.05
    sampled_errors = []
    difference = (prediction - target).abs().mean(2)
    for step in range(steps):
        grid = uv[:, :, step].mul(2).sub(1).unsqueeze(2)
        sampled = F.grid_sample(difference[:, step:step + 1], grid, mode="bilinear", align_corners=True)
        sampled_errors.append(sampled[:, 0, :, 0])
    error = torch.stack(sampled_errors, 2)
    weight = confidence * valid.float() * dynamic[:, :, None].float()
    return (error * weight).sum() / weight.sum().clamp_min(1)


def common_loss(prediction, target, last, correction, lpips_model, args):
    difference = prediction - target
    rgb = difference.abs().mean() + 0.1 * difference.square().mean()
    pred_delta = torch.cat((prediction[:, :1] - last[:, None], prediction[:, 1:] - prediction[:, :-1]), 1)
    target_delta = torch.cat((target[:, :1] - last[:, None], target[:, 1:] - target[:, :-1]), 1)
    temporal = (pred_delta - target_delta).abs().mean()
    b, k = prediction.shape[:2]
    perceptual = lpips_model(
        prediction.reshape(b * k, 3, *prediction.shape[-2:]).mul(2).sub(1),
        target.reshape(b * k, 3, *target.shape[-2:]).mul(2).sub(1),
    ).mean()
    correction_size = correction.abs().mean()
    total = rgb + args.lpips_weight * perceptual + args.temporal_weight * temporal + args.correction_weight * correction_size
    return total, {"rgb": rgb, "lpips": perceptual, "temporal": temporal, "correction": correction_size}


def gt_geometry_targets(names, labels, mean, std, device):
    values = torch.tensor(
        [[labels[name]["ground_truth"][key] for key in GEOMETRY_KEYS] for name in names],
        dtype=torch.float32, device=device,
    ).log1p()
    return ((values - mean) / std).clamp(-6, 6)


@torch.no_grad()
def validation_mse(adapter, loader, backbone, device):
    adapter.eval(); squared = count = 0.0
    for raw in loader:
        batch = move(raw, device)
        baseline = backbone(batch["context_rgb"], None, None, None)
        prediction, _ = adapter(baseline, batch["context_rgb"][:, -1])
        squared += (prediction - batch["future_rgb"]).square().sum().item()
        count += batch["future_rgb"].numel()
    return squared / max(count, 1)


def train_adapters(args, critic, train_loader, val_loader, train_labels, backbone,
                   mean, std, device, output, lpips_model):
    dual = ResidualAdapter(args.max_correction).to(device)
    control = copy.deepcopy(dual)
    models = {"dual_geometry_critic": dual, "zero_geometry_control": control}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.adapter_lr, weight_decay=1e-4)
                  for name, model in models.items()}
    best = {name: float("inf") for name in models}
    history = []
    critic.eval()
    for epoch in range(1, args.adapter_epochs + 1):
        for model in models.values(): model.train()
        totals = {name: defaultdict(float) for name in models}; batches = 0; started = time.time()
        for raw in train_loader:
            batch = move(raw, device)
            with torch.no_grad():
                baseline = backbone(batch["context_rgb"], None, None, None)
            for name, model in models.items():
                prediction, correction = model(baseline, batch["context_rgb"][:, -1])
                loss, pieces = common_loss(
                    prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                    correction, lpips_model, args,
                )
                geometry = prediction.new_zeros(())
                track = prediction.new_zeros(())
                if name == "dual_geometry_critic":
                    video = torch.cat((batch["context_rgb"], prediction), 1)
                    predicted_scores = critic(video)
                    target_scores = gt_geometry_targets(batch["name"], train_labels, mean, std, device)
                    geometry = F.smooth_l1_loss(predicted_scores, target_scores)
                    track = dynamic_track_loss(prediction, batch["future_rgb"], batch)
                    loss = loss + args.vggt_weight * geometry + args.d4rt_weight * track
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                for key, value in pieces.items(): totals[name][key] += value.item()
                totals[name]["geometry"] += geometry.item(); totals[name]["track"] += track.item()
            batches += 1
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, model in models.items():
            value = validation_mse(model, val_loader, backbone, device)
            row[f"{name}_val_MSE"] = value
            for key, total in totals[name].items(): row[f"{name}_{key}"] = total / max(batches, 1)
            saved = {"model": model.state_dict(), "epoch": epoch, "config": vars(args),
                     "trainable_params": sum(p.numel() for p in model.parameters())}
            torch.save(saved, output / f"{name}_last.pt")
            if value < best[name]:
                best[name] = value; torch.save(saved, output / f"{name}_best.pt")
        history.append(row)
        (output / "adapter_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "adapter", **row}), flush=True)
    for name, model in models.items():
        model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device)["model"])
    return models, history


@torch.no_grad()
def evaluate(models, loader, backbone, test_labels, critic, mean, std, device, future_frames):
    from video_metrics import VideoMetricAccumulator
    names = ("fair_simvp", *models.keys())
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in names}
    dynamic = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0, "track": []} for name in names}
    geometry_gaps = {name: [] for name in names}
    for raw in loader:
        batch = move(raw, device)
        baseline = backbone(batch["context_rgb"], None, None, None)
        predictions = {"fair_simvp": baseline}
        for name, model in models.items():
            predictions[name] = model(baseline, batch["context_rgb"][:, -1])[0]
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2, keepdim=True) > 0.05
        gt_scores = gt_geometry_targets(batch["name"], test_labels, mean, std, device)
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, last)
            difference = prediction - target
            selected = target_motion.expand_as(difference)
            dynamic[name]["abs"] += difference.abs()[selected].sum().item()
            dynamic[name]["sq"] += difference.square()[selected].sum().item()
            dynamic[name]["count"] += selected.sum().item()
            dynamic[name]["track"].append(float(dynamic_track_loss(prediction, target, batch)))
            score = critic(torch.cat((batch["context_rgb"], prediction), 1))
            geometry_gaps[name].extend((score - gt_scores).abs().sum(1).cpu().tolist())
    results = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute(); count = max(dynamic[name]["count"], 1)
        dmse = dynamic[name]["sq"] / count
        metrics.update({
            "dynamic_MAE": dynamic[name]["abs"] / count,
            "dynamic_MSE": dmse,
            "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)),
            "dynamic_track_loss": float(np.mean(dynamic[name]["track"])),
            "surrogate_geometry_gap_to_GT": float(np.mean(geometry_gaps[name])),
        })
        results[name] = metrics
    return results


def main():
    args = parse_args(); seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from train_pointodyssey_dmvfn_injection import combined_splits
    from train_vggt_omega_injection import GeometryDataset, VGGTResidual, geometry_collate
    from video_forecast_dataset import VideoForecastDataset
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    labels = {
        "train": read_labels(args.teacher_train_jsonl),
        "val": read_labels(args.teacher_val_jsonl),
        "test": read_labels(args.teacher_test_jsonl),
    }
    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
    datasets = {}
    for split in ("train", "val", "test"):
        paths = paths_for_labels(splits[split], labels[split])
        base = VideoForecastDataset(args.cache_root, split, 64, paths=paths, cache_items=False)
        datasets[split] = GeometryDataset(base, args.geometry_cache)
    train_loader = DataLoader(datasets["train"], args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=geometry_collate, num_workers=0)
    val_loader = DataLoader(datasets["val"], args.batch_size, collate_fn=geometry_collate, num_workers=0)
    test_loader = DataLoader(datasets["test"], 1, collate_fn=geometry_collate, num_workers=0)

    backbone_checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(backbone_checkpoint).to(device).eval()
    backbone.load_state_dict(backbone_checkpoint["model"])
    context_frames, future_frames = int(backbone_checkpoint["context_frames"]), int(backbone_checkpoint["future_frames"])
    old_geometry = VGGTResidual(context_frames, future_frames).to(device).eval()
    old_control = VGGTResidual(context_frames, future_frames).to(device).eval()
    old_geometry.load_state_dict(torch.load(args.old_geometry_checkpoint, map_location="cpu")["model"])
    old_control.load_state_dict(torch.load(args.old_control_checkpoint, map_location="cpu")["model"])
    for module in (backbone, old_geometry, old_control):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    mean, std = score_normalizer(labels["train"], device)
    critic = LatentGeometryScoreCritic().to(device)
    critic_history, critic_validation = train_critic(
        args, critic, train_loader, val_loader, labels["train"], labels["val"],
        backbone, old_geometry, old_control, mean, std, device, output,
    )
    if critic_validation["mean_spearman"] < args.min_critic_spearman:
        rejected = {
            "status": "critic_rejected",
            "reason": "held-out mean Spearman below threshold",
            "critic_validation": critic_validation,
            "threshold": args.min_critic_spearman,
            "meta": {"used": {k: len(v) for k, v in datasets.items()}, "config": vars(args)},
        }
        (output / "summary.json").write_text(json.dumps(rejected, indent=2))
        (output / "CRITIC_REJECTED").write_text("critic rejected\n")
        print(json.dumps(rejected), flush=True)
        return
    for parameter in critic.parameters(): parameter.requires_grad_(False)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters(): parameter.requires_grad_(False)
    models, adapter_history = train_adapters(
        args, critic, train_loader, val_loader, labels["train"], backbone,
        mean, std, device, output, lpips_model,
    )
    metrics = evaluate(models, test_loader, backbone, labels["test"], critic, mean, std, device, future_frames)
    summary = {
        "metrics": metrics, "critic_validation": critic_validation,
        "meta": {"used": {k: len(v) for k, v in datasets.items()}, "split": split_meta,
                 "geometry_keys": GEOMETRY_KEYS, "normalizer_mean": mean.cpu().tolist(),
                 "normalizer_std": std.cpu().tolist(), "config": vars(args)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
