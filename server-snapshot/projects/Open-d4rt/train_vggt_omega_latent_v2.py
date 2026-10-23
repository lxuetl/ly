#!/usr/bin/env python3
"""Train a causal VGGT-Omega latent adapter inside a frozen TAU predictor.

The adapter uses three complementary context-only signals:
1) per-frame camera/register tokens queried by future-horizon cross-attention;
2) physically relative camera-pose motion;
3) confidence-gated temporal depth statistics for spatial correction.
It is zero initialized and bounded, so disabling it exactly recovers TAU.
"""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_simvp_feature_warp as core


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", default="")
    p.add_argument("--omega-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--token-hidden", type=int, default=96)
    p.add_argument("--context-hidden", type=int, default=256)
    p.add_argument("--feature-scale", type=float, default=0.12)
    p.add_argument("--dynamic-weight", type=float, default=2.0)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--static-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--rank-weight", type=float, default=0.50)
    p.add_argument("--wrong-reject-weight", type=float, default=0.05)
    p.add_argument("--rank-margin", type=float, default=2e-4)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OmegaLatentDataset(Dataset):
    def __init__(self, base, omega_root):
        self.base = base
        self.omega_root = Path(omega_root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        source = Path(self.base.paths[index])
        path = self.omega_root / source.parent.name / f"{source.stem}.pt"
        omega = torch.load(path, map_location="cpu", weights_only=True)
        item["omega_tokens"] = omega["tokens"].float()
        item["omega_depth"] = omega["depth"].float()
        item["omega_confidence"] = omega["depth_confidence"].float()
        item["omega_pose"] = omega["pose_enc"].float()
        return item


def collate(items):
    from video_forecast_dataset import video_pad_collate

    output = video_pad_collate(items)
    for key in ("omega_tokens", "omega_depth", "omega_confidence", "omega_pose"):
        output[key] = torch.stack([item[key] for item in items])
    return output


def make_loaders(a):
    from video_forecast_dataset import VideoForecastDataset
    from train_pointodyssey_dmvfn_injection import combined_splits
    from types import SimpleNamespace

    splits, metadata = combined_splits(SimpleNamespace(**vars(a)))
    limits = {"train": a.max_train, "val": a.max_val, "test": a.max_test}
    datasets = {}
    for split in ("train", "val", "test"):
        paths = splits[split][: limits[split] or None]
        base = VideoForecastDataset(
            a.cache_root, split, a.image_size, paths=paths, cache_items=False
        )
        datasets[split] = OmegaLatentDataset(base, a.omega_cache)
    generator = torch.Generator().manual_seed(a.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], a.batch_size, shuffle=True, generator=generator,
            collate_fn=collate, num_workers=a.num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], a.batch_size, collate_fn=collate,
            num_workers=a.num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], a.batch_size, collate_fn=collate,
            num_workers=a.num_workers, pin_memory=True,
        ),
    }
    return loaders, datasets, metadata


def move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def quaternion_relative(quaternion):
    quaternion = F.normalize(quaternion, dim=-1)
    reference = quaternion[:, :1]
    sign = torch.sign((quaternion * reference).sum(-1, keepdim=True)).clamp(min=-1, max=1)
    quaternion = quaternion * torch.where(sign == 0, torch.ones_like(sign), sign)
    w0, x0, y0, z0 = reference.unbind(-1)
    wt, xt, yt, zt = quaternion.unbind(-1)
    return torch.stack((
        wt * w0 + xt * x0 + yt * y0 + zt * z0,
        -wt * x0 + xt * w0 - yt * z0 + zt * y0,
        -wt * y0 + xt * z0 + yt * w0 - zt * x0,
        -wt * z0 - xt * y0 + yt * x0 + zt * w0,
    ), -1)


class OmegaLatentAdapterV2(nn.Module):
    """Future-query cross-attention plus confidence-gated spatial depth injection."""

    def __init__(self, channels, future_frames, token_hidden=96,
                 context_hidden=256, feature_scale=0.12):
        super().__init__()
        self.channels = int(channels)
        self.future_frames = int(future_frames)
        self.feature_scale = float(feature_scale)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, token_hidden), nn.GELU()
        )
        self.register_projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, token_hidden), nn.GELU()
        )
        self.pose_projection = nn.Sequential(
            nn.LayerNorm(18), nn.Linear(18, context_hidden - 2 * token_hidden), nn.GELU()
        )
        self.context_norm = nn.LayerNorm(context_hidden)
        self.context_time = nn.Parameter(torch.zeros(1, 32, context_hidden))
        self.horizon_query = nn.Sequential(
            nn.Linear(6, context_hidden), nn.GELU(), nn.Linear(context_hidden, context_hidden)
        )
        self.cross_attention = nn.MultiheadAttention(
            context_hidden, num_heads=4, dropout=0.05, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(context_hidden)
        self.film = nn.Sequential(
            nn.Linear(context_hidden, context_hidden), nn.GELU(),
            nn.Linear(context_hidden, channels * 2),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(context_hidden, channels), nn.Sigmoid()
        )
        depth_hidden = max(16, channels // 2)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(5, depth_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(depth_hidden, channels, 3, padding=1),
        )
        self.depth_horizon = nn.Sequential(
            nn.Linear(context_hidden, channels), nn.Tanh()
        )
        self.risk = nn.Sequential(
            nn.Linear(context_hidden, channels), nn.Sigmoid()
        )
        self.global_gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.context_time, std=0.01)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        nn.init.zeros_(self.depth_encoder[-1].weight)
        nn.init.zeros_(self.depth_encoder[-1].bias)
        self.last = {}

    @staticmethod
    def pose_motion(pose):
        translation = pose[..., :3] - pose[:, :1, :3]
        relative_quaternion = quaternion_relative(pose[..., 3:7])
        fov = pose[..., 7:9] - pose[:, :1, 7:9]
        relative = torch.cat((translation, relative_quaternion, fov), -1)
        velocity = torch.cat((
            torch.zeros_like(relative[:, :1]), relative[:, 1:] - relative[:, :-1]
        ), 1)
        motion = torch.cat((relative, velocity), -1)
        scale = motion.square().mean(1, keepdim=True).sqrt().clamp_min(1e-4)
        return (motion / scale).clamp(-10, 10)

    @staticmethod
    def depth_statistics(depth, confidence):
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-4).log()
        mean = depth.mean((-2, -1), keepdim=True)
        std = depth.std((-2, -1), keepdim=True).clamp_min(1e-4)
        depth = ((depth - mean) / std).clamp(-5, 5)
        temporal_mean = depth.mean(1)
        temporal_std = depth.std(1)
        temporal_delta = depth[:, -1] - depth[:, 0]
        last = depth[:, -1]
        confidence = torch.nan_to_num(confidence, nan=0.0).clamp_min(0)
        confidence = confidence / confidence.amax((-2, -1), keepdim=True).clamp_min(1e-4)
        last_confidence = confidence[:, -1]
        statistics = torch.stack((last, temporal_mean, temporal_std, temporal_delta,
                                  last_confidence), 1)
        return statistics, last_confidence[:, None]

    def forward(self, hidden, tokens, depth, confidence, pose,
                use_tokens=True, use_pose=True, use_depth=True):
        batch, future_steps, channels, height, width = hidden.shape
        if channels != self.channels or future_steps != self.future_frames:
            raise ValueError(f"unexpected hidden shape {tuple(hidden.shape)}")
        tokens = tokens.to(hidden.dtype)
        pose = pose.to(hidden.dtype)
        camera = self.camera_projection(tokens[:, :, 0])
        registers = self.register_projection(tokens[:, :, 1:].mean(2))
        if not use_tokens:
            camera = torch.zeros_like(camera)
            registers = torch.zeros_like(registers)
        pose_feature = self.pose_projection(self.pose_motion(pose))
        if not use_pose:
            pose_feature = torch.zeros_like(pose_feature)
        context = torch.cat((camera, registers, pose_feature), -1)
        context = self.context_norm(context + self.context_time[:, :context.shape[1]])

        horizon = torch.linspace(0, 1, future_steps, device=hidden.device, dtype=hidden.dtype)
        basis = torch.stack((
            horizon, horizon.square(), horizon.pow(3),
            torch.sin(math.pi * horizon), torch.cos(math.pi * horizon),
            torch.ones_like(horizon),
        ), -1)
        query = self.horizon_query(basis)[None].expand(batch, -1, -1)
        attended, attention = self.cross_attention(query, context, context,
                                                   need_weights=True)
        attended = self.attention_norm(attended + query)
        gamma, beta = self.film(attended).chunk(2, -1)
        film_delta = hidden * torch.tanh(gamma[..., None, None]) + beta[..., None, None]

        if use_depth:
            statistics, confidence_map = self.depth_statistics(depth, confidence)
            statistics = F.interpolate(
                statistics, (height, width), mode="bilinear", align_corners=False
            )
            confidence_map = F.interpolate(
                confidence_map, (height, width), mode="bilinear", align_corners=False
            )
            depth_feature = self.depth_encoder(statistics)[:, None]
            depth_scale = self.depth_horizon(attended)[..., None, None]
            depth_delta = depth_feature * depth_scale
        else:
            confidence_map = hidden.new_ones((batch, 1, height, width))
            depth_delta = torch.zeros_like(hidden)

        channel_gate = self.channel_gate(attended)[..., None, None]
        risk_gate = self.risk(attended)[..., None, None]
        spatial_gate = 0.25 + 0.75 * confidence_map[:, None]
        gate = torch.sigmoid(self.global_gate_logit) * channel_gate * risk_gate * spatial_gate
        correction = self.feature_scale * torch.tanh(film_delta + depth_delta) * gate
        self.last = {
            "correction": correction,
            "gate": gate.detach(),
            "attention": attention.detach(),
            "token_gate": torch.sigmoid(self.global_gate_logit).detach(),
        }
        return hidden + correction


def omega_inputs(batch, shuffled=False):
    keys = ("omega_tokens", "omega_depth", "omega_confidence", "omega_pose")
    values = [batch[key] for key in keys]
    if shuffled:
        if values[0].shape[0] > 1:
            values = [value.roll(1, 0) for value in values]
        else:
            values = [value.flip(1) for value in values]
    return values


def dynamic_error(prediction, target, last, threshold):
    mask = ((target - last[:, None]).abs().mean(2, keepdim=True) > threshold).float()
    denominator = (mask.sum() * prediction.shape[2]).clamp_min(1)
    return ((prediction - target).square() * mask).sum() / denominator


def objective(correct, shuffled, baseline, target, last, correction, lpips_model, a):
    motion = (target - last[:, None]).abs().mean(2, keepdim=True)
    dynamic = (motion > a.dynamic_threshold).float()
    error = correct - target
    reconstruction = (error.square() * (1 + a.dynamic_weight * dynamic)).mean()
    flat_correct = correct.flatten(0, 1) * 2 - 1
    flat_target = target.flatten(0, 1) * 2 - 1
    perceptual = lpips_model(flat_correct, flat_target).mean()
    temporal = ((correct[:, 1:] - correct[:, :-1]) -
                (target[:, 1:] - target[:, :-1])).abs().mean()
    static = 1 - dynamic
    pixel_correction = correct - baseline
    static_preserve = (pixel_correction.abs() * static).sum() / (
        static.sum() * pixel_correction.shape[2]
    ).clamp_min(1)
    correct_dynamic = dynamic_error(correct, target, last, a.dynamic_threshold)
    shuffled_dynamic = dynamic_error(shuffled, target, last, a.dynamic_threshold)
    rank = F.relu(correct_dynamic - shuffled_dynamic + a.rank_margin)
    wrong_reject = (shuffled - baseline).abs().mean()
    loss = (
        reconstruction + a.lpips_weight * perceptual + a.temporal_weight * temporal
        + a.static_weight * static_preserve
        + a.correction_weight * correction.abs().mean()
        + a.rank_weight * rank + a.wrong_reject_weight * wrong_reject
    )
    pieces = {
        "reconstruction": reconstruction, "lpips": perceptual,
        "temporal": temporal, "static_preserve": static_preserve,
        "correction": correction.abs().mean(), "rank": rank,
        "wrong_reject": wrong_reject, "correct_dynamic_mse": correct_dynamic,
        "shuffled_dynamic_mse": shuffled_dynamic,
    }
    return loss, pieces


@torch.no_grad()
def validate(model, loader, backbone, device, a):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw in loader:
        batch = move(raw, device)
        _, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        baseline = core.decode_frozen(backbone, hidden, skip)
        correct_hidden = model(hidden, *omega_inputs(batch))
        correct = core.decode_frozen(backbone, correct_hidden, skip)
        shuffled_hidden = model(hidden, *omega_inputs(batch, shuffled=True))
        shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
        target = batch["future_rgb"]
        totals["MSE"] += (correct - target).square().mean().item()
        totals["MAE"] += (correct - target).abs().mean().item()
        totals["baseline_MSE"] += (baseline - target).square().mean().item()
        totals["shuffled_MSE"] += (shuffled - target).square().mean().item()
        totals["correct_vs_baseline_l1"] += (correct - baseline).abs().mean().item()
        totals["correct_vs_shuffled_l1"] += (correct - shuffled).abs().mean().item()
        totals["gate"] += model.last["gate"].mean().item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(model, loaders, backbone, lpips_model, device, a, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    history = []
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw in loaders["train"]:
            batch = move(raw, device)
            with torch.no_grad():
                _, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
                baseline = core.decode_frozen(backbone, hidden, skip)
            correct_hidden = model(hidden, *omega_inputs(batch))
            correction = model.last["correction"]
            correct = core.decode_frozen(backbone, correct_hidden, skip)
            shuffled_hidden = model(hidden, *omega_inputs(batch, shuffled=True))
            shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
            loss, pieces = objective(
                correct, shuffled, baseline, batch["future_rgb"],
                batch["context_rgb"][:, -1], correction, lpips_model, a,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        validation = validate(model, loaders["val"], backbone, device, a)
        selection = validation["MSE"] + 2 * max(
            validation["MSE"] - validation["shuffled_MSE"], 0
        )
        row = {
            "epoch": epoch, "epoch_seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0], "selection": selection,
            **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch,
                 "validation": validation, "config": vars(a)}
        torch.save(state, output / "last.pt")
        if selection < best:
            best = selection
            torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "omega_latent_v2", **row}), flush=True)
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    return history, int(selected["epoch"])


@torch.no_grad()
def evaluate(model, loader, backbone, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator

    variants = (
        "tau_baseline", "omega_full", "omega_shuffled",
        "omega_no_tokens", "omega_no_pose", "omega_no_depth",
    )
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in variants}
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    model.eval()
    for raw in loader:
        batch = move(raw, device)
        _, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        predictions = {"tau_baseline": core.decode_frozen(backbone, hidden, skip)}
        configurations = {
            "omega_full": (False, True, True, True),
            "omega_shuffled": (True, True, True, True),
            "omega_no_tokens": (False, False, True, True),
            "omega_no_pose": (False, True, False, True),
            "omega_no_depth": (False, True, True, False),
        }
        for name, (shuffled, use_tokens, use_pose, use_depth) in configurations.items():
            corrected = model(
                hidden, *omega_inputs(batch, shuffled),
                use_tokens=use_tokens, use_pose=use_pose, use_depth=use_depth,
            )
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
        utilization["full_vs_baseline_l1"].append(
            (predictions["omega_full"] - predictions["tau_baseline"]).abs().mean().item())
        utilization["full_vs_shuffled_l1"].append(
            (predictions["omega_full"] - predictions["omega_shuffled"]).abs().mean().item())
        utilization["gate"].append(model.last["gate"].mean().item())
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > 0.05
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, last)
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[name]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[name]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[name]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, sequence_name in enumerate(batch["name"]):
                records.append({
                    "name": sequence_name, "variant": name,
                    **core.sequence_metrics(
                        prediction[index:index + 1], target[index:index + 1],
                        last[index:index + 1],
                    ),
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
        reference: core.paired_bootstrap(records, reference, "omega_full", seed)
        for reference in ("tau_baseline", "omega_shuffled")
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(values)) for key, values in utilization.items()
    }


def main():
    a = arguments()
    seed_all(a.seed)
    root = Path(a.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    import lpips

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = make_loaders(a)

    checkpoint = torch.load(a.backbone_checkpoint, map_location="cpu", weights_only=False)
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("VGGT-Omega latent v2 requires a TAU checkpoint")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])
    channels = int(backbone.backbone_config["hid_s"])
    model = OmegaLatentAdapterV2(
        channels, future_frames, a.token_hidden, a.context_hidden, a.feature_scale
    ).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    history, best_epoch = train(model, loaders, backbone, lpips_model, device, a, output)
    metrics, records, bootstrap, utilization = evaluate(
        model, loaders["test"], backbone, device, future_frames, a.seed
    )
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(a), "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "best_epoch": best_epoch,
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "frozen_backbone_params": sum(p.numel() for p in backbone.parameters()),
            "architecture": (
                "VGGT-Omega framewise camera/register cross-attention + relative-pose "
                "motion + confidence-gated temporal-depth injection at TAU future latent"
            ),
            "leakage_policy": "VGGT-Omega sees observed context frames only",
            "negative_control": "cross-video shuffled Omega features trained for rejection",
            "identity_control": "skipping the adapter exactly recovers frozen TAU",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
