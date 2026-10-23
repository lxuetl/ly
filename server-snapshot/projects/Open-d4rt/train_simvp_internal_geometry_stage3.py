#!/usr/bin/env python3
"""Stage-3: inject D4RT geometry inside SimVP's temporal core.

The frozen v1 prediction remains the safe path.  A geometry-conditioned adapter
is inserted immediately before the last SimVP temporal block, and only that
last block plus the adapter/permission head are trainable.
"""

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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
    p.add_argument("--v1-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--backbone-lr", type=float, default=8e-6)
    p.add_argument("--max-train", type=int, default=216)
    p.add_argument("--max-val", type=int, default=80)
    p.add_argument("--max-test", type=int, default=80)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--track-weight", type=float, default=0.25)
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--feature-correction-weight", type=float, default=0.002)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--flow-kernel", type=int, default=9)
    p.add_argument("--feature-scale", type=float, default=0.15)
    p.add_argument("--internal-scale", type=float, default=0.10)
    p.add_argument("--teacher-weight", type=float, default=0.05)
    p.add_argument("--rank-weight", type=float, default=0.15)
    p.add_argument("--rank-margin", type=float, default=2e-4)
    p.add_argument("--wrong-preserve-weight", type=float, default=0.10)
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


def shuffled_bundle(bundle):
    raw, uv, reliability, anchor = bundle
    if raw.shape[0] > 1:
        order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
        return raw[order], uv[order], reliability[order], anchor[order]
    order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
    return raw[:, order], uv[:, order], reliability[:, order], anchor[:, order]


class InternalTemporalGeometry(nn.Module):
    """Geometry adapter before the last SimVP temporal block with v1 fallback."""

    def __init__(self, backbone, v1, future_frames, internal_scale=0.10):
        super().__init__()
        self.backbone = backbone
        self.v1 = v1
        self.future_frames = int(future_frames)
        self.internal_scale = float(internal_scale)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        temporal = self.backbone.backbone.hid.enc
        if len(temporal) < 2:
            raise RuntimeError("SimVP temporal core has no separable last block")
        last = temporal[-1]
        mid_channels = int(last.block.norm1.num_features)
        if mid_channels % self.future_frames:
            raise RuntimeError(
                f"Temporal width {mid_channels} is not divisible by K={self.future_frames}")
        self.mid_channels = mid_channels
        self.channels_per_horizon = mid_channels // self.future_frames
        latent_channels = int(self.backbone.backbone_config["hid_s"])
        geometry_channels = latent_channels * 2 + 3
        self.mid_adapter = nn.Sequential(
            nn.Conv3d(geometry_channels, 48, 3, padding=1), nn.GELU(),
            nn.Conv3d(48, self.channels_per_horizon, 3, padding=1),
        )
        self.permission_head = nn.Sequential(
            nn.Conv3d(geometry_channels, 24, 3, padding=1), nn.GELU(),
            nn.Conv3d(24, 1, 1),
        )
        nn.init.zeros_(self.mid_adapter[-1].weight)
        nn.init.zeros_(self.mid_adapter[-1].bias)
        nn.init.zeros_(self.permission_head[-1].weight)
        nn.init.constant_(self.permission_head[-1].bias, -1.5)
        self.output_scale_logit = nn.Parameter(torch.tensor(-0.5))
        for parameter in temporal[-1].parameters():
            parameter.requires_grad_(True)
        for module in (self.mid_adapter, self.permission_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.last = {}

    @property
    def last_temporal_block(self):
        return self.backbone.backbone.hid.enc[-1]

    def set_train_mode(self):
        self.eval()
        self.mid_adapter.train()
        self.permission_head.train()
        self.last_temporal_block.train()

    def encode_prefix(self, context):
        """Run the frozen encoder and all temporal blocks except the last."""
        b, t, c, h0, w0 = context.shape
        with torch.no_grad():
            embed, skip = self.backbone.backbone.enc(
                context.reshape(b * t, c, h0, w0))
            ch, h, w = embed.shape[1:]
            source = embed.reshape(b, t, ch, h, w)[:, -1]
            z = embed.reshape(b, t * ch, h, w)
            for block in self.backbone.backbone.hid.enc[:-1]:
                z = block(z)
            skip = self.backbone.skip_time(skip.reshape(b, t, *skip.shape[1:]))
        return source.detach(), z.detach(), skip.detach(), t, ch

    def geometry_injection(self, source, bundle):
        geometry, future_uv, reliability, anchor_uv = bundle
        _, _, h, w = source.shape
        flow, dense_support = self.v1.raster(
            anchor_uv, future_uv, reliability, h, w)
        geometry_map, sparse_support = self.v1.geometry(
            geometry, future_uv, reliability, h, w, return_support=True)
        support = torch.maximum(dense_support, sparse_support).clamp(0, 1)
        warped = core.backward_warp(source, flow)
        features = torch.cat((warped, geometry_map, flow, support), 2)
        features = features.permute(0, 2, 1, 3, 4)
        delta = torch.tanh(self.mid_adapter(features)).permute(0, 2, 1, 3, 4)
        permission = torch.sigmoid(self.permission_head(features)).permute(0, 2, 1, 3, 4)
        permission = permission * support
        delta = delta * permission
        b, k, q, h, w = delta.shape
        return delta.reshape(b, k * q, h, w), permission, support, flow

    def finish(self, source, prefix, skip, context_steps, latent_channels,
               bundle, teacher_corrected):
        delta_mid, permission, support, flow = self.geometry_injection(source, bundle)
        if delta_mid.shape[1] != prefix.shape[1]:
            raise RuntimeError(f"Adapter width {delta_mid.shape[1]} != temporal width {prefix.shape[1]}")
        z = self.last_temporal_block(prefix + self.internal_scale * delta_mid)
        b, _, h, w = z.shape
        hidden = z.reshape(b, context_steps, latent_channels, h, w)
        hidden = self.backbone.latent_time(hidden)
        candidate, _, _ = self.v1(source, hidden, *bundle)
        output_permission = permission * torch.sigmoid(self.output_scale_logit)
        corrected = teacher_corrected + output_permission * (candidate - teacher_corrected)
        correction = corrected - teacher_corrected
        self.last = {
            "permission": output_permission.detach(), "support": support.detach(),
            "flow": flow.detach(), "internal_delta": delta_mid,
            "candidate_delta": candidate - teacher_corrected,
            "correction": correction,
        }
        return corrected, correction


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate
    splits, metadata = core.select_paths(args)
    datasets = {
        split: VideoForecastDataset(args.cache_root, split, 64, paths=paths, cache_items=False)
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
def teacher_features(teacher_backbone, v1, context, bundle):
    source, hidden, skip = core.extract_frozen_features(teacher_backbone, context)
    corrected, _, _ = v1(source, hidden, *bundle)
    return source, corrected, skip


def forward_student(model, context, bundle, teacher_corrected):
    source, prefix, skip, steps, channels = model.encode_prefix(context)
    corrected, correction = model.finish(
        source, prefix, skip, steps, channels, bundle, teacher_corrected)
    prediction = core.decode_frozen(model.backbone, corrected, skip)
    return prediction, correction, (source, prefix, skip, steps, channels)


def forward_from_cache(model, cached, bundle, teacher_corrected):
    source, prefix, skip, steps, channels = cached
    corrected, correction = model.finish(
        source, prefix, skip, steps, channels, bundle, teacher_corrected)
    prediction = core.decode_frozen(model.backbone, corrected, skip)
    return prediction, correction


def selective_wrong_preservation(wrong, teacher, target):
    with torch.no_grad():
        teacher_error = (teacher - target).abs().mean(2, keepdim=True)
        wrong_error = (wrong - target).abs().mean(2, keepdim=True)
        harmful = (wrong_error > teacher_error + 0.001).float()
        harmful = F.max_pool2d(
            harmful.flatten(0, 1), 3, stride=1, padding=1).reshape_as(harmful)
    if harmful.sum() < 1:
        return wrong.sum() * 0, harmful.mean()
    loss = ((wrong - teacher).abs().mean(2, keepdim=True) * harmful).sum()
    return loss / harmful.sum().clamp_min(1), harmful.mean()


@torch.no_grad()
def validate(model, teacher_backbone, v1, loader, joint, device, future_frames):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw_batch in loader:
        batch = move(raw_batch, device)
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        _, teacher_hidden, _ = teacher_features(
            teacher_backbone, v1, batch["context_rgb"], bundle)
        teacher_prediction = core.decode_frozen(teacher_backbone, teacher_hidden,
                                                 core.extract_frozen_features(
                                                     teacher_backbone, batch["context_rgb"])[2])
        prediction, _, cached = forward_student(
            model, batch["context_rgb"], bundle, teacher_hidden)
        wrong, _ = forward_from_cache(
            model, cached, shuffled_bundle(bundle), teacher_hidden)
        target = batch["future_rgb"]
        for name, value in (
            ("MSE", (prediction - target).square().mean()),
            ("MAE", (prediction - target).abs().mean()),
            ("teacher_MSE", (teacher_prediction - target).square().mean()),
            ("teacher_MAE", (teacher_prediction - target).abs().mean()),
            ("wrong_MSE", (wrong - target).square().mean()),
            ("correct_effect", (prediction - teacher_prediction).abs().mean()),
            ("wrong_effect", (wrong - teacher_prediction).abs().mean()),
            ("permission", model.last["permission"].mean()),
        ):
            totals[name] += float(value)
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(model, teacher_backbone, v1, train_loader, val_loader, backbone_for_decode,
          joint, lpips_model, device, args, output, future_frames):
    adapter_params = list(model.mid_adapter.parameters()) + list(model.permission_head.parameters())
    adapter_params.append(model.output_scale_logit)
    temporal_params = list(model.last_temporal_block.parameters())
    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": args.lr},
        {"params": temporal_params, "lr": args.backbone_lr},
    ], weight_decay=1e-4)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.set_train_mode()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            bundle = core.geometry_bundle("correct", joint, batch, future_frames)
            with torch.no_grad():
                _, teacher_hidden, teacher_skip = teacher_features(
                    teacher_backbone, v1, batch["context_rgb"], bundle)
                teacher_prediction = core.decode_frozen(
                    teacher_backbone, teacher_hidden, teacher_skip)
            prediction, correction, cached = forward_student(
                model, batch["context_rgb"], bundle, teacher_hidden)
            loss, pieces = core.objective(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                teacher_prediction, torch.zeros_like(correction), correction,
                batch, lpips_model, args)
            wrong, _ = forward_from_cache(
                model, cached, shuffled_bundle(bundle), teacher_hidden)
            target = batch["future_rgb"]
            correct_error = (prediction - target).abs().mean((1, 2, 3, 4))
            wrong_error = (wrong - target).abs().mean((1, 2, 3, 4))
            rank = F.relu(correct_error - wrong_error + args.rank_margin).mean()
            wrong_preserve, harmful_fraction = selective_wrong_preservation(
                wrong, teacher_prediction, target)
            teacher_distill = (prediction - teacher_prediction).abs().mean()
            loss = (loss + args.teacher_weight * teacher_distill
                    + args.rank_weight * rank
                    + args.wrong_preserve_weight * wrong_preserve)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params + temporal_params, 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["teacher_distill"] += teacher_distill.item()
            totals["rank"] += rank.item()
            totals["wrong_preserve"] += wrong_preserve.item()
            totals["harmful_fraction"] += harmful_fraction.item()
            totals["permission"] += model.last["permission"].mean().item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            batches += 1
        metrics = validate(
            model, teacher_backbone, v1, val_loader, joint, device, future_frames)
        score = metrics["MSE"] + 0.1 * max(metrics["MAE"] - metrics["teacher_MAE"], 0)
        row = {
            "epoch": epoch, "epoch_seconds": time.time() - started, "score": score,
            **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch, "score": score,
                 "validation": metrics, "config": vars(args)}
        torch.save(state, output / "last.pt")
        torch.save(state, output / f"epoch_{epoch:02d}.pt")
        if score < best:
            best = score
            torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "internal_geometry_stage3", **row}), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device)["model"])
    return history


@torch.no_grad()
def evaluate(model, teacher_backbone, v1, loader, joint, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator
    variants = ("frozen_simvp", "v1_teacher", "stage3_correct", "stage3_wrong_internal")
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in variants}
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    model.eval()
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, base_hidden, base_skip = core.extract_frozen_features(
            teacher_backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        teacher_hidden, _, _ = v1(source, base_hidden, *bundle)
        predictions = {
            "frozen_simvp": core.decode_frozen(teacher_backbone, base_hidden, base_skip),
            "v1_teacher": core.decode_frozen(teacher_backbone, teacher_hidden, base_skip),
        }
        correct, _, cached = forward_student(
            model, batch["context_rgb"], bundle, teacher_hidden)
        wrong, _ = forward_from_cache(
            model, cached, shuffled_bundle(bundle), teacher_hidden)
        predictions["stage3_correct"] = correct
        predictions["stage3_wrong_internal"] = wrong
        utilization["correct_vs_v1_l1"].append(
            (correct - predictions["v1_teacher"]).abs().mean().item())
        utilization["correct_vs_wrong_l1"].append((correct - wrong).abs().mean().item())
        utilization["permission"].append(model.last["permission"].mean().item())
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
        reference: core.paired_bootstrap(records, reference, "stage3_correct", seed)
        for reference in ("frozen_simvp", "v1_teacher", "stage3_wrong_internal")
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(value)) for key, value in utilization.items()}


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

    backbone_checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    teacher_backbone = build_video_predictor_from_checkpoint(backbone_checkpoint).to(device)
    teacher_backbone.load_state_dict(backbone_checkpoint["model"])
    teacher_backbone.eval()
    for parameter in teacher_backbone.parameters():
        parameter.requires_grad_(False)
    student_backbone = copy.deepcopy(teacher_backbone)
    future_frames = int(backbone_checkpoint["future_frames"])

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

    v1 = core.DualPathFeatureWarp(16, args.flow_kernel, args.feature_scale).to(device)
    v1.load_state_dict(torch.load(args.v1_checkpoint, map_location="cpu")["model"])
    v1.eval()
    for parameter in v1.parameters():
        parameter.requires_grad_(False)

    model = InternalTemporalGeometry(
        student_backbone, v1, future_frames, args.internal_scale).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    train(model, teacher_backbone, v1, loaders["train"], loaders["val"],
          student_backbone, joint, lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = evaluate(
        model, teacher_backbone, v1, loaders["test"], joint, device,
        future_frames, args.seed)
    summary = {
        "metrics": metrics, "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args), "split": split_metadata,
            "used": {key: len(value) for key, value in datasets.items()},
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            "architecture": "v1 safe path + pre-last-temporal geometry injection",
            "unfrozen": "last SimVP temporal block + internal adapter + permission head",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
