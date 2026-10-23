#!/usr/bin/env python3
"""Spatially align VGGT-Omega patch tokens with TAU forecasting features.

Compared with semantic v3, this adapter removes pose addition from the default
path and replaces global-only FiLM with per-location temporal cross-attention.
The adapter remains bounded and zero initialized, and TAU stays frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_simvp_feature_warp as core
import train_vggt_omega_latent_v2 as v2
import train_vggt_omega_semantic_v3 as v3


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--omega-cache", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini"
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--alignment-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alignment-lr", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--semantic-dim", type=int, default=192)
    parser.add_argument("--feature-scale", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--alignment-weight", type=float, default=0.02)
    parser.add_argument("--dynamic-weight", type=float, default=2.0)
    parser.add_argument("--lpips-weight", type=float, default=0.03)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--static-weight", type=float, default=0.10)
    parser.add_argument("--correction-weight", type=float, default=0.01)
    parser.add_argument("--rank-weight", type=float, default=0.50)
    parser.add_argument("--wrong-reject-weight", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=2e-4)
    parser.add_argument("--dynamic-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


class OmegaSpatialDataset(Dataset):
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
        required = (
            "tokens", "patch_tokens", "depth", "depth_confidence", "pose_enc"
        )
        missing = [key for key in required if key not in omega]
        if missing:
            raise KeyError(f"{path} lacks spatial-v4 fields: {missing}")
        item["omega_tokens"] = omega["tokens"].float()
        item["omega_patch_tokens"] = omega["patch_tokens"].float()
        item["omega_depth"] = omega["depth"].float()
        item["omega_confidence"] = omega["depth_confidence"].float()
        item["omega_pose"] = omega["pose_enc"].float()
        return item


def collate(items):
    from video_forecast_dataset import video_pad_collate

    output = video_pad_collate(items)
    for key in (
        "omega_tokens", "omega_patch_tokens", "omega_depth",
        "omega_confidence", "omega_pose",
    ):
        output[key] = torch.stack([item[key] for item in items])
    return output


def make_loaders(args):
    from types import SimpleNamespace
    from video_forecast_dataset import VideoForecastDataset
    from train_pointodyssey_dmvfn_injection import combined_splits

    splits, metadata = combined_splits(SimpleNamespace(**vars(args)))
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    datasets = {}
    for split in ("train", "val", "test"):
        paths = splits[split][: limits[split] or None]
        base = VideoForecastDataset(
            args.cache_root, split, args.image_size, paths=paths, cache_items=False
        )
        datasets[split] = OmegaSpatialDataset(base, args.omega_cache)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], args.batch_size, shuffle=True, generator=generator,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], args.batch_size, collate_fn=collate,
            num_workers=args.num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], args.batch_size, collate_fn=collate,
            num_workers=args.num_workers, pin_memory=True,
        ),
    }
    return loaders, datasets, metadata


def omega_inputs(batch, shuffled=False):
    keys = (
        "omega_tokens", "omega_depth", "omega_confidence",
        "omega_pose", "omega_patch_tokens",
    )
    values = [batch[key] for key in keys]
    if shuffled:
        if values[0].shape[0] > 1:
            values = [value.roll(1, 0) for value in values]
        else:
            values = [value.flip(1) for value in values]
    return values


class SpatialOmegaAdapterV4(nn.Module):
    """Local TAU-to-Omega temporal attention followed by bounded feature FiLM."""

    def __init__(self, channels, future_frames, semantic_dim=192, feature_scale=0.12):
        super().__init__()
        if semantic_dim % 4:
            raise ValueError("semantic_dim must be divisible by four")
        self.channels = int(channels)
        self.future_frames = int(future_frames)
        self.semantic_dim = int(semantic_dim)
        self.feature_scale = float(feature_scale)

        self.tau_spatial = nn.Sequential(
            nn.Conv2d(channels, semantic_dim, 1), nn.GroupNorm(8, semantic_dim),
            nn.GELU(),
        )
        self.patch_projection = nn.Sequential(
            nn.Conv2d(2048, semantic_dim, 1), nn.GroupNorm(8, semantic_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.horizon_projection = nn.Sequential(
            nn.Linear(6, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        self.context_time = nn.Parameter(torch.zeros(1, 32, semantic_dim, 1, 1))
        self.slot_type = nn.Parameter(torch.zeros(1, 1, 17, semantic_dim))
        self.local_attention = nn.MultiheadAttention(
            semantic_dim, 4, dropout=0.05, batch_first=True
        )
        self.global_attention = nn.MultiheadAttention(
            semantic_dim, 4, dropout=0.05, batch_first=True
        )
        self.local_norm = nn.LayerNorm(semantic_dim)
        self.global_norm = nn.LayerNorm(semantic_dim)
        self.match_head = nn.Sequential(
            nn.Linear(semantic_dim * 4, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, 1),
        )
        self.local_film = nn.Conv2d(semantic_dim, channels * 2, 1)
        self.global_film = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim), nn.GELU(),
            nn.Linear(semantic_dim, channels * 2),
        )
        self.channel_gate = nn.Sequential(nn.Linear(semantic_dim, channels), nn.Sigmoid())
        depth_hidden = max(16, channels // 2)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(5, depth_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(depth_hidden, channels, 3, padding=1),
        )
        self.depth_horizon = nn.Sequential(
            nn.Linear(semantic_dim, channels), nn.Tanh()
        )
        self.global_gate_logit = nn.Parameter(torch.tensor(-1.5))

        nn.init.normal_(self.context_time, std=0.01)
        nn.init.normal_(self.slot_type, std=0.01)
        for final in (self.local_film, self.global_film[-1], self.depth_encoder[-1]):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.last = {}

    def horizon(self, steps, device, dtype):
        time = torch.linspace(0, 1, steps, device=device, dtype=dtype)
        basis = torch.stack((
            time, time.square(), time.pow(3),
            torch.sin(math.pi * time), torch.cos(math.pi * time),
            torch.ones_like(time),
        ), -1)
        return self.horizon_projection(basis)

    def forward(
        self, hidden, tokens, depth, confidence, pose, patch_tokens,
        use_tokens=True, use_pose=False, use_depth=True,
        use_patch=True, use_global=True,
    ):
        # Pose is deliberately excluded from v4's default path: v3 no-pose won.
        del pose, use_pose
        batch, future_steps, channels, height, width = hidden.shape
        if channels != self.channels or future_steps != self.future_frames:
            raise ValueError(f"unexpected hidden shape {tuple(hidden.shape)}")

        horizon = self.horizon(future_steps, hidden.device, hidden.dtype)
        tau = self.tau_spatial(hidden.flatten(0, 1)).reshape(
            batch, future_steps, self.semantic_dim, height, width
        )
        tau = tau + horizon[None, :, :, None, None]
        query = tau.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, future_steps, self.semantic_dim
        )

        if use_tokens and use_patch:
            context_steps = patch_tokens.shape[1]
            patch = self.patch_projection(
                patch_tokens.flatten(0, 1).to(hidden.dtype)
            )
            patch = F.interpolate(
                patch, (height, width), mode="bilinear", align_corners=False
            ).reshape(batch, context_steps, self.semantic_dim, height, width)
            patch = patch + self.context_time[:, :context_steps].to(patch.dtype)
            memory = patch.permute(0, 3, 4, 1, 2).reshape(
                batch * height * width, context_steps, self.semantic_dim
            )
            local, local_attention = self.local_attention(
                query, memory, memory, need_weights=True
            )
            local = self.local_norm(query + local)
        else:
            local = self.local_norm(query)
            local_attention = None

        global_query = tau.mean((-2, -1))
        if use_tokens and use_global:
            global_memory = self.global_projection(tokens.to(hidden.dtype))
            global_memory = global_memory + self.slot_type[
                :, :, :global_memory.shape[2]
            ].to(global_memory.dtype)
            global_memory = global_memory.reshape(batch, -1, self.semantic_dim)
            global_context, global_attention = self.global_attention(
                global_query, global_memory, global_memory, need_weights=True
            )
            global_context = self.global_norm(global_query + global_context)
        else:
            global_memory = None
            global_context = self.global_norm(global_query)
            global_attention = None

        global_grid = global_context[:, None, None].expand(
            batch, height, width, future_steps, self.semantic_dim
        ).reshape(batch * height * width, future_steps, self.semantic_dim)
        aligned = self.local_norm(local + global_grid)
        match_features = torch.cat((
            query, aligned, (query - aligned).abs(), query * aligned
        ), -1)
        match_logits = self.match_head(match_features).squeeze(-1)
        match_logits = match_logits.reshape(
            batch, height, width, future_steps
        ).permute(0, 3, 1, 2)

        aligned_grid = aligned.reshape(
            batch, height, width, future_steps, self.semantic_dim
        ).permute(0, 3, 4, 1, 2)
        gamma, beta = self.local_film(aligned_grid.flatten(0, 1)).chunk(2, 1)
        local_delta = (
            hidden.flatten(0, 1) * torch.tanh(gamma) + beta
        ).reshape_as(hidden)
        if not (use_tokens and use_patch):
            local_delta = torch.zeros_like(local_delta)

        global_gamma, global_beta = self.global_film(global_context).chunk(2, -1)
        global_delta = (
            hidden * torch.tanh(global_gamma[..., None, None])
            + global_beta[..., None, None]
        )
        if not (use_tokens and use_global):
            global_delta = torch.zeros_like(global_delta)

        if use_depth:
            statistics, confidence_map = v2.OmegaLatentAdapterV2.depth_statistics(
                depth, confidence
            )
            statistics = F.interpolate(
                statistics, (height, width), mode="bilinear", align_corners=False
            )
            confidence_map = F.interpolate(
                confidence_map, (height, width), mode="bilinear", align_corners=False
            )
            depth_feature = self.depth_encoder(statistics)[:, None]
            depth_delta = depth_feature * self.depth_horizon(global_context)[
                ..., None, None
            ]
        else:
            confidence_map = hidden.new_ones((batch, 1, height, width))
            depth_delta = torch.zeros_like(hidden)

        match_gate = torch.sigmoid(match_logits)[:, :, None]
        channel_gate = self.channel_gate(global_context)[..., None, None]
        spatial_gate = 0.10 + 0.90 * confidence_map[:, None]
        gate = (
            torch.sigmoid(self.global_gate_logit)
            * match_gate * channel_gate * spatial_gate
        )
        correction = self.feature_scale * torch.tanh(
            local_delta + global_delta + depth_delta
        ) * gate

        tau_descriptor = F.normalize(global_query.mean(1), dim=-1)
        descriptors = []
        if global_memory is not None:
            descriptors.append(global_memory.mean(1))
        if use_tokens and use_patch:
            descriptors.append(patch.mean((1, 3, 4)))
        omega_descriptor = F.normalize(
            torch.stack(descriptors).mean(0) if descriptors
            else torch.zeros_like(tau_descriptor),
            dim=-1,
        )
        self.last = {
            "correction": correction,
            "gate": gate.detach(),
            "match_logits": match_logits,
            "match_probability": torch.sigmoid(match_logits).detach(),
            "tau_descriptor": tau_descriptor,
            "omega_descriptor": omega_descriptor,
            "local_attention": (
                local_attention.detach().mean() if local_attention is not None
                else hidden.new_zeros(())
            ),
            "global_attention": (
                global_attention.detach().mean() if global_attention is not None
                else hidden.new_zeros(())
            ),
        }
        return hidden + correction


@torch.no_grad()
def evaluate(model, loader, backbone, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator

    variants = (
        "tau_baseline", "omega_spatial_full", "omega_shuffled",
        "omega_no_patch", "omega_patch_only", "omega_no_depth",
    )
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in variants
    }
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    model.eval()
    for raw in loader:
        batch = v2.move(raw, device)
        _, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        predictions = {
            "tau_baseline": core.decode_frozen(backbone, hidden, skip)
        }
        configurations = {
            "omega_spatial_full": dict(),
            "omega_shuffled": dict(shuffled=True),
            "omega_no_patch": dict(use_patch=False),
            "omega_patch_only": dict(use_global=False, use_depth=False),
            "omega_no_depth": dict(use_depth=False),
        }
        for name, config in configurations.items():
            shuffled = config.pop("shuffled", False)
            corrected = model(
                hidden, *omega_inputs(batch, shuffled=shuffled), **config
            )
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
            if name == "omega_spatial_full":
                full_gate_value = model.last["gate"].mean().item()
        utilization["full_vs_baseline_l1"].append(
            (predictions["omega_spatial_full"] - predictions["tau_baseline"])
            .abs().mean().item()
        )
        utilization["full_vs_shuffled_l1"].append(
            (predictions["omega_spatial_full"] - predictions["omega_shuffled"])
            .abs().mean().item()
        )
        utilization["gate"].append(full_gate_value)
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
                    "name": sequence_name,
                    "variant": name,
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
        reference: core.paired_bootstrap(
            records, reference, "omega_spatial_full", seed
        )
        for reference in ("tau_baseline", "omega_shuffled", "omega_no_patch")
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(values)) for key, values in utilization.items()
    }


def main():
    args = arguments()
    v2.seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = make_loaders(args)
    checkpoint = torch.load(
        args.backbone_checkpoint, map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("spatial Omega v4 requires a TAU checkpoint")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    future_frames = int(checkpoint["future_frames"])
    channels = int(backbone.backbone_config["hid_s"])
    model = SpatialOmegaAdapterV4(
        channels, future_frames, args.semantic_dim, args.feature_scale
    ).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    # Reuse the proven v3 optimization loop with the spatial-v4 input contract.
    original_inputs = v2.omega_inputs
    v2.omega_inputs = omega_inputs
    try:
        alignment_history = v3.run_alignment_stage(
            model, loaders["train"], backbone, device, args, output
        )
        forecast_history, best_epoch = v3.run_forecast_stage(
            model, loaders, backbone, lpips_model, device, args, output
        )
    finally:
        v2.omega_inputs = original_inputs

    metrics, records, bootstrap, utilization = evaluate(
        model, loaders["test"], backbone, device, future_frames, args.seed
    )
    original_inputs = v2.omega_inputs
    v2.omega_inputs = omega_inputs
    try:
        semantic_test = v3.validate(model, loaders["test"], backbone, device, args)
    finally:
        v2.omega_inputs = original_inputs
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "semantic_alignment": semantic_test,
        "meta": {
            "config": vars(args),
            "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "best_epoch": best_epoch,
            "alignment_epochs": len(alignment_history),
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "architecture": (
                "TAU spatial feature queries over VGGT final-layer patch-token grids; "
                "per-location temporal cross-attention plus global token context; "
                "no pose in default path; confidence and mismatch-gated bounded feature FiLM"
            ),
            "leakage_policy": "VGGT-Omega sees observed context frames only",
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
