#!/usr/bin/env python3
"""Directional-transport tests for D4RT-guided frozen-TAU prediction.

Four equal-capacity adapters share exactly the same geometry-derived spatial
support, deterministic gate budget, and feature-correction L1 budget:

* correct: correct D4RT UV flow, XYZ/UV geometry, and feature warp;
* mask_only: only the correct spatial support (no flow/RGB source residual);
* shuffled_flow: wrong cross-video motion at the correct spatial support;
* local_rgb: local RGB/TAU feature residual at the correct support, no geometry.

The correct/wrong flow is now an explicit, non-bypassable feature transport
path.  A learned synthesis residual is restricted to low-confidence regions.
All four models receive the same correct endpoint supervision, while only the
geometry models receive per-sample displacement as an input.  Direction and
magnitude counterfactuals test whether the learned predictor genuinely depends
on the UV vector rather than only its spatial support.

PointOdyssey dynamic evaluation uses a camera-compensated teacher-trajectory
mask.  A robust global affine motion is removed before residual-moving points
are rasterized and dilated.  The legacy RGB frame-difference metrics are also
kept under explicit ``rgb_motion_*`` names.
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
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_simvp_feature_warp as core
import train_tau_uv_warp_causal_v1 as v1


MODES = ("correct", "mask_only", "shuffled_flow", "local_rgb")
FLOW_COUNTERFACTUAL_MODES = ("reverse_flow", "half_flow", "onehalf_flow")
ALL_MODES = (*MODES, *FLOW_COUNTERFACTUAL_MODES)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", default="")
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=16)
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
    p.add_argument("--min-point-motion", type=float, default=0.015)
    p.add_argument("--full-point-motion", type=float, default=0.060)
    p.add_argument("--edge-margin", type=float, default=0.025)
    p.add_argument("--gate-open", type=float, default=0.80)
    p.add_argument("--feature-budget", type=float, default=0.006)
    p.add_argument("--pixel-budget", type=float, default=0.0010)
    p.add_argument("--pixel-budget-weight", type=float, default=0.50)
    p.add_argument("--transport-fraction", type=float, default=0.75)
    p.add_argument("--endpoint-weight", type=float, default=0.05)
    p.add_argument("--endpoint-rank-weight", type=float, default=0.05)
    p.add_argument("--endpoint-margin", type=float, default=0.02)
    p.add_argument("--object-motion-threshold-px", type=float, default=2.0)
    p.add_argument("--object-mask-radius", type=int, default=3)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DirectionalTransportTAUAdapter(nn.Module):
    """Make UV displacement an explicit transport operator, not a soft hint."""

    def __init__(
        self,
        channels,
        future_frames,
        flow_kernel=17,
        min_motion=0.015,
        full_motion=0.060,
        edge_margin=0.025,
        gate_open=0.80,
        feature_budget=0.006,
        transport_fraction=0.75,
        mode="correct",
    ):
        super().__init__()
        from video_prediction_model import SpatialGeometryAdapter

        if mode not in ALL_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.raster = core.SparseDenseFlow(flow_kernel)
        self.geometry = SpatialGeometryAdapter(channels)
        self.min_motion = float(min_motion)
        self.motion_span = max(float(full_motion) - float(min_motion), 1e-4)
        self.edge_margin = max(float(edge_margin), 1e-4)
        self.gate_open = float(gate_open)
        self.feature_budget = float(feature_budget)
        self.transport_fraction = float(transport_fraction)
        if not 0.0 <= self.transport_fraction <= 1.0:
            raise ValueError("transport_fraction must be in [0, 1]")
        self.horizon = nn.Parameter(torch.zeros(1, future_frames, channels, 1, 1))

        # hidden, source/warped residual, geometry, horizon, flow, support
        input_channels = channels * 4 + 3
        self.proposal = nn.Sequential(
            nn.Conv3d(input_channels, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(96, channels, 3, padding=1),
        )
        nn.init.normal_(self.horizon, std=0.01)
        nn.init.normal_(self.proposal[-1].weight, std=0.01)
        nn.init.zeros_(self.proposal[-1].bias)
        self.last = {}

    @staticmethod
    def _normalize_budget(delta, budget):
        """Give a branch a fixed feature-L1 budget without backprop through scale."""
        norm = delta.abs().mean((1, 2, 3, 4), keepdim=True)
        factor = float(budget) / norm.detach().clamp_min(1e-6)
        active = (norm > 1e-6).to(delta.dtype)
        return delta * factor * active

    def _permission(self, future_uv, reliability, anchor_uv):
        displacement = (future_uv - anchor_uv[:, :, None]).norm(dim=-1)
        motion = ((displacement - self.min_motion) / self.motion_span).clamp(0, 1)
        edge_distance = torch.minimum(
            torch.minimum(future_uv[..., 0], 1 - future_uv[..., 0]),
            torch.minimum(future_uv[..., 1], 1 - future_uv[..., 1]),
        )
        visible = (edge_distance / self.edge_margin).clamp(0, 1)
        return reliability * motion * visible

    def _raster_values(self, values, positions, weight, height, width):
        numerator, denominator = self.raster._splat(
            values, positions, weight, height, width
        )
        padding = self.raster.kernel.shape[-1] // 2
        channels = numerator.shape[1]
        numerator = F.conv2d(
            numerator,
            self.raster.kernel.expand(channels, 1, -1, -1),
            padding=padding,
            groups=channels,
        )
        denominator = F.conv2d(denominator, self.raster.kernel, padding=padding)
        flow = numerator / denominator.clamp_min(1e-5)
        support = (1 - torch.exp(-4 * denominator)).clamp(0, 1)
        b, _, k, _ = values.shape
        return (
            flow.reshape(b, k, channels, height, width),
            support.reshape(b, k, 1, height, width),
        )

    def forward(self, source, hidden, correct_bundle, shuffled_bundle):
        raw, future_uv, reliability, anchor_uv = correct_bundle
        shuffled_raw, shuffled_uv, _, shuffled_anchor = shuffled_bundle
        batch, steps, channels, height, width = hidden.shape
        horizon = self.horizon.expand(batch, -1, -1, height, width)

        permission = self._permission(future_uv, reliability, anchor_uv)
        correct_flow, dense_support = self.raster(
            anchor_uv, future_uv, permission, height, width
        )
        correct_geometry, sparse_support = self.geometry(
            raw, future_uv, permission, height, width, return_support=True
        )
        # This support is the single immutable spatial permission for all modes.
        support = torch.maximum(dense_support, sparse_support).clamp(0, 1).detach()
        zeros_flow = hidden.new_zeros((batch, steps, 2, height, width))
        zeros_geometry = hidden.new_zeros(hidden.shape)
        zeros_residual = hidden.new_zeros(hidden.shape)

        if self.mode == "correct":
            flow = correct_flow
            geometry_map = correct_geometry
            transported = core.backward_warp(source, flow)
        elif self.mode == "shuffled_flow":
            # Keep correct locations/support but transplant wrong displacement.
            wrong_displacement = shuffled_uv - shuffled_anchor[:, :, None]
            flow, _ = self._raster_values(
                wrong_displacement, future_uv, permission, height, width
            )
            geometry_map = self.geometry(
                shuffled_raw, future_uv, permission, height, width
            )
            transported = core.backward_warp(source, flow)
        elif self.mode in FLOW_COUNTERFACTUAL_MODES:
            scale = {
                "reverse_flow": -1.0,
                "half_flow": 0.5,
                "onehalf_flow": 1.5,
            }[self.mode]
            flow = correct_flow * scale
            geometry_map = correct_geometry
            transported = core.backward_warp(source, flow)
        elif self.mode == "mask_only":
            flow = zeros_flow
            geometry_map = zeros_geometry
            transported = hidden
        elif self.mode == "local_rgb":
            flow = zeros_flow
            geometry_map = zeros_geometry
            transported = source[:, None].expand(-1, steps, -1, -1, -1)
        else:  # pragma: no cover
            raise ValueError(self.mode)

        warped_residual = transported - hidden

        # Dense confidence is computed at the immutable correct future support.
        # It controls only propagate-vs-synthesize allocation and never changes
        # the supplied direction or magnitude.
        dense_confidence, _ = self._raster_values(
            reliability[..., None], future_uv, permission, height, width
        )
        visibility = (support * dense_confidence.clamp(0, 1)).detach()
        transport_gate = visibility * self.gate_open
        synthesis_gate = support * (1 - visibility) * self.gate_open

        features = torch.cat(
            (hidden, warped_residual, geometry_map, horizon, flow, support), dim=2
        ).permute(0, 2, 1, 3, 4)
        proposal = self.proposal(features).permute(0, 2, 1, 3, 4)

        # The propagation branch receives most of the budget and cannot be
        # cancelled by the synthesis branch, whose norm is independently capped.
        transport_budget = self.feature_budget * self.transport_fraction
        synthesis_budget = self.feature_budget * (1 - self.transport_fraction)
        transport_delta = self._normalize_budget(
            warped_residual * transport_gate, transport_budget
        )
        synthesis_delta = self._normalize_budget(
            torch.tanh(proposal) * synthesis_gate, synthesis_budget
        )
        correction = transport_delta + synthesis_delta
        self.last = {
            "flow": flow.detach(),
            "support": support.detach(),
            "gate": (transport_gate + synthesis_gate).detach(),
            "visibility": visibility.detach(),
            "permission": permission.detach(),
            "feature_delta": correction,
            "feature_l1": correction.detach().abs().mean(),
            "transport_l1": transport_delta.detach().abs().mean(),
            "synthesis_l1": synthesis_delta.detach().abs().mean(),
        }
        return hidden + correction, torch.zeros_like(correction), correction


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate

    split_args = SimpleNamespace(**vars(args))
    split_args.seed = args.split_seed
    splits, metadata = core.select_paths(split_args)
    datasets = {
        split: VideoForecastDataset(
            args.cache_root,
            split,
            args.image_size,
            paths=paths,
            cache_items=False,
        )
        for split, paths in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], args.batch_size, shuffle=True, generator=generator,
            collate_fn=video_pad_collate, num_workers=args.num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"], args.batch_size, collate_fn=video_pad_collate,
            num_workers=args.num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], args.batch_size, collate_fn=video_pad_collate,
            num_workers=args.num_workers, pin_memory=True,
        ),
    }
    metadata["split_seed"] = args.split_seed
    return loaders, datasets, metadata, splits


def geometry_bundles(joint, batch, future_frames):
    return (
        core.geometry_bundle("correct", joint, batch, future_frames),
        core.geometry_bundle("shuffled", joint, batch, future_frames),
    )


def match_pixel_budget(prediction, baseline, target_budget):
    """Normalize each sample's decoded correction to the same pixel-L1 budget."""
    delta = prediction - baseline
    norm = delta.abs().mean((1, 2, 3, 4), keepdim=True)
    factor = float(target_budget) / norm.detach().clamp_min(1e-7)
    return (baseline + delta * factor).clamp(0, 1)


def _sample_source_points(feature, uv):
    """Bilinearly sample ``[B,C,H,W]`` features at normalized ``[B,N,2]`` UV."""
    safe_uv = torch.nan_to_num(uv, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
    grid = safe_uv.mul(2).sub(1).unsqueeze(2)
    sampled = F.grid_sample(
        feature, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def _sample_future_points(feature, uv):
    """Sample ``[B,K,C,H,W]`` features at normalized ``[B,N,K,2]`` UV."""
    batch, steps, channels, height, width = feature.shape
    safe_uv = torch.nan_to_num(uv, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
    points = safe_uv.permute(0, 2, 1, 3)
    grid = points.reshape(batch * steps, points.shape[2], 1, 2).mul(2).sub(1)
    sampled = F.grid_sample(
        feature.reshape(batch * steps, channels, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).reshape(
        batch, steps, points.shape[2], channels
    ).permute(0, 2, 1, 3)


def trajectory_endpoint_losses(source, corrected, correct_bundle, permission, margin):
    """Enforce correct future endpoints and rank direction/magnitude negatives."""
    _, future_uv, reliability, anchor_uv = correct_bundle
    source_points = _sample_source_points(source, anchor_uv)
    future_points = _sample_future_points(corrected, future_uv)
    source_points = source_points[:, :, None]
    positive = F.smooth_l1_loss(
        future_points, source_points.expand_as(future_points), reduction="none"
    ).mean(-1)

    displacement = future_uv - anchor_uv[:, :, None]
    negative_uvs = (
        anchor_uv[:, :, None] - displacement,
        anchor_uv[:, :, None] + 0.5 * displacement,
        anchor_uv[:, :, None] + 1.5 * displacement,
    )
    weight = (permission * reliability).detach().clamp_min(0)
    endpoint = (positive * weight).sum() / weight.sum().clamp_min(1e-6)

    ranking_total = positive.new_zeros(())
    ranking_weight = positive.new_zeros(())
    for negative_uv in negative_uvs:
        valid = (
            (negative_uv[..., 0] >= 0)
            & (negative_uv[..., 0] <= 1)
            & (negative_uv[..., 1] >= 0)
            & (negative_uv[..., 1] <= 1)
            & torch.isfinite(negative_uv).all(-1)
        )
        negative_points = _sample_future_points(corrected, negative_uv)
        negative = F.smooth_l1_loss(
            negative_points,
            source_points.expand_as(negative_points),
            reduction="none",
        ).mean(-1)
        current_weight = weight * valid.detach().to(weight.dtype)
        ranking_total = ranking_total + (
            F.relu(float(margin) + positive - negative) * current_weight
        ).sum()
        ranking_weight = ranking_weight + current_weight.sum()
    ranking = ranking_total / ranking_weight.clamp_min(1e-6)
    return endpoint, ranking


@torch.no_grad()
def validate(models, loader, backbone, joint, device, future_frames, pixel_budget):
    totals = {name: defaultdict(float) for name in models}
    batches_seen = 0
    for raw_batch in loader:
        batch = core.move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        baseline = core.decode_frozen(backbone, hidden, skip)
        pair = geometry_bundles(joint, batch, future_frames)
        for name, model in models.items():
            corrected, _, _ = model(source, hidden, *pair)
            prediction = match_pixel_budget(
                core.decode_frozen(backbone, corrected, skip),
                baseline,
                pixel_budget,
            )
            error = prediction - batch["future_rgb"]
            totals[name]["MSE"] += error.square().mean().item()
            totals[name]["MAE"] += error.abs().mean().item()
            totals[name]["support"] += model.last["support"].mean().item()
            totals[name]["gate"] += model.last["gate"].mean().item()
            totals[name]["feature_delta"] += model.last["feature_l1"].item()
            totals[name]["transport_delta"] += model.last["transport_l1"].item()
            totals[name]["synthesis_delta"] += model.last["synthesis_l1"].item()
            totals[name]["visibility"] += model.last["visibility"].mean().item()
            totals[name]["pixel_delta"] += (prediction - baseline).abs().mean().item()
        batches_seen += 1
    return {
        name: {key: value / max(batches_seen, 1) for key, value in values.items()}
        for name, values in totals.items()
    }


def train(
    models, train_loader, val_loader, backbone, joint, lpips_model,
    device, args, output, future_frames,
):
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for name, model in models.items()
    }
    best = {name: float("inf") for name in models}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        totals = {name: defaultdict(float) for name in models}
        for model in models.values():
            model.train()
        batches_seen = 0
        for raw_batch in train_loader:
            batch = core.move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(
                    backbone, batch["context_rgb"]
                )
                baseline = core.decode_frozen(backbone, hidden, skip)
                pair = geometry_bundles(joint, batch, future_frames)
            for name, model in models.items():
                corrected, app_delta, geo_delta = model(source, hidden, *pair)
                prediction = match_pixel_budget(
                    core.decode_frozen(backbone, corrected, skip),
                    baseline,
                    args.pixel_budget,
                )
                loss, pieces = core.objective(
                    prediction,
                    batch["future_rgb"],
                    batch["context_rgb"][:, -1],
                    baseline,
                    app_delta,
                    geo_delta,
                    batch,
                    lpips_model,
                    args,
                )
                endpoint_loss, endpoint_rank_loss = trajectory_endpoint_losses(
                    source,
                    corrected,
                    pair[0],
                    model.last["permission"],
                    args.endpoint_margin,
                )
                loss = (
                    loss
                    + args.endpoint_weight * endpoint_loss
                    + args.endpoint_rank_weight * endpoint_rank_loss
                )
                pixel_delta = (prediction - baseline).abs().mean()
                budget_penalty = (pixel_delta - args.pixel_budget).abs()
                loss = loss + args.pixel_budget_weight * budget_penalty
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                totals[name]["budget_penalty"] += budget_penalty.item()
                totals[name]["feature_budget"] += model.last["feature_l1"].item()
                totals[name]["transport_budget"] += model.last["transport_l1"].item()
                totals[name]["synthesis_budget"] += model.last["synthesis_l1"].item()
                totals[name]["endpoint"] += endpoint_loss.item()
                totals[name]["endpoint_rank"] += endpoint_rank_loss.item()
                for key, value in pieces.items():
                    totals[name][key] += float(value.detach())
            batches_seen += 1

        validation = validate(
            models, val_loader, backbone, joint, device, future_frames,
            args.pixel_budget,
        )
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, model in models.items():
            for key, value in totals[name].items():
                row[f"{name}_train_{key}"] = value / max(batches_seen, 1)
            for key, value in validation[name].items():
                row[f"{name}_val_{key}"] = value
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "validation": validation[name],
                "config": vars(args),
                "mode": model.mode,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
            torch.save(state, output / f"{name}_last.pt")
            if validation[name]["MSE"] < best[name]:
                best[name] = validation[name]["MSE"]
                torch.save(state, output / f"{name}_best.pt")
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "tau_directional_transport_v1", **row}), flush=True)

    for name, model in models.items():
        state = torch.load(
            output / f"{name}_best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])


def camera_compensated_dynamic_mask(batch, height, width, threshold_px, radius):
    """Rasterize teacher-track residual motion after robust affine compensation."""
    raw_uv = batch["future_uv_teacher"].detach().float().cpu().numpy()
    valid = batch["future_valid"].detach().bool().cpu().numpy()
    anchor, anchor_valid = core.last_observed_uv(batch)
    anchor = anchor.detach().float().cpu().numpy()
    anchor_valid = anchor_valid.detach().bool().cpu().numpy()
    b, n, k, _ = raw_uv.shape
    # OpenCV drawing does not accept NumPy bool arrays; convert to bool on return.
    masks = np.zeros((b, k, height, width), dtype=np.uint8)
    stats = []
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], np.float32)

    for bi in range(b):
        sequence_valid = 0
        sequence_dynamic = 0
        residuals = []
        for step in range(k):
            ok = (
                valid[bi, :, step]
                & anchor_valid[bi]
                & np.isfinite(raw_uv[bi, :, step]).all(-1)
                & np.isfinite(anchor[bi]).all(-1)
            )
            src = anchor[bi, ok] * scale
            dst = raw_uv[bi, ok, step] * scale
            inside = (
                (src[:, 0] >= 0) & (src[:, 0] <= width - 1)
                & (src[:, 1] >= 0) & (src[:, 1] <= height - 1)
                & (dst[:, 0] >= 0) & (dst[:, 0] <= width - 1)
                & (dst[:, 1] >= 0) & (dst[:, 1] <= height - 1)
            )
            src, dst = src[inside], dst[inside]
            if len(src) < 6:
                continue
            matrix, _ = cv2.estimateAffine2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=max(float(threshold_px), 0.5),
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if matrix is None:
                translation = np.median(dst - src, axis=0)
                predicted = src + translation
            else:
                predicted = src @ matrix[:, :2].T + matrix[:, 2]
            residual = np.linalg.norm(dst - predicted, axis=1)
            moving = residual > float(threshold_px)
            sequence_valid += len(dst)
            sequence_dynamic += int(moving.sum())
            residuals.extend(residual.tolist())
            for point in dst[moving]:
                x = int(np.clip(round(float(point[0])), 0, width - 1))
                y = int(np.clip(round(float(point[1])), 0, height - 1))
                cv2.circle(masks[bi, step], (x, y), int(radius), 1, thickness=-1)
        stats.append(
            {
                "valid_points": int(sequence_valid),
                "dynamic_points": int(sequence_dynamic),
                "dynamic_point_fraction": float(sequence_dynamic / max(sequence_valid, 1)),
                "median_residual_px": float(np.median(residuals)) if residuals else float("nan"),
                "mask_fraction": float(masks[bi].mean()),
            }
        )
    return torch.from_numpy(masks.astype(np.bool_)).to(batch["future_rgb"].device), stats


def masked_sequence_metrics(prediction, target, mask):
    error = prediction - target
    selected = mask[:, :, None].expand_as(error)
    mse = error.square().mean().item()
    mae = error.abs().mean().item()
    dmse = error.square()[selected].mean().item() if selected.any() else float("nan")
    dmae = error.abs()[selected].mean().item() if selected.any() else float("nan")
    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": -10 * math.log10(max(mse, 1e-12)),
        "dynamic_MSE": dmse,
        "dynamic_MAE": dmae,
        "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)) if math.isfinite(dmse) else float("nan"),
    }


@torch.no_grad()
def evaluate(models, loader, backbone, joint, device, future_frames, args):
    from video_metrics import VideoMetricAccumulator

    same_weight_variants = (
        "correct_model_mask_only",
        "correct_model_shuffled_flow",
        "correct_model_local_rgb",
        "correct_model_reverse_flow",
        "correct_model_half_flow",
        "correct_model_onehalf_flow",
        "correct_model_oracle",
    )
    variants = ("frozen_tau", *MODES, *same_weight_variants)
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in variants
    }
    rgb_motion_counts = {name: defaultdict(float) for name in variants}
    object_correction_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = {name: defaultdict(list) for name in MODES}
    mask_records = []

    for model in models.values():
        model.eval()
    for raw_batch in loader:
        batch = core.move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        baseline = core.decode_frozen(backbone, hidden, skip)
        pair = geometry_bundles(joint, batch, future_frames)
        predictions = {"frozen_tau": baseline}
        for name, model in models.items():
            corrected, _, _ = model(source, hidden, *pair)
            predictions[name] = match_pixel_budget(
                core.decode_frozen(backbone, corrected, skip),
                baseline,
                args.pixel_budget,
            )
            utilization[name]["support"].append(model.last["support"].mean().item())
            utilization[name]["gate"].append(model.last["gate"].mean().item())
            utilization[name]["feature_delta"].append(model.last["feature_l1"].item())
            utilization[name]["transport_delta"].append(
                model.last["transport_l1"].item()
            )
            utilization[name]["synthesis_delta"].append(
                model.last["synthesis_l1"].item()
            )
            utilization[name]["visibility"].append(
                model.last["visibility"].mean().item()
            )
            utilization[name]["pixel_delta"].append(
                (predictions[name] - baseline).abs().mean().item()
            )

        correct_model = models["correct"]
        original_mode = correct_model.mode
        for mode, variant in (
            ("mask_only", "correct_model_mask_only"),
            ("shuffled_flow", "correct_model_shuffled_flow"),
            ("local_rgb", "correct_model_local_rgb"),
            ("reverse_flow", "correct_model_reverse_flow"),
            ("half_flow", "correct_model_half_flow"),
            ("onehalf_flow", "correct_model_onehalf_flow"),
        ):
            correct_model.mode = mode
            corrected, _, _ = correct_model(source, hidden, *pair)
            predictions[variant] = match_pixel_budget(
                core.decode_frozen(backbone, corrected, skip),
                baseline,
                args.pixel_budget,
            )
        correct_model.mode = "correct"
        oracle = v1.oracle_bundle(joint, batch, future_frames)
        corrected, _, _ = correct_model(source, hidden, oracle, pair[1])
        predictions["correct_model_oracle"] = match_pixel_budget(
            core.decode_frozen(backbone, corrected, skip),
            baseline,
            args.pixel_budget,
        )
        correct_model.mode = original_mode

        object_mask, batch_mask_stats = camera_compensated_dynamic_mask(
            batch,
            batch["future_rgb"].shape[-2],
            batch["future_rgb"].shape[-1],
            args.object_motion_threshold_px,
            args.object_mask_radius,
        )
        for index, stat in enumerate(batch_mask_stats):
            mask_records.append({"name": batch["name"][index], **stat})

        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        rgb_target_motion = (target - last[:, None]).abs().mean(2) > args.dynamic_threshold
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            rgb_predicted_motion = (
                (prediction - last[:, None]).abs().mean(2) > args.dynamic_threshold
            )
            rgb_motion_counts[variant]["tp"] += (
                rgb_predicted_motion & rgb_target_motion
            ).sum().item()
            rgb_motion_counts[variant]["fp"] += (
                rgb_predicted_motion & ~rgb_target_motion
            ).sum().item()
            rgb_motion_counts[variant]["fn"] += (
                ~rgb_predicted_motion & rgb_target_motion
            ).sum().item()

            correction_mask = (
                (prediction - baseline).abs().mean(2) > args.pixel_budget
            )
            object_correction_counts[variant]["tp"] += (
                correction_mask & object_mask
            ).sum().item()
            object_correction_counts[variant]["fp"] += (
                correction_mask & ~object_mask
            ).sum().item()
            object_correction_counts[variant]["fn"] += (
                ~correction_mask & object_mask
            ).sum().item()

            for index, sequence_name in enumerate(batch["name"]):
                records.append(
                    {
                        "name": sequence_name,
                        "variant": variant,
                        **masked_sequence_metrics(
                            prediction[index:index + 1],
                            target[index:index + 1],
                            object_mask[index:index + 1],
                        ),
                    }
                )

    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            values = np.asarray([row[key] for row in rows], np.float64)
            finite = values[np.isfinite(values)]
            metrics[name][key] = float(finite.mean()) if len(finite) else float("nan")
        metrics[name]["dynamic_finite_sequences"] = int(
            sum(math.isfinite(row["dynamic_MSE"]) for row in rows)
        )
        tp, fp, fn = (rgb_motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["rgb_motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["rgb_motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        tp, fp, fn = (
            object_correction_counts[name][key] for key in ("tp", "fp", "fn")
        )
        metrics[name]["object_correction_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["object_correction_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)

    vs_tau = {
        candidate: core.paired_bootstrap(records, "frozen_tau", candidate, args.seed)
        for candidate in variants if candidate != "frozen_tau"
    }
    fair_controls = {
        control: core.paired_bootstrap(records, control, "correct", args.seed)
        for control in ("mask_only", "shuffled_flow", "local_rgb")
    }
    same_weight = {
        control: core.paired_bootstrap(records, control, "correct", args.seed)
        for control in (
            "correct_model_mask_only",
            "correct_model_shuffled_flow",
            "correct_model_local_rgb",
            "correct_model_reverse_flow",
            "correct_model_half_flow",
            "correct_model_onehalf_flow",
        )
    }
    oracle = core.paired_bootstrap(
        records, "correct", "correct_model_oracle", args.seed
    )
    utilization_summary = {
        name: {key: float(np.mean(values)) for key, values in values_by_key.items()}
        for name, values_by_key in utilization.items()
    }
    mask_summary = {
        "num_sequences": len(mask_records),
        "finite_dynamic_sequences": int(sum(x["mask_fraction"] > 0 for x in mask_records)),
        "mean_mask_fraction": float(np.mean([x["mask_fraction"] for x in mask_records])),
        "mean_dynamic_point_fraction": float(np.mean([x["dynamic_point_fraction"] for x in mask_records])),
        "median_camera_residual_px": float(np.nanmedian([x["median_residual_px"] for x in mask_records])),
    }
    return (
        metrics, records, vs_tau, fair_controls, same_weight, oracle,
        utilization_summary, mask_records, mask_summary,
    )


def main():
    args = arguments()
    seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import (
        ImplicitConditionedJointForecaster,
        VelocityResidualImplicitForecaster,
    )
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata, splits = make_loaders(args)
    (output / "split_paths.json").write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "paths": {split: [str(path) for path in paths] for split, paths in splits.items()},
            },
            indent=2,
        )
    )

    checkpoint = torch.load(
        args.backbone_checkpoint, map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("This experiment requires a TAU checkpoint")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError(
            f"checkpoint image size {checkpoint['image_size']} != {args.image_size}"
        )
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])

    trajectory_checkpoint = torch.load(
        args.trajectory_checkpoint, map_location="cpu", weights_only=False
    )
    trajectory_class = (
        VelocityResidualImplicitForecaster
        if trajectory_checkpoint.get("model_variant") == "velocity_residual"
        else ImplicitConditionedJointForecaster
    )
    joint = trajectory_class(
        trajectory_checkpoint["future_frames"],
        trajectory_checkpoint["hidden_dim"],
    ).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"])
    joint.eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)

    channels = int(backbone.backbone_config["hid_s"])
    prototype = DirectionalTransportTAUAdapter(
        channels,
        future_frames,
        args.flow_kernel,
        args.min_point_motion,
        args.full_point_motion,
        args.edge_margin,
        args.gate_open,
        args.feature_budget,
        args.transport_fraction,
        mode="correct",
    ).to(device)
    models = {name: copy.deepcopy(prototype) for name in MODES}
    for name, model in models.items():
        model.mode = name

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    if args.eval_only:
        for name, model in models.items():
            state = torch.load(
                output / f"{name}_best.pt", map_location=device, weights_only=False
            )
            model.load_state_dict(state["model"])
    else:
        train(
            models,
            loaders["train"],
            loaders["val"],
            backbone,
            joint,
            lpips_model,
            device,
            args,
            output,
            future_frames,
        )

    (
        metrics,
        records,
        vs_tau,
        fair_controls,
        same_weight,
        oracle,
        utilization,
        mask_records,
        mask_summary,
    ) = evaluate(
        models, loaders["test"], backbone, joint, device, future_frames, args
    )
    summary = {
        "metrics": metrics,
        "paired_bootstrap_vs_frozen_tau": vs_tau,
        "correct_vs_budget_matched_controls": fair_controls,
        "correct_same_weight_counterfactuals": same_weight,
        "oracle_headroom": oracle,
        "budget_utilization": utilization,
        "camera_compensated_dynamic_mask": mask_summary,
        "meta": {
            "config": vars(args),
            "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "trainable_params_each": sum(p.numel() for p in prototype.parameters()),
            "frozen_backbone_params": sum(p.numel() for p in backbone.parameters()),
            "frozen_trajectory_params": sum(p.numel() for p in joint.parameters()),
            "architecture": (
                "non-bypassable D4RT feature transport with independently budgeted "
                "low-confidence synthesis; shared endpoint/ranking supervision; correct "
                "flow vs mask-only, wrong flow, local RGB, direction and magnitude controls"
            ),
            "dynamic_evaluation": (
                "teacher UV residual after per-frame robust global affine camera compensation; "
                "dynamic points rasterized at future UV and dilated"
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (output / "dynamic_mask_per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in mask_records)
    )
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
