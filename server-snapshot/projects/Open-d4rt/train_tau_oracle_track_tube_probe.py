#!/usr/bin/env python3
"""Low-cost oracle trajectory-tube attention probe for frozen TAU.

Four equal-capacity adapters share exactly the same geometry-derived spatial
support, deterministic gate budget, and feature-correction L1 budget:

* correct: correct D4RT UV flow, XYZ/UV geometry, and feature warp;
* mask_only: only the correct spatial support (no flow/RGB source residual);
* shuffled_flow: wrong cross-video motion at the correct spatial support;
* local_rgb: local RGB/TAU feature residual at the correct support, no geometry.

The probe preserves point identity instead of rasterizing trajectories into a
dense flow.  Anchor appearance, per-horizon oracle XYZ/UV, displacement,
confidence, and time form track tokens.  Each future TAU latent query attends to
nearby reliable track tokens with an explicit spatial bias and a learned null
token, then adds a bounded residual before the frozen decoder.  Only the correct
model is trained; mask/shuffle/reverse/scale controls reuse exactly its weights.

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
    p.add_argument("--max-transport-gate", type=float, default=0.25)
    p.add_argument("--transport-gate-bias", type=float, default=-4.0)
    p.add_argument("--direction-rank-weight", type=float, default=1.0)
    p.add_argument("--direction-rank-margin", type=float, default=5e-6)
    p.add_argument("--adapter-init-dir", default="")
    p.add_argument("--max-tracks", type=int, default=128)
    p.add_argument("--max-tube-gate", type=float, default=0.50)
    p.add_argument("--tube-gate-bias", type=float, default=-2.0)
    p.add_argument("--attention-temperature", type=float, default=0.02)
    p.add_argument("--tube-alignment-weight", type=float, default=0.02)
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


class OracleTrajectoryTubeAdapter(nn.Module):
    """Identity-preserving sparse trajectory attention over future TAU latents."""

    def __init__(
        self,
        channels,
        future_frames,
        feature_budget=0.006,
        max_tracks=128,
        max_tube_gate=0.50,
        tube_gate_bias=-2.0,
        attention_temperature=0.02,
        mode="correct",
    ):
        super().__init__()
        if mode not in ALL_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.feature_budget = float(feature_budget)
        self.max_tracks = int(max_tracks)
        self.max_tube_gate = float(max_tube_gate)
        if not 0.0 <= self.max_tube_gate <= 1.0:
            raise ValueError("max_tube_gate must be in [0, 1]")
        self.horizon = nn.Parameter(torch.zeros(1, future_frames, channels, 1, 1))
        self.spatial_position = nn.Sequential(
            nn.Linear(2, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.track_token = nn.Sequential(
            nn.Linear(channels + 10, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.null_key = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.null_value = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.null_bias = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(max(float(attention_temperature), 1e-4)))
        )
        self.proposal = nn.Sequential(
            nn.Conv3d(channels * 2, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(96, channels, 3, padding=1),
        )
        self.tube_gate = nn.Sequential(
            nn.Conv3d(channels * 2 + 1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 1, 1),
        )
        nn.init.normal_(self.horizon, std=0.01)
        nn.init.normal_(self.proposal[-1].weight, std=0.01)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.zeros_(self.tube_gate[-1].weight)
        nn.init.constant_(self.tube_gate[-1].bias, float(tube_gate_bias))
        self.last = {}

    @staticmethod
    def _normalize_budget(delta, budget):
        norm = delta.abs().mean((1, 2, 3, 4), keepdim=True)
        factor = float(budget) / norm.detach().clamp_min(1e-6)
        active = (norm > 1e-6).to(delta.dtype)
        return delta * factor * active

    @staticmethod
    def _sample_points(feature, uv):
        grid = uv.mul(2).sub(1).unsqueeze(2)
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled[..., 0].transpose(1, 2)

    def _select_tracks(self, raw, future_uv, reliability, anchor_uv):
        count = min(self.max_tracks, raw.shape[1])
        score = reliability.amax(2)
        index = score.topk(count, dim=1).indices
        raw = raw.gather(1, index[:, :, None, None].expand(-1, -1, raw.shape[2], raw.shape[3]))
        future_uv = future_uv.gather(
            1, index[:, :, None, None].expand(-1, -1, future_uv.shape[2], 2)
        )
        reliability = reliability.gather(
            1, index[:, :, None].expand(-1, -1, reliability.shape[2])
        )
        anchor_uv = anchor_uv.gather(1, index[:, :, None].expand(-1, -1, 2))
        return raw, future_uv, reliability, anchor_uv

    def forward(self, source, hidden, correct_bundle, shuffled_bundle):
        raw, future_uv, reliability, anchor_uv = correct_bundle
        shuffled_raw, shuffled_uv, shuffled_reliability, shuffled_anchor = shuffled_bundle
        batch, steps, channels, height, width = hidden.shape
        if self.mode == "shuffled_flow":
            raw, future_uv, reliability, anchor_uv = (
                shuffled_raw, shuffled_uv, shuffled_reliability, shuffled_anchor
            )
        elif self.mode in FLOW_COUNTERFACTUAL_MODES:
            scale = {"reverse_flow": -1.0, "half_flow": 0.5, "onehalf_flow": 1.5}[self.mode]
            displacement = future_uv - anchor_uv[:, :, None]
            future_uv = (anchor_uv[:, :, None] + scale * displacement).clamp(0, 1)
            raw = raw.clone()
            raw[..., :3] = raw[..., :3] * scale
            raw[..., 3:5] = future_uv

        raw, future_uv, reliability, anchor_uv = self._select_tracks(
            raw, future_uv, reliability, anchor_uv
        )
        appearance = self._sample_points(source, anchor_uv)
        if self.mode == "mask_only":
            raw = torch.zeros_like(raw)
            appearance = torch.zeros_like(appearance)
        elif self.mode == "local_rgb":
            raw = torch.zeros_like(raw)
            future_uv = anchor_uv[:, :, None].expand(-1, -1, steps, -1)

        displacement = future_uv - anchor_uv[:, :, None]
        token_input = torch.cat(
            (
                appearance[:, :, None].expand(-1, -1, steps, -1),
                raw,
                displacement,
                reliability[..., None],
            ),
            -1,
        )
        tokens = self.track_token(token_input).permute(0, 2, 1, 3)
        token_position = self.spatial_position(future_uv).permute(0, 2, 1, 3)
        keys = self.key(tokens + token_position)
        values = self.value(tokens)

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=hidden.device, dtype=hidden.dtype),
            torch.linspace(0, 1, width, device=hidden.device, dtype=hidden.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((xx, yy), -1).reshape(1, 1, height * width, 2)
        hidden_flat = hidden.permute(0, 1, 3, 4, 2).reshape(
            batch, steps, height * width, channels
        )
        horizon = self.horizon.permute(0, 1, 3, 4, 2).reshape(1, steps, 1, channels)
        queries = self.query(
            hidden_flat + self.spatial_position(coordinates) + horizon
        )
        logits = torch.einsum("bkqc,bknc->bkqn", queries, keys) / math.sqrt(channels)
        uv_by_step = future_uv.permute(0, 2, 1, 3)
        distance = (coordinates[:, :, :, None] - uv_by_step[:, :, None]).square().sum(-1)
        temperature = self.log_temperature.exp().clamp(1e-4, 0.25)
        logits = logits - distance / temperature
        logits = logits + reliability.permute(0, 2, 1)[:, :, None].clamp_min(1e-6).log()
        null_logits = self.null_bias.expand(batch, steps, height * width, 1)
        weights = torch.softmax(torch.cat((logits, null_logits), -1), -1)
        attended = torch.einsum("bkqn,bknc->bkqc", weights[..., :-1], values)
        support = (1 - weights[..., -1]).reshape(batch, steps, 1, height, width)
        attended = attended.reshape(batch, steps, height, width, channels).permute(0, 1, 4, 2, 3)

        features = torch.cat((hidden, attended), 2).permute(0, 2, 1, 3, 4)
        proposal = self.proposal(features).permute(0, 2, 1, 3, 4)
        gate_features = torch.cat((hidden, attended, support), 2).permute(0, 2, 1, 3, 4)
        tube_gate = self.max_tube_gate * torch.sigmoid(
            self.tube_gate(gate_features).permute(0, 2, 1, 3, 4)
        ) * support
        correction = self._normalize_budget(torch.tanh(proposal) * tube_gate, self.feature_budget)
        corrected = hidden + correction

        corrected_flat = corrected.reshape(batch * steps, channels, height, width)
        uv_flat = future_uv.permute(0, 2, 1, 3).reshape(batch * steps, -1, 2)
        sampled_future = self._sample_points(corrected_flat, uv_flat).reshape(
            batch, steps, -1, channels
        ).permute(0, 2, 1, 3)
        alignment_error = F.smooth_l1_loss(
            sampled_future, appearance[:, :, None].expand_as(sampled_future), reduction="none"
        ).mean(-1)
        alignment_loss = (
            alignment_error * reliability
        ).sum() / reliability.sum().clamp_min(1.0)
        self.last = {
            "support": support.detach(),
            "gate": tube_gate.detach(),
            "feature_delta": correction,
            "feature_l1": correction.detach().abs().mean(),
            "residual_l1": correction.detach().abs().mean(),
            "transport_candidate_l1": attended.detach().abs().mean(),
            "transport_alpha": tube_gate.detach(),
            "transport_alpha_mean": tube_gate.detach().mean(),
            "transport_alpha_max": tube_gate.detach().amax(),
            "alignment_loss": alignment_loss,
        }
        return corrected, torch.zeros_like(correction), correction


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
    correct = v1.oracle_bundle(joint, batch, future_frames)
    raw, uv, reliability, anchor = correct
    if raw.shape[0] > 1:
        order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
        shuffled = raw[order], uv[order], reliability[order], anchor[order]
    else:
        order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
        shuffled = raw[:, order], uv[:, order], reliability[:, order], anchor[:, order]
    return correct, shuffled


def match_pixel_budget(prediction, baseline, target_budget):
    """Use pixel L1 as a ceiling so the oracle probe may reject all tracks."""
    delta = prediction - baseline
    norm = delta.abs().mean((1, 2, 3, 4), keepdim=True)
    factor = (float(target_budget) / norm.detach().clamp_min(1e-7)).clamp(max=1.0)
    return (baseline + delta * factor).clamp(0, 1)


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
            totals[name]["residual_l1"] += model.last["residual_l1"].item()
            totals[name]["transport_candidate_l1"] += model.last[
                "transport_candidate_l1"
            ].item()
            totals[name]["transport_alpha_mean"] += model.last[
                "transport_alpha_mean"
            ].item()
            totals[name]["transport_alpha_max"] += model.last[
                "transport_alpha_max"
            ].item()
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
                tube_alignment = model.last.get("alignment_loss", loss.new_zeros(()))
                loss = loss + args.tube_alignment_weight * tube_alignment
                positive_last = model.last
                direction_rank = loss.new_zeros(())
                if name == "correct" and args.direction_rank_weight > 0:
                    negative_mode = (
                        "reverse_flow" if batches_seen % 2 == 0 else "shuffled_flow"
                    )
                    original_mode = model.mode
                    model.mode = negative_mode
                    negative_corrected, _, _ = model(source, hidden, *pair)
                    negative_prediction = match_pixel_budget(
                        core.decode_frozen(backbone, negative_corrected, skip),
                        baseline,
                        args.pixel_budget,
                    )
                    model.mode = original_mode
                    model.last = positive_last
                    positive_error = (prediction - batch["future_rgb"]).square().mean(
                        (1, 2, 3, 4)
                    )
                    negative_error = (
                        negative_prediction - batch["future_rgb"]
                    ).square().mean((1, 2, 3, 4))
                    direction_rank = F.relu(
                        args.direction_rank_margin + positive_error - negative_error
                    ).mean()
                    loss = loss + args.direction_rank_weight * direction_rank
                pixel_delta = (prediction - baseline).abs().mean()
                budget_penalty = F.relu(pixel_delta - args.pixel_budget)
                loss = loss + args.pixel_budget_weight * budget_penalty
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                totals[name]["budget_penalty"] += budget_penalty.item()
                totals[name]["feature_budget"] += model.last["feature_l1"].item()
                totals[name]["residual_l1"] += model.last["residual_l1"].item()
                totals[name]["transport_candidate_l1"] += model.last[
                    "transport_candidate_l1"
                ].item()
                totals[name]["transport_alpha_mean"] += model.last[
                    "transport_alpha_mean"
                ].item()
                totals[name]["transport_alpha_max"] += model.last[
                    "transport_alpha_max"
                ].item()
                totals[name]["direction_rank"] += direction_rank.item()
                totals[name]["tube_alignment"] += tube_alignment.item()
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
        print(json.dumps({"stage": "tau_oracle_track_tube_probe", **row}), flush=True)

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
        "correct_model_reverse_flow",
        "correct_model_half_flow",
        "correct_model_onehalf_flow",
        "correct_model_local_rgb",
    )
    variants = ("frozen_tau", "correct", *same_weight_variants)
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
            utilization[name]["residual_l1"].append(model.last["residual_l1"].item())
            utilization[name]["transport_candidate_l1"].append(
                model.last["transport_candidate_l1"].item()
            )
            utilization[name]["transport_alpha_mean"].append(
                model.last["transport_alpha_mean"].item()
            )
            utilization[name]["transport_alpha_max"].append(
                model.last["transport_alpha_max"].item()
            )
            utilization[name]["pixel_delta"].append(
                (predictions[name] - baseline).abs().mean().item()
            )

        correct_model = models["correct"]
        original_mode = correct_model.mode
        for mode, variant in (
            ("mask_only", "correct_model_mask_only"),
            ("shuffled_flow", "correct_model_shuffled_flow"),
            ("reverse_flow", "correct_model_reverse_flow"),
            ("half_flow", "correct_model_half_flow"),
            ("onehalf_flow", "correct_model_onehalf_flow"),
            ("local_rgb", "correct_model_local_rgb"),
        ):
            correct_model.mode = mode
            corrected, _, _ = correct_model(source, hidden, *pair)
            predictions[variant] = match_pixel_budget(
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
    fair_controls = {}
    same_weight = {
        control: core.paired_bootstrap(records, control, "correct", args.seed)
        for control in (
            "correct_model_mask_only",
            "correct_model_shuffled_flow",
            "correct_model_reverse_flow",
            "correct_model_half_flow",
            "correct_model_onehalf_flow",
            "correct_model_local_rgb",
        )
    }
    oracle = {}
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
    prototype = OracleTrajectoryTubeAdapter(
        channels,
        future_frames,
        args.feature_budget,
        args.max_tracks,
        args.max_tube_gate,
        args.tube_gate_bias,
        args.attention_temperature,
        mode="correct",
    ).to(device)
    models = {"correct": copy.deepcopy(prototype)}
    for name, model in models.items():
        model.mode = name

    init_report = {}
    if args.adapter_init_dir:
        init_dir = Path(args.adapter_init_dir)
        for name, model in models.items():
            checkpoint_path = init_dir / f"{name}_best.pt"
            state = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            incompatible = model.load_state_dict(state["model"], strict=False)
            init_report[name] = {
                "checkpoint": str(checkpoint_path),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
            }
        print(
            json.dumps({"stage": "adapter_warm_start", "report": init_report}),
            flush=True,
        )

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
            "adapter_initialization": init_report,
            "architecture": (
                "oracle identity-preserving trajectory-tube attention over frozen TAU "
                "future latents; anchor appearance plus XYZ/UV/displacement/confidence/time "
                "tokens; spatially biased attention with a learned null route and bounded gate"
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
