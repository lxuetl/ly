"""MSE-aligned causal hierarchical action-risk prediction.

The four mutually exclusive actions are the RGB baseline, D4RT future UV/XYZ,
robust sparse affine motion, and RAFT-Small motion.  The first routing stage
decides whether motion correction is safe at all.  The second stage uses sparse
affine as the conservative motion default and lets D4RT or RAFT override it only
when their predicted risk is lower by a configurable margin.  Baseline remains
explicit abstention when the selected motion action is predicted to be harmful.

At inference the calibrator never observes future RGB.  Each source receives an
independent leave-one-step-out hindcast error and consistency map computed from
observed history.  Future ground truth is used only as training supervision and
for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from evaluate_camera_source_judgement import (
    fit_global_transform, last_two_valid, repeat_static_xyz,
)
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


SOURCE_NAMES = ("d4rt", "sparse_affine", "raft_small")
ACTION_NAMES = ("baseline", *SOURCE_NAMES)
BASELINE_INDEX = 0
D4RT_INDEX = 1
AFFINE_INDEX = 2
RAFT_INDEX = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--correction-checkpoint", default="")
    p.add_argument("--d4rt-correction", default="")
    p.add_argument("--affine-correction", default="")
    p.add_argument("--raft-correction", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--max-train", type=int, default=192)
    p.add_argument("--max-val", type=int, default=48)
    p.add_argument("--max-test", type=int, default=96)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--selection-temperature", type=float, default=0.20)
    p.add_argument("--abstain-margin", type=float, default=0.0)
    p.add_argument("--specialist-margin", type=float, default=0.05)
    p.add_argument("--harm-negative-weight", type=float, default=3.0)
    p.add_argument("--gate-regret-weight", type=float, default=0.20)
    return p.parse_args()


def robust_matrices(previous_uv, anchor_uv, pair_valid):
    """Return normalized-coordinate affine matrices and fit residual maps."""
    matrices, residuals = [], []
    for batch_index in range(anchor_uv.shape[0]):
        keep = pair_valid[batch_index].detach().cpu().numpy().astype(bool)
        source = previous_uv[batch_index].detach().cpu().numpy()[keep].astype(np.float32)
        destination = anchor_uv[batch_index].detach().cpu().numpy()[keep].astype(np.float32)
        matrix = None
        if len(source) >= 3:
            fitted, _ = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=0.01,
            )
            if fitted is not None:
                matrix = np.eye(3, dtype=np.float32)
                matrix[:2] = fitted
        if matrix is None or not np.isfinite(matrix).all():
            matrix = np.eye(3, dtype=np.float32)
            if len(source):
                matrix[:2, 2] = np.median(destination - source, axis=0)
        if len(source):
            homogeneous = np.concatenate((source, np.ones((len(source), 1), np.float32)), -1)
            projected = homogeneous @ matrix.T
            projected = projected[:, :2] / np.clip(projected[:, 2:3], 1e-6, None)
            residual = float(np.linalg.norm(projected - destination, axis=-1).mean())
        else:
            residual = 1.0
        matrices.append(matrix)
        residuals.append(residual)
    return (
        torch.as_tensor(np.stack(matrices), device=anchor_uv.device, dtype=anchor_uv.dtype),
        torch.as_tensor(residuals, device=anchor_uv.device, dtype=anchor_uv.dtype),
    )


def image_grid(batch, height, width, device, dtype):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype), indexing="ij",
    )
    return torch.stack((xx, yy), -1)[None].expand(batch, -1, -1, -1)


def dense_point_stat(uv, values, valid, height, width, sigma=0.05):
    """Gaussian-rasterize a sparse per-point statistic with a global fallback."""
    grid = image_grid(uv.shape[0], height, width, uv.device, uv.dtype)
    distance = (grid[:, None] - torch.nan_to_num(uv)[:, :, None, None]).square().sum(-1)
    weight = torch.exp(-distance / (2 * sigma ** 2)) * valid[:, :, None, None].float()
    denominator = weight.sum(1, keepdim=True)
    numerator = (weight * torch.nan_to_num(values)[:, :, None, None]).sum(1, keepdim=True)
    global_mean = (
        (torch.nan_to_num(values) * valid.float()).sum(1)
        / valid.float().sum(1).clamp_min(1)
    )[:, None, None, None]
    return torch.where(denominator > 1e-4, numerator / denominator.clamp_min(1e-6), global_mean)


def apply_normalized_matrix(matrix, uv):
    homogeneous = torch.cat((uv, torch.ones_like(uv[..., :1])), -1)
    projected = torch.einsum("bij,bnj->bni", matrix, homogeneous)
    return projected[..., :2] / projected[..., 2:3].clamp_min(1e-6)


def observed_history(batch):
    """Remove the final observed instant for leave-one-step-out hindcasting."""
    result = dict(batch)
    result["context_rgb"] = batch["context_rgb"][:, :-1]
    for key in ("past_xyz", "past_uv", "past_valid", "past_confidence"):
        result[key] = batch[key][:, :, :-1]
    return result


def warp_with_normalized_matrix(source_rgb, forward_matrix):
    """Backward sample a source image using a normalized source->target matrix."""
    b, _, h, w = source_rgb.shape
    grid = image_grid(b, h, w, source_rgb.device, source_rgb.dtype)
    target_h = torch.cat((grid, torch.ones_like(grid[..., :1])), -1)
    inverse = torch.linalg.inv(forward_matrix)
    source_h = torch.einsum("bij,bhwj->bhwi", inverse, target_h)
    source_uv = source_h[..., :2] / source_h[..., 2:3].clamp_min(1e-6)
    return F.grid_sample(
        source_rgb, source_uv.mul(2).sub(1), mode="bilinear",
        padding_mode="border", align_corners=True,
    )


def warp_with_normalized_flow(source_rgb, normalized_flow):
    b, _, h, w = source_rgb.shape
    grid = image_grid(b, h, w, source_rgb.device, source_rgb.dtype)
    source_grid = grid - normalized_flow.permute(0, 2, 3, 1)
    return F.grid_sample(
        source_rgb, source_grid.mul(2).sub(1), mode="bilinear",
        padding_mode="border", align_corners=True,
    )


@torch.no_grad()
def raft_history(raft, transform, previous, current, anchor_uv, future_frames):
    original_size = previous.shape[-2:]
    previous_128 = F.interpolate(previous, (128, 128), mode="bilinear", align_corners=False)
    current_128 = F.interpolate(current, (128, 128), mode="bilinear", align_corners=False)
    p, c = transform(previous_128, current_128)
    forward = raft(p, c)[-1]
    backward = raft(c, p)[-1]
    forward_norm = torch.stack((forward[:, 0] / 127, forward[:, 1] / 127), 1)
    backward_norm = torch.stack((backward[:, 0] / 127, backward[:, 1] / 127), 1)
    forward_norm = F.interpolate(forward_norm, original_size, mode="bilinear", align_corners=False)
    backward_norm = F.interpolate(backward_norm, original_size, mode="bilinear", align_corners=False)

    replay = warp_with_normalized_flow(previous, forward_norm)
    base_grid = image_grid(
        previous.shape[0], original_size[0], original_size[1],
        previous.device, previous.dtype,
    )
    destination = (base_grid + forward_norm.permute(0, 2, 3, 1)).mul(2).sub(1)
    backward_at_destination = F.grid_sample(
        backward_norm, destination, mode="bilinear", padding_mode="border", align_corners=True,
    )
    cycle = torch.linalg.vector_norm(forward_norm + backward_at_destination, dim=1, keepdim=True)

    point_grid = anchor_uv.mul(2).sub(1).unsqueeze(2)
    point_flow = F.grid_sample(
        forward_norm, point_grid, mode="bilinear", padding_mode="border", align_corners=True,
    ).squeeze(-1).transpose(1, 2)
    horizon = torch.arange(
        1, future_frames + 1, device=previous.device, dtype=previous.dtype,
    ).view(1, 1, -1, 1)
    future_uv = (anchor_uv[:, :, None] + horizon * point_flow[:, :, None]).clamp(0, 1)
    return future_uv, replay, cycle, forward_norm


def structured_candidate(model, baseline, last_rgb, batch, xyz, uv):
    prediction, _ = model(
        baseline, last_rgb, xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(),
        batch["past_valid"], batch["past_confidence"].float(), mode="structured",
    )
    return prediction.clamp(0, 1)


@torch.no_grad()
def build_candidates(base, trajectory, camera_model, raft, raft_transform, batch):
    if isinstance(camera_model, dict):
        d4rt_model = camera_model["d4rt"]
        affine_model = camera_model["sparse_affine"]
        raft_model = camera_model["raft_small"]
    else:
        d4rt_model = affine_model = raft_model = camera_model
    baseline, xyz, d4rt_uv = frozen_prediction(base, trajectory, batch)
    previous_uv, anchor_uv, anchor_xyz, pair_valid = last_two_valid(batch)
    future_frames = d4rt_uv.shape[2]
    static_xyz = repeat_static_xyz(anchor_xyz, future_frames)
    affine_uv = fit_global_transform(
        previous_uv, anchor_uv, pair_valid, future_frames, "affine",
    )
    previous_rgb, current_rgb = batch["context_rgb"][:, -2], batch["context_rgb"][:, -1]
    raft_uv, _, _, _ = raft_history(
        raft, raft_transform, previous_rgb, current_rgb, anchor_uv, future_frames,
    )

    d4rt_candidate = structured_candidate(
        d4rt_model, baseline, current_rgb, batch, xyz, d4rt_uv,
    )
    affine_candidate = structured_candidate(
        affine_model, baseline, current_rgb, batch, static_xyz, affine_uv,
    )
    raft_candidate = structured_candidate(
        raft_model, baseline, current_rgb, batch, static_xyz, raft_uv,
    )
    motion_candidates = torch.stack((d4rt_candidate, affine_candidate, raft_candidate), 2)

    # Source-specific leave-one-step-out replay.  Every source must predict the
    # final observed frame using information ending one instant earlier.
    history = observed_history(batch)
    earlier_rgb = batch["context_rgb"][:, -3]
    hist_previous_uv, hist_anchor_uv, hist_anchor_xyz, hist_pair_valid = last_two_valid(history)

    hindcast_xyz, hindcast_uv = trajectory(
        history["past_xyz"].float(), history["past_uv"].float(),
        history["past_valid"], history["past_confidence"].float(),
    )
    hindcast_xyz, hindcast_uv = hindcast_xyz[:, :, :1], hindcast_uv[:, :, :1]
    d4rt_replay = structured_candidate(
        d4rt_model, previous_rgb[:, None], previous_rgb, history,
        hindcast_xyz, hindcast_uv,
    )[:, 0]

    affine_hindcast_uv = fit_global_transform(
        hist_previous_uv, hist_anchor_uv, hist_pair_valid, 1, "affine",
    )
    hist_matrices, _ = robust_matrices(hist_previous_uv, hist_anchor_uv, hist_pair_valid)
    affine_replay = warp_with_normalized_matrix(previous_rgb, hist_matrices)

    raft_hindcast_uv, _, raft_cycle, raft_flow = raft_history(
        raft, raft_transform, earlier_rgb, previous_rgb, hist_anchor_uv, 1,
    )
    raft_replay = warp_with_normalized_flow(previous_rgb, raft_flow)

    d4rt_error = (d4rt_replay - current_rgb).abs().mean(1, keepdim=True)
    affine_error = (affine_replay - current_rgb).abs().mean(1, keepdim=True)
    raft_error = (raft_replay - current_rgb).abs().mean(1, keepdim=True)
    motion_replay_error = torch.stack((d4rt_error, affine_error, raft_error), 1)

    h, w = current_rgb.shape[-2:]
    d4rt_point_error = torch.linalg.vector_norm(hindcast_uv[:, :, 0] - anchor_uv, dim=-1)
    affine_point_error = torch.linalg.vector_norm(
        affine_hindcast_uv[:, :, 0] - anchor_uv, dim=-1,
    )
    raft_point_error = torch.linalg.vector_norm(raft_hindcast_uv[:, :, 0] - anchor_uv, dim=-1)
    d4rt_consistency = dense_point_stat(anchor_uv, d4rt_point_error, pair_valid, h, w)
    affine_consistency = dense_point_stat(anchor_uv, affine_point_error, pair_valid, h, w)
    raft_point_consistency = dense_point_stat(anchor_uv, raft_point_error, pair_valid, h, w)
    raft_consistency = raft_point_consistency + F.interpolate(
        raft_cycle, (h, w), mode="bilinear", align_corners=False,
    )
    motion_consistency = torch.stack(
        (d4rt_consistency, affine_consistency, raft_consistency), 1,
    )

    # Baseline is a decision action, not an RGB mixture source.  Its observed
    # reliability cues are temporal persistence error and local acceleration.
    identity_error = (previous_rgb - current_rgb).abs().mean(1, keepdim=True)
    temporal_acceleration = (
        (current_rgb - previous_rgb) - (previous_rgb - earlier_rgb)
    ).abs().mean(1, keepdim=True)
    replay_error = torch.cat((identity_error[:, None], motion_replay_error), 1)
    consistency = torch.cat((temporal_acceleration[:, None], motion_consistency), 1)
    replay_error = replay_error[:, None].expand(-1, future_frames, -1, -1, -1, -1)
    consistency = consistency[:, None].expand(-1, future_frames, -1, -1, -1, -1)

    actions = torch.cat((baseline[:, :, None], motion_candidates), 2)
    consensus = actions.mean(2, keepdim=True)
    disagreement = (actions - consensus).abs().mean(3, keepdim=True)
    correction_magnitude = (actions - baseline[:, :, None]).abs().mean(3, keepdim=True)
    return baseline, actions, replay_error, consistency, disagreement, correction_magnitude


class HierarchicalActionRiskCalibrator(nn.Module):
    """Predict affine-relative risks and execute a two-stage safe action policy."""

    def __init__(
        self, actions=len(ACTION_NAMES), hidden=24, temperature=0.20,
        abstain_margin=0.0, specialist_margin=0.05,
    ):
        super().__init__()
        self.actions = actions
        self.temperature = float(temperature)
        self.abstain_margin = float(abstain_margin)
        self.specialist_margin = float(specialist_margin)
        # candidate RGB, baseline RGB, replay, consistency, disagreement,
        # correction magnitude, horizon, and source identity.
        channels = 3 + 3 + 4 + 1 + actions
        self.local = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(channels * 2, 48), nn.GELU(), nn.Linear(48, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def forward(self, baseline, actions, replay, consistency, disagreement, magnitude):
        b, k, a, _, h, w = actions.shape
        baseline_actions = baseline[:, :, None].expand(-1, -1, a, -1, -1, -1)
        horizon = torch.linspace(0, 1, k, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, k, 1, 1, 1, 1).expand(b, -1, a, -1, h, w)
        one_hot = torch.eye(a, device=baseline.device, dtype=baseline.dtype)
        one_hot = one_hot.view(1, 1, a, a, 1, 1).expand(b, k, -1, -1, h, w)
        features = torch.cat((
            actions, baseline_actions, replay, consistency,
            disagreement, magnitude, horizon, one_hot,
        ), 3)
        flat = features.flatten(0, 2)
        local = self.local(flat).view(b, k, a, 1, h, w)
        pooled = torch.cat((flat.mean((-2, -1)), flat.std((-2, -1))), -1)
        global_risk = self.global_head(pooled).view(b, k, a, 1, 1, 1)
        predicted_risk = local + global_risk

        baseline_risk = predicted_risk[:, :, BASELINE_INDEX]
        affine_risk = predicted_risk[:, :, AFFINE_INDEX]
        specialist_risk = predicted_risk[:, :, (D4RT_INDEX, RAFT_INDEX)]

        # Stage 2: affine is the motion default.  A specialist must beat it by
        # specialist_margin before it is allowed to replace the affine action.
        specialist_soft = torch.softmax(-specialist_risk / self.temperature, dim=2)
        soft_best_specialist_risk = (specialist_soft * specialist_risk).sum(2)
        specialist_probability = torch.sigmoid(
            (affine_risk - soft_best_specialist_risk - self.specialist_margin)
            / self.temperature
        )
        soft_motion_choice = torch.stack((
            specialist_probability * specialist_soft[:, :, 0],
            1 - specialist_probability,
            specialist_probability * specialist_soft[:, :, 1],
        ), 2)

        hard_specialist_relative = specialist_risk.argmin(2)
        hard_specialist_action = torch.where(
            hard_specialist_relative == 0, D4RT_INDEX, RAFT_INDEX,
        )
        hard_specialist_risk = specialist_risk.min(2).values
        hard_override = hard_specialist_risk + self.specialist_margin < affine_risk
        hard_motion_action = torch.where(
            hard_override, hard_specialist_action,
            torch.full_like(hard_specialist_action, AFFINE_INDEX),
        )
        hard_motion_risk = predicted_risk.gather(
            2, hard_motion_action[:, :, None],
        ).squeeze(2)

        # Stage 1: abstain only after Stage 2 has produced its safest motion
        # action.  This prevents a weak specialist from forcing motion use.
        soft_motion_risk = (soft_motion_choice * predicted_risk[:, :, 1:]).sum(2)
        motion_probability = torch.sigmoid(
            (baseline_risk - soft_motion_risk - self.abstain_margin)
            / self.temperature
        )
        hard_accept_motion = hard_motion_risk + self.abstain_margin < baseline_risk
        hard_action = torch.where(
            hard_accept_motion, hard_motion_action,
            torch.full_like(hard_motion_action, BASELINE_INDEX),
        )

        soft_choice = torch.cat((
            (1 - motion_probability)[:, :, None],
            motion_probability[:, :, None] * soft_motion_choice,
        ), 2)
        hard_choice = F.one_hot(hard_action.squeeze(2), self.actions)
        hard_choice = hard_choice.permute(0, 1, 4, 2, 3)[:, :, :, None].to(predicted_risk.dtype)
        choice = hard_choice + soft_choice - soft_choice.detach() if self.training else hard_choice
        prediction = (choice * actions).sum(2)
        return {
            "prediction": prediction.clamp(0, 1),
            "predicted_risk": predicted_risk,
            "choice": choice,
            "hard_choice": hard_choice,
            "motion_probability": motion_probability,
            "predicted_motion_index": (hard_motion_action - 1)[:, :, None],
            "specialist_probability": specialist_probability,
            "hard_specialist_override": hard_override,
        }


def selector_loss(
    state, baseline, actions, target, last, temperature,
    harm_negative_weight=3.0, gate_regret_weight=0.20,
):
    prediction = state["prediction"]
    # The paper's primary metrics are MSE/PSNR, so risk, ranking, gating and
    # checkpoint selection must all use the same squared-error action cost.
    action_error = (actions - target[:, :, None]).square().mean(3, keepdim=True)
    baseline_error = action_error[:, :, BASELINE_INDEX]
    affine_error = action_error[:, :, AFFINE_INDEX]
    denominator = affine_error[:, :, None].detach().clamp_min(0.002)
    normalized_regret = ((action_error - affine_error[:, :, None]) / denominator).clamp(-3, 3)
    predicted_risk = state["predicted_risk"]

    dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).float()
    risk_weight = (1 + dynamic[:, :, None]) * (1 + 0.5 * normalized_regret.abs())
    risk = (
        F.smooth_l1_loss(predicted_risk, normalized_regret, reduction="none")
        * risk_weight
    ).mean()

    motion_error = action_error[:, :, 1:]
    motion_oracle = motion_error.argmin(2).squeeze(2)
    motion_rank = F.cross_entropy(
        (-predicted_risk[:, :, 1:] / temperature)
        .permute(0, 1, 3, 4, 5, 2).reshape(-1, len(SOURCE_NAMES)),
        motion_oracle.reshape(-1),
    )
    # Stage 1 is supervised against the action Stage 2 will actually execute,
    # rather than an unavailable best-of-all-motion oracle.  Harmful false
    # accepts receive extra weight to counter the roughly 3:1 class imbalance.
    selected_motion_error = motion_error.gather(
        2, state["predicted_motion_index"],
    ).squeeze(2)
    motion_helpful = (selected_motion_error < baseline_error).float()
    gate_weight = torch.where(
        motion_helpful > 0.5,
        torch.ones_like(motion_helpful),
        torch.full_like(motion_helpful, float(harm_negative_weight)),
    )
    abstain_rank = (
        F.binary_cross_entropy(
        state["motion_probability"].clamp(1e-5, 1 - 1e-5),
            motion_helpful, reduction="none",
        ) * gate_weight
    ).mean()
    baseline_denominator = baseline_error.detach().clamp_min(0.002)
    selected_regret = (selected_motion_error - baseline_error) / baseline_denominator
    gate_regret = (
        state["motion_probability"] * selected_regret.relu()
        + 0.25 * (1 - state["motion_probability"]) * (-selected_regret).relu()
    ).mean()

    squared_error = (prediction - target).square().mean(2, keepdim=True)
    reconstruction = 10.0 * (squared_error * (1 + 2 * dynamic)).mean()
    temporal = (
        (prediction[:, 1:] - prediction[:, :-1])
        - (target[:, 1:] - target[:, :-1])
    ).abs().mean()
    correction = (prediction - baseline).abs().mean(2, keepdim=True)
    static_correction = (correction * (1 - dynamic)).mean()
    action_probability = torch.softmax(-predicted_risk / temperature, 2)
    tv = (
        (action_probability[..., 1:] - action_probability[..., :-1]).abs().mean()
        + (action_probability[..., 1:, :] - action_probability[..., :-1, :]).abs().mean()
    )
    loss = (
        reconstruction + 0.50 * risk + 0.10 * motion_rank + 0.10 * abstain_rank
        + gate_regret_weight * gate_regret
        + 0.05 * temporal + 0.03 * static_correction + 0.01 * tv
    )
    return loss, {
        "loss": loss, "reconstruction": reconstruction, "risk": risk,
        "motion_rank": motion_rank, "abstain_rank": abstain_rank,
        "gate_regret": gate_regret, "temporal": temporal,
        "static_correction": static_correction, "action_tv": tv,
    }


class Metrics:
    def __init__(self):
        self.v = defaultdict(float)

    def update(self, prediction, target, last):
        difference = prediction - target
        dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).expand_as(difference)
        self.v["abs"] += difference.abs().sum().item()
        self.v["sq"] += difference.square().sum().item()
        self.v["count"] += difference.numel()
        self.v["dyn_sq"] += difference.square()[dynamic].sum().item()
        self.v["dyn_count"] += dynamic.sum().item()

    def result(self):
        mse = self.v["sq"] / max(self.v["count"], 1)
        return {
            "MAE": self.v["abs"] / max(self.v["count"], 1), "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "dynamic_MSE": self.v["dyn_sq"] / max(self.v["dyn_count"], 1),
        }


class CalibrationMetrics:
    """Streaming approximate calibration and selective-risk measurements."""

    def __init__(self, bins=20):
        self.bins = int(bins)
        self.help_count = np.zeros(bins, np.float64)
        self.help_probability = np.zeros(bins, np.float64)
        self.help_positive = np.zeros(bins, np.float64)
        self.accept_count = np.zeros(bins, np.float64)
        self.accept_harm = np.zeros(bins, np.float64)
        self.accept_regret = np.zeros(bins, np.float64)

    def update(self, help_probability, helpful, selected_probability, selected_regret):
        def add(probability, value, count, total):
            probability = probability.detach().float().flatten().cpu().numpy()
            value = value.detach().float().flatten().cpu().numpy()
            index = np.minimum((probability * self.bins).astype(np.int64), self.bins - 1)
            np.add.at(count, index, 1)
            np.add.at(total, index, value)

        add(help_probability, helpful, self.help_count, self.help_positive)
        probability_sum, _ = np.histogram(
            help_probability.detach().float().flatten().cpu().numpy(),
            bins=self.bins, range=(0, 1),
            weights=help_probability.detach().float().flatten().cpu().numpy(),
        )
        self.help_probability += probability_sum
        add(
            selected_probability, (selected_regret >= 0).float(),
            self.accept_count, self.accept_harm,
        )
        probability = selected_probability.detach().float().flatten().cpu().numpy()
        regret = selected_regret.detach().float().flatten().cpu().numpy()
        index = np.minimum((probability * self.bins).astype(np.int64), self.bins - 1)
        np.add.at(self.accept_regret, index, regret)

    def result(self):
        total = self.help_count.sum()
        valid = self.help_count > 0
        confidence = np.divide(
            self.help_probability, self.help_count,
            out=np.zeros_like(self.help_probability), where=valid,
        )
        accuracy = np.divide(
            self.help_positive, self.help_count,
            out=np.zeros_like(self.help_positive), where=valid,
        )
        ece = float((self.help_count * np.abs(confidence - accuracy)).sum() / max(total, 1))

        positives = self.help_positive.sum()
        negatives_by_bin = self.help_count - self.help_positive
        negatives = negatives_by_bin.sum()
        cumulative_negatives = 0.0
        concordant = 0.0
        for pos, neg in zip(self.help_positive, negatives_by_bin):
            concordant += pos * (cumulative_negatives + 0.5 * neg)
            cumulative_negatives += neg
        auroc = float(concordant / max(positives * negatives, 1))

        cumulative_count = cumulative_harm = cumulative_regret = 0.0
        curve, aurc = [], 0.0
        all_count = self.accept_count.sum()
        for index in range(self.bins - 1, -1, -1):
            previous_coverage = cumulative_count / max(all_count, 1)
            cumulative_count += self.accept_count[index]
            cumulative_harm += self.accept_harm[index]
            cumulative_regret += self.accept_regret[index]
            coverage = cumulative_count / max(all_count, 1)
            harmful_rate = cumulative_harm / max(cumulative_count, 1)
            mean_regret = cumulative_regret / max(cumulative_count, 1)
            aurc += (coverage - previous_coverage) * harmful_rate
            curve.append({
                "threshold": index / self.bins, "coverage": coverage,
                "harmful_rate": harmful_rate, "mean_regret": mean_regret,
            })
        return {"help_ece": ece, "help_auroc": auroc, "harm_aurc": float(aurc),
                "risk_coverage_curve": curve}


@torch.no_grad()
def evaluate(selector, base, trajectory, camera_model, raft, raft_transform, loader, device):
    selector.eval()
    metrics = {name: Metrics() for name in (*ACTION_NAMES, "selected", "oracle")}
    calibration_metrics = CalibrationMetrics()
    action_choice = torch.zeros(len(ACTION_NAMES), device=device)
    oracle_choice = torch.zeros(len(ACTION_NAMES), device=device)
    totals = defaultdict(float)
    for batch in loader:
        batch = move(batch, device)
        baseline, actions, replay, cycle, disagreement, magnitude = build_candidates(
            base, trajectory, camera_model, raft, raft_transform, batch,
        )
        state = selector(baseline, actions, replay, cycle, disagreement, magnitude)
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        for index, name in enumerate(ACTION_NAMES):
            metrics[name].update(actions[:, :, index], target, last)
        metrics["selected"].update(state["prediction"], target, last)

        action_error = (actions - target[:, :, None]).square().mean(3, keepdim=True)
        oracle_index = action_error.argmin(2, keepdim=True)
        oracle = actions.gather(
            2, oracle_index.expand(-1, -1, -1, 3, -1, -1),
        ).squeeze(2)
        metrics["oracle"].update(oracle, target, last)

        oracle_action = action_error.argmin(2)
        selected_action = state["hard_choice"].argmax(2)
        totals["action_correct"] += (selected_action == oracle_action).sum().item()
        totals["action_count"] += oracle_action.numel()
        action_choice += state["hard_choice"].sum((0, 1, 3, 4, 5))
        oracle_choice.scatter_add_(
            0, oracle_action.flatten(), torch.ones_like(oracle_action.flatten(), dtype=torch.float),
        )
        totals["choice_count"] += state["hard_choice"].sum().item()

        baseline_error = action_error[:, :, 0]
        motion_regret = action_error[:, :, 1:] - baseline_error[:, :, None]
        predicted_motion = state["predicted_motion_index"]
        predicted_motion_regret = motion_regret.gather(2, predicted_motion).squeeze(2)
        motion_helpful = (motion_regret.min(2).values < 0).float()
        calibration_metrics.update(
            state["motion_probability"], motion_helpful,
            state["motion_probability"], predicted_motion_regret,
        )
        totals["brier_sum"] += (
            state["motion_probability"] - motion_helpful
        ).square().sum().item()
        totals["brier_count"] += motion_helpful.numel()
        selected_motion = selected_action != 0
        predicted_motion_action = state["predicted_motion_index"].squeeze(2)
        motion_oracle_action = action_error[:, :, 1:].argmin(2)
        totals["motion_stage_correct"] += (
            predicted_motion_action == motion_oracle_action
        ).sum().item()
        totals["motion_stage_count"] += motion_oracle_action.numel()
        selected_motion_error = action_error[:, :, 1:].gather(
            2, state["predicted_motion_index"],
        ).squeeze(2)
        chosen_motion_helpful = selected_motion_error < baseline_error
        totals["stage1_correct"] += (
            selected_motion == chosen_motion_helpful
        ).sum().item()
        totals["stage1_count"] += selected_motion.numel()
        specialist_override = (selected_action == D4RT_INDEX) | (selected_action == RAFT_INDEX)
        selected_error = action_error.gather(2, selected_action[:, :, None]).squeeze(2)
        affine_error = action_error[:, :, AFFINE_INDEX]
        totals["specialist_override"] += specialist_override.sum().item()
        totals["specialist_helpful"] += (
            specialist_override & (selected_error < affine_error)
        ).sum().item()
        totals["coverage"] += selected_motion.sum().item()
        totals["pixels"] += selected_motion.numel()
        totals["accepted_helpful"] += (
            selected_motion & (predicted_motion_regret < 0)
        ).sum().item()
        totals["motion_regret"] += predicted_motion_regret.sum().item()
        totals["motion_regret_count"] += predicted_motion_regret.numel()

    result = {name: value.result() for name, value in metrics.items()}
    result["action_choice_fraction"] = {
        name: float(action_choice[index] / max(totals["choice_count"], 1))
        for index, name in enumerate(ACTION_NAMES)
    }
    result["oracle_action_fraction"] = {
        name: float(oracle_choice[index] / oracle_choice.sum().clamp_min(1))
        for index, name in enumerate(ACTION_NAMES)
    }
    result["calibration"] = {
        "action_accuracy": totals["action_correct"] / max(totals["action_count"], 1),
        "motion_help_brier": totals["brier_sum"] / max(totals["brier_count"], 1),
        "coverage": totals["coverage"] / max(totals["pixels"], 1),
        "accepted_help_rate": totals["accepted_helpful"] / max(totals["coverage"], 1),
        "stage1_accuracy": totals["stage1_correct"] / max(totals["stage1_count"], 1),
        "motion_stage_accuracy": (
            totals["motion_stage_correct"] / max(totals["motion_stage_count"], 1)
        ),
        "specialist_override_rate": (
            totals["specialist_override"] / max(totals["pixels"], 1)
        ),
        "specialist_override_help_rate": (
            totals["specialist_helpful"] / max(totals["specialist_override"], 1)
        ),
        "mean_predicted_motion_regret": (
            totals["motion_regret"] / max(totals["motion_regret_count"], 1)
        ),
        **calibration_metrics.result(),
    }
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(args)

    def dataset(split, maximum):
        paths = list(splits[split])[:maximum] if maximum > 0 else list(splits[split])
        return VideoForecastDataset(
            args.cache_root, split, args.image_size, paths=paths, cache_items=False
        )

    datasets = {
        "train": dataset("train", args.max_train),
        "val": dataset("val", args.max_val),
        "test": dataset("test", args.max_test),
    }
    train_loader = DataLoader(
        datasets["train"], args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=video_pad_collate, num_workers=0,
    )
    val_loader = DataLoader(datasets["val"], args.batch_size, collate_fn=video_pad_collate, num_workers=0)
    test_loader = DataLoader(datasets["test"], args.batch_size, collate_fn=video_pad_collate, num_workers=0)

    base, trajectory, _, _ = load_frozen_models(args, device)
    def load_camera(path):
        correction = torch.load(path, map_location=device)
        config = correction.get("config", {})
        model = StructuredCaPSR(
            config.get("sigma", 0.035), config.get("hidden", 48),
            num_motion_clusters=config.get("num_motion_clusters", 4),
        ).to(device)
        model.load_state_dict(correction["model"])
        return model

    source_paths = {
        "d4rt": args.d4rt_correction,
        "sparse_affine": args.affine_correction,
        "raft_small": args.raft_correction,
    }
    if all(source_paths.values()):
        camera_model = {name: load_camera(path) for name, path in source_paths.items()}
    elif args.correction_checkpoint:
        shared = load_camera(args.correction_checkpoint)
        camera_model = shared
    else:
        raise ValueError("provide either --correction-checkpoint or all three source corrections")
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    raft_weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=raft_weights, progress=True).to(device)
    raft_transform = raft_weights.transforms()
    frozen_models = [base, trajectory, raft]
    frozen_models.extend(camera_model.values() if isinstance(camera_model, dict) else [camera_model])
    for frozen in frozen_models:
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    selector = HierarchicalActionRiskCalibrator(
        temperature=args.selection_temperature,
        abstain_margin=args.abstain_margin,
        specialist_margin=args.specialist_margin,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(parameter.numel() for parameter in selector.parameters())
    history, start, best = [], time.time(), float("inf")
    print(json.dumps({
        "trainable_params": trainable, "actions": ACTION_NAMES,
        "image_size": args.image_size,
        "source_specific_corrections": isinstance(camera_model, dict),
        "used": {name: len(value) for name, value in datasets.items()}, "split": split_meta,
    }), flush=True)

    for epoch in range(1, args.epochs + 1):
        selector.train()
        totals = defaultdict(float)
        for batch in train_loader:
            batch = move(batch, device)
            with torch.no_grad():
                baseline, actions, replay, cycle, disagreement, magnitude = build_candidates(
                    base, trajectory, camera_model, raft, raft_transform, batch,
                )
            state = selector(
                baseline, actions, replay, cycle, disagreement, magnitude,
            )
            loss, pieces = selector_loss(
                state, baseline, actions, batch["future_rgb"],
                batch["context_rgb"][:, -1], args.selection_temperature,
                args.harm_negative_weight, args.gate_regret_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            for name, value in pieces.items():
                totals[name] += float(value.detach())
        validation = evaluate(
            selector, base, trajectory, camera_model, raft, raft_transform, val_loader, device,
        )
        score = validation["selected"]["MSE"]
        row = {
            "epoch": epoch, "elapsed_seconds": time.time() - start,
            **{f"train_{name}": value / len(train_loader) for name, value in totals.items()},
            "val": validation,
        }
        history.append(row)
        checkpoint = {"model": selector.state_dict(), "epoch": epoch, "config": vars(args)}
        torch.save(checkpoint, output / "last.pt")
        if score < best:
            best = score
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)

    best_checkpoint = torch.load(output / "best.pt", map_location=device)
    selector.load_state_dict(best_checkpoint["model"])
    test = evaluate(
        selector, base, trajectory, camera_model, raft, raft_transform, test_loader, device,
    )
    result = {
        "test": test,
        "meta": {
            "best_epoch": best_checkpoint["epoch"], "trainable_params": trainable,
            "actions": ACTION_NAMES, "historical_replay_only_at_inference": True,
            "hard_action_selection": True, "baseline_action_is_abstention": True,
            "hierarchical_affine_default": True,
            "mse_aligned_regret": True,
            "selected_motion_gate_target": True,
            "used": {name: len(value) for name, value in datasets.items()},
            "elapsed_seconds": time.time() - start, "split": split_meta,
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
