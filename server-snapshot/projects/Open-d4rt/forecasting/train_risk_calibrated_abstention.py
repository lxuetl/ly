"""Causal replay-calibrated motion selection with explicit abstention.

Frozen motion candidates are D4RT future UV/XYZ, robust sparse affine motion,
and RAFT-Small motion.  The RGB predictor is a safe fallback, not another
source to be softly blended.

At inference the calibrator never observes future RGB.  Each source receives an
independent leave-one-step-out hindcast error and consistency map computed from
observed history.  It predicts future pixel regret, makes a hard source choice,
and applies that source's residual only where a calibrated permission gate says
it is safer than abstaining to the RGB baseline.  Future ground truth is used
only as training supervision and for evaluation.
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
CANDIDATE_NAMES = ("baseline", *SOURCE_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--correction-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--max-train", type=int, default=192)
    p.add_argument("--max-val", type=int, default=48)
    p.add_argument("--max-test", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--help-margin", type=float, default=0.002)
    p.add_argument("--selection-temperature", type=float, default=0.20)
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
        camera_model, baseline, current_rgb, batch, xyz, d4rt_uv,
    )
    affine_candidate = structured_candidate(
        camera_model, baseline, current_rgb, batch, static_xyz, affine_uv,
    )
    raft_candidate = structured_candidate(
        camera_model, baseline, current_rgb, batch, static_xyz, raft_uv,
    )
    candidates = torch.stack((d4rt_candidate, affine_candidate, raft_candidate), 2)

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
        camera_model, previous_rgb[:, None], previous_rgb, history,
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
    replay_error = torch.stack((d4rt_error, affine_error, raft_error), 1)
    replay_error = replay_error[:, None].expand(-1, future_frames, -1, -1, -1, -1)

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
    consistency = torch.stack((d4rt_consistency, affine_consistency, raft_consistency), 1)
    consistency = consistency[:, None].expand(-1, future_frames, -1, -1, -1, -1)

    consensus = candidates.mean(2, keepdim=True)
    disagreement = (candidates - consensus).abs().mean(3, keepdim=True)
    correction_magnitude = (candidates - baseline[:, :, None]).abs().mean(3, keepdim=True)
    return baseline, candidates, replay_error, consistency, disagreement, correction_magnitude


class ReplayRiskCalibrator(nn.Module):
    """Predict per-source regret, select one source, and abstain when unsafe."""

    def __init__(self, sources=len(SOURCE_NAMES), hidden=24, temperature=0.20):
        super().__init__()
        self.sources = sources
        self.temperature = float(temperature)
        # candidate RGB, baseline RGB, replay, consistency, disagreement,
        # correction magnitude, horizon, and source identity.
        channels = 3 + 3 + 4 + 1 + sources
        self.local = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(channels * 2, 48), nn.GELU(), nn.Linear(48, 2),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        # Start conservatively: predicted regret is neutral and permission is low.
        self.local[-1].bias.data[1] = -1.5

    @staticmethod
    def _hard_choice(predicted_regret):
        winner = predicted_regret.argmin(2).squeeze(2)
        return F.one_hot(winner, predicted_regret.shape[2]).permute(0, 1, 4, 2, 3)[:, :, :, None]

    def forward(self, baseline, candidates, replay, consistency, disagreement, magnitude):
        b, k, s, _, h, w = candidates.shape
        baseline_sources = baseline[:, :, None].expand(-1, -1, s, -1, -1, -1)
        horizon = torch.linspace(0, 1, k, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, k, 1, 1, 1, 1).expand(b, -1, s, -1, h, w)
        one_hot = torch.eye(s, device=baseline.device, dtype=baseline.dtype)
        one_hot = one_hot.view(1, 1, s, s, 1, 1).expand(b, k, -1, -1, h, w)
        features = torch.cat((
            candidates, baseline_sources, replay, consistency,
            disagreement, magnitude, horizon, one_hot,
        ), 3)
        flat = features.flatten(0, 2)
        local = self.local(flat).view(b, k, s, 2, h, w)
        pooled = torch.cat((flat.mean((-2, -1)), flat.std((-2, -1))), -1)
        global_values = self.global_head(pooled).view(b, k, s, 2, 1, 1)
        values = local + global_values
        predicted_regret = values[:, :, :, :1]
        help_logits = values[:, :, :, 1:]

        soft_choice = torch.softmax(-predicted_regret / self.temperature, dim=2)
        hard_choice = self._hard_choice(predicted_regret).to(predicted_regret.dtype)
        choice = hard_choice + soft_choice - soft_choice.detach() if self.training else hard_choice
        selected_candidate = (choice * candidates).sum(2)
        selected_help_logit = (choice * help_logits).sum(2)
        permission_probability = torch.sigmoid(selected_help_logit)
        hard_permission = (permission_probability >= 0.5).to(permission_probability.dtype)
        permission = (
            hard_permission + permission_probability - permission_probability.detach()
            if self.training else hard_permission
        )
        prediction = baseline + permission * (selected_candidate - baseline)
        return {
            "prediction": prediction.clamp(0, 1),
            "selected_candidate": selected_candidate,
            "predicted_regret": predicted_regret,
            "help_logits": help_logits,
            "choice": choice,
            "hard_choice": hard_choice,
            "permission": permission,
            "permission_probability": permission_probability,
        }


def selector_loss(state, baseline, candidates, target, last, help_margin):
    prediction = state["prediction"]
    baseline_error = (baseline - target).abs().mean(2, keepdim=True)
    source_error = (candidates - target[:, :, None]).abs().mean(3, keepdim=True)
    actual_regret = source_error - baseline_error[:, :, None]
    predicted_regret = state["predicted_regret"]

    risk = F.smooth_l1_loss(predicted_regret, actual_regret)
    oracle_source = source_error.argmin(2).squeeze(2)
    source_rank = F.cross_entropy(
        (-predicted_regret / 0.20).permute(0, 1, 3, 4, 5, 2).reshape(-1, candidates.shape[2]),
        oracle_source.reshape(-1),
    )

    dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).float()
    helpful = ((actual_regret < -help_margin) & (dynamic[:, :, None] > 0)).float()
    positive = helpful.sum()
    negative = helpful.numel() - positive
    positive_weight = (negative / positive.clamp_min(1)).clamp(1, 12)
    help_loss = F.binary_cross_entropy_with_logits(
        state["help_logits"], helpful,
        pos_weight=positive_weight.to(state["help_logits"].dtype),
    )

    error = (prediction - target).abs().mean(2, keepdim=True)
    reconstruction = (error * (1 + 2 * dynamic)).mean()
    temporal = (
        (prediction[:, 1:] - prediction[:, :-1])
        - (target[:, 1:] - target[:, :-1])
    ).abs().mean()
    static_permission = (state["permission_probability"] * (1 - dynamic)).mean()
    probability = torch.sigmoid(state["help_logits"])
    tv = (
        (probability[..., 1:] - probability[..., :-1]).abs().mean()
        + (probability[..., 1:, :] - probability[..., :-1, :]).abs().mean()
    )
    loss = (
        reconstruction + 0.40 * risk + 0.08 * source_rank + 0.08 * help_loss
        + 0.05 * temporal + 0.10 * static_permission + 0.02 * tv
    )
    return loss, {
        "loss": loss, "reconstruction": reconstruction, "risk": risk,
        "source_rank": source_rank, "help": help_loss, "temporal": temporal,
        "static_permission": static_permission, "permission_tv": tv,
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
def evaluate(selector, base, trajectory, camera_model, raft, raft_transform, loader, device,
             help_margin=0.002):
    selector.eval()
    metrics = {name: Metrics() for name in (*CANDIDATE_NAMES, "chosen_motion", "selected", "oracle")}
    calibration_metrics = CalibrationMetrics()
    source_choice = torch.zeros(len(SOURCE_NAMES), device=device)
    totals = defaultdict(float)
    for batch in loader:
        batch = move(batch, device)
        baseline, candidates, replay, cycle, disagreement, magnitude = build_candidates(
            base, trajectory, camera_model, raft, raft_transform, batch,
        )
        state = selector(
            baseline, candidates, replay, cycle, disagreement, magnitude,
        )
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        metrics["baseline"].update(baseline, target, last)
        for index, name in enumerate(SOURCE_NAMES):
            metrics[name].update(candidates[:, :, index], target, last)
        metrics["chosen_motion"].update(state["selected_candidate"], target, last)
        metrics["selected"].update(state["prediction"], target, last)

        all_candidates = torch.cat((baseline[:, :, None], candidates), 2)
        all_error = (all_candidates - target[:, :, None]).abs().mean(3, keepdim=True)
        oracle_index = all_error.argmin(2, keepdim=True)
        oracle = all_candidates.gather(
            2, oracle_index.expand(-1, -1, -1, 3, -1, -1),
        ).squeeze(2)
        metrics["oracle"].update(oracle, target, last)

        source_error = (candidates - target[:, :, None]).abs().mean(3, keepdim=True)
        baseline_error = (baseline - target).abs().mean(2, keepdim=True)
        actual_regret = source_error - baseline_error[:, :, None]
        oracle_source = source_error.argmin(2)
        selected_source = state["hard_choice"].argmax(2)
        totals["source_correct"] += (selected_source == oracle_source).sum().item()
        totals["source_count"] += oracle_source.numel()
        source_choice += state["hard_choice"].sum((0, 1, 3, 4, 5))
        totals["choice_count"] += state["hard_choice"].sum().item()

        dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05)
        helpful = ((actual_regret < -help_margin) & dynamic[:, :, None]).float()
        help_probability = torch.sigmoid(state["help_logits"])
        totals["brier_sum"] += (help_probability - helpful).square().sum().item()
        totals["brier_count"] += helpful.numel()
        selected_regret = (state["hard_choice"] * actual_regret).sum(2)
        calibration_metrics.update(
            help_probability, helpful, state["permission_probability"], selected_regret,
        )
        accepted = state["permission"] > 0.5
        totals["accepted"] += accepted.sum().item()
        totals["pixels"] += accepted.numel()
        totals["accepted_helpful"] += ((selected_regret < 0) & accepted).sum().item()
        totals["selected_regret"] += selected_regret.sum().item()
        totals["selected_regret_count"] += selected_regret.numel()
    result = {name: value.result() for name, value in metrics.items()}
    result["source_choice_fraction"] = {
        name: float(source_choice[index] / max(totals["choice_count"], 1))
        for index, name in enumerate(SOURCE_NAMES)
    }
    result["calibration"] = {
        "source_accuracy": totals["source_correct"] / max(totals["source_count"], 1),
        "help_brier": totals["brier_sum"] / max(totals["brier_count"], 1),
        "coverage": totals["accepted"] / max(totals["pixels"], 1),
        "accepted_help_rate": totals["accepted_helpful"] / max(totals["accepted"], 1),
        "mean_selected_regret": totals["selected_regret"] / max(totals["selected_regret_count"], 1),
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
        return VideoForecastDataset(args.cache_root, split, 64, paths=paths, cache_items=False)

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
    correction = torch.load(args.correction_checkpoint, map_location=device)
    config = correction.get("config", {})
    camera_model = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    camera_model.load_state_dict(correction["model"])
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    raft_weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=raft_weights, progress=True).to(device)
    raft_transform = raft_weights.transforms()
    for frozen in (base, trajectory, camera_model, raft):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)

    selector = ReplayRiskCalibrator(
        temperature=args.selection_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(parameter.numel() for parameter in selector.parameters())
    history, start, best = [], time.time(), float("inf")
    print(json.dumps({
        "trainable_params": trainable, "sources": SOURCE_NAMES,
        "used": {name: len(value) for name, value in datasets.items()}, "split": split_meta,
    }), flush=True)

    for epoch in range(1, args.epochs + 1):
        selector.train()
        totals = defaultdict(float)
        for batch in train_loader:
            batch = move(batch, device)
            with torch.no_grad():
                baseline, candidates, replay, cycle, disagreement, magnitude = build_candidates(
                    base, trajectory, camera_model, raft, raft_transform, batch,
                )
            state = selector(
                baseline, candidates, replay, cycle, disagreement, magnitude,
            )
            loss, pieces = selector_loss(
                state, baseline, candidates, batch["future_rgb"],
                batch["context_rgb"][:, -1], args.help_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            for name, value in pieces.items():
                totals[name] += float(value.detach())
        validation = evaluate(
            selector, base, trajectory, camera_model, raft, raft_transform, val_loader, device,
            args.help_margin,
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
        args.help_margin,
    )
    result = {
        "test": test,
        "meta": {
            "best_epoch": best_checkpoint["epoch"], "trainable_params": trainable,
            "sources": SOURCE_NAMES, "historical_replay_only_at_inference": True,
            "hard_source_selection": True, "explicit_abstention": True,
            "used": {name: len(value) for name, value in datasets.items()},
            "elapsed_seconds": time.time() - start, "split": split_meta,
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
