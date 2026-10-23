"""Historical-replay calibrated multi-motion-source routing.

Frozen candidate sources:
  0. strong video-prediction baseline (no motion correction),
  1. D4RT future UV/XYZ camera correction,
  2. robust sparse affine camera correction,
  3. RAFT-Small camera correction.

At inference the selector never observes future RGB.  Its reliability cues are
computed only from observed history: held-out last-frame replay error, RAFT
forward/backward cycle error, sparse affine fit residual, and disagreement among
future candidates.  Future ground truth is used only to train the selector.
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


SOURCE_NAMES = ("baseline", "d4rt", "sparse_affine", "raft_small")


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
    return future_uv, replay, cycle


def camera_candidate(model, baseline, batch, xyz, uv):
    _, state = model(
        baseline, batch["context_rgb"][:, -1], xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(),
        batch["past_valid"], batch["past_confidence"].float(), mode="structured",
    )
    return (
        baseline + state["camera_weight"] * (state["camera_rgb"] - baseline)
    ).clamp(0, 1)


@torch.no_grad()
def build_candidates(base, trajectory, camera_model, raft, raft_transform, batch):
    baseline, xyz, d4rt_uv = frozen_prediction(base, trajectory, batch)
    previous_uv, anchor_uv, anchor_xyz, pair_valid = last_two_valid(batch)
    future_frames = d4rt_uv.shape[2]
    static_xyz = repeat_static_xyz(anchor_xyz, future_frames)
    affine_uv = fit_global_transform(
        previous_uv, anchor_uv, pair_valid, future_frames, "affine",
    )
    matrices, sparse_residual = robust_matrices(previous_uv, anchor_uv, pair_valid)
    previous_rgb, current_rgb = batch["context_rgb"][:, -2], batch["context_rgb"][:, -1]
    affine_replay = warp_with_normalized_matrix(previous_rgb, matrices)
    raft_uv, raft_replay, raft_cycle = raft_history(
        raft, raft_transform, previous_rgb, current_rgb, anchor_uv, future_frames,
    )

    d4rt_candidate = camera_candidate(camera_model, baseline, batch, xyz, d4rt_uv)
    affine_candidate = camera_candidate(camera_model, baseline, batch, static_xyz, affine_uv)
    raft_candidate = camera_candidate(camera_model, baseline, batch, static_xyz, raft_uv)
    candidates = torch.stack((baseline, d4rt_candidate, affine_candidate, raft_candidate), 2)

    identity_error = (previous_rgb - current_rgb).abs().mean(1, keepdim=True)
    affine_error = (affine_replay - current_rgb).abs().mean(1, keepdim=True)
    raft_error = (raft_replay - current_rgb).abs().mean(1, keepdim=True)
    replay_error = torch.stack((identity_error, affine_error, affine_error, raft_error), 1)
    replay_error = replay_error[:, None].expand(-1, future_frames, -1, -1, -1, -1)

    zero = torch.zeros_like(identity_error)
    sparse_map = sparse_residual[:, None, None, None].expand_as(identity_error)
    consistency = torch.stack((zero, sparse_map, sparse_map, raft_cycle), 1)
    consistency = consistency[:, None].expand(-1, future_frames, -1, -1, -1, -1)

    consensus = candidates.mean(2, keepdim=True)
    disagreement = (candidates - consensus).abs().mean(3, keepdim=True)
    correction_magnitude = (candidates - baseline[:, :, None]).abs().mean(3, keepdim=True)
    return baseline, candidates, replay_error, consistency, disagreement, correction_magnitude


class ReplayMultiSourceSelector(nn.Module):
    """Global horizon source selection plus local pixel refinement."""

    def __init__(self, sources=len(SOURCE_NAMES), hidden=24):
        super().__init__()
        self.sources = sources
        # RGB candidate, RGB baseline, four reliability maps, horizon, source one-hot.
        channels = 3 + 3 + 4 + 1 + sources
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
        local_logits = self.local(flat).view(b, k, s, 1, h, w)
        pooled = torch.cat((flat.mean((-2, -1)), flat.std((-2, -1))), -1)
        global_logits = self.global_head(pooled).view(b, k, s, 1, 1, 1)
        weights = torch.softmax(global_logits + local_logits, dim=2)
        prediction = (weights * candidates).sum(2)
        return prediction, weights, global_logits.squeeze(-1).squeeze(-1).squeeze(-1)


def selector_loss(prediction, weights, global_logits, candidates, target, last):
    source_error = (candidates - target[:, :, None]).abs().mean((3, 4, 5))
    oracle_source = source_error.argmin(2)
    global_rank = F.cross_entropy(global_logits.flatten(0, 1), oracle_source.flatten())
    pixel_error = (candidates - target[:, :, None]).abs().mean(3, keepdim=True)
    pixel_oracle = pixel_error.argmin(2).squeeze(2)
    local_rank = F.nll_loss(
        weights.clamp_min(1e-7).log().permute(0, 1, 3, 4, 5, 2).reshape(-1, weights.shape[2]),
        pixel_oracle.reshape(-1),
    )
    dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).float()
    error = (prediction - target).abs().mean(2, keepdim=True)
    reconstruction = (error * (1 + 2 * dynamic)).mean()
    temporal = (
        (prediction[:, 1:] - prediction[:, :-1])
        - (target[:, 1:] - target[:, :-1])
    ).abs().mean()
    entropy = -(weights.clamp_min(1e-7) * weights.clamp_min(1e-7).log()).sum(2).mean()
    loss = reconstruction + 0.12 * global_rank + 0.05 * local_rank + 0.05 * temporal + 0.002 * entropy
    return loss, {
        "loss": loss, "reconstruction": reconstruction, "global_rank": global_rank,
        "local_rank": local_rank, "temporal": temporal, "entropy": entropy,
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


@torch.no_grad()
def evaluate(selector, base, trajectory, camera_model, raft, raft_transform, loader, device):
    selector.eval()
    metrics = {name: Metrics() for name in (*SOURCE_NAMES, "selected")}
    source_weight = torch.zeros(len(SOURCE_NAMES), device=device)
    source_count = 0
    for batch in loader:
        batch = move(batch, device)
        baseline, candidates, replay, cycle, disagreement, magnitude = build_candidates(
            base, trajectory, camera_model, raft, raft_transform, batch,
        )
        prediction, weights, _ = selector(
            baseline, candidates, replay, cycle, disagreement, magnitude,
        )
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        for index, name in enumerate(SOURCE_NAMES):
            metrics[name].update(candidates[:, :, index], target, last)
        metrics["selected"].update(prediction, target, last)
        source_weight += weights.mean((0, 1, 3, 4, 5))
        source_count += 1
    result = {name: value.result() for name, value in metrics.items()}
    result["mean_source_weight"] = {
        name: float(source_weight[index] / max(source_count, 1))
        for index, name in enumerate(SOURCE_NAMES)
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

    selector = ReplayMultiSourceSelector().to(device)
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
            prediction, weights, global_logits = selector(
                baseline, candidates, replay, cycle, disagreement, magnitude,
            )
            loss, pieces = selector_loss(
                prediction, weights, global_logits, candidates,
                batch["future_rgb"], batch["context_rgb"][:, -1],
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
    test = evaluate(selector, base, trajectory, camera_model, raft, raft_transform, test_loader, device)
    result = {
        "test": test,
        "meta": {
            "best_epoch": best_checkpoint["epoch"], "trainable_params": trainable,
            "sources": SOURCE_NAMES, "historical_replay_only_at_inference": True,
            "used": {name: len(value) for name, value in datasets.items()},
            "elapsed_seconds": time.time() - start, "split": split_meta,
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
