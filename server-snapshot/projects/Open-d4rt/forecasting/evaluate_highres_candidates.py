#!/usr/bin/env python3
"""Evaluate fixed geometry candidates at a requested spatial resolution.

This deliberately contains no learned router: it measures the RGB backbone,
D4RT, sparse affine, RAFT-Small, and a per-pixel oracle on the exact same clips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from train_capsr import load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from train_unified_action_risk import ACTION_NAMES, build_candidates
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--correction-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-test", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class MetricAccumulator:
    def __init__(self):
        self.values = defaultdict(float)

    def update(self, prediction, target, last):
        difference = prediction - target
        dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05)
        dynamic_rgb = dynamic.expand_as(difference)
        self.values["abs"] += difference.abs().sum().item()
        self.values["sq"] += difference.square().sum().item()
        self.values["count"] += difference.numel()
        self.values["dyn_abs"] += difference.abs()[dynamic_rgb].sum().item()
        self.values["dyn_sq"] += difference.square()[dynamic_rgb].sum().item()
        self.values["dyn_count"] += dynamic_rgb.sum().item()

    def result(self):
        count = max(self.values["count"], 1)
        dyn_count = max(self.values["dyn_count"], 1)
        mse = self.values["sq"] / count
        dynamic_mse = self.values["dyn_sq"] / dyn_count
        return {
            "MAE": self.values["abs"] / count,
            "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "dynamic_MAE": self.values["dyn_abs"] / dyn_count,
            "dynamic_MSE": dynamic_mse,
            "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
        }


@torch.no_grad()
def evaluate(base, trajectory, correction, raft, transform, loader, device):
    names = (*ACTION_NAMES, "pixel_oracle", "frame_oracle")
    metrics = {name: MetricAccumulator() for name in names}
    pixel_choices = torch.zeros(len(ACTION_NAMES), device=device)
    frame_choices = torch.zeros(len(ACTION_NAMES), device=device)
    clip_rows = []
    for clip_index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, actions, _, _, _, _ = build_candidates(
            base, trajectory, correction, raft, transform, batch,
        )
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        for action_index, name in enumerate(ACTION_NAMES):
            metrics[name].update(actions[:, :, action_index], target, last)

        squared = (actions - target[:, :, None]).square()
        pixel_error = squared.mean(3, keepdim=True)
        pixel_index = pixel_error.argmin(2, keepdim=True)
        pixel_oracle = actions.gather(
            2, pixel_index.expand(-1, -1, -1, 3, -1, -1),
        ).squeeze(2)
        metrics["pixel_oracle"].update(pixel_oracle, target, last)
        pixel_choices.scatter_add_(
            0, pixel_index.flatten(),
            torch.ones_like(pixel_index.flatten(), dtype=torch.float),
        )

        frame_error = squared.mean((3, 4, 5))
        frame_index = frame_error.argmin(2)
        frame_oracle = actions.gather(
            2, frame_index[:, :, None, None, None, None].expand(
                -1, -1, -1, 3, actions.shape[-2], actions.shape[-1]
            ),
        ).squeeze(2)
        metrics["frame_oracle"].update(frame_oracle, target, last)
        frame_choices.scatter_add_(
            0, frame_index.flatten(),
            torch.ones_like(frame_index.flatten(), dtype=torch.float),
        )

        row = {"clip_index": clip_index}
        for action_index, name in enumerate(ACTION_NAMES):
            row[name + "_MSE"] = float(squared[:, :, action_index].mean())
        clip_rows.append(row)

    return {
        "metrics": {name: accumulator.result() for name, accumulator in metrics.items()},
        "pixel_oracle_fraction": {
            name: float(pixel_choices[index] / pixel_choices.sum().clamp_min(1))
            for index, name in enumerate(ACTION_NAMES)
        },
        "frame_oracle_fraction": {
            name: float(frame_choices[index] / frame_choices.sum().clamp_min(1))
            for index, name in enumerate(ACTION_NAMES)
        },
        "per_clip": clip_rows,
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(args)
    paths = list(splits["test"])
    if args.max_test > 0:
        paths = paths[:args.max_test]
    dataset = VideoForecastDataset(
        args.cache_root, "test", args.image_size, paths=paths, cache_items=False,
    )
    loader = DataLoader(
        dataset, args.batch_size, collate_fn=video_pad_collate, num_workers=0,
    )
    base, trajectory, _, _ = load_frozen_models(args, device)
    checkpoint = torch.load(args.correction_checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    correction = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    correction.load_state_dict(checkpoint["model"])
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights, progress=True).to(device)
    transform = weights.transforms()
    for frozen in (base, trajectory, correction, raft):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    result = evaluate(base, trajectory, correction, raft, transform, loader, device)
    result["meta"] = {
        "image_size": args.image_size,
        "test_clips": len(dataset),
        "actions": ACTION_NAMES,
        "no_learned_router": True,
        "split": split_meta,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
