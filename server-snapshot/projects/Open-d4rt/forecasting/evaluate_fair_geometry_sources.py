#!/usr/bin/env python3
"""Evaluate independently trained D4RT, affine, and RAFT correction models."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from evaluate_camera_source_judgement import fit_global_transform, last_two_valid, repeat_static_xyz
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from train_unified_action_risk import raft_history, structured_candidate
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


NAMES = ("baseline", "d4rt", "sparse_affine", "raft_small")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--d4rt-correction", required=True)
    p.add_argument("--affine-correction", required=True)
    p.add_argument("--raft-correction", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--max-test", type=int, default=48)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def domain(path):
    text = str(path).replace("\\", "/")
    if "/po50/" in text:
        return "po50"
    for name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
        if f"/{name}/" in text:
            return name
    return Path(path).parent.name


class Metrics:
    def __init__(self):
        self.v = defaultdict(float)

    def update(self, prediction, target, last):
        difference = prediction - target
        dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).expand_as(difference)
        self.v["abs"] += difference.abs().sum().item()
        self.v["sq"] += difference.square().sum().item()
        self.v["count"] += difference.numel()
        self.v["dyn_abs"] += difference.abs()[dynamic].sum().item()
        self.v["dyn_sq"] += difference.square()[dynamic].sum().item()
        self.v["dyn_count"] += dynamic.sum().item()

    def result(self):
        count, dynamic_count = max(self.v["count"], 1), max(self.v["dyn_count"], 1)
        mse = self.v["sq"] / count
        dynamic_mse = self.v["dyn_sq"] / dynamic_count
        return {
            "MAE": self.v["abs"] / count,
            "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "dynamic_MAE": self.v["dyn_abs"] / dynamic_count,
            "dynamic_MSE": dynamic_mse,
            "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
        }


def load_correction(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def bootstrap(rows, seed=42, samples=20000):
    rng = np.random.default_rng(seed)
    size = len(rows)
    indices = rng.integers(0, size, (samples, size))
    arrays = {name: np.asarray([row[name + "_MSE"] for row in rows]) for name in NAMES}
    result = {}
    for name in NAMES[1:]:
        delta = arrays[name] - arrays["baseline"]
        distribution = delta[indices].mean(1)
        result[name + "_vs_baseline"] = {
            "mean_delta": float(delta.mean()),
            "relative_improvement_percent": float(-100 * delta.mean() / arrays["baseline"].mean()),
            "ci95_delta": [float(value) for value in np.quantile(distribution, (0.025, 0.975))],
            "clips_better": int((delta < 0).sum()),
            "clips": size,
        }
    for name in ("sparse_affine", "raft_small"):
        delta = arrays[name] - arrays["d4rt"]
        distribution = delta[indices].mean(1)
        result[name + "_vs_d4rt"] = {
            "mean_delta": float(delta.mean()),
            "relative_improvement_percent": float(-100 * delta.mean() / arrays["d4rt"].mean()),
            "ci95_delta": [float(value) for value in np.quantile(distribution, (0.025, 0.975))],
            "clips_better": int((delta < 0).sum()),
            "clips": size,
        }
    return result


@torch.no_grad()
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
    dataset = VideoForecastDataset(args.cache_root, "test", args.image_size, paths=paths, cache_items=False)
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, _ = load_frozen_models(args, device)
    models = {
        "d4rt": load_correction(args.d4rt_correction, device),
        "sparse_affine": load_correction(args.affine_correction, device),
        "raft_small": load_correction(args.raft_correction, device),
    }
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights, progress=True).to(device).eval()
    transform = weights.transforms()
    for parameter in raft.parameters():
        parameter.requires_grad_(False)

    metrics = {name: Metrics() for name in (*NAMES, "pixel_oracle", "frame_oracle")}
    domain_metrics = defaultdict(lambda: {name: Metrics() for name in NAMES})
    pixel_choice = torch.zeros(len(NAMES), device=device)
    frame_choice = torch.zeros(len(NAMES), device=device)
    rows = []
    for index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, xyz, d4rt_uv = frozen_prediction(base, trajectory, batch)
        previous_uv, anchor_uv, anchor_xyz, pair_valid = last_two_valid(batch)
        future_frames = d4rt_uv.shape[2]
        static_xyz = repeat_static_xyz(anchor_xyz, future_frames)
        affine_uv = fit_global_transform(previous_uv, anchor_uv, pair_valid, future_frames, "affine")
        raft_uv, _, _, _ = raft_history(
            raft, transform, batch["context_rgb"][:, -2], batch["context_rgb"][:, -1],
            anchor_uv, future_frames,
        )
        last = batch["context_rgb"][:, -1]
        candidates = (
            baseline,
            structured_candidate(models["d4rt"], baseline, last, batch, xyz, d4rt_uv),
            structured_candidate(models["sparse_affine"], baseline, last, batch, static_xyz, affine_uv),
            structured_candidate(models["raft_small"], baseline, last, batch, static_xyz, raft_uv),
        )
        actions = torch.stack(candidates, 2)
        target = batch["future_rgb"]
        source_domain = domain(paths[index])
        row = {"clip_index": index, "name": batch["name"][0], "domain": source_domain}
        for action_index, name in enumerate(NAMES):
            value = actions[:, :, action_index]
            metrics[name].update(value, target, last)
            domain_metrics[source_domain][name].update(value, target, last)
            row[name + "_MSE"] = float((value - target).square().mean())

        squared = (actions - target[:, :, None]).square()
        pixel_error = squared.mean(3, keepdim=True)
        pixel_index = pixel_error.argmin(2, keepdim=True)
        pixel_oracle = actions.gather(2, pixel_index.expand(-1, -1, -1, 3, -1, -1)).squeeze(2)
        metrics["pixel_oracle"].update(pixel_oracle, target, last)
        pixel_choice.scatter_add_(0, pixel_index.flatten(), torch.ones_like(pixel_index.flatten(), dtype=torch.float))
        frame_error = squared.mean((3, 4, 5))
        frame_index = frame_error.argmin(2)
        frame_oracle = actions.gather(
            2, frame_index[:, :, None, None, None, None].expand(-1, -1, -1, 3, actions.shape[-2], actions.shape[-1]),
        ).squeeze(2)
        metrics["frame_oracle"].update(frame_oracle, target, last)
        frame_choice.scatter_add_(0, frame_index.flatten(), torch.ones_like(frame_index.flatten(), dtype=torch.float))
        rows.append(row)

    result = {
        "metrics": {name: accumulator.result() for name, accumulator in metrics.items()},
        "per_domain": {
            source: {name: accumulator.result() for name, accumulator in values.items()}
            for source, values in domain_metrics.items()
        },
        "pixel_oracle_fraction": {
            name: float(pixel_choice[i] / pixel_choice.sum().clamp_min(1)) for i, name in enumerate(NAMES)
        },
        "frame_oracle_fraction": {
            name: float(frame_choice[i] / frame_choice.sum().clamp_min(1)) for i, name in enumerate(NAMES)
        },
        "paired_bootstrap": bootstrap(rows, args.seed),
        "per_clip": rows,
        "meta": {"image_size": args.image_size, "test_clips": len(dataset), "split": split_meta},
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
