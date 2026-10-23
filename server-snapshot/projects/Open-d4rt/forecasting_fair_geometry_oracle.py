#!/usr/bin/env python3
"""Measure geometry candidate headroom on top of the fair SimVP baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "forecasting"))
sys.path.insert(0, str(HERE))

from capsr_stable_router_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--geometry-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pixel_oracle(target, candidates):
    stack = torch.stack(candidates, dim=2)
    cost = (stack - target[:, :, None]).square().mean(3)
    choice = cost.argmin(2, keepdim=True)
    selected = stack.gather(
        2, choice[:, :, :, None].expand(-1, -1, -1, 3, -1, -1)
    ).squeeze(2)
    return selected, choice.squeeze(2), cost


def main():
    a = arguments()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(SimpleNamespace(**vars(a)))
    paths = splits["test"][: a.max_test or None]
    dataset = VideoForecastDataset(
        a.cache_root, "test", 64, paths=paths, cache_items=False
    )
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(a, device)
    checkpoint = torch.load(a.geometry_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    geometry = StructuredCaPSR(
        float(config.get("sigma", 0.035)),
        int(config.get("hidden", 48)),
        num_motion_clusters=int(config.get("num_motion_clusters", 4)),
    ).to(device)
    geometry.load_state_dict(checkpoint["model"])
    geometry.eval()
    for parameter in geometry.parameters():
        parameter.requires_grad_(False)

    k = int(trajectory_checkpoint["future_frames"])
    names = (
        "fair_simvp", "trained_router", "camera", "object", "synthesis",
        "oracle_base_object", "oracle_all",
    )
    accumulators = {name: VideoMetricAccumulator(device, k) for name in names}
    dynamic = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in names}
    selected = torch.zeros(4, dtype=torch.float64)
    selected_dynamic = torch.zeros(4, dtype=torch.float64)
    total_pixels = dynamic_pixels = 0.0
    base_minus_oracle = 0.0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = move(batch, device)
            base_prediction, xyz, uv = frozen_prediction(base, trajectory, batch)
            routed, state = geometry(
                base_prediction, batch["context_rgb"][:, -1], xyz, uv,
                batch["past_xyz"].float(), batch["past_uv"].float(),
                batch["past_valid"], batch["past_confidence"].float(),
                mode="structured",
            )
            stack = state["candidate_stack"]
            candidates = [stack[:, :, i] for i in range(4)]
            object_oracle, object_choice, _ = pixel_oracle(
                batch["future_rgb"], [candidates[0], candidates[2]]
            )
            all_oracle, all_choice, all_cost = pixel_oracle(
                batch["future_rgb"], candidates
            )
            target = batch["future_rgb"]
            last = batch["context_rgb"][:, -1:]
            motion = (target - last).abs().mean(2, keepdim=True) > 0.05
            predictions = {
                "fair_simvp": base_prediction,
                "trained_router": routed,
                "camera": candidates[1],
                "object": candidates[2],
                "synthesis": candidates[3],
                "oracle_base_object": object_oracle,
                "oracle_all": all_oracle,
            }
            for name, prediction in predictions.items():
                accumulators[name].update(prediction, target, batch["context_rgb"][:, -1])
                difference = prediction - target
                mask = motion.expand_as(target)
                dynamic[name]["abs"] += difference.abs()[mask].sum().item()
                dynamic[name]["sq"] += difference.square()[mask].sum().item()
                dynamic[name]["count"] += mask.sum().item()
            for path in range(4):
                chosen = all_choice == path
                selected[path] += chosen.sum().cpu()
                selected_dynamic[path] += (chosen & motion.squeeze(2)).sum().cpu()
            total_pixels += all_choice.numel()
            dynamic_pixels += motion.sum().item()
            base_minus_oracle += (all_cost[:, :, 0] - all_cost.amin(2)).sum().item()
            if (index + 1) % 25 == 0:
                print(json.dumps({"processed": index + 1, "total": len(dataset)}), flush=True)

    metrics = {}
    for name, accumulator in accumulators.items():
        values = accumulator.compute()
        count = max(dynamic[name]["count"], 1.0)
        dmse = dynamic[name]["sq"] / count
        values.update({
            "dynamic_MAE": dynamic[name]["abs"] / count,
            "dynamic_MSE": dmse,
            "dynamic_PSNR": -10.0 * math.log10(max(dmse, 1e-12)),
        })
        metrics[name] = values
    result = {
        "protocol": {
            "test_clips": len(dataset), "split": split_meta,
            "backbone_checkpoint": a.backbone_checkpoint,
            "geometry_checkpoint": a.geometry_checkpoint,
            "training": False,
        },
        "metrics": metrics,
        "oracle_path_fraction": {
            name: float(selected[i] / max(total_pixels, 1.0))
            for i, name in enumerate(("base", "camera", "object", "synthesis"))
        },
        "oracle_dynamic_path_fraction": {
            name: float(selected_dynamic[i] / max(dynamic_pixels, 1.0))
            for i, name in enumerate(("base", "camera", "object", "synthesis"))
        },
        "oracle_mean_pixel_mse_headroom": base_minus_oracle / max(total_pixels, 1.0),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
