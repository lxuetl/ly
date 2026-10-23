#!/usr/bin/env python3
"""Evaluate one frozen CaPSR variant and retain sequence-level sufficient statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR as CalibratedCaPSR
from capsr_multipath_router_model import StructuredCaPSR as MultipathCaPSR
from capsr_stable_router_model import StructuredCaPSR as StableCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v5", "router_v1", "stable_v2"), required=True)
    parser.add_argument("--mode", choices=("structured", "no_geometry"), default="structured")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--trajectory-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    parser.add_argument("--motion-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def domain_from_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/po50/" in text:
        return "po50"
    for name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
        if f"/{name}/" in text:
            return name
    return path.parent.name


def build_model(variant: str, checkpoint: dict, device: torch.device):
    classes = {
        "v5": CalibratedCaPSR,
        "router_v1": MultipathCaPSR,
        "stable_v2": StableCaPSR,
    }
    model = classes[variant](0.035, 48, num_motion_clusters=4).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_args = SimpleNamespace(**vars(args))
    splits, split_meta = combined_splits(split_args)
    dataset = VideoForecastDataset(
        args.cache_root, "test", 64, paths=splits["test"], cache_items=False
    )
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.variant, checkpoint, device)

    import lpips
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    rows = []
    for index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        prediction, state = model(
            baseline, batch["context_rgb"][:, -1], xyz, uv,
            batch["past_xyz"].float(), batch["past_uv"].float(),
            batch["past_valid"], batch["past_confidence"].float(), mode=args.mode,
        )
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        difference = prediction - target
        temporal_previous_prediction = torch.cat((last, prediction[:, :-1]), 1)
        temporal_previous_target = torch.cat((last, target[:, :-1]), 1)
        temporal_error = (
            (prediction - temporal_previous_prediction)
            - (target - temporal_previous_target)
        )
        target_motion = (target - last).abs().mean(2, keepdim=True)
        target_motion_mask = target_motion > args.motion_threshold
        predicted_motion_mask = (
            (prediction - last).abs().mean(2, keepdim=True) > args.motion_threshold
        )
        dynamic = target_motion_mask.expand_as(difference)
        dynamic_abs_sum = difference.abs()[dynamic].sum().item()
        dynamic_sq_sum = difference.square()[dynamic].sum().item()
        dynamic_count = dynamic.sum().item()
        tp = (predicted_motion_mask & target_motion_mask).sum().item()
        fp = (predicted_motion_mask & ~target_motion_mask).sum().item()
        fn = (~predicted_motion_mask & target_motion_mask).sum().item()

        pred_np = prediction[0].detach().cpu().numpy()
        target_np = target[0].detach().cpu().numpy()
        ssim_values = [
            structural_similarity(
                target_np[step].transpose(1, 2, 0),
                pred_np[step].transpose(1, 2, 0),
                channel_axis=-1, data_range=1.0,
            )
            for step in range(pred_np.shape[0])
        ]
        flat_prediction = prediction.flatten(0, 1) * 2 - 1
        flat_target = target.flatten(0, 1) * 2 - 1
        lpips_value = perceptual(flat_prediction, flat_target).mean().item()
        mse = difference.square().mean().item()
        dynamic_mse = dynamic_sq_sum / max(dynamic_count, 1)
        path = Path(dataset.paths[index])
        row = {
            "name": batch["name"][0],
            "domain": domain_from_path(path),
            "cache_path": str(path),
            "MAE": difference.abs().mean().item(),
            "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "SSIM": float(np.mean(ssim_values)),
            "LPIPS": lpips_value,
            "temporal_delta_MAE": temporal_error.abs().mean().item(),
            "dynamic_abs_sum": dynamic_abs_sum,
            "dynamic_sq_sum": dynamic_sq_sum,
            "dynamic_count": dynamic_count,
            "dynamic_MAE": dynamic_abs_sum / max(dynamic_count, 1),
            "dynamic_MSE": dynamic_mse,
            "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
            "tp": tp, "fp": fp, "fn": fn,
            "Motion_IoU": tp / max(tp + fp + fn, 1),
            "Motion_F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "camera_fraction": state["camera_weight"].mean().item(),
            "object_fraction": state["object_weight"].mean().item(),
            "synthesis_fraction": state["synthesis_weight"].mean().item(),
        }
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(json.dumps({"variant": args.variant, "done": index + 1, "total": len(dataset)}), flush=True)

    payload = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "future_frames": trajectory_checkpoint["future_frames"],
        "split": split_meta,
        "sequences": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"variant": args.variant, "completed": len(rows), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
