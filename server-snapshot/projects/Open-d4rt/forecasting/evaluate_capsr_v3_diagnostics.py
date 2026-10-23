"""Counterfactual path ablation and object-routing diagnosis for CaPSR v3."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_structured_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


BUCKETS = {
    "static_0_0.02": (0.0, 0.02),
    "small_0.02_0.05": (0.02, 0.05),
    "medium_0.05_0.10": (0.05, 0.10),
    "large_ge_0.10": (0.10, float("inf")),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--trajectory-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--correction-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    parser.add_argument("--motion-threshold", type=float, default=0.05)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_bucket(magnitude, lower, upper):
    selected = magnitude >= lower
    if math.isfinite(upper):
        selected = selected & (magnitude < upper)
    return selected


def empty_metric_state():
    return {
        "dynamic_abs": 0.0, "dynamic_sq": 0.0, "dynamic_count": 0.0,
        "tp": 0.0, "fp": 0.0, "fn": 0.0,
        "buckets": {
            name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in BUCKETS
        },
    }


def update_auxiliary(state, value, target, last, threshold):
    difference = value - target
    target_motion = (target - last).abs().mean(2, keepdim=True)
    target_dynamic = target_motion > threshold
    predicted_dynamic = (value - last).abs().mean(2, keepdim=True) > threshold
    expanded = target_dynamic.expand_as(target)
    state["dynamic_abs"] += difference.abs()[expanded].sum().item()
    state["dynamic_sq"] += difference.square()[expanded].sum().item()
    state["dynamic_count"] += expanded.sum().item()
    state["tp"] += (predicted_dynamic & target_dynamic).sum().item()
    state["fp"] += (predicted_dynamic & ~target_dynamic).sum().item()
    state["fn"] += (~predicted_dynamic & target_dynamic).sum().item()
    pixel_abs = difference.abs().mean(2, keepdim=True)
    pixel_sq = difference.square().mean(2, keepdim=True)
    for name, (lower, upper) in BUCKETS.items():
        selected = selected_bucket(target_motion, lower, upper)
        state["buckets"][name]["abs"] += pixel_abs[selected].sum().item()
        state["buckets"][name]["sq"] += pixel_sq[selected].sum().item()
        state["buckets"][name]["count"] += selected.sum().item()


def finish_auxiliary(state):
    dynamic_count = max(state["dynamic_count"], 1)
    dynamic_mse = state["dynamic_sq"] / dynamic_count
    result = {
        "dynamic_MAE": state["dynamic_abs"] / dynamic_count,
        "dynamic_MSE": dynamic_mse,
        "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
        "Motion_IoU": state["tp"] / max(state["tp"] + state["fp"] + state["fn"], 1),
        "Motion_F1": 2 * state["tp"] / max(2 * state["tp"] + state["fp"] + state["fn"], 1),
        "motion_magnitude_buckets": {},
    }
    for name, values in state["buckets"].items():
        count = max(values["count"], 1)
        mse = values["sq"] / count
        result["motion_magnitude_buckets"][name] = {
            "MAE": values["abs"] / count,
            "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "pixel_count": values["count"],
        }
    return result


def empty_diagnostic_state():
    keys = (
        "pixels", "raw_support", "after_cycle", "after_occlusion", "permission",
        "object_weight", "beneficial_raw", "beneficial_permission",
        "positive_gain_permission", "candidate_delta_permission",
        "gate_on_beneficial", "beneficial_permission_mass",
        "gate_on_adverse", "adverse_permission_mass",
    )
    return {name: {key: 0.0 for key in keys} for name in BUCKETS}


def update_diagnostics(storage, state, baseline, target, last):
    motion = (target - last).abs().mean(2, keepdim=True)
    base_error = (baseline - target).abs().mean(2, keepdim=True)
    object_error = (state["object_rgb"] - target).abs().mean(2, keepdim=True)
    beneficial = object_error + 0.002 < base_error
    positive_gain = (base_error - object_error).clamp_min(0)
    raw = state["raw_object_support"].clamp(0, 1)
    cycle = state["cycle_confidence"].clamp(0, 1)
    non_occluded = state["occlusion_confidence"].clamp(0, 1)
    after_cycle = raw * cycle
    after_occlusion = after_cycle * non_occluded
    permission = state["dynamic_object_permission"].clamp(0, 1)
    gate = state["object_gate"].clamp(0, 1)
    weight = state["object_weight"].clamp(0, 1)
    for name, (lower, upper) in BUCKETS.items():
        selected = selected_bucket(motion, lower, upper)
        target_values = storage[name]
        selected_f = selected.float()
        raw_selected = raw * selected_f
        permission_selected = permission * selected_f
        beneficial_f = beneficial.float()
        adverse_f = 1 - beneficial_f
        target_values["pixels"] += selected.sum().item()
        target_values["raw_support"] += raw_selected.sum().item()
        target_values["after_cycle"] += (after_cycle * selected_f).sum().item()
        target_values["after_occlusion"] += (after_occlusion * selected_f).sum().item()
        target_values["permission"] += permission_selected.sum().item()
        target_values["object_weight"] += (weight * selected_f).sum().item()
        target_values["beneficial_raw"] += (beneficial_f * raw_selected).sum().item()
        target_values["beneficial_permission"] += (
            beneficial_f * permission_selected
        ).sum().item()
        target_values["positive_gain_permission"] += (
            positive_gain * permission_selected
        ).sum().item()
        target_values["candidate_delta_permission"] += (
            (object_error - base_error) * permission_selected
        ).sum().item()
        beneficial_mass = beneficial_f * permission_selected
        adverse_mass = adverse_f * permission_selected
        target_values["gate_on_beneficial"] += (gate * beneficial_mass).sum().item()
        target_values["beneficial_permission_mass"] += beneficial_mass.sum().item()
        target_values["gate_on_adverse"] += (gate * adverse_mass).sum().item()
        target_values["adverse_permission_mass"] += adverse_mass.sum().item()


def finish_diagnostics(storage):
    result = {}
    for name, values in storage.items():
        pixels = max(values["pixels"], 1)
        raw = max(values["raw_support"], 1e-8)
        after_cycle = max(values["after_cycle"], 1e-8)
        after_occlusion = max(values["after_occlusion"], 1e-8)
        permission = max(values["permission"], 1e-8)
        result[name] = {
            "raw_support_fraction": values["raw_support"] / pixels,
            "cycle_filtered_fraction": 1 - values["after_cycle"] / raw,
            "occlusion_filtered_fraction_after_cycle": 1 - values["after_occlusion"] / after_cycle,
            "motion_disocclusion_filtered_fraction": 1 - values["permission"] / after_occlusion,
            "final_permission_fraction": values["permission"] / pixels,
            "final_object_weight_fraction": values["object_weight"] / pixels,
            "routing_fraction_of_permission": values["object_weight"] / permission,
            "candidate_beneficial_rate_on_raw_support": values["beneficial_raw"] / raw,
            "candidate_beneficial_rate_on_permission": values["beneficial_permission"] / permission,
            "positive_oracle_gain_on_permission": values["positive_gain_permission"] / permission,
            "candidate_delta_MAE_on_permission": values["candidate_delta_permission"] / permission,
            "gate_on_beneficial": values["gate_on_beneficial"] / max(
                values["beneficial_permission_mass"], 1e-8
            ),
            "gate_on_adverse": values["gate_on_adverse"] / max(
                values["adverse_permission_mass"], 1e-8
            ),
            "pixel_count": values["pixels"],
        }
    return result


@torch.no_grad()
def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits, split_meta = combined_splits(args)
    dataset = VideoForecastDataset(
        args.cache_root, "test", 64, paths=splits["test"], cache_items=False
    )
    if args.max_test > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_test, len(dataset))))
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(args, device)
    checkpoint = torch.load(args.correction_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    future_frames = trajectory_checkpoint["future_frames"]
    names = ("full", "no_object", "no_camera", "no_synthesis")
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in names
    }
    auxiliary = {name: empty_metric_state() for name in names}
    diagnostics = empty_diagnostic_state()
    for index, batch in enumerate(loader, 1):
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        full, state = model(
            baseline, batch["context_rgb"][:, -1], xyz, uv,
            batch["past_xyz"].float(), batch["past_uv"].float(),
            batch["past_valid"], batch["past_confidence"].float(), mode="structured",
        )
        synthesis = (baseline + state["synthesis_residual"]).clamp(0, 1)
        variants = {
            "full": full,
            "no_object": (full + state["object_weight"] * (baseline - state["object_rgb"])).clamp(0, 1),
            "no_camera": (full + state["camera_weight"] * (baseline - state["camera_rgb"])).clamp(0, 1),
            "no_synthesis": (full + state["synthesis_weight"] * (baseline - synthesis)).clamp(0, 1),
        }
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        for name, value in variants.items():
            accumulators[name].update(value, target, batch["context_rgb"][:, -1])
            update_auxiliary(auxiliary[name], value, target, last, args.motion_threshold)
        update_diagnostics(diagnostics, state, baseline, target, last)
        if index % 25 == 0 or index == len(loader):
            print(json.dumps({"processed": index, "total": len(loader)}), flush=True)
    results = {}
    for name in names:
        results[name] = accumulators[name].compute()
        results[name].update(finish_auxiliary(auxiliary[name]))
    output = {
        "path_ablation": results,
        "object_routing_diagnostics": finish_diagnostics(diagnostics),
        "meta": {
            "checkpoint": args.correction_checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "test_clips": len(dataset),
            "split": split_meta,
            "ablation_rule": "disabled path weight is reassigned to the frozen base prediction",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
