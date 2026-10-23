#!/usr/bin/env python3
"""Conservatively add Router-v1 object transport to the frozen v5 prediction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR as CalibratedCaPSR
from capsr_multipath_router_model import StructuredCaPSR as MultipathCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--trajectory-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--v5-checkpoint", required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    parser.add_argument("--motion-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(0.035, 48, num_motion_clusters=4).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def collect(loader, base, trajectory, v5, router, device):
    records = []
    for index, batch in enumerate(loader):
        names = list(batch["name"])
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        arguments = (
            baseline, batch["context_rgb"][:, -1], xyz, uv,
            batch["past_xyz"].float(), batch["past_uv"].float(),
            batch["past_valid"], batch["past_confidence"].float(),
        )
        v5_prediction, _ = v5(*arguments)
        _, router_state = router(*arguments)
        object_candidate = router_state["candidate_stack"][:, :, 2]
        geometric_confidence = (
            router_state["object_weight"].clamp(0, 1)
            * router_state["dynamic_object_permission"].clamp(0, 1)
            * router_state["cycle_confidence"].clamp(0, 1)
            * router_state["occlusion_confidence"].clamp(0, 1)
        ).clamp_min(0).pow(0.25)
        support = (
            (router_state["object_support"] > 1e-4)
            & (router_state["dynamic_object_permission"] > 1e-4)
        )
        records.append({
            "name": names,
            "v5": v5_prediction.cpu(),
            "object": object_candidate.cpu(),
            "confidence": geometric_confidence.cpu(),
            "support": support.cpu(),
            "target": batch["future_rgb"].cpu(),
            "last": batch["context_rgb"][:, -1:].cpu(),
        })
        if (index + 1) % 25 == 0:
            print(json.dumps({"collected": index + 1, "total": len(loader.dataset)}), flush=True)
    return records


def predict(record, threshold, alpha):
    selected = record["support"] & (record["confidence"] >= threshold)
    gate = selected.float() * float(alpha)
    prediction = (record["v5"] + gate * (record["object"] - record["v5"])).clamp(0, 1)
    return prediction, selected


def cheap_metrics(records, threshold, alpha, motion_threshold):
    abs_sum = sq_sum = count = 0.0
    dynamic_abs = dynamic_sq = dynamic_count = 0.0
    temporal_abs = temporal_count = selected_count = support_count = 0.0
    for record in records:
        prediction, selected = predict(record, threshold, alpha)
        target, last = record["target"], record["last"]
        difference = prediction - target
        abs_sum += difference.abs().sum().item()
        sq_sum += difference.square().sum().item()
        count += difference.numel()
        dynamic = ((target - last).abs().mean(2, keepdim=True) > motion_threshold).expand_as(difference)
        dynamic_abs += difference.abs()[dynamic].sum().item()
        dynamic_sq += difference.square()[dynamic].sum().item()
        dynamic_count += dynamic.sum().item()
        previous_prediction = torch.cat((last, prediction[:, :-1]), 1)
        previous_target = torch.cat((last, target[:, :-1]), 1)
        temporal = (prediction - previous_prediction) - (target - previous_target)
        temporal_abs += temporal.abs().sum().item()
        temporal_count += temporal.numel()
        selected_count += selected.sum().item()
        support_count += record["support"].sum().item()
    return {
        "MAE": abs_sum / count, "MSE": sq_sum / count,
        "dynamic_MAE": dynamic_abs / max(dynamic_count, 1),
        "dynamic_MSE": dynamic_sq / max(dynamic_count, 1),
        "temporal_delta_MAE": temporal_abs / temporal_count,
        "selected_fraction": selected_count / (count / 3),
        "selected_fraction_on_object_support": selected_count / max(support_count, 1),
    }


@torch.no_grad()
def full_metrics(records, threshold, alpha, motion_threshold, device, future_frames):
    accumulator = VideoMetricAccumulator(device, future_frames)
    dynamic_abs = dynamic_sq = dynamic_count = 0.0
    tp = fp = fn = selected_count = support_count = 0.0
    for record in records:
        prediction_cpu, selected = predict(record, threshold, alpha)
        prediction = prediction_cpu.to(device)
        target = record["target"].to(device)
        last = record["last"].to(device)
        accumulator.update(prediction, target, last[:, 0])
        difference = prediction - target
        target_motion = (target - last).abs().mean(2, keepdim=True) > motion_threshold
        predicted_motion = (prediction - last).abs().mean(2, keepdim=True) > motion_threshold
        dynamic = target_motion.expand_as(difference)
        dynamic_abs += difference.abs()[dynamic].sum().item()
        dynamic_sq += difference.square()[dynamic].sum().item()
        dynamic_count += dynamic.sum().item()
        tp += (predicted_motion & target_motion).sum().item()
        fp += (predicted_motion & ~target_motion).sum().item()
        fn += (~predicted_motion & target_motion).sum().item()
        selected_count += selected.sum().item()
        support_count += record["support"].sum().item()
    metrics = accumulator.compute()
    dynamic_mse = dynamic_sq / max(dynamic_count, 1)
    metrics.update({
        "dynamic_MAE": dynamic_abs / max(dynamic_count, 1),
        "dynamic_MSE": dynamic_mse,
        "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
        "Motion_IoU": tp / max(tp + fp + fn, 1),
        "Motion_F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "selected_fraction": selected_count / max(accumulator.count / 3, 1),
        "selected_fraction_on_object_support": selected_count / max(support_count, 1),
        "threshold": float(threshold), "alpha": float(alpha),
    })
    return metrics


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
    make_loader = lambda split: DataLoader(
        VideoForecastDataset(args.cache_root, split, 64, paths=splits[split], cache_items=False),
        1, collate_fn=video_pad_collate, num_workers=0,
    )
    val_loader, test_loader = make_loader("val"), make_loader("test")
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(args, device)
    v5, v5_checkpoint = load_model(CalibratedCaPSR, args.v5_checkpoint, device)
    router, router_checkpoint = load_model(MultipathCaPSR, args.router_checkpoint, device)
    val_records = collect(val_loader, base, trajectory, v5, router, device)
    positive_scores = torch.cat([
        record["confidence"][record["support"]] for record in val_records
        if record["support"].any()
    ])
    quantiles = (0.70, 0.80, 0.90, 0.95, 0.975)
    thresholds = [float(torch.quantile(positive_scores, quantile)) for quantile in quantiles]
    base_metrics = cheap_metrics(val_records, float("inf"), 0.0, args.motion_threshold)
    sweep = [{"quantile": 1.0, "threshold": float("inf"), "alpha": 0.0, **base_metrics}]
    for quantile, threshold in zip(quantiles, thresholds):
        for alpha in (0.25, 0.50, 0.75, 1.0):
            sweep.append({
                "quantile": quantile, "threshold": threshold, "alpha": alpha,
                **cheap_metrics(val_records, threshold, alpha, args.motion_threshold),
            })
    feasible = [
        row for row in sweep[1:]
        if row["MAE"] <= base_metrics["MAE"] * 1.002
        and row["MSE"] <= base_metrics["MSE"] * 1.002
        and row["temporal_delta_MAE"] <= base_metrics["temporal_delta_MAE"] * 1.01
    ]
    if feasible:
        selected = min(feasible, key=lambda row: row["dynamic_MSE"])
        selection_rule = "minimum dynamic MSE under global/temporal preservation constraints"
    else:
        selected = min(
            sweep,
            key=lambda row: row["MSE"] + 0.25 * row["dynamic_MSE"]
            + 0.02 * row["temporal_delta_MAE"],
        )
        selection_rule = "fallback weighted validation score"
    print(json.dumps({"selected": selected, "rule": selection_rule}), flush=True)
    test_records = collect(test_loader, base, trajectory, v5, router, device)
    test_v5 = full_metrics(
        test_records, float("inf"), 0.0, args.motion_threshold, device,
        trajectory_checkpoint["future_frames"],
    )
    test_hybrid = full_metrics(
        test_records, selected["threshold"], selected["alpha"],
        args.motion_threshold, device, trajectory_checkpoint["future_frames"],
    )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summary = {
        "validation_sweep": sweep,
        "selected_validation_configuration": selected,
        "selection_rule": selection_rule,
        "test": {"v5": test_v5, "v5_object_hybrid": test_hybrid},
        "meta": {
            "split": split_meta, "v5_epoch": v5_checkpoint.get("epoch"),
            "router_epoch": router_checkpoint.get("epoch"),
            "confidence": "fourth_root(router_object_weight * object_permission * cycle * non_occlusion)",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
