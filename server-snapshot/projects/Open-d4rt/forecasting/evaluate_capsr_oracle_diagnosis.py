"""Diagnose CaPSR with future-coordinate and candidate-selection oracles.

No parameter is trained.  GT UV comes from the source WorldTrack tracks;
teacher XYZ comes from the cached full-context D4RT teacher aligned to the
context-only D4RT gauge.  Unavailable teacher XYZ is replaced by the predicted
value so each comparison changes only labelled coordinates.
"""

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

from capsr_model import _last_valid
from capsr_structured_model_v3 import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--capsr-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-test", type=int, default=0)
    return p.parse_args()


def capsr_forward(model, base, xyz, uv, batch):
    return model(
        base, batch["context_rgb"][:, -1], xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(),
        batch["past_valid"], batch["past_confidence"].float(),
        mode="structured",
    )


def pixel_oracle(target, candidates):
    stack = torch.stack(candidates, 0)
    error = (stack - target[None]).square().mean(3, keepdim=True)
    choice = error.argmin(0, keepdim=True)
    return stack.gather(0, choice.expand(-1, -1, -1, 3, -1, -1)).squeeze(0)


def main():
    a = parse_args()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    splits, split_meta = combined_splits(a)
    paths = splits["test"][:a.max_test or None]
    dataset = VideoForecastDataset(a.cache_root, "test", 64, paths=paths, cache_items=False)
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base_model, trajectory, _, trajectory_checkpoint = load_frozen_models(a, device)
    capsr_checkpoint = torch.load(a.capsr_checkpoint, map_location=device)
    config = capsr_checkpoint.get("config", {})
    model = StructuredCaPSR(
        sigma=float(config.get("sigma", 0.035)),
        hidden=int(config.get("hidden", 48)),
        num_motion_clusters=int(config.get("num_motion_clusters", 4)),
    ).to(device)
    model.load_state_dict(capsr_checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    k = int(trajectory_checkpoint["future_frames"])
    names = (
        "base", "capsr_predicted", "capsr_gt_uv", "capsr_teacher_xyz",
        "capsr_gt_uv_teacher_xyz", "oracle_base_object_predicted",
        "oracle_base_object_teacher", "oracle_all_predicted", "oracle_all_teacher",
    )
    accumulators = {name: VideoMetricAccumulator(device, k) for name in names}
    dynamic = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in names}
    buckets = {
        "static_0_0.02": (0.0, 0.02), "small_0.02_0.05": (0.02, 0.05),
        "medium_0.05_0.10": (0.05, 0.10), "large_ge_0.10": (0.10, float("inf")),
    }
    bucket_stats = {
        name: {bucket: {"abs": 0.0, "sq": 0.0, "count": 0.0} for bucket in buckets}
        for name in names
    }
    benefit = {
        geometry: {
            bucket: {"beneficial": 0.0, "count": 0.0, "gain": 0.0}
            for bucket in buckets
        } for geometry in ("predicted", "teacher")
    }
    benefit_horizon = {
        geometry: [{"beneficial": 0.0, "count": 0.0, "gain": 0.0} for _ in range(k)]
        for geometry in ("predicted", "teacher")
    }
    uv_sum = uv_count = xyz_sum = xyz_count = 0.0
    uv_horizon_sum = torch.zeros(k)
    uv_horizon_count = torch.zeros(k)
    xyz_horizon_sum = torch.zeros(k)
    xyz_horizon_count = torch.zeros(k)

    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = move(batch, device)
            base, predicted_xyz, predicted_uv = frozen_prediction(base_model, trajectory, batch)
            finite_uv = torch.isfinite(batch["future_uv_teacher"]).all(-1)
            gt_uv = torch.nan_to_num(batch["future_uv_teacher"].float()).clamp(0, 1)
            gt_uv = torch.where(finite_uv[..., None], gt_uv, predicted_uv)
            finite_xyz = torch.isfinite(batch["future_xyz_teacher"]).all(-1)
            teacher_xyz_valid = batch["future_valid"].bool() & finite_xyz
            teacher_xyz = torch.nan_to_num(batch["future_xyz_teacher"].float())
            teacher_xyz = torch.where(
                teacher_xyz_valid[..., None], teacher_xyz, predicted_xyz
            )

            predicted_output, predicted_state = capsr_forward(
                model, base, predicted_xyz, predicted_uv, batch
            )
            uv_output, _ = capsr_forward(model, base, predicted_xyz, gt_uv, batch)
            xyz_output, _ = capsr_forward(model, base, teacher_xyz, predicted_uv, batch)
            teacher_output, teacher_state = capsr_forward(
                model, base, teacher_xyz, gt_uv, batch
            )
            target = batch["future_rgb"]
            teacher_synthesis = (
                base + teacher_state["synthesis_residual"]
            ).clamp(0, 1)
            predicted_synthesis = (
                base + predicted_state["synthesis_residual"]
            ).clamp(0, 1)
            predictions = {
                "base": base,
                "capsr_predicted": predicted_output,
                "capsr_gt_uv": uv_output,
                "capsr_teacher_xyz": xyz_output,
                "capsr_gt_uv_teacher_xyz": teacher_output,
                "oracle_base_object_predicted": pixel_oracle(
                    target, [base, predicted_state["object_rgb"]]
                ),
                "oracle_base_object_teacher": pixel_oracle(
                    target, [base, teacher_state["object_rgb"]]
                ),
                "oracle_all_predicted": pixel_oracle(target, [
                    base, predicted_state["camera_rgb"], predicted_state["object_rgb"],
                    predicted_synthesis,
                ]),
                "oracle_all_teacher": pixel_oracle(target, [
                    base, teacher_state["camera_rgb"], teacher_state["object_rgb"],
                    teacher_synthesis,
                ]),
            }
            last = batch["context_rgb"][:, -1:]
            target_motion = (target - last).abs().mean(2, keepdim=True)
            dynamic_mask = target_motion > 0.05
            for name, prediction in predictions.items():
                accumulators[name].update(prediction, target, batch["context_rgb"][:, -1])
                difference = prediction - target
                selected_dynamic = dynamic_mask.expand_as(target)
                dynamic[name]["abs"] += difference.abs()[selected_dynamic].sum().item()
                dynamic[name]["sq"] += difference.square()[selected_dynamic].sum().item()
                dynamic[name]["count"] += selected_dynamic.sum().item()
                pixel_abs = difference.abs().mean(2, keepdim=True)
                pixel_sq = difference.square().mean(2, keepdim=True)
                for bucket, (lower, upper) in buckets.items():
                    selected = target_motion >= lower
                    if math.isfinite(upper):
                        selected &= target_motion < upper
                    bucket_stats[name][bucket]["abs"] += pixel_abs[selected].sum().item()
                    bucket_stats[name][bucket]["sq"] += pixel_sq[selected].sum().item()
                    bucket_stats[name][bucket]["count"] += selected.sum().item()

            base_error = (base - target).square().mean(2, keepdim=True)
            for geometry, state in (("predicted", predicted_state), ("teacher", teacher_state)):
                object_error = (state["object_rgb"] - target).square().mean(2, keepdim=True)
                is_beneficial = object_error + 1e-5 < base_error
                gain = base_error - object_error
                for bucket, (lower, upper) in buckets.items():
                    selected = target_motion >= lower
                    if math.isfinite(upper):
                        selected &= target_motion < upper
                    benefit[geometry][bucket]["beneficial"] += is_beneficial[selected].sum().item()
                    benefit[geometry][bucket]["gain"] += gain[selected].sum().item()
                    benefit[geometry][bucket]["count"] += selected.sum().item()
                for step in range(k):
                    values = benefit_horizon[geometry][step]
                    values["beneficial"] += is_beneficial[:, step].sum().item()
                    values["gain"] += gain[:, step].sum().item()
                    values["count"] += is_beneficial[:, step].numel()

            uv_error = torch.linalg.vector_norm(predicted_uv - gt_uv, dim=-1)
            uv_valid = finite_uv & batch["past_valid"].any(2)[:, :, None]
            xyz_error = torch.linalg.vector_norm(predicted_xyz - teacher_xyz, dim=-1)
            uv_sum += uv_error[uv_valid].sum().item()
            uv_count += uv_valid.sum().item()
            xyz_sum += xyz_error[teacher_xyz_valid].sum().item()
            xyz_count += teacher_xyz_valid.sum().item()
            for step in range(k):
                uv_horizon_sum[step] += uv_error[:, :, step][uv_valid[:, :, step]].sum().cpu()
                uv_horizon_count[step] += uv_valid[:, :, step].sum().cpu()
                xyz_horizon_sum[step] += xyz_error[:, :, step][teacher_xyz_valid[:, :, step]].sum().cpu()
                xyz_horizon_count[step] += teacher_xyz_valid[:, :, step].sum().cpu()
            if (index + 1) % 25 == 0:
                print(json.dumps({"processed": index + 1, "total": len(dataset)}), flush=True)

    metrics = {}
    for name, accumulator in accumulators.items():
        values = accumulator.compute()
        count = max(dynamic[name]["count"], 1)
        dmse = dynamic[name]["sq"] / count
        values.update({
            "dynamic_MAE": dynamic[name]["abs"] / count,
            "dynamic_MSE": dmse,
            "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)),
            "motion_magnitude_buckets": {},
        })
        for bucket, stats in bucket_stats[name].items():
            count = max(stats["count"], 1)
            mse = stats["sq"] / count
            values["motion_magnitude_buckets"][bucket] = {
                "MAE": stats["abs"] / count, "MSE": mse,
                "PSNR": -10 * math.log10(max(mse, 1e-12)),
                "pixel_count": stats["count"],
            }
        metrics[name] = values

    benefit_result = {}
    for geometry, geometry_values in benefit.items():
        benefit_result[geometry] = {}
        for bucket, values in geometry_values.items():
            count = max(values["count"], 1)
            benefit_result[geometry][bucket] = {
                "beneficial_fraction": values["beneficial"] / count,
                "mean_base_minus_object_MSE": values["gain"] / count,
                "pixel_count": values["count"],
            }
        benefit_result[geometry]["per_horizon"] = [
            {
                "horizon": step + 1,
                "beneficial_fraction": values["beneficial"] / max(values["count"], 1),
                "mean_base_minus_object_MSE": values["gain"] / max(values["count"], 1),
            } for step, values in enumerate(benefit_horizon[geometry])
        ]

    result = {
        "protocol": {
            "training": False,
            "num_test_clips": len(dataset),
            "capsr_checkpoint": a.capsr_checkpoint,
            "trajectory_checkpoint": a.trajectory_checkpoint,
            "gt_uv": "source WorldTrack annotated tracks",
            "teacher_xyz": "full-context D4RT aligned to context-only D4RT gauge",
            "teacher_xyz_unavailable_policy": "use predicted XYZ",
            "split": split_meta,
        },
        "trajectory_error": {
            "UV_ADE_normalized": uv_sum / max(uv_count, 1),
            "UV_FDE_normalized": (
                uv_horizon_sum[-1] / uv_horizon_count[-1].clamp_min(1)
            ).item(),
            "UV_ADE_per_horizon": (
                uv_horizon_sum / uv_horizon_count.clamp_min(1)
            ).tolist(),
            "XYZ_ADE_teacher_units": xyz_sum / max(xyz_count, 1),
            "XYZ_FDE_teacher_units": (
                xyz_horizon_sum[-1] / xyz_horizon_count[-1].clamp_min(1)
            ).item(),
            "XYZ_ADE_per_horizon": (
                xyz_horizon_sum / xyz_horizon_count.clamp_min(1)
            ).tolist(),
        },
        "metrics": metrics,
        "object_candidate_benefit": benefit_result,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
