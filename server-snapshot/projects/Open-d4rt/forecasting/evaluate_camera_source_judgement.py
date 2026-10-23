"""Final camera-source judgement for the frozen calibrated CaPSR checkpoint.

All conditions share the same SimVP backbone, calibrated renderer, test split,
and metric implementation.  Only the future camera-motion source changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


ERROR_METRICS = ("MAE", "MSE", "LPIPS", "temporal_delta_MAE", "dynamic_MSE")
QUALITY_METRICS = ("PSNR", "SSIM")


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
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-lpips", action="store_true")
    return p.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sequence(name, domain):
    """Collapse correlated clip windows to their original video/sequence id."""
    if domain == "po50":
        return re.sub(r"_clip_\d+$", "", name)
    return re.sub(r"_\d+$", "", name)


def generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def model_forward(model, baseline, batch, xyz, uv, *, past=None):
    source = batch if past is None else past
    return model(
        baseline, batch["context_rgb"][:, -1], xyz, uv,
        source["past_xyz"].float(), source["past_uv"].float(),
        source["past_valid"], source["past_confidence"].float(),
        mode="structured",
    )


def finite_noise(value, sigma, g, clamp=None):
    result = value.clone()
    valid = torch.isfinite(result)
    noise = torch.randn(result.shape, device=result.device, dtype=result.dtype, generator=g)
    result[valid] += float(sigma) * noise[valid]
    if clamp is not None:
        result = result.clamp(*clamp)
    return result


def gather_time(value, index):
    channels = value.shape[-1]
    return value.gather(2, index[..., None, None].expand(-1, -1, 1, channels)).squeeze(2)


def last_two_valid(batch):
    valid = batch["past_valid"].bool()
    steps = valid.shape[-1]
    timeline = torch.arange(steps, device=valid.device).view(1, 1, steps)
    last_index = torch.where(valid, timeline, -1).amax(-1).clamp_min(0)
    previous_mask = valid & (timeline < last_index[..., None])
    previous_index = torch.where(previous_mask, timeline, -1).amax(-1).clamp_min(0)
    pair_valid = valid.any(-1) & previous_mask.any(-1)
    previous_uv = gather_time(batch["past_uv"].float(), previous_index)
    anchor_uv = gather_time(batch["past_uv"].float(), last_index)
    anchor_xyz = gather_time(batch["past_xyz"].float(), last_index)
    return previous_uv, anchor_uv, anchor_xyz, pair_valid


def repeat_static_xyz(anchor_xyz, future_frames):
    return anchor_xyz[:, :, None].expand(-1, -1, future_frames, -1).clone()


def fallback_translation(source, destination):
    if len(source) == 0:
        return np.eye(3, dtype=np.float32)
    delta = np.median(destination - source, axis=0)
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, 2] = delta
    return matrix


def fit_global_transform(previous_uv, anchor_uv, pair_valid, future_frames, kind):
    """Fit one robust 2-D global transform and extrapolate it recursively."""
    outputs = []
    for batch_index in range(anchor_uv.shape[0]):
        keep = pair_valid[batch_index].detach().cpu().numpy().astype(bool)
        source = previous_uv[batch_index].detach().cpu().numpy()[keep].astype(np.float32)
        destination = anchor_uv[batch_index].detach().cpu().numpy()[keep].astype(np.float32)
        matrix = None
        if kind == "affine" and len(source) >= 3:
            affine, _ = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=0.01,
            )
            if affine is not None:
                matrix = np.eye(3, dtype=np.float32)
                matrix[:2] = affine
        elif kind == "homography" and len(source) >= 4:
            matrix, _ = cv2.findHomography(source, destination, cv2.RANSAC, 0.01)
        if matrix is None or not np.isfinite(matrix).all():
            matrix = fallback_translation(source, destination)
        points = anchor_uv[batch_index].detach().cpu().numpy().astype(np.float32)
        homogeneous = np.concatenate((points, np.ones((len(points), 1), np.float32)), axis=-1)
        future = []
        for _ in range(future_frames):
            homogeneous = homogeneous @ matrix.T
            homogeneous = homogeneous / np.clip(homogeneous[:, 2:3], 1e-6, None)
            future.append(np.clip(homogeneous[:, :2], 0.0, 1.0))
        outputs.append(np.stack(future, axis=1))
    return torch.as_tensor(np.stack(outputs), device=anchor_uv.device, dtype=anchor_uv.dtype)


def raft_future_uv(raft, raft_transform, context_rgb, anchor_uv, future_frames):
    previous, current = context_rgb[:, -2], context_rgb[:, -1]
    # Torchvision RAFT requires feature maps of at least 16x16; its encoder
    # downsamples by eight, so the 64x64 forecasting inputs are evaluated at
    # 128x128.  Flow is converted back to normalized coordinates below.
    previous = F.interpolate(previous, size=(128, 128), mode="bilinear", align_corners=False)
    current = F.interpolate(current, size=(128, 128), mode="bilinear", align_corners=False)
    previous, current = raft_transform(previous, current)
    flow = raft(previous, current)[-1]
    grid = anchor_uv.mul(2).sub(1).unsqueeze(2)
    sampled = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2)
    sampled[..., 0] /= max(flow.shape[-1] - 1, 1)
    sampled[..., 1] /= max(flow.shape[-2] - 1, 1)
    horizon = torch.arange(1, future_frames + 1, device=flow.device, dtype=flow.dtype).view(1, 1, -1, 1)
    return (anchor_uv[:, :, None] + horizon * sampled[:, :, None]).clamp(0, 1)


def camera_prediction(model, baseline, batch, xyz, uv):
    _, state = model_forward(model, baseline, batch, xyz, uv)
    return (baseline + state["camera_weight"] * (state["camera_rgb"] - baseline)).clamp(0, 1)


def sequence_metrics(prediction, target, last, lpips_value, ssim_value):
    difference = prediction - target
    mse = difference.square().mean().item()
    previous_prediction = torch.cat((last, prediction[:, :-1]), 1)
    previous_target = torch.cat((last, target[:, :-1]), 1)
    temporal = ((prediction - previous_prediction) - (target - previous_target)).abs().mean().item()
    target_motion = (target - last).abs().mean(2, keepdim=True) > 0.05
    dynamic = target_motion.expand_as(difference)
    dynamic_abs_sum = difference.abs()[dynamic].sum().item()
    dynamic_sq_sum = difference.square()[dynamic].sum().item()
    dynamic_count = float(dynamic.sum().item())
    return {
        "MAE": difference.abs().mean().item(),
        "MSE": mse,
        "PSNR": -10 * math.log10(max(mse, 1e-12)),
        "SSIM": float(ssim_value),
        "LPIPS": float(lpips_value),
        "temporal_delta_MAE": temporal,
        "dynamic_MAE": dynamic_abs_sum / max(dynamic_count, 1.0),
        "dynamic_MSE": dynamic_sq_sum / max(dynamic_count, 1.0),
        "dynamic_abs_sum": dynamic_abs_sum,
        "dynamic_sq_sum": dynamic_sq_sum,
        "dynamic_count": dynamic_count,
    }


@torch.no_grad()
def perceptual_metrics(predictions, target, lpips_model):
    names = list(predictions)
    stack = torch.cat([predictions[name] for name in names], 0)
    count, steps, channels, height, width = stack.shape
    repeated_target = target.expand(count, -1, -1, -1, -1)
    if lpips_model is None:
        lpips_values = np.full(count, np.nan, dtype=np.float64)
    else:
        values = lpips_model(
            stack.reshape(count * steps, channels, height, width) * 2 - 1,
            repeated_target.reshape(count * steps, channels, height, width) * 2 - 1,
        ).reshape(count, steps)
        lpips_values = values.mean(1).detach().cpu().numpy()
    pred_np = stack.detach().cpu().numpy()
    target_np = target[0].detach().cpu().numpy()
    ssim_values = []
    for condition in range(count):
        values = []
        for step in range(steps):
            values.append(structural_similarity(
                target_np[step].transpose(1, 2, 0),
                pred_np[condition, step].transpose(1, 2, 0),
                channel_axis=-1, data_range=1.0,
            ))
        ssim_values.append(float(np.mean(values)))
    return {
        name: {"LPIPS": float(lpips_values[index]), "SSIM": ssim_values[index]}
        for index, name in enumerate(names)
    }


def aggregate(rows):
    if not rows:
        return {}
    mse = float(np.mean([row["MSE"] for row in rows]))
    dynamic_count = sum(row["dynamic_count"] for row in rows)
    dynamic_sq = sum(row["dynamic_sq_sum"] for row in rows)
    dynamic_abs = sum(row["dynamic_abs_sum"] for row in rows)
    result = {
        "num_sequences": len(rows),
        "MAE": float(np.mean([row["MAE"] for row in rows])),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "PSNR": -10 * math.log10(max(mse, 1e-12)),
        "SSIM": float(np.mean([row["SSIM"] for row in rows])),
        "LPIPS": float(np.nanmean([row["LPIPS"] for row in rows])),
        "temporal_delta_MAE": float(np.mean([row["temporal_delta_MAE"] for row in rows])),
        "dynamic_MAE": dynamic_abs / max(dynamic_count, 1.0),
        "dynamic_MSE": dynamic_sq / max(dynamic_count, 1.0),
        "dynamic_pixel_values": dynamic_count,
    }
    result["dynamic_PSNR"] = -10 * math.log10(max(result["dynamic_MSE"], 1e-12))
    return result


def collapse_sequence_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_sequence"]].append(row)
    collapsed = []
    for sequence, values in sorted(grouped.items()):
        item = aggregate(values)
        item.update({
            "source_sequence": sequence,
            "domain": values[0]["domain"],
            "condition": values[0]["condition"],
            "num_clips": len(values),
        })
        collapsed.append(item)
    return collapsed


def bootstrap_degradation(correct, other, samples, seed):
    """Positive degradation means the counterfactual is worse than correct geometry."""
    rng = np.random.default_rng(seed)
    correct_map = {x["source_sequence"]: x for x in collapse_sequence_rows(correct)}
    other_map = {x["source_sequence"]: x for x in collapse_sequence_rows(other)}
    common = sorted(set(correct_map) & set(other_map))
    if not common:
        raise RuntimeError("No paired source sequences for bootstrap")
    correct = [correct_map[name] for name in common]
    other = [other_map[name] for name in common]
    n = len(correct)
    draws = rng.integers(0, n, size=(samples, n))
    result = {}
    for metric in ERROR_METRICS:
        if metric == "dynamic_MSE":
            c = np.asarray([x[metric] for x in correct], dtype=np.float64)
            o = np.asarray([x[metric] for x in other], dtype=np.float64)
            values = (o[draws] - c[draws]).mean(1)
            estimate = float((o - c).mean())
        else:
            c = np.asarray([x[metric] for x in correct], dtype=np.float64)
            o = np.asarray([x[metric] for x in other], dtype=np.float64)
            values = (o[draws] - c[draws]).mean(1)
            estimate = float((o - c).mean())
        result[metric] = {
            "degradation": float(estimate),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_degradation_gt_0": float(np.mean(values > 0)),
            "bootstrap_unit": "source_sequence_equal_weight",
            "num_source_sequences": n,
        }
    for metric in QUALITY_METRICS:
        if metric == "PSNR":
            c = np.asarray([x["MSE"] for x in correct])
            o = np.asarray([x["MSE"] for x in other])
            values = -10 * np.log10(c[draws].mean(1).clip(1e-12))
            values -= -10 * np.log10(o[draws].mean(1).clip(1e-12))
            estimate = -10 * math.log10(max(c.mean(), 1e-12)) + 10 * math.log10(max(o.mean(), 1e-12))
        else:
            c = np.asarray([x[metric] for x in correct])
            o = np.asarray([x[metric] for x in other])
            values = (c[draws] - o[draws]).mean(1)
            estimate = float((c - o).mean())
        result[metric] = {
            "degradation": float(estimate),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_degradation_gt_0": float(np.mean(values > 0)),
            "bootstrap_unit": "source_sequence_equal_weight",
            "num_source_sequences": n,
        }
    return result


def write_csv(path, rows):
    fields = [
        "sequence", "source_sequence", "domain", "source_path", "cross_video_donor", "condition",
        "MAE", "MSE", "PSNR", "SSIM", "LPIPS", "temporal_delta_MAE",
        "dynamic_MAE", "dynamic_MSE", "dynamic_count",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary):
    correct = summary["aggregate"]["d4rt_uv_xyz"]
    lines = [
        "# CaPSR v3 geometry counterfactual audit", "",
        f"Test sequences: {correct['num_sequences']}", "",
        "Positive degradation means the condition is worse than D4RT UV+XYZ.", "",
        "| Condition | MSE | MSE degradation (95% CI) | LPIPS degradation (95% CI) | Temporal degradation (95% CI) |", 
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["aggregate"].items():
        if name == "d4rt_uv_xyz":
            lines.append(f"| d4rt_uv_xyz | {metrics['MSE']:.6f} | reference | reference | reference |")
            continue
        ci = summary["paired_bootstrap_vs_correct"][name]
        def show(metric):
            item = ci[metric]
            return f"{item['degradation']:+.6f} [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}]"
        lines.append(f"| {name} | {metrics['MSE']:.6f} | {show('MSE')} | {show('LPIPS')} | {show('temporal_delta_MAE')} |")
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    splits, split_meta = combined_splits(args)
    test_paths = list(splits["test"])
    if args.max_test > 0:
        test_paths = test_paths[:args.max_test]
    dataset = VideoForecastDataset(args.cache_root, "test", 64, paths=test_paths, cache_items=False)
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(args, device)
    checkpoint = torch.load(args.correction_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lpips_model = None
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)

    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    raft_weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=raft_weights, progress=True).to(device).eval()
    raft_transform = raft_weights.transforms()
    for parameter in raft.parameters():
        parameter.requires_grad_(False)

    rows = []
    condition_rows = defaultdict(list)
    domain_rows = defaultdict(lambda: defaultdict(list))
    for index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        previous_uv, anchor_uv, anchor_xyz, pair_valid = last_two_valid(batch)
        future_frames = uv.shape[2]
        static_xyz = repeat_static_xyz(anchor_xyz, future_frames)
        affine_uv = fit_global_transform(
            previous_uv, anchor_uv, pair_valid, future_frames, "affine",
        )
        homography_uv = fit_global_transform(
            previous_uv, anchor_uv, pair_valid, future_frames, "homography",
        )
        raft_uv = raft_future_uv(
            raft, raft_transform, batch["context_rgb"], anchor_uv, future_frames,
        )
        predictions = {
            "base_only": baseline,
            "d4rt_uv_xyz": camera_prediction(model, baseline, batch, xyz, uv),
            "d4rt_uv_only": camera_prediction(model, baseline, batch, static_xyz, uv),
            "affine_2d": camera_prediction(model, baseline, batch, static_xyz, affine_uv),
            "homography_2d": camera_prediction(model, baseline, batch, static_xyz, homography_uv),
            "raft_small_2d": camera_prediction(model, baseline, batch, static_xyz, raft_uv),
        }

        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        perceptual = perceptual_metrics(predictions, target, lpips_model)
        source_path = str(dataset.paths[index])
        domain = Path(source_path).parent.name
        for condition, prediction in predictions.items():
            metrics = sequence_metrics(
                prediction, target, last,
                perceptual[condition]["LPIPS"], perceptual[condition]["SSIM"],
            )
            row = {
                "sequence": batch["name"][0], "domain": domain,
                "source_sequence": source_sequence(batch["name"][0], domain),
                "source_path": source_path,
                "cross_video_donor": "",
                "condition": condition, **metrics,
            }
            rows.append(row)
            condition_rows[condition].append(row)
            domain_rows[domain][condition].append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(json.dumps({"processed": index + 1, "total": len(dataset)}), flush=True)

    aggregate_results = {name: aggregate(values) for name, values in condition_rows.items()}
    domain_results = {
        domain: {name: aggregate(values) for name, values in conditions.items()}
        for domain, conditions in domain_rows.items()
    }
    per_source_sequence = {
        name: collapse_sequence_rows(values) for name, values in condition_rows.items()
    }
    correct_rows = condition_rows["d4rt_uv_xyz"]
    bootstrap = {
        name: bootstrap_degradation(
            correct_rows, values, args.bootstrap_samples,
            args.seed + 10000 + condition_index,
        )
        for condition_index, (name, values) in enumerate(condition_rows.items())
        if name != "d4rt_uv_xyz"
    }
    result = {
        "aggregate": aggregate_results,
        "per_domain": domain_results,
        "per_source_sequence": per_source_sequence,
        "paired_bootstrap_vs_correct": bootstrap,
        "meta": {
            "test_sequences": len(dataset), "split": split_meta,
            "checkpoint": args.correction_checkpoint,
            "checkpoint_sha256": sha256(args.correction_checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "model_source": str(HERE / "capsr_calibrated_model.py"),
            "model_source_sha256": sha256(HERE / "capsr_calibrated_model.py"),
            "backbone_checkpoint": args.backbone_checkpoint,
            "trajectory_checkpoint": args.trajectory_checkpoint,
            "future_frames": trajectory_checkpoint["future_frames"],
            "conditions": {
                "d4rt_uv_xyz": "D4RT-predicted future UV and XYZ; camera path only",
                "d4rt_uv_only": "D4RT-predicted future UV with last valid XYZ repeated",
                "affine_2d": "robust 2-D partial affine fitted from the last two observed sparse UV frames",
                "homography_2d": "robust 2-D homography fitted from the last two observed sparse UV frames",
                "raft_small_2d": "pretrained torchvision RAFT-Small flow from the last two RGB frames, linearly extrapolated",
                "base_only": "strong loss-matched SimVP checkpoint",
            },
            "path_variants": "all motion-source conditions use the frozen camera-only renderer",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed, "training": False,
        },
    }
    write_csv(output / "per_sequence.csv", rows)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    write_markdown(output / "summary.md", result)
    print(json.dumps({"complete": True, "output": str(output), "conditions": list(condition_rows)}), flush=True)


if __name__ == "__main__":
    main()
