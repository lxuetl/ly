#!/usr/bin/env python3
"""Low-cost, training-free geometry-critic validation for video forecasts.

The script compares ground truth, a frozen SimVP baseline, a VGGT-conditioned
residual, and an equally-sized RGB-only residual.  VGGT-Omega is used only as a
frozen evaluator on context+future clips.  Per-sequence scores are written
incrementally so the run can resume after interruption.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = ("ground_truth", "fair_simvp", "vggt_omega_geometry", "rgb_capacity_control")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--geometry-cache", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--geometry-residual-checkpoint", required=True)
    p.add_argument("--capacity-control-checkpoint", required=True)
    p.add_argument("--vggt-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--max-items", type=int, default=80)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--metric-size", type=int, default=64)
    p.add_argument("--confidence-quantile", type=float, default=0.30)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def stratified_paths(paths, limit, seed):
    if not limit or limit >= len(paths):
        return list(paths)
    groups = defaultdict(list)
    for path in paths:
        groups[Path(path).parent.name].append(Path(path))
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def prepare_vggt_video(video, resolution):
    # All forecasting variants are evaluated from the same 64x64 representation,
    # preventing the ground-truth branch from receiving privileged high-res input.
    return F.interpolate(
        video, size=(resolution, resolution), mode="bicubic", align_corners=False,
        antialias=True,
    ).clamp(0, 1)


def so3_log(rotation):
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(((trace - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6))
    vee = torch.stack((
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ), -1)
    scale = theta / (2 * torch.sin(theta).clamp_min(1e-6))
    small = theta < 1e-3
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    return vee * scale[..., None]


def camera_metrics(extrinsics, context_frames):
    # Extrinsics follow the world-to-camera convention [R | t].  Include the
    # final two context poses so discontinuity at the forecast boundary counts.
    start = max(int(context_frames) - 2, 0)
    ext = extrinsics[start:]
    rotation = ext[:, :3, :3]
    translation = ext[:, :3, 3]
    orientation = rotation.transpose(-1, -2)
    center = -(orientation @ translation[..., None]).squeeze(-1)
    velocity = center[1:] - center[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    trans_den = velocity[1:].norm(dim=-1) + velocity[:-1].norm(dim=-1)
    trans_error = (acceleration.norm(dim=-1) / trans_den.clamp_min(1e-6)).mean()

    relative = orientation[:-1].transpose(-1, -2) @ orientation[1:]
    angular_velocity = so3_log(relative)
    angular_acceleration = angular_velocity[1:] - angular_velocity[:-1]
    rot_den = angular_velocity[1:].norm(dim=-1) + angular_velocity[:-1].norm(dim=-1)
    rot_error = (angular_acceleration.norm(dim=-1) / rot_den.clamp_min(1e-6)).mean()
    reward = 0.5 * (1 / (1 + trans_error) + 1 / (1 + rot_error))
    return {
        "camera_translation_acceleration": float(trans_error),
        "camera_rotation_acceleration": float(rot_error),
        "camera_smoothness_reward": float(reward),
    }


def resize_depth_camera(depth, confidence, intrinsics, size):
    height, width = depth.shape[-2:]
    depth = F.interpolate(depth[:, None], (size, size), mode="bilinear", align_corners=False)[:, 0]
    confidence = F.interpolate(
        confidence[:, None], (size, size), mode="bilinear", align_corners=False,
    )[:, 0]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= size / float(width)
    intrinsics[:, 1] *= size / float(height)
    return depth, confidence, intrinsics


def one_way_reprojection(depth, confidence, extrinsics, intrinsics, source, target, quantile):
    size = depth.shape[-1]
    device, dtype = depth.device, depth.dtype
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype), indexing="ij",
    )
    pixels = torch.stack((xx.reshape(-1), yy.reshape(-1), torch.ones(size * size, device=device, dtype=dtype)))
    source_depth = depth[source].reshape(-1)
    source_confidence = confidence[source].reshape(-1)
    threshold = torch.quantile(source_confidence, quantile)
    valid_source = (source_depth > 1e-5) & torch.isfinite(source_depth) & (source_confidence >= threshold)

    k_inverse = torch.linalg.inv(intrinsics[source])
    camera_source = (k_inverse @ pixels) * source_depth[None]
    r_source, t_source = extrinsics[source, :3, :3], extrinsics[source, :3, 3]
    world = r_source.transpose(0, 1) @ (camera_source - t_source[:, None])
    r_target, t_target = extrinsics[target, :3, :3], extrinsics[target, :3, 3]
    camera_target = r_target @ world + t_target[:, None]
    projected = intrinsics[target] @ camera_target
    z = camera_target[2]
    u = torch.round(projected[0] / projected[2].clamp_min(1e-6)).long()
    v = torch.round(projected[1] / projected[2].clamp_min(1e-6)).long()
    inside = valid_source & (z > 1e-5) & (u >= 0) & (u < size) & (v >= 0) & (v < size)
    if inside.sum() < 64:
        return float("nan"), 0.0
    index = v[inside] * size + u[inside]
    rendered = torch.full((size * size,), float("inf"), device=device, dtype=dtype)
    rendered.scatter_reduce_(0, index, z[inside], reduce="amin", include_self=True)
    target_depth = depth[target].reshape(-1)
    target_confidence = confidence[target].reshape(-1)
    target_threshold = torch.quantile(target_confidence, quantile)
    valid = (
        torch.isfinite(rendered) & torch.isfinite(target_depth) & (target_depth > 1e-5)
        & (target_confidence >= target_threshold)
    )
    if valid.sum() < 64:
        return float("nan"), float(valid.float().mean())
    error = (rendered[valid].clamp_min(1e-5).log() - target_depth[valid].clamp_min(1e-5).log()).abs()
    # Trim the largest 10% of per-pixel errors to reduce unavoidable occlusion
    # contamination while retaining a worst-frame statistic below.
    cutoff = torch.quantile(error, 0.90)
    error = error[error <= cutoff]
    return float(error.mean()), float(valid.float().mean())


def reprojection_metrics(depth, confidence, extrinsics, intrinsics, context_frames, size, quantile):
    depth, confidence, intrinsics = resize_depth_camera(depth, confidence, intrinsics, size)
    first = max(int(context_frames) - 1, 0)
    pair_errors, fractions = [], []
    for source in range(first, depth.shape[0] - 1):
        target = source + 1
        for left, right in ((source, target), (target, source)):
            error, fraction = one_way_reprojection(
                depth, confidence, extrinsics, intrinsics, left, right, quantile,
            )
            if math.isfinite(error):
                pair_errors.append(error)
                fractions.append(fraction)
    if not pair_errors:
        return {
            "depth_reprojection_error": float("nan"),
            "depth_reprojection_worst3": float("nan"),
            "depth_reprojection_valid_fraction": 0.0,
            "depth_reprojection_reward": float("nan"),
        }
    values = np.asarray(pair_errors, dtype=np.float64)
    worst = np.sort(values)[-min(3, len(values)):]
    worst3 = float(worst.mean())
    return {
        "depth_reprojection_error": float(values.mean()),
        "depth_reprojection_worst3": worst3,
        "depth_reprojection_valid_fraction": float(np.mean(fractions)),
        "depth_reprojection_reward": 1.0 / (1.0 + worst3),
    }


def track_aligned_errors(prediction, target, item):
    future_uv = torch.nan_to_num(item["future_uv_teacher"].float()).clamp(0, 1).to(prediction.device)
    future_valid = item["future_valid"].bool().to(prediction.device)
    future_valid &= torch.isfinite(item["future_uv_teacher"].float()).all(-1).to(prediction.device)
    confidence = torch.nan_to_num(item["future_confidence"].float()).clamp(0, 1).to(prediction.device)
    past_uv = torch.nan_to_num(item["past_uv"].float()).clamp(0, 1).to(prediction.device)
    past_valid = item["past_valid"].bool().to(prediction.device)
    n, future_frames = future_uv.shape[:2]
    if n == 0:
        return {"track_MAE": float("nan"), "dynamic_track_MAE": float("nan"), "static_track_MAE": float("nan"), "dynamic_track_fraction": 0.0}
    indices = torch.arange(past_valid.shape[1], device=prediction.device)[None].expand(n, -1)
    last_index = torch.where(past_valid, indices, torch.zeros_like(indices)).max(1).values
    point_index = torch.arange(n, device=prediction.device)
    anchor = past_uv[point_index, last_index]
    amplitude = (future_uv - anchor[:, None]).norm(dim=-1)
    amplitude = torch.where(future_valid, amplitude, torch.zeros_like(amplitude)).amax(1)
    dynamic = amplitude > 0.05

    errors = []
    for step in range(future_frames):
        difference = (prediction[step] - target[step]).abs().mean(0, keepdim=True)[None]
        grid = future_uv[:, step].mul(2).sub(1).view(1, n, 1, 2)
        sampled = F.grid_sample(difference, grid, mode="bilinear", align_corners=True)[0, 0, :, 0]
        errors.append(sampled)
    error = torch.stack(errors, 1)
    weight = confidence * future_valid.float()

    def weighted(mask):
        selected = weight * mask[:, None].float()
        if selected.sum() <= 0:
            return float("nan")
        return float((error * selected).sum() / selected.sum())

    usable = future_valid.any(1)
    return {
        "track_MAE": weighted(usable),
        "dynamic_track_MAE": weighted(dynamic & usable),
        "static_track_MAE": weighted((~dynamic) & usable),
        "dynamic_track_fraction": float((dynamic & usable).sum() / usable.sum().clamp_min(1)),
    }


def rgb_metrics(prediction, target, last):
    difference = prediction - target
    mse = float(difference.square().mean())
    predicted_delta = torch.cat((prediction[:1] - last[None], prediction[1:] - prediction[:-1]))
    target_delta = torch.cat((target[:1] - last[None], target[1:] - target[:-1]))
    return {
        "MAE": float(difference.abs().mean()),
        "MSE": mse,
        "PSNR": -10 * math.log10(max(mse, 1e-12)),
        "temporal_delta_MAE": float((predicted_delta - target_delta).abs().mean()),
    }


@torch.inference_mode()
def vggt_scores(model, encoding_to_camera, sequence, context_frames, args):
    images = prepare_vggt_video(sequence, args.resolution)
    output = model(images)
    depth = output["depth"].squeeze(0).squeeze(-1).float()
    confidence = output["depth_conf"].squeeze(0).float()
    extrinsics, intrinsics = encoding_to_camera(output["pose_enc"], output["images"].shape[-2:])
    extrinsics, intrinsics = extrinsics.squeeze(0).float(), intrinsics.squeeze(0).float()
    result = camera_metrics(extrinsics, context_frames)
    result.update(reprojection_metrics(
        depth, confidence, extrinsics, intrinsics, context_frames,
        args.metric_size, args.confidence_quantile,
    ))
    return result


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[valid]), rankdata(y[valid])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def paired_bootstrap_improvement(baseline, candidate, seed=42, samples=10000):
    """Positive values mean the candidate reduced a lower-is-better error."""
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    delta = baseline[valid] - candidate[valid]
    if not len(delta):
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "win_rate": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "win_rate": float((delta > 0).mean()),
        "n": int(len(delta)),
    }


def geometry_gap(row, ground_truth):
    camera = (
        abs(row["camera_translation_acceleration"] - ground_truth["camera_translation_acceleration"])
        + abs(row["camera_rotation_acceleration"] - ground_truth["camera_rotation_acceleration"])
    )
    reprojection = abs(row["depth_reprojection_worst3"] - ground_truth["depth_reprojection_worst3"])
    return camera, reprojection, camera + reprojection


def summarize(records, seed=42):
    by_name = defaultdict(dict)
    for row in records:
        by_name[row["name"]][row["variant"]] = row
    complete = {name: rows for name, rows in by_name.items() if all(v in rows for v in VARIANTS)}
    metrics = sorted({key for rows in complete.values() for row in rows.values() for key, value in row.items() if isinstance(value, (int, float))})
    aggregate = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        for metric in metrics:
            values = np.asarray([rows[variant].get(metric, np.nan) for rows in complete.values()], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if len(finite):
                aggregate[variant][metric] = {"mean": float(finite.mean()), "std": float(finite.std()), "n": int(len(finite))}

    paired = {}
    for variant in ("fair_simvp", "vggt_omega_geometry", "rgb_capacity_control"):
        geometry_gaps, mse, dynamic_track = [], [], []
        camera_gap, reprojection_gap = [], []
        for rows in complete.values():
            row, gt = rows[variant], rows["ground_truth"]
            cam, reproj, total = geometry_gap(row, gt)
            geometry_gaps.append(total)
            camera_gap.append(cam)
            reprojection_gap.append(reproj)
            mse.append(row["MSE"])
            dynamic_track.append(row["dynamic_track_MAE"])
        paired[variant] = {
            "spearman_geometry_gap_vs_MSE": spearman(geometry_gaps, mse),
            "spearman_geometry_gap_vs_dynamic_track_MAE": spearman(geometry_gaps, dynamic_track),
            "spearman_camera_gap_vs_MSE": spearman(camera_gap, mse),
            "spearman_reprojection_gap_vs_MSE": spearman(reprojection_gap, mse),
            "mean_geometry_gap_to_GT": float(np.nanmean(geometry_gaps)),
        }

    prediction_rows = []
    for rows in complete.values():
        gt = rows["ground_truth"]
        for variant in ("fair_simvp", "vggt_omega_geometry", "rgb_capacity_control"):
            row = rows[variant]
            prediction_rows.append((
                abs(row["camera_translation_acceleration"] - gt["camera_translation_acceleration"])
                + abs(row["camera_rotation_acceleration"] - gt["camera_rotation_acceleration"])
                + abs(row["depth_reprojection_worst3"] - gt["depth_reprojection_worst3"]),
                row["MSE"], row["dynamic_track_MAE"],
            ))
    across = {
        "spearman_geometry_gap_vs_MSE": spearman([x[0] for x in prediction_rows], [x[1] for x in prediction_rows]),
        "spearman_geometry_gap_vs_dynamic_track_MAE": spearman([x[0] for x in prediction_rows], [x[2] for x in prediction_rows]),
    }

    comparisons = {}
    lower_is_better = ("MAE", "MSE", "temporal_delta_MAE", "track_MAE", "dynamic_track_MAE", "static_track_MAE")
    for baseline_name, candidate_name in (
        ("fair_simvp", "vggt_omega_geometry"),
        ("fair_simvp", "rgb_capacity_control"),
        ("rgb_capacity_control", "vggt_omega_geometry"),
    ):
        key = f"{candidate_name}_vs_{baseline_name}"
        comparisons[key] = {}
        for metric in lower_is_better:
            baseline_values = [rows[baseline_name][metric] for rows in complete.values()]
            candidate_values = [rows[candidate_name][metric] for rows in complete.values()]
            comparisons[key][metric] = paired_bootstrap_improvement(
                baseline_values, candidate_values, seed=seed,
            )
        base_camera, base_reprojection, base_total = [], [], []
        cand_camera, cand_reprojection, cand_total = [], [], []
        for rows in complete.values():
            gt = rows["ground_truth"]
            bg = geometry_gap(rows[baseline_name], gt)
            cg = geometry_gap(rows[candidate_name], gt)
            base_camera.append(bg[0]); base_reprojection.append(bg[1]); base_total.append(bg[2])
            cand_camera.append(cg[0]); cand_reprojection.append(cg[1]); cand_total.append(cg[2])
        for metric, baseline_values, candidate_values in (
            ("camera_gap_to_GT", base_camera, cand_camera),
            ("reprojection_gap_to_GT", base_reprojection, cand_reprojection),
            ("combined_geometry_gap_to_GT", base_total, cand_total),
        ):
            comparisons[key][metric] = paired_bootstrap_improvement(
                baseline_values, candidate_values, seed=seed,
            )

    domain_aggregate = {}
    domains = sorted({row["domain"] for rows in complete.values() for row in rows.values()})
    domain_metrics = ("MSE", "dynamic_track_MAE", "camera_translation_acceleration", "camera_rotation_acceleration", "depth_reprojection_worst3")
    for domain in domains:
        domain_aggregate[domain] = {}
        domain_sequences = [rows for rows in complete.values() if rows["ground_truth"]["domain"] == domain]
        for variant in VARIANTS:
            domain_aggregate[domain][variant] = {}
            for metric in domain_metrics:
                values = np.asarray([rows[variant][metric] for rows in domain_sequences], dtype=np.float64)
                values = values[np.isfinite(values)]
                domain_aggregate[domain][variant][metric] = float(values.mean()) if len(values) else float("nan")
        domain_aggregate[domain]["n"] = len(domain_sequences)

    return {
        "num_complete_sequences": len(complete), "aggregate": aggregate,
        "correlations_by_variant": paired, "correlations_across_predictions": across,
        "paired_bootstrap_improvements": comparisons, "domain_aggregate": domain_aggregate,
    }


def main():
    args = parse_args()
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "per_sequence.jsonl"
    existing_records = []
    if records_path.exists():
        existing_records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if args.summarize_only:
        previous_meta = {}
        summary_path = output / "summary.json"
        if summary_path.exists():
            previous_meta = json.loads(summary_path.read_text()).get("meta", {})
        summary = summarize(existing_records, seed=args.seed)
        previous_meta["postprocessed_with_bootstrap"] = True
        summary["meta"] = previous_meta
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True))
        print(json.dumps({"summarized": True, "sequences": summary["num_complete_sequences"], "output": str(summary_path)}))
        return

    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from train_pointodyssey_dmvfn_injection import combined_splits
    from video_forecast_dataset import VideoForecastDataset
    from train_vggt_omega_injection import GeometryDataset, VGGTResidual
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    completed = defaultdict(set)
    for row in existing_records:
        completed[row["name"]].add(row["variant"])

    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
    selected = stratified_paths(splits[args.split], args.max_items, args.seed)
    base = VideoForecastDataset(args.cache_root, args.split, 64, paths=selected, cache_items=False)
    dataset = GeometryDataset(base, args.geometry_cache)

    backbone_checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(backbone_checkpoint).to(device).eval()
    backbone.load_state_dict(backbone_checkpoint["model"])
    context_frames = int(backbone_checkpoint["context_frames"])
    future_frames = int(backbone_checkpoint["future_frames"])
    geometry_residual = VGGTResidual(context_frames, future_frames).to(device).eval()
    capacity_control = VGGTResidual(context_frames, future_frames).to(device).eval()
    geometry_residual.load_state_dict(torch.load(args.geometry_residual_checkpoint, map_location="cpu")["model"])
    capacity_control.load_state_dict(torch.load(args.capacity_control_checkpoint, map_location="cpu")["model"])
    for module in (backbone, geometry_residual, capacity_control):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    vggt = VGGTOmega().eval()
    vggt.load_state_dict(torch.load(args.vggt_checkpoint, map_location="cpu"), strict=True)
    vggt = vggt.to(device)
    started = time.time()
    domain_counts = defaultdict(int)

    for index in range(len(dataset)):
        item = dataset[index]
        name = item["name"]
        if all(variant in completed[name] for variant in VARIANTS):
            continue
        context = item["context_rgb"].to(device)
        target = item["future_rgb"].to(device)
        with torch.inference_mode():
            baseline = backbone(context[None], None, None, None)
            geometry_prediction = geometry_residual(
                baseline, context[-1][None], item["vggt_depth"][None].to(device),
                item["vggt_confidence"][None].to(device), item["vggt_pose"][None].to(device), True,
            )[0]
            control_prediction = capacity_control(
                baseline, context[-1][None], item["vggt_depth"][None].to(device),
                item["vggt_confidence"][None].to(device), item["vggt_pose"][None].to(device), False,
            )[0]
        predictions = {
            "ground_truth": target,
            "fair_simvp": baseline[0],
            "vggt_omega_geometry": geometry_prediction[0],
            "rgb_capacity_control": control_prediction[0],
        }
        domain = Path(base.paths[index]).parent.name
        rows = []
        for variant, prediction in predictions.items():
            sequence = torch.cat((context, prediction), 0)
            row = {"name": name, "domain": domain, "variant": variant}
            row.update(rgb_metrics(prediction, target, context[-1]))
            row.update(track_aligned_errors(prediction, target, item))
            row.update(vggt_scores(vggt, encoding_to_camera, sequence, context_frames, args))
            rows.append(row)
            del sequence
        with records_path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=True) + "\n")
                existing_records.append(row)
                completed[name].add(row["variant"])
        domain_counts[domain] += 1
        if (index + 1) % 5 == 0 or index == 0:
            elapsed = time.time() - started
            print(json.dumps({
                "done": index + 1, "total": len(dataset), "elapsed_seconds": elapsed,
                "seconds_per_sequence": elapsed / (index + 1),
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0,
            }), flush=True)

    summary = summarize(existing_records, seed=args.seed)
    summary["meta"] = {
        "selected": len(selected), "selected_split": args.split,
        "domain_counts": dict(domain_counts), "split": split_meta,
        "resolution": args.resolution, "metric_size": args.metric_size,
        "confidence_quantile": args.confidence_quantile,
        "variants": list(VARIANTS), "args": vars(args),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps({"complete": True, "output": str(output), "sequences": summary["num_complete_sequences"]}), flush=True)


if __name__ == "__main__":
    main()
