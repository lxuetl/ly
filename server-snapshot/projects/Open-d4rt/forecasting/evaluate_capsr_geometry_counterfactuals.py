"""Low-cost geometry counterfactual audit for a frozen CaPSR v3 checkpoint.

The frozen video backbone and trajectory forecaster are evaluated once per clip.
No D4RT extraction or training is performed.  Results include per-sequence rows,
per-domain aggregates, and paired sequence-bootstrap confidence intervals.
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
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_structured_model_v3 import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


UV_NOISE = (0.005, 0.01, 0.02, 0.05)
XYZ_NOISE_REL = (0.01, 0.05, 0.10, 0.20)
DROPOUT = (0.25, 0.50, 0.75, 0.90)
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
    correct = summary["aggregate"]["correct"]
    lines = [
        "# CaPSR v3 geometry counterfactual audit", "",
        f"Test sequences: {correct['num_sequences']}", "",
        "Positive degradation means the condition is worse than correct geometry.", "",
        "| Condition | MSE | MSE degradation (95% CI) | LPIPS degradation (95% CI) | Temporal degradation (95% CI) |", 
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["aggregate"].items():
        if name == "correct":
            lines.append(f"| correct | {metrics['MSE']:.6f} | reference | reference | reference |")
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

    rng = np.random.default_rng(args.seed + 1009)
    permutation = rng.permutation(len(dataset)).tolist()
    groups = [source_sequence(path.stem, path.parent.name) for path in dataset.paths]
    donor_indices = []
    same_domain_donor_indices = []
    for index, group in enumerate(groups):
        candidates = [candidate for candidate in permutation if groups[candidate] != group]
        if not candidates:
            raise RuntimeError("Cross-video counterfactual requires at least two source sequences")
        donor_indices.append(candidates[index % len(candidates)])
        domain = dataset.paths[index].parent.name
        same_domain = [
            candidate for candidate in candidates
            if dataset.paths[candidate].parent.name == domain
        ]
        same_domain_donor_indices.append(
            same_domain[index % len(same_domain)] if same_domain else None
        )

    rows = []
    condition_rows = defaultdict(list)
    domain_rows = defaultdict(lambda: defaultdict(list))
    for index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        correct, state = model_forward(model, baseline, batch, xyz, uv)
        predictions = {"correct": correct, "base_only": baseline}
        predictions["camera_only"] = (
            baseline + state["camera_weight"] * (state["camera_rgb"] - baseline)
        ).clamp(0, 1)
        predictions["object_only"] = (
            baseline + state["object_weight"] * (state["object_rgb"] - baseline)
        ).clamp(0, 1)

        reverse = torch.arange(xyz.shape[2] - 1, -1, -1, device=device)
        predictions["time_reverse"] = model_forward(
            model, baseline, batch, xyz[:, :, reverse], uv[:, :, reverse]
        )[0]

        shuffle_generator = generator(device, args.seed + index * 1000 + 50)
        shuffled = torch.randperm(xyz.shape[2], device=device, generator=shuffle_generator)
        predictions["time_shuffle"] = model_forward(
            model, baseline, batch, xyz[:, :, shuffled], uv[:, :, shuffled]
        )[0]

        donor_index = int(donor_indices[index])
        donor = move(video_pad_collate([dataset[donor_index]]), device)
        _, donor_xyz, donor_uv = frozen_prediction(base, trajectory, donor)
        predictions["cross_video_global"] = model_forward(
            model, baseline, batch, donor_xyz, donor_uv, past=donor
        )[0]
        same_domain_donor_index = same_domain_donor_indices[index]
        if same_domain_donor_index is not None:
            same_domain_donor = move(video_pad_collate([dataset[same_domain_donor_index]]), device)
            _, same_domain_xyz, same_domain_uv = frozen_prediction(base, trajectory, same_domain_donor)
            predictions["cross_video_same_domain"] = model_forward(
                model, baseline, batch, same_domain_xyz, same_domain_uv, past=same_domain_donor
            )[0]

        for level_index, sigma in enumerate(UV_NOISE):
            g = generator(device, args.seed + index * 1000 + 100 + level_index)
            noisy_uv = finite_noise(uv, sigma, g, clamp=(0.0, 1.0))
            predictions[f"uv_noise_{sigma:g}"] = model_forward(
                model, baseline, batch, xyz, noisy_uv
            )[0]

        xyz_scale = torch.nan_to_num(xyz).std().clamp_min(1e-3).item()
        for level_index, relative in enumerate(XYZ_NOISE_REL):
            g = generator(device, args.seed + index * 1000 + 200 + level_index)
            noisy_xyz = finite_noise(xyz, relative * xyz_scale, g)
            predictions[f"xyz_noise_rel_{relative:g}"] = model_forward(
                model, baseline, batch, noisy_xyz, uv
            )[0]

        score_generator = generator(device, args.seed + index * 1000 + 300)
        scores = torch.rand(
            (batch["past_valid"].shape[0], batch["past_valid"].shape[1], 1),
            device=device, generator=score_generator,
        )
        for fraction in DROPOUT:
            dropped = dict(batch)
            keep = scores >= fraction
            dropped["past_valid"] = batch["past_valid"] & keep
            dropped["past_confidence"] = batch["past_confidence"] * keep.float()
            predictions[f"dropout_{fraction:g}"] = model_forward(
                model, baseline, dropped, xyz, uv
            )[0]

        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        perceptual = perceptual_metrics(predictions, target, lpips_model)
        source_path = str(dataset.paths[index])
        domain = Path(source_path).parent.name
        donor_name = dataset.paths[donor_index].stem
        for condition, prediction in predictions.items():
            metrics = sequence_metrics(
                prediction, target, last,
                perceptual[condition]["LPIPS"], perceptual[condition]["SSIM"],
            )
            row = {
                "sequence": batch["name"][0], "domain": domain,
                "source_sequence": source_sequence(batch["name"][0], domain),
                "source_path": source_path,
                "cross_video_donor": (
                    dataset.paths[same_domain_donor_index].stem
                    if condition == "cross_video_same_domain" and same_domain_donor_index is not None
                    else donor_name
                ),
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
    correct_rows = condition_rows["correct"]
    bootstrap = {
        name: bootstrap_degradation(
            correct_rows, values, args.bootstrap_samples,
            args.seed + 10000 + condition_index,
        )
        for condition_index, (name, values) in enumerate(condition_rows.items())
        if name != "correct"
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
            "model_source": str(HERE / "capsr_structured_model_v3.py"),
            "model_source_sha256": sha256(HERE / "capsr_structured_model_v3.py"),
            "backbone_checkpoint": args.backbone_checkpoint,
            "trajectory_checkpoint": args.trajectory_checkpoint,
            "future_frames": trajectory_checkpoint["future_frames"],
            "uv_noise_sigma_normalized": list(UV_NOISE),
            "xyz_noise_sigma_relative_to_prediction_std": list(XYZ_NOISE_REL),
            "trajectory_dropout_fraction": list(DROPOUT),
            "time_shuffle": "future horizon reversal",
            "cross_video_global": "deterministic deranged donor geometry; domain unrestricted",
            "cross_video_same_domain": "different source sequence within the same domain; singleton-source domains excluded",
            "path_variants": "post-hoc single-path contribution from the correct full checkpoint",
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
