"""Train structured CaPSR v2 or its equal-parameter RGB-only control."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_model import visibility_targets
from capsr_structured_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("structured", "no_geometry"), required=True)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dynamic-weight", type=float, default=3.0)
    p.add_argument("--visibility-weight", type=float, default=0.15)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--gradient-weight", type=float, default=0.05)
    p.add_argument("--static-identity-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.005)
    p.add_argument("--mask-budget-weight", type=float, default=0.005)
    p.add_argument("--object-routing-weight", type=float, default=0.05)
    p.add_argument("--motion-threshold", type=float, default=0.05)
    p.add_argument("--num-motion-clusters", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.035)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def limited(dataset, maximum):
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, range(maximum))


def forward_model(model, mode, base, trajectory, batch):
    base_prediction, xyz, uv = frozen_prediction(base, trajectory, batch)
    prediction, state = model(
        base_prediction, batch["context_rgb"][:, -1], xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(), batch["past_valid"],
        batch["past_confidence"].float(), mode=mode,
    )
    return prediction, base_prediction, state


def balanced_visibility_loss(logits, past_valid, future_valid):
    labels, mask = visibility_targets(past_valid, future_valid)
    counts = torch.stack([((labels == index) & mask).sum() for index in range(3)]).float()
    weights = torch.sqrt(mask.sum().float() / counts.clamp_min(1)).clamp(0.5, 10.0)
    probability = logits.softmax(-1).gather(-1, labels[..., None]).squeeze(-1).clamp_min(1e-6)
    raw = -weights[labels] * (1 - probability).square() * probability.log()
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def loss_fn(prediction, baseline, state, batch, a, lpips_model):
    target = batch["future_rgb"]
    last = batch["context_rgb"][:, -1:]
    dynamic = ((target - last).abs().mean(2, keepdim=True) > 0.05).float()
    error = (prediction - target).abs() + 0.1 * (prediction - target).square()
    reconstruction = (error * (1 + a.dynamic_weight * dynamic)).mean()
    visibility = prediction.new_zeros(())
    if a.mode == "structured":
        visibility = balanced_visibility_loss(
            state["visibility_logits"], batch["past_valid"], batch["future_valid"]
        )
    pred_previous = torch.cat((last, prediction[:, :-1]), 1)
    target_previous = torch.cat((last, target[:, :-1]), 1)
    temporal = ((prediction - pred_previous) - (target - target_previous)).abs().mean()
    b, k, c, h, w = prediction.shape
    perceptual = lpips_model(
        prediction.reshape(b * k, c, h, w) * 2 - 1,
        target.reshape(b * k, c, h, w) * 2 - 1,
    ).mean()
    pred_x, target_x = prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]
    pred_y, target_y = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    gradient = 0.5 * ((pred_x - target_x).abs().mean() + (pred_y - target_y).abs().mean())
    correction = prediction - baseline
    static = 1 - dynamic
    static_identity = (correction.abs().mean(2, keepdim=True) * static).sum() / static.sum().clamp_min(1)
    correction_size = correction.abs().mean()
    mask_budget = (
        state["camera_weight"] + state["object_weight"] + state["synthesis_weight"]
    ).mean()
    object_routing = prediction.new_zeros(())
    if a.mode == "structured":
        base_error = (baseline - target).abs().mean(2, keepdim=True)
        object_error = (state["object_rgb"] - target).abs().mean(2, keepdim=True)
        beneficial = (object_error + 0.002 < base_error).float()
        permission = state["dynamic_object_permission"].detach()
        route_mask = permission * dynamic
        routing_raw = F.binary_cross_entropy(
            state["object_gate"].clamp(1e-5, 1 - 1e-5), beneficial,
            reduction="none",
        )
        object_routing = (routing_raw * route_mask).sum() / route_mask.sum().clamp_min(1)
    total = (
        reconstruction + a.visibility_weight * visibility + a.temporal_weight * temporal
        + a.lpips_weight * perceptual + a.gradient_weight * gradient
        + a.static_identity_weight * static_identity + a.correction_weight * correction_size
        + a.mask_budget_weight * mask_budget + a.object_routing_weight * object_routing
    )
    pieces = {
        "reconstruction": reconstruction, "visibility": visibility, "temporal": temporal,
        "lpips": perceptual, "gradient": gradient, "static_identity": static_identity,
        "correction": correction_size, "mask_budget": mask_budget,
        "object_routing": object_routing,
    }
    return total, {name: float(value.detach()) for name, value in pieces.items()}


@torch.no_grad()
def validation(model, mode, base, trajectory, loader, device):
    model.eval()
    mse = count = dyn_abs = dyn_count = temporal = batches = 0.0
    for batch in loader:
        batch = move(batch, device)
        prediction, _, _ = forward_model(model, mode, base, trajectory, batch)
        target = batch["future_rgb"]
        difference = prediction - target
        mse += difference.square().sum().item()
        count += target.numel()
        last = batch["context_rgb"][:, -1:]
        dynamic = ((target - last).abs().mean(2, keepdim=True) > 0.05).expand_as(target)
        dyn_abs += difference.abs()[dynamic].sum().item()
        dyn_count += dynamic.sum().item()
        pred_previous = torch.cat((last, prediction[:, :-1]), 1)
        target_previous = torch.cat((last, target[:, :-1]), 1)
        temporal += ((prediction - pred_previous) - (target - target_previous)).abs().mean().item()
        batches += 1
    return {
        "MSE": mse / max(count, 1),
        "dynamic_MAE": dyn_abs / max(dyn_count, 1),
        "temporal_MAE": temporal / max(batches, 1),
    }


@torch.no_grad()
def evaluate(model, mode, base, trajectory, loader, device, future_frames,
             motion_threshold=0.05):
    model.eval()
    accumulators = {
        "no_geometry_backbone": VideoMetricAccumulator(device, future_frames),
        mode: VideoMetricAccumulator(device, future_frames),
    }
    extras = {name: {"abs": 0.0, "sq": 0.0, "count": 0.0} for name in accumulators}
    motion_counts = {
        name: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for name in accumulators
    }
    bucket_ranges = {
        "static_0_0.02": (0.0, 0.02),
        "small_0.02_0.05": (0.02, 0.05),
        "medium_0.05_0.10": (0.05, 0.10),
        "large_ge_0.10": (0.10, float("inf")),
    }
    bucket_stats = {
        name: {bucket: {"abs": 0.0, "sq": 0.0, "count": 0.0}
               for bucket in bucket_ranges} for name in accumulators
    }
    object_by_bucket = {
        bucket: {"weight": 0.0, "permission": 0.0, "count": 0.0}
        for bucket in bucket_ranges
    }
    mask_sums = {"camera": 0.0, "object": 0.0, "synthesis": 0.0,
                 "disocclusion": 0.0, "pixels": 0.0}
    confidence_sums = {"cycle": 0.0, "occlusion": 0.0, "support": 0.0,
                       "cluster_entropy": 0.0, "cluster_count": 0.0}
    for batch in loader:
        batch = move(batch, device)
        prediction, baseline, state = forward_model(model, mode, base, trajectory, batch)
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        target_motion = (target - last).abs().mean(2, keepdim=True)
        target_motion_mask = target_motion > motion_threshold
        dynamic = target_motion_mask.expand_as(target)
        for name, value in (("no_geometry_backbone", baseline), (mode, prediction)):
            accumulators[name].update(value, target, batch["context_rgb"][:, -1])
            difference = value - target
            extras[name]["abs"] += difference.abs()[dynamic].sum().item()
            extras[name]["sq"] += difference.square()[dynamic].sum().item()
            extras[name]["count"] += dynamic.sum().item()
            predicted_motion_mask = (
                (value - last).abs().mean(2, keepdim=True) > motion_threshold
            )
            motion_counts[name]["tp"] += (predicted_motion_mask & target_motion_mask).sum().item()
            motion_counts[name]["fp"] += (predicted_motion_mask & ~target_motion_mask).sum().item()
            motion_counts[name]["fn"] += (~predicted_motion_mask & target_motion_mask).sum().item()
            pixel_abs = difference.abs().mean(2, keepdim=True)
            pixel_sq = difference.square().mean(2, keepdim=True)
            for bucket, (lower, upper) in bucket_ranges.items():
                selected = target_motion >= lower
                if math.isfinite(upper):
                    selected = selected & (target_motion < upper)
                bucket_stats[name][bucket]["abs"] += pixel_abs[selected].sum().item()
                bucket_stats[name][bucket]["sq"] += pixel_sq[selected].sum().item()
                bucket_stats[name][bucket]["count"] += selected.sum().item()
        mask_sums["camera"] += state["camera_weight"].sum().item()
        mask_sums["object"] += state["object_weight"].sum().item()
        mask_sums["synthesis"] += state["synthesis_weight"].sum().item()
        mask_sums["disocclusion"] += state["disocclusion"].sum().item()
        mask_sums["pixels"] += state["synthesis_weight"].numel()
        support = state["object_support"]
        confidence_sums["cycle"] += (state["cycle_confidence"] * support).sum().item()
        confidence_sums["occlusion"] += (state["occlusion_confidence"] * support).sum().item()
        confidence_sums["support"] += support.sum().item()
        if "cluster_entropy" in state:
            confidence_sums["cluster_entropy"] += state["cluster_entropy"].sum().item()
            confidence_sums["cluster_count"] += state["cluster_entropy"].numel()
        for bucket, (lower, upper) in bucket_ranges.items():
            selected = target_motion >= lower
            if math.isfinite(upper):
                selected = selected & (target_motion < upper)
            object_by_bucket[bucket]["weight"] += state["object_weight"][selected].sum().item()
            object_by_bucket[bucket]["permission"] += state["dynamic_object_permission"][selected].sum().item()
            object_by_bucket[bucket]["count"] += selected.sum().item()
    results = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute()
        dynamic_mse = extras[name]["sq"] / max(extras[name]["count"], 1)
        metrics.update({
            "dynamic_MAE": extras[name]["abs"] / max(extras[name]["count"], 1),
            "dynamic_MSE": dynamic_mse,
            "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
            "Motion_IoU": motion_counts[name]["tp"] / max(
                motion_counts[name]["tp"] + motion_counts[name]["fp"] + motion_counts[name]["fn"], 1
            ),
            "Motion_F1": 2 * motion_counts[name]["tp"] / max(
                2 * motion_counts[name]["tp"] + motion_counts[name]["fp"] + motion_counts[name]["fn"], 1
            ),
        })
        metrics["motion_magnitude_buckets"] = {}
        for bucket, values in bucket_stats[name].items():
            count = max(values["count"], 1)
            mse = values["sq"] / count
            metrics["motion_magnitude_buckets"][bucket] = {
                "MAE": values["abs"] / count,
                "MSE": mse,
                "PSNR": -10 * math.log10(max(mse, 1e-12)),
                "pixel_count": values["count"],
            }
        results[name] = metrics
    results["masks"] = {
        name + "_fraction": mask_sums[name] / max(mask_sums["pixels"], 1)
        for name in ("camera", "object", "synthesis", "disocclusion")
    }
    results["object_path_by_motion_bucket"] = {
        bucket: {
            "object_weight": values["weight"] / max(values["count"], 1),
            "object_permission": values["permission"] / max(values["count"], 1),
            "pixel_count": values["count"],
        }
        for bucket, values in object_by_bucket.items()
    }
    results["transport_diagnostics"] = {
        "cycle_confidence_on_support": confidence_sums["cycle"] / max(confidence_sums["support"], 1),
        "non_occlusion_confidence_on_support": confidence_sums["occlusion"] / max(confidence_sums["support"], 1),
        "cluster_assignment_entropy": confidence_sums["cluster_entropy"] / max(confidence_sums["cluster_count"], 1),
    }
    return results


def main():
    a = parse_args()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(a)
    make_dataset = lambda split: VideoForecastDataset(
        a.cache_root, split, a.image_size, paths=splits[split], cache_items=False
    )
    datasets = {
        "train": limited(make_dataset("train"), a.max_train),
        "val": limited(make_dataset("val"), a.max_val),
        "test": limited(make_dataset("test"), a.max_test),
    }
    train_loader = DataLoader(
        datasets["train"], a.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(a.seed), collate_fn=video_pad_collate,
        num_workers=0,
    )
    val_loader = DataLoader(datasets["val"], a.batch_size, collate_fn=video_pad_collate, num_workers=0)
    test_loader = DataLoader(datasets["test"], 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(a, device)
    model = StructuredCaPSR(
        a.sigma, a.hidden, num_motion_clusters=a.num_motion_clusters
    ).to(device)
    if a.init_checkpoint:
        initial = torch.load(a.init_checkpoint, map_location="cpu")
        state_dict = initial.get("model", initial)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"incompatible init checkpoint: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    import lpips
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    best, start_epoch, history = float("inf"), 1, []
    last_path, best_path = output / "last.pt", output / "best.pt"
    if a.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        if (output / "history.json").exists():
            history = json.loads((output / "history.json").read_text())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({"mode": a.mode, "split": split_meta,
                      "used": {name: len(value) for name, value in datasets.items()},
                      "image_size": a.image_size,
                      "trainable_params": trainable}), flush=True)
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in (
            "loss", "reconstruction", "visibility", "temporal", "lpips", "gradient",
            "static_identity", "correction", "mask_budget", "object_routing",
        )}
        for batch in train_loader:
            batch = move(batch, device)
            prediction, baseline, state = forward_model(model, a.mode, base, trajectory, batch)
            loss, pieces = loss_fn(prediction, baseline, state, batch, a, lpips_model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            for name, value in pieces.items():
                totals[name] += value
        val = validation(model, a.mode, base, trajectory, val_loader, device)
        score = val["MSE"] + 0.01 * val["dynamic_MAE"] + 0.05 * val["temporal_MAE"]
        row = {"epoch": epoch, **{f"train_{name}": value / len(train_loader)
                                   for name, value in totals.items()},
               **{f"val_{name}": value for name, value in val.items()}, "score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best": min(best, score), "config": vars(a),
                      "trainable_params": trainable}
        torch.save(checkpoint, last_path)
        if score < best:
            best = score
            torch.save(checkpoint, best_path)
        (output / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, a.mode, base, trajectory, test_loader, device,
                       trajectory_checkpoint["future_frames"], a.motion_threshold)
    summary = {
        "metrics": metrics,
        "meta": {"mode": a.mode, "best_epoch": checkpoint["epoch"],
                 "image_size": a.image_size,
                 "trainable_params": trainable, "split": split_meta,
                 "used": {name: len(value) for name, value in datasets.items()},
                 "equal_parameter_model_class": "StructuredCaPSR"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
