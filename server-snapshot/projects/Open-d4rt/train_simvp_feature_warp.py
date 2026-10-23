#!/usr/bin/env python3
"""First-round causal D4RT -> frozen-SimVP feature-warp experiment.

The SimVP backbone and the future-trajectory model are frozen.  Three equal-
capacity correction models are trained with identical losses and initialization:
correct predicted geometry, zero geometry, and cross-video shuffled geometry.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", default="")
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train", type=int, default=216)
    p.add_argument("--max-val", type=int, default=80)
    p.add_argument("--max-test", type=int, default=80)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--track-weight", type=float, default=0.25)
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--feature-correction-weight", type=float, default=0.002)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--flow-kernel", type=int, default=9)
    p.add_argument("--feature-scale", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def last_observed_uv(batch):
    valid = batch["past_valid"].bool()
    steps = valid.shape[2]
    index = (valid.long() * torch.arange(1, steps + 1, device=valid.device)).argmax(2)
    uv = torch.nan_to_num(batch["past_uv"].float()).clamp(0, 1)
    anchor = uv.gather(2, index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    return anchor, valid.any(2)


def extract_frozen_features(backbone, context):
    """Run frozen SimVP up to its future latent and temporal skip tensors."""
    b, t, c, h0, w0 = context.shape
    embed, skip = backbone.backbone.enc(context.reshape(b * t, c, h0, w0))
    ch, h, w = embed.shape[1:]
    source = embed.reshape(b, t, ch, h, w)[:, -1]
    hidden = backbone.backbone.hid(embed.reshape(b, t, ch, h, w))
    hidden = backbone.latent_time(hidden)
    skip = backbone.skip_time(skip.reshape(b, t, *skip.shape[1:]))
    return source.detach(), hidden.detach(), skip.detach()


def decode_frozen(backbone, hidden, skip):
    b, k, c, h, w = hidden.shape
    decoded = backbone.backbone.dec(
        hidden.reshape(b * k, c, h, w),
        skip.reshape(b * k, *skip.shape[2:]),
    )
    return torch.sigmoid(decoded.reshape(b, k, 3, *decoded.shape[-2:]))


class SparseDenseFlow(nn.Module):
    """Bilinearly splat sparse trajectories then normalized-convolve locally."""

    def __init__(self, kernel_size=9):
        super().__init__()
        size = int(kernel_size) // 2 * 2 + 1
        coord = torch.arange(size, dtype=torch.float32) - size // 2
        sigma = max(size / 4.0, 1.0)
        kernel = torch.exp(-(coord[:, None].square() + coord[None, :].square()) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None])

    @staticmethod
    def _splat(values, uv, weight, height, width):
        # values B,N,K,C; uv B,N,K,2; weight B,N,K
        b, n, k, c = values.shape
        xf = uv[..., 0].clamp(0, 1) * (width - 1)
        yf = uv[..., 1].clamp(0, 1) * (height - 1)
        x0, y0 = xf.floor().long(), yf.floor().long()
        x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
        dx, dy = xf - x0.float(), yf - y0.float()
        numerator = values.new_zeros((b, k, height * width, c))
        denominator = values.new_zeros((b, k, height * width, 1))
        source = values.permute(0, 2, 1, 3)
        active = weight.permute(0, 2, 1)[..., None]
        for xx, yy, bilinear in (
            (x0, y0, (1 - dx) * (1 - dy)),
            (x1, y0, dx * (1 - dy)),
            (x0, y1, (1 - dx) * dy),
            (x1, y1, dx * dy),
        ):
            wgt = bilinear.permute(0, 2, 1)[..., None] * active
            index = (yy * width + xx).permute(0, 2, 1)[..., None]
            numerator.scatter_add_(2, index.expand(-1, -1, -1, c), source * wgt)
            denominator.scatter_add_(2, index, wgt)
        numerator = numerator.permute(0, 1, 3, 2).reshape(b * k, c, height, width)
        denominator = denominator.permute(0, 1, 3, 2).reshape(b * k, 1, height, width)
        return numerator, denominator

    def forward(self, anchor_uv, future_uv, reliability, height, width):
        displacement = future_uv - anchor_uv[:, :, None]
        numerator, denominator = self._splat(displacement, future_uv, reliability, height, width)
        padding = self.kernel.shape[-1] // 2
        channels = numerator.shape[1]
        numerator = F.conv2d(numerator, self.kernel.expand(channels, 1, -1, -1),
                             padding=padding, groups=channels)
        denominator = F.conv2d(denominator, self.kernel, padding=padding)
        flow = numerator / denominator.clamp_min(1e-5)
        support = (1 - torch.exp(-4 * denominator)).clamp(0, 1)
        b, _, k = anchor_uv.shape[0], future_uv.shape[1], future_uv.shape[2]
        return (flow.reshape(b, k, 2, height, width),
                support.reshape(b, k, 1, height, width))


def backward_warp(source, flow):
    """Move source content to future targets using a target->source grid."""
    b, k, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=source.device, dtype=source.dtype),
        torch.linspace(-1, 1, w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    identity = torch.stack((xx, yy), -1)[None, None].expand(b, k, -1, -1, -1)
    grid = identity - 2 * flow.permute(0, 1, 3, 4, 2)
    src = source[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, *source.shape[1:])
    return F.grid_sample(src, grid.reshape(b * k, h, w, 2), mode="bilinear",
                         padding_mode="border", align_corners=True).reshape(b, k, *source.shape[1:])


class DualPathFeatureWarp(nn.Module):
    """Appearance generation plus permission-gated geometric propagation."""

    def __init__(self, channels=16, flow_kernel=9, scale=0.15):
        super().__init__()
        from video_prediction_model import SpatialGeometryAdapter
        self.raster = SparseDenseFlow(flow_kernel)
        self.geometry = SpatialGeometryAdapter(channels)
        self.scale = float(scale)
        self.appearance = nn.Sequential(
            nn.Conv3d(channels, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, channels, 3, padding=1),
        )
        geo_channels = channels * 3 + 3
        self.propagation = nn.Sequential(
            nn.Conv3d(geo_channels, 48, 3, padding=1), nn.GELU(),
            nn.Conv3d(48, channels, 3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(geo_channels, 24, 3, padding=1), nn.GELU(),
            nn.Conv3d(24, 1, 1),
        )
        nn.init.zeros_(self.appearance[-1].weight)
        nn.init.zeros_(self.appearance[-1].bias)
        nn.init.zeros_(self.propagation[-1].weight)
        nn.init.zeros_(self.propagation[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)
        self.last = {}

    def forward(self, source, hidden, geometry, future_uv, reliability, anchor_uv):
        b, k, c, h, w = hidden.shape
        flow, dense_support = self.raster(anchor_uv, future_uv, reliability, h, w)
        geometry_map, sparse_support = self.geometry(
            geometry, future_uv, reliability, h, w, return_support=True,
        )
        # A local dense flow may cover neighbours, but it cannot exceed the
        # geometry-derived motion permission.
        support = torch.maximum(dense_support, sparse_support).clamp(0, 1)
        warped = backward_warp(source, flow)
        geo_input = torch.cat((hidden, warped, geometry_map, flow, support), 2).permute(0, 2, 1, 3, 4)
        app_delta = self.appearance(hidden.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        geo_delta = self.propagation(geo_input).permute(0, 2, 1, 3, 4)
        gate = torch.sigmoid(self.gate(geo_input).permute(0, 2, 1, 3, 4)) * support
        app_delta = self.scale * torch.tanh(app_delta)
        geo_delta = self.scale * torch.tanh(geo_delta) * gate
        corrected = hidden + app_delta + geo_delta
        self.last = {"flow": flow.detach(), "support": support.detach(), "gate": gate.detach(),
                     "appearance_delta": app_delta, "geometry_delta": geo_delta,
                     "warped": warped.detach()}
        return corrected, app_delta, geo_delta


def geometry_bundle(mode, joint, batch, future_frames):
    from train_pointodyssey_dmvfn_injection import geometry_for_source
    raw, reliability, future_uv = geometry_for_source(
        "predicted", joint, batch, future_frames, "xyz_uv", True, True,
    )
    anchor, point_ok = last_observed_uv(batch)
    reliability = reliability * point_ok[:, :, None].float()
    if mode == "zero":
        return torch.zeros_like(raw), future_uv, torch.zeros_like(reliability), anchor
    if mode == "shuffled":
        if raw.shape[0] > 1:
            order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
            return raw[order], future_uv[order], reliability[order], anchor[order]
        # A deterministic within-video permutation is only a last-batch fallback.
        order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
        return raw[:, order], future_uv[:, order], reliability[:, order], anchor[:, order]
    if mode != "correct":
        raise ValueError(mode)
    return raw, future_uv, reliability, anchor


def sample_video(video, uv):
    outputs = []
    for step in range(video.shape[1]):
        grid = uv[:, :, step].mul(2).sub(1).unsqueeze(2)
        value = F.grid_sample(video[:, step], grid, mode="bilinear", padding_mode="border", align_corners=True)
        outputs.append(value[:, :, :, 0].transpose(1, 2))
    return torch.stack(outputs, 2)


def d4rt_track_loss(prediction, target, batch, temporal_weight):
    raw_uv = batch["future_uv_teacher"].float()
    uv = torch.nan_to_num(raw_uv).clamp(0, 1)
    valid = batch["future_valid"].bool() & torch.isfinite(raw_uv).all(-1)
    confidence = torch.nan_to_num(batch["future_confidence"].float()).clamp(0, 1)
    anchor, past_valid = last_observed_uv(batch)
    displacement = torch.where(valid, (uv - anchor[:, :, None]).norm(dim=-1), torch.zeros_like(confidence))
    dynamic = displacement.amax(2) > 0.05
    weight = confidence * valid.float() * dynamic[:, :, None].float() * past_valid[:, :, None].float()
    pred = sample_video(prediction, uv)
    truth = sample_video(target, uv)
    point = (pred - truth).abs().mean(-1)
    pred_velocity = torch.cat((pred[:, :, :1] * 0, pred[:, :, 1:] - pred[:, :, :-1]), 2)
    truth_velocity = torch.cat((truth[:, :, :1] * 0, truth[:, :, 1:] - truth[:, :, :-1]), 2)
    temporal = (pred_velocity - truth_velocity).abs().mean(-1)
    denominator = weight.sum().clamp_min(1)
    return ((point * weight).sum() + temporal_weight * (temporal * weight).sum()) / denominator


def objective(prediction, target, last, baseline, app_delta, geo_delta, batch, lpips_model, args):
    error = prediction - target
    rgb = error.abs().mean() + 0.1 * error.square().mean()
    pred_prev = torch.cat((last[:, None], prediction[:, :-1]), 1)
    target_prev = torch.cat((last[:, None], target[:, :-1]), 1)
    temporal = ((prediction - pred_prev) - (target - target_prev)).abs().mean()
    b, k = prediction.shape[:2]
    perceptual = lpips_model(
        prediction.reshape(b * k, 3, *prediction.shape[-2:]).mul(2).sub(1),
        target.reshape(b * k, 3, *target.shape[-2:]).mul(2).sub(1),
    ).mean()
    track = d4rt_track_loss(prediction, target, batch, args.track_temporal_weight)
    correction = (prediction - baseline).abs().mean()
    feature_correction = app_delta.abs().mean() + geo_delta.abs().mean()
    loss = (rgb + args.lpips_weight * perceptual + args.temporal_weight * temporal
            + args.track_weight * track + args.correction_weight * correction
            + args.feature_correction_weight * feature_correction)
    return loss, {"rgb": rgb, "lpips": perceptual, "temporal": temporal, "track": track,
                  "pixel_correction": correction, "feature_correction": feature_correction}


@torch.no_grad()
def validation(models, loader, backbone, joint, device, future_frames):
    totals = {name: defaultdict(float) for name in models}
    batches = 0
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = extract_frozen_features(backbone, batch["context_rgb"])
        for name, model in models.items():
            geometry = geometry_bundle(name, joint, batch, future_frames)
            corrected, _, _ = model(source, hidden, *geometry)
            prediction = decode_frozen(backbone, corrected, skip)
            error = prediction - batch["future_rgb"]
            totals[name]["MSE"] += error.square().mean().item()
            totals[name]["MAE"] += error.abs().mean().item()
            totals[name]["support"] += model.last["support"].mean().item()
            totals[name]["gate"] += model.last["gate"].mean().item()
        batches += 1
    return {name: {key: value / max(batches, 1) for key, value in values.items()}
            for name, values in totals.items()}


def train(models, train_loader, val_loader, backbone, joint, lpips_model, device, args, output, future_frames):
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
                  for name, model in models.items()}
    best = {name: float("inf") for name in models}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        totals = {name: defaultdict(float) for name in models}
        for model in models.values():
            model.train()
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)
        batches = 0
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = extract_frozen_features(backbone, batch["context_rgb"])
                baseline = decode_frozen(backbone, hidden, skip)
            for name, model in models.items():
                geometry = geometry_bundle(name, joint, batch, future_frames)
                corrected, app_delta, geo_delta = model(source, hidden, *geometry)
                prediction = decode_frozen(backbone, corrected, skip)
                loss, pieces = objective(
                    prediction, batch["future_rgb"], batch["context_rgb"][:, -1], baseline,
                    app_delta, geo_delta, batch, lpips_model, args,
                )
                if hasattr(model, "auxiliary_loss"):
                    auxiliary, auxiliary_pieces = model.auxiliary_loss(
                        batch["future_rgb"], batch["context_rgb"][:, -1],
                    )
                    loss = loss + auxiliary
                    pieces.update(auxiliary_pieces)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                for key, value in pieces.items():
                    totals[name][key] += float(value.detach())
            batches += 1
        val = validation(models, val_loader, backbone, joint, device, future_frames)
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, model in models.items():
            for key, value in totals[name].items():
                row[f"{name}_train_{key}"] = value / max(batches, 1)
            for key, value in val[name].items():
                row[f"{name}_val_{key}"] = value
            state = {"model": model.state_dict(), "epoch": epoch, "validation": val[name],
                     "config": vars(args), "trainable_params": sum(p.numel() for p in model.parameters())}
            torch.save(state, output / f"{name}_last.pt")
            if val[name]["MSE"] < best[name]:
                best[name] = val[name]["MSE"]
                torch.save(state, output / f"{name}_best.pt")
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "feature_warp", **row}), flush=True)
    for name, model in models.items():
        model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device)["model"])
    return history


def sequence_metrics(prediction, target, last):
    error = prediction - target
    motion = (target - last[:, None]).abs().mean(2, keepdim=True) > 0.05
    selected = motion.expand_as(error)
    mse = error.square().mean().item()
    mae = error.abs().mean().item()
    dmse = error.square()[selected].mean().item() if selected.any() else float("nan")
    dmae = error.abs()[selected].mean().item() if selected.any() else float("nan")
    return {"MSE": mse, "MAE": mae, "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "dynamic_MSE": dmse, "dynamic_MAE": dmae,
            "dynamic_PSNR": -10 * math.log10(max(dmse, 1e-12)) if math.isfinite(dmse) else float("nan")}


def paired_bootstrap(records, reference, candidate, seed=42, samples=10000):
    lookup = defaultdict(dict)
    for row in records:
        lookup[row["name"]][row["variant"]] = row
    pairs = [values for values in lookup.values() if reference in values and candidate in values]
    rng = np.random.default_rng(seed)
    output = {"reference": reference, "candidate": candidate, "num_sequences": len(pairs)}
    for metric in ("MSE", "MAE", "dynamic_MSE", "dynamic_MAE"):
        differences = np.asarray([p[reference][metric] - p[candidate][metric] for p in pairs], np.float64)
        differences = differences[np.isfinite(differences)]
        if not len(differences):
            continue
        means = differences[rng.integers(0, len(differences), (samples, len(differences)))].mean(1)
        output[metric] = {"improvement": float(differences.mean()),
                          "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
                          "win_rate": float((differences > 0).mean())}
    return output


@torch.no_grad()
def evaluate(models, loader, backbone, joint, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator
    variants = ["frozen_simvp", *models.keys(), "correct_model_zero_input", "correct_model_shuffled_input"]
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in variants}
    records = []
    utilization = defaultdict(list)
    motion_counts = {name: defaultdict(float) for name in variants}
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = extract_frozen_features(backbone, batch["context_rgb"])
        baseline = decode_frozen(backbone, hidden, skip)
        predictions = {"frozen_simvp": baseline}
        for name, model in models.items():
            bundle = geometry_bundle(name, joint, batch, future_frames)
            corrected, _, _ = model(source, hidden, *bundle)
            predictions[name] = decode_frozen(backbone, corrected, skip)
        correct_model = models["correct"]
        for mode, variant in (("zero", "correct_model_zero_input"),
                              ("shuffled", "correct_model_shuffled_input")):
            corrected, _, _ = correct_model(source, hidden, *geometry_bundle(mode, joint, batch, future_frames))
            predictions[variant] = decode_frozen(backbone, corrected, skip)
        utilization["correct_vs_zero_output_l1"].append(
            (predictions["correct"] - predictions["correct_model_zero_input"]).abs().mean().item())
        utilization["correct_vs_shuffled_output_l1"].append(
            (predictions["correct"] - predictions["correct_model_shuffled_input"]).abs().mean().item())
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            target_motion = (target - last[:, None]).abs().mean(2) > 0.05
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[variant]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[variant]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[variant]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, name in enumerate(batch["name"]):
                records.append({"name": name, "variant": variant,
                                **sequence_metrics(prediction[index:index + 1], target[index:index + 1],
                                                   last[index:index + 1])})
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in metrics:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    bootstrap = {
        candidate: paired_bootstrap(records, "frozen_simvp", candidate, seed)
        for candidate in variants if candidate != "frozen_simvp"
    }
    control_bootstrap = {
        control: paired_bootstrap(records, control, "correct", seed)
        for control in ("zero", "shuffled", "correct_model_zero_input", "correct_model_shuffled_input")
    }
    return (metrics, records, bootstrap, control_bootstrap,
            {k: float(np.mean(v)) for k, v in utilization.items()})


def select_paths(args):
    from train_pointodyssey_dmvfn_injection import combined_splits
    splits, metadata = combined_splits(SimpleNamespace(**vars(args)))
    if args.sample_manifest:
        spec = json.loads(Path(args.sample_manifest).read_text())
        selected = spec.get("selected", spec.get("paths", {}))
        for split in ("train", "val", "test"):
            if split in selected:
                allowed = {str(Path(path)) for path in splits[split]}
                chosen = [Path(path) for path in selected[split] if str(Path(path)) in allowed]
                if not chosen:
                    raise RuntimeError(f"sample manifest selected no valid {split} paths")
                splits[split] = chosen
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    for split, limit in limits.items():
        splits[split] = splits[split][:limit] if limit > 0 else splits[split]
    metadata["used"] = {split: len(paths) for split, paths in splits.items()}
    return splits, metadata


def main():
    args = arguments()
    seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_metadata = select_paths(args)
    datasets = {split: VideoForecastDataset(args.cache_root, split, 64, paths=paths, cache_items=False)
                for split, paths in splits.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True, generator=generator,
                            collate_fn=video_pad_collate, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], args.batch_size, collate_fn=video_pad_collate,
                          num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], args.batch_size, collate_fn=video_pad_collate,
                           num_workers=args.num_workers),
    }
    backbone_checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(backbone_checkpoint).to(device)
    backbone.load_state_dict(backbone_checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(backbone_checkpoint["future_frames"])
    trajectory_checkpoint = torch.load(args.trajectory_checkpoint, map_location="cpu")
    cls = (VelocityResidualImplicitForecaster
           if trajectory_checkpoint.get("model_variant") == "velocity_residual"
           else ImplicitConditionedJointForecaster)
    joint = cls(trajectory_checkpoint["future_frames"], trajectory_checkpoint["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"])
    joint.eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)
    prototype = DualPathFeatureWarp(16, args.flow_kernel, args.feature_scale).to(device)
    models = {"correct": prototype, "zero": copy.deepcopy(prototype), "shuffled": copy.deepcopy(prototype)}
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    if args.eval_only:
        for name, model in models.items():
            model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device)["model"])
    else:
        train(models, loaders["train"], loaders["val"], backbone, joint, lpips_model,
              device, args, output, future_frames)
    metrics, records, bootstrap, control_bootstrap, utilization = evaluate(
        models, loaders["test"], backbone, joint, device, future_frames, args.seed,
    )
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "correct_vs_controls_paired_bootstrap": control_bootstrap,
        "geometry_utilization": utilization,
        "meta": {"config": vars(args), "split": split_metadata,
                 "trainable_params_each": sum(p.numel() for p in prototype.parameters()),
                 "frozen_backbone_params": sum(p.numel() for p in backbone.parameters()),
                 "frozen_trajectory_params": sum(p.numel() for p in joint.parameters()),
                 "fairness": "same samples, initialization, capacity, loss and optimizer; only geometry input differs",
                 "causality": "all injected future geometry is predicted from history; teacher tracks are loss/evaluation only"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
