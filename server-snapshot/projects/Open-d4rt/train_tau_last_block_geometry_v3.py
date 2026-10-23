#!/usr/bin/env python3
"""Full-data TAU experiment with geometry inside the final temporal block.

Two equal-schedule models start from the same TAU checkpoint:
  * geometry: D4RT future geometry is injected between TAU temporal blocks 5/6;
  * control: identical partial TAU fine-tuning with the geometry branch disabled.

Only the final TAU temporal block, latent-time projection, decoder readout and
the geometry adapter are trainable.  A frozen TAU copy is retained as teacher
and evaluation baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_simvp_feature_warp as core
import train_simvp_feature_warp_128 as fair


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
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--tau-lr", type=float, default=1e-5)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--track-weight", type=float, default=0.25)
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--feature-correction-weight", type=float, default=0.002)
    p.add_argument("--anchor-weight", type=float, default=0.02)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--flow-kernel", type=int, default=17)
    p.add_argument("--feature-scale", type=float, default=0.10)
    p.add_argument("--min-point-motion", type=float, default=0.015)
    p.add_argument("--full-point-motion", type=float, default=0.060)
    p.add_argument("--edge-margin", type=float, default=0.025)
    p.add_argument("--rank-weight", type=float, default=0.50)
    p.add_argument("--rank-margin", type=float, default=0.001)
    p.add_argument("--selection-dynamic-weight", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LastBlockGeometryAdapter(nn.Module):
    """Rasterize K-step geometry into TAU's 256-channel temporal bottleneck."""

    def __init__(self, future_frames, bottleneck_channels=256, geometry_channels=32,
                 flow_kernel=17, scale=0.10, min_motion=0.015,
                 full_motion=0.060, edge_margin=0.025):
        super().__init__()
        from video_prediction_model import SpatialGeometryAdapter

        if future_frames * geometry_channels != bottleneck_channels:
            raise ValueError("future_frames * geometry_channels must equal bottleneck_channels")
        self.future_frames = int(future_frames)
        self.raster = core.SparseDenseFlow(flow_kernel)
        self.geometry = SpatialGeometryAdapter(geometry_channels)
        self.scale = float(scale)
        self.min_motion = float(min_motion)
        self.motion_span = max(float(full_motion) - float(min_motion), 1e-4)
        self.edge_margin = max(float(edge_margin), 1e-4)

        input_channels = bottleneck_channels * 2 + future_frames * 3
        self.proposal = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, bottleneck_channels, 3, padding=1),
        )
        self.risk_gate = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        # Near-identity start, but unlike exact zero initialization all adapter
        # parameters receive gradients from the first update.
        self.alpha = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.proposal[-1].weight)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.zeros_(self.risk_gate[-1].weight)
        nn.init.constant_(self.risk_gate[-1].bias, -2.0)
        self.last = {}

    def permission(self, future_uv, reliability, anchor_uv):
        displacement = (future_uv - anchor_uv[:, :, None]).norm(dim=-1)
        motion = ((displacement - self.min_motion) / self.motion_span).clamp(0, 1)
        edge_distance = torch.minimum(
            torch.minimum(future_uv[..., 0], 1 - future_uv[..., 0]),
            torch.minimum(future_uv[..., 1], 1 - future_uv[..., 1]),
        )
        visible = (edge_distance / self.edge_margin).clamp(0, 1)
        return reliability * motion * visible

    def forward(self, temporal, geometry, future_uv, reliability, anchor_uv):
        b, _, height, width = temporal.shape
        permission = self.permission(future_uv, reliability, anchor_uv)
        flow, dense_support = self.raster(
            anchor_uv, future_uv, permission, height, width)
        geometry_map, sparse_support = self.geometry(
            geometry, future_uv, permission, height, width, return_support=True)
        support = torch.maximum(dense_support, sparse_support).clamp(0, 1)
        geometry_flat = geometry_map.reshape(b, -1, height, width)
        flow_flat = flow.reshape(b, -1, height, width)
        support_flat = support.reshape(b, -1, height, width)
        features = torch.cat(
            (temporal, geometry_flat, flow_flat, support_flat), dim=1)
        proposal = self.proposal(features)
        learned_gate = torch.sigmoid(self.risk_gate(features))
        spatial_permission = support.amax(dim=1)
        gate = learned_gate * spatial_permission
        amplitude = self.scale * torch.tanh(self.alpha)
        correction = amplitude * torch.tanh(proposal) * gate
        self.last = {
            "permission": permission.detach(),
            "support": support.detach(),
            "gate": gate.detach(),
            "correction": correction,
            "amplitude": amplitude.detach(),
        }
        return temporal + correction, correction


class PartialTAUTail(nn.Module):
    """Trainable final temporal block and small decoder head from a TAU model."""

    def __init__(self, backbone, future_frames, args):
        super().__init__()
        self.context_frames = int(backbone.context_frames)
        self.feature_channels = int(backbone.backbone_config["hid_s"])
        self.last_temporal = copy.deepcopy(backbone.backbone.hid.enc[-1])
        self.latent_time = copy.deepcopy(backbone.latent_time)
        self.skip_time = copy.deepcopy(backbone.skip_time)
        self.decoder = copy.deepcopy(backbone.backbone.dec)
        self.injector = LastBlockGeometryAdapter(
            future_frames=future_frames,
            flow_kernel=args.flow_kernel,
            scale=args.feature_scale,
            min_motion=args.min_point_motion,
            full_motion=args.full_point_motion,
            edge_margin=args.edge_margin,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.last_temporal, self.latent_time,
                       self.decoder.readout, self.injector):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def set_train_mode(self):
        self.eval()
        self.last_temporal.train()
        self.latent_time.train()
        self.decoder.readout.train()
        self.injector.train()

    def optimizer(self, args):
        adapter = [p for p in self.injector.parameters() if p.requires_grad]
        tau = [p for name, p in self.named_parameters()
               if p.requires_grad and not name.startswith("injector.")]
        return torch.optim.AdamW([
            {"params": adapter, "lr": args.adapter_lr},
            {"params": tau, "lr": args.tau_lr},
        ], weight_decay=1e-4)

    def forward(self, prefix, skip, bundle=None, use_geometry=True):
        temporal = prefix
        if use_geometry:
            if bundle is None:
                raise ValueError("geometry bundle required")
            temporal, correction = self.injector(temporal, *bundle)
        else:
            correction = temporal.new_zeros(temporal.shape)
            self.injector.last = {
                "support": temporal.new_zeros(
                    temporal.shape[0], 1, 1, temporal.shape[2], temporal.shape[3]),
                "gate": temporal.new_zeros(
                    temporal.shape[0], 1, temporal.shape[2], temporal.shape[3]),
                "correction": correction,
                "amplitude": temporal.new_zeros(()),
            }
        packed = self.last_temporal(temporal)
        b, _, height, width = packed.shape
        hidden = packed.reshape(
            b, self.context_frames, self.feature_channels, height, width)
        future = self.latent_time(hidden)
        future_skip = self.skip_time(skip)
        k = future.shape[1]
        decoded = self.decoder(
            future.reshape(b * k, self.feature_channels, height, width),
            future_skip.reshape(b * k, *future_skip.shape[2:]),
        )
        return torch.sigmoid(decoded.reshape(b, k, 3, *decoded.shape[-2:])), correction


@torch.no_grad()
def frozen_prefix(backbone, context):
    b, t, channels, height, width = context.shape
    embed, skip = backbone.backbone.enc(
        context.reshape(b * t, channels, height, width))
    feature_channels, fh, fw = embed.shape[1:]
    packed = embed.reshape(b, t * feature_channels, fh, fw)
    for block in backbone.backbone.hid.enc[:-1]:
        packed = block(packed)
    skip = skip.reshape(b, t, *skip.shape[1:])
    return packed.detach(), skip.detach()


@torch.no_grad()
def frozen_prediction(backbone, prefix, skip):
    packed = backbone.backbone.hid.enc[-1](prefix)
    b, _, height, width = packed.shape
    t = int(backbone.context_frames)
    channels = int(backbone.backbone_config["hid_s"])
    hidden = packed.reshape(b, t, channels, height, width)
    future = backbone.latent_time(hidden)
    future_skip = backbone.skip_time(skip)
    k = future.shape[1]
    decoded = backbone.backbone.dec(
        future.reshape(b * k, channels, height, width),
        future_skip.reshape(b * k, *future_skip.shape[2:]),
    )
    return torch.sigmoid(decoded.reshape(b, k, 3, *decoded.shape[-2:]))


def shuffled_bundle(bundle):
    geometry, future_uv, reliability, anchor = bundle
    if geometry.shape[0] > 1:
        order = torch.roll(torch.arange(geometry.shape[0], device=geometry.device), 1)
        return geometry[order], future_uv[order], reliability[order], anchor[order]
    order = torch.arange(geometry.shape[1] - 1, -1, -1, device=geometry.device)
    return (geometry[:, order], future_uv[:, order],
            reliability[:, order], anchor[:, order])


def zero_bundle(bundle):
    geometry, future_uv, reliability, anchor = bundle
    return torch.zeros_like(geometry), future_uv, torch.zeros_like(reliability), anchor


def dynamic_l1(prediction, target, last, threshold):
    mask = ((target - last[:, None]).abs().mean(2) > threshold).float()[:, :, None]
    denominator = (mask.sum() * prediction.shape[2]).clamp_min(1)
    return ((prediction - target).abs() * mask).sum() / denominator


def prediction_objective(prediction, target, last, teacher, correction,
                         batch, lpips_model, args):
    loss, pieces = core.objective(
        prediction, target, last, teacher,
        torch.zeros_like(correction), correction, batch, lpips_model, args)
    anchor = (prediction - teacher).abs().mean()
    loss = loss + args.anchor_weight * anchor
    pieces["teacher_anchor"] = anchor
    return loss, pieces


@torch.no_grad()
def validate(models, loader, backbone, joint, device, future_frames, args):
    for model in models.values():
        model.eval()
    totals = {name: defaultdict(float) for name in ("frozen", *models)}
    batches = 0
    for raw_batch in loader:
        batch = fair.move(raw_batch, device)
        prefix, skip = frozen_prefix(backbone, batch["context_rgb"])
        teacher = frozen_prediction(backbone, prefix, skip)
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        predictions = {
            "frozen": teacher,
            "geometry": models["geometry"](prefix, skip, bundle, True)[0],
            "control": models["control"](prefix, skip, None, False)[0],
        }
        for name, prediction in predictions.items():
            error = prediction - batch["future_rgb"]
            totals[name]["MSE"] += error.square().mean().item()
            totals[name]["MAE"] += error.abs().mean().item()
            totals[name]["dynamic_MAE"] += dynamic_l1(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                args.dynamic_threshold).item()
        totals["geometry"]["support"] += models["geometry"].injector.last[
            "support"].mean().item()
        totals["geometry"]["gate"] += models["geometry"].injector.last[
            "gate"].mean().item()
        totals["geometry"]["amplitude"] += models["geometry"].injector.last[
            "amplitude"].item()
        batches += 1
    return {
        name: {key: value / max(batches, 1) for key, value in values.items()}
        for name, values in totals.items()
    }


def train(models, train_loader, val_loader, backbone, joint, lpips_model,
          device, args, output, future_frames):
    optimizers = {name: model.optimizer(args) for name, model in models.items()}
    best = {name: float("inf") for name in models}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        totals = {name: defaultdict(float) for name in models}
        for model in models.values():
            model.set_train_mode()
        batches = 0
        for raw_batch in train_loader:
            batch = fair.move(raw_batch, device)
            with torch.no_grad():
                prefix, skip = frozen_prefix(backbone, batch["context_rgb"])
                teacher = frozen_prediction(backbone, prefix, skip)
                bundle = core.geometry_bundle("correct", joint, batch, future_frames)

            # Geometry-conditioned partial fine-tuning.
            geometry_model = models["geometry"]
            prediction, correction = geometry_model(prefix, skip, bundle, True)
            reconstruction, pieces = prediction_objective(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                teacher, correction, batch, lpips_model, args)
            wrong_prediction, _ = geometry_model(
                prefix, skip, shuffled_bundle(bundle), True)
            correct_dynamic = dynamic_l1(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                args.dynamic_threshold)
            wrong_dynamic = dynamic_l1(
                wrong_prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                args.dynamic_threshold)
            rank = F.relu(args.rank_margin + correct_dynamic - wrong_dynamic)
            geometry_loss = reconstruction + args.rank_weight * rank
            optimizers["geometry"].zero_grad(set_to_none=True)
            geometry_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in geometry_model.parameters() if p.requires_grad], 1.0)
            optimizers["geometry"].step()
            totals["geometry"]["loss"] += geometry_loss.item()
            totals["geometry"]["rank"] += rank.item()
            totals["geometry"]["dynamic_correct"] += correct_dynamic.item()
            totals["geometry"]["dynamic_shuffled"] += wrong_dynamic.item()
            for key, value in pieces.items():
                totals["geometry"][key] += float(value.detach())

            # Same partial TAU fine-tuning schedule, with geometry disabled.
            control_model = models["control"]
            control_prediction, control_correction = control_model(
                prefix, skip, None, False)
            control_loss, control_pieces = prediction_objective(
                control_prediction, batch["future_rgb"],
                batch["context_rgb"][:, -1], teacher, control_correction,
                batch, lpips_model, args)
            optimizers["control"].zero_grad(set_to_none=True)
            control_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in control_model.parameters() if p.requires_grad], 1.0)
            optimizers["control"].step()
            totals["control"]["loss"] += control_loss.item()
            for key, value in control_pieces.items():
                totals["control"][key] += float(value.detach())
            batches += 1

        metrics = validate(
            models, val_loader, backbone, joint, device, future_frames, args)
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name in models:
            score = (metrics[name]["MSE"]
                     + args.selection_dynamic_weight * metrics[name]["dynamic_MAE"])
            row[f"{name}_selection_score"] = score
            for key, value in totals[name].items():
                row[f"train_{name}_{key}"] = value / max(batches, 1)
            for key, value in metrics[name].items():
                row[f"val_{name}_{key}"] = value
            state = {"model": models[name].state_dict(), "epoch": epoch,
                     "validation": metrics[name], "selection_score": score,
                     "config": vars(args)}
            torch.save(state, output / f"{name}_last.pt")
            if score < best[name]:
                best[name] = score
                torch.save(state, output / f"{name}_best.pt")
        for key, value in metrics["frozen"].items():
            row[f"val_frozen_{key}"] = value
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "tau_last_block_geometry_v3", **row}), flush=True)

    for name, model in models.items():
        state = torch.load(output / f"{name}_best.pt", map_location=device)
        model.load_state_dict(state["model"])


@torch.no_grad()
def evaluate(models, loader, backbone, joint, device, future_frames, args):
    from video_metrics import VideoMetricAccumulator

    variants = (
        "frozen_tau", "partial_control", "partial_geometry_correct",
        "partial_geometry_zero", "partial_geometry_shuffled",
    )
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in variants}
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    for model in models.values():
        model.eval()
    for raw_batch in loader:
        batch = fair.move(raw_batch, device)
        prefix, skip = frozen_prefix(backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        shuffled = shuffled_bundle(bundle)
        zero = zero_bundle(bundle)
        predictions = {
            "frozen_tau": frozen_prediction(backbone, prefix, skip),
            "partial_control": models["control"](prefix, skip, None, False)[0],
            "partial_geometry_correct": models["geometry"](
                prefix, skip, bundle, True)[0],
            "partial_geometry_zero": models["geometry"](
                prefix, skip, zero, True)[0],
            "partial_geometry_shuffled": models["geometry"](
                prefix, skip, shuffled, True)[0],
        }
        utilization["correct_vs_zero_l1"].append(
            (predictions["partial_geometry_correct"]
             - predictions["partial_geometry_zero"]).abs().mean().item())
        utilization["correct_vs_shuffled_l1"].append(
            (predictions["partial_geometry_correct"]
             - predictions["partial_geometry_shuffled"]).abs().mean().item())
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > args.dynamic_threshold
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            predicted_motion = ((prediction - last[:, None]).abs().mean(2)
                                > args.dynamic_threshold)
            motion_counts[variant]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[variant]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[variant]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, name in enumerate(batch["name"]):
                records.append({
                    "name": name, "variant": variant,
                    **core.sequence_metrics(
                        prediction[index:index + 1], target[index:index + 1],
                        last[index:index + 1]),
                })
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    references = (
        "frozen_tau", "partial_control", "partial_geometry_zero",
        "partial_geometry_shuffled",
    )
    bootstrap = {
        reference: core.paired_bootstrap(
            records, reference, "partial_geometry_correct", args.seed)
        for reference in references
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(values)) for key, values in utilization.items()}


def main():
    args = arguments()
    seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import (ImplicitConditionedJointForecaster,
                                  VelocityResidualImplicitForecaster)
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = fair.make_loaders(args)

    checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    if str(checkpoint.get("backbone", "")).lower() != "tau":
        raise ValueError("TAU checkpoint required")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError("backbone/image-size mismatch")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])

    trajectory_checkpoint = torch.load(args.trajectory_checkpoint, map_location="cpu")
    cls = (VelocityResidualImplicitForecaster
           if trajectory_checkpoint.get("model_variant") == "velocity_residual"
           else ImplicitConditionedJointForecaster)
    joint = cls(trajectory_checkpoint["future_frames"],
                trajectory_checkpoint["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"])
    joint.eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)

    prototype = PartialTAUTail(backbone, future_frames, args).to(device)
    models = {"geometry": prototype, "control": copy.deepcopy(prototype)}
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    train(models, loaders["train"], loaders["val"], backbone, joint,
          lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = evaluate(
        models, loaders["test"], backbone, joint, device, future_frames, args)
    trainable_counts = {
        name: sum(p.numel() for p in model.parameters() if p.requires_grad)
        for name, model in models.items()
    }
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args), "split": split_metadata,
            "used": {key: len(value) for key, value in datasets.items()},
            "trainable_params": trainable_counts,
            "frozen_backbone_params": sum(p.numel() for p in backbone.parameters()),
            "frozen_trajectory_params": sum(p.numel() for p in joint.parameters()),
            "injection": "between TAU temporal blocks 5 and 6",
            "unfrozen": "TAU block 6, latent_time, decoder readout, geometry adapter",
            "control": "same TAU layers and schedule, geometry branch disabled",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
