#!/usr/bin/env python3
"""Causal D4RT UV feature-warp experiment for a frozen TAU backbone.

This experiment fixes the central limitation of token-only injection: geometry
must transport source features to predicted future UV locations.  It trains
three identically initialized, equal-parameter adapters on one fixed split:

* correct: predicted D4RT future UV/XYZ with permission-gated feature warp;
* query_only: the same active network using only TAU/source RGB features;
* shuffled: cross-video geometry with the same architecture and objective.

Oracle future tracks are used only at evaluation time to measure the attainable
headroom of the propagation architecture.  The TAU backbone and trajectory
forecaster are always frozen.
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
from torch.utils.data import DataLoader

import train_simvp_feature_warp as core


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
    p.add_argument(
        "--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini"
    )
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--track-weight", type=float, default=0.25)
    p.add_argument("--track-temporal-weight", type=float, default=0.50)
    p.add_argument("--correction-weight", type=float, default=0.01)
    p.add_argument("--feature-correction-weight", type=float, default=0.002)
    p.add_argument("--dynamic-threshold", type=float, default=0.05)
    p.add_argument("--flow-kernel", type=int, default=17)
    p.add_argument("--feature-scale", type=float, default=0.10)
    p.add_argument("--min-point-motion", type=float, default=0.015)
    p.add_argument("--full-point-motion", type=float, default=0.060)
    p.add_argument("--edge-margin", type=float, default=0.025)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CausalTAUUVWarp(nn.Module):
    """Explicit UV transport with an equally active query-only control mode."""

    def __init__(
        self,
        channels,
        future_frames,
        flow_kernel=17,
        scale=0.10,
        min_motion=0.015,
        full_motion=0.060,
        edge_margin=0.025,
        input_mode="geometry",
    ):
        super().__init__()
        from video_prediction_model import SpatialGeometryAdapter

        if input_mode not in ("geometry", "query_only"):
            raise ValueError(input_mode)
        self.input_mode = input_mode
        self.raster = core.SparseDenseFlow(flow_kernel)
        self.geometry = SpatialGeometryAdapter(channels)
        self.scale = float(scale)
        self.min_motion = float(min_motion)
        self.motion_span = max(float(full_motion) - float(min_motion), 1e-4)
        self.edge_margin = max(float(edge_margin), 1e-4)
        self.horizon = nn.Parameter(
            torch.zeros(1, future_frames, channels, 1, 1)
        )

        # hidden, source/warped residual, geometry, horizon, UV flow, support
        input_channels = channels * 4 + 3
        self.proposal = nn.Sequential(
            nn.Conv3d(input_channels, 96, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(96, channels, 3, padding=1),
        )
        self.risk_gate = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 1, 1),
        )
        nn.init.normal_(self.horizon, std=0.01)
        nn.init.zeros_(self.proposal[-1].weight)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.zeros_(self.risk_gate[-1].weight)
        nn.init.constant_(self.risk_gate[-1].bias, -2.0)
        self.last = {}

    def _permission(self, future_uv, reliability, anchor_uv):
        displacement = (future_uv - anchor_uv[:, :, None]).norm(dim=-1)
        motion = (
            (displacement - self.min_motion) / self.motion_span
        ).clamp(0, 1)
        edge_distance = torch.minimum(
            torch.minimum(future_uv[..., 0], 1 - future_uv[..., 0]),
            torch.minimum(future_uv[..., 1], 1 - future_uv[..., 1]),
        )
        visible = (edge_distance / self.edge_margin).clamp(0, 1)
        return reliability * motion * visible

    def forward(self, source, hidden, geometry, future_uv, reliability, anchor_uv):
        batch, steps, channels, height, width = hidden.shape
        horizon = self.horizon.expand(batch, -1, -1, height, width)

        if self.input_mode == "query_only":
            # Same active proposal/gate capacity, but no geometry, flow or mask.
            flow = hidden.new_zeros((batch, steps, 2, height, width))
            support = hidden.new_ones((batch, steps, 1, height, width))
            permission = reliability.new_zeros(reliability.shape)
            geometry_map = hidden.new_zeros(hidden.shape)
            warped = source[:, None].expand(-1, steps, -1, -1, -1)
        else:
            permission = self._permission(future_uv, reliability, anchor_uv)
            flow, dense_support = self.raster(
                anchor_uv, future_uv, permission, height, width
            )
            geometry_map, sparse_support = self.geometry(
                geometry,
                future_uv,
                permission,
                height,
                width,
                return_support=True,
            )
            support = torch.maximum(dense_support, sparse_support).clamp(0, 1)
            warped = core.backward_warp(source, flow)

        warped_residual = warped - hidden
        features = torch.cat(
            (hidden, warped_residual, geometry_map, horizon, flow, support), dim=2
        ).permute(0, 2, 1, 3, 4)
        proposal = self.proposal(features).permute(0, 2, 1, 3, 4)
        learned_gate = torch.sigmoid(
            self.risk_gate(features).permute(0, 2, 1, 3, 4)
        )
        gate = learned_gate if self.input_mode == "query_only" else learned_gate * support
        correction = self.scale * torch.tanh(proposal) * gate
        self.last = {
            "flow": flow.detach(),
            "support": support.detach(),
            "gate": gate.detach(),
            "learned_gate": learned_gate.detach(),
            "permission": permission.detach(),
            "geometry_delta": correction,
            "warped": warped.detach(),
        }
        return hidden + correction, torch.zeros_like(correction), correction


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate

    split_args = SimpleNamespace(**vars(args))
    split_args.seed = args.split_seed
    splits, metadata = core.select_paths(split_args)
    datasets = {
        split: VideoForecastDataset(
            args.cache_root,
            split,
            args.image_size,
            paths=paths,
            cache_items=False,
        )
        for split, paths in splits.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=video_pad_collate,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            args.batch_size,
            collate_fn=video_pad_collate,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            args.batch_size,
            collate_fn=video_pad_collate,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }
    metadata["split_seed"] = args.split_seed
    return loaders, datasets, metadata, splits


def bundles(joint, batch, future_frames):
    correct = core.geometry_bundle("correct", joint, batch, future_frames)
    shuffled = core.geometry_bundle("shuffled", joint, batch, future_frames)
    raw, uv, reliability, anchor = correct
    zero = (
        torch.zeros_like(raw),
        uv,
        torch.zeros_like(reliability),
        anchor,
    )
    return {"correct": correct, "query_only": zero, "shuffled": shuffled}


def oracle_bundle(joint, batch, future_frames):
    from train_pointodyssey_dmvfn_injection import geometry_for_source

    raw, reliability, future_uv = geometry_for_source(
        "oracle", joint, batch, future_frames, "xyz_uv", True, True
    )
    anchor, point_ok = core.last_observed_uv(batch)
    reliability = reliability * point_ok[:, :, None].float()
    return raw, future_uv, reliability, anchor


@torch.no_grad()
def validate(models, loader, backbone, joint, device, future_frames):
    totals = {name: defaultdict(float) for name in models}
    batches_seen = 0
    for raw_batch in loader:
        batch = core.move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        current_bundles = bundles(joint, batch, future_frames)
        for name, model in models.items():
            corrected, _, _ = model(source, hidden, *current_bundles[name])
            prediction = core.decode_frozen(backbone, corrected, skip)
            error = prediction - batch["future_rgb"]
            totals[name]["MSE"] += error.square().mean().item()
            totals[name]["MAE"] += error.abs().mean().item()
            totals[name]["support"] += model.last["support"].mean().item()
            totals[name]["gate"] += model.last["gate"].mean().item()
            totals[name]["pixel_delta"] += (
                prediction
                - core.decode_frozen(backbone, hidden, skip)
            ).abs().mean().item()
        batches_seen += 1
    return {
        name: {key: value / max(batches_seen, 1) for key, value in values.items()}
        for name, values in totals.items()
    }


def train(
    models,
    train_loader,
    val_loader,
    backbone,
    joint,
    lpips_model,
    device,
    args,
    output,
    future_frames,
):
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for name, model in models.items()
    }
    best = {name: float("inf") for name in models}
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        totals = {name: defaultdict(float) for name in models}
        for model in models.values():
            model.train()
        batches_seen = 0
        for raw_batch in train_loader:
            batch = core.move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(
                    backbone, batch["context_rgb"]
                )
                baseline = core.decode_frozen(backbone, hidden, skip)
                current_bundles = bundles(joint, batch, future_frames)
            for name, model in models.items():
                corrected, app_delta, geo_delta = model(
                    source, hidden, *current_bundles[name]
                )
                prediction = core.decode_frozen(backbone, corrected, skip)
                loss, pieces = core.objective(
                    prediction,
                    batch["future_rgb"],
                    batch["context_rgb"][:, -1],
                    baseline,
                    app_delta,
                    geo_delta,
                    batch,
                    lpips_model,
                    args,
                )
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                for key, value in pieces.items():
                    totals[name][key] += float(value.detach())
            batches_seen += 1

        validation = validate(
            models, val_loader, backbone, joint, device, future_frames
        )
        row = {"epoch": epoch, "epoch_seconds": time.time() - started}
        for name, model in models.items():
            for key, value in totals[name].items():
                row[f"{name}_train_{key}"] = value / max(batches_seen, 1)
            for key, value in validation[name].items():
                row[f"{name}_val_{key}"] = value
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "validation": validation[name],
                "config": vars(args),
                "input_mode": model.input_mode,
                "trainable_params": sum(p.numel() for p in model.parameters()),
            }
            torch.save(state, output / f"{name}_last.pt")
            if validation[name]["MSE"] < best[name]:
                best[name] = validation[name]["MSE"]
                torch.save(state, output / f"{name}_best.pt")
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps({"stage": "tau_uv_warp_causal_v1", **row}), flush=True)

    for name, model in models.items():
        state = torch.load(
            output / f"{name}_best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])


@torch.no_grad()
def evaluate(models, loader, backbone, joint, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator

    variants = (
        "frozen_tau",
        "correct",
        "query_only",
        "shuffled",
        "correct_model_zero_input",
        "correct_model_shuffled_input",
        "correct_model_oracle_input",
    )
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in variants
    }
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)

    for model in models.values():
        model.eval()
    for raw_batch in loader:
        batch = core.move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        baseline = core.decode_frozen(backbone, hidden, skip)
        current_bundles = bundles(joint, batch, future_frames)
        predictions = {"frozen_tau": baseline}
        correct_support = None
        correct_gate = None
        for name, model in models.items():
            corrected, _, _ = model(source, hidden, *current_bundles[name])
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
            if name == "correct":
                correct_support = model.last["support"].mean().item()
                correct_gate = model.last["gate"].mean().item()

        correct_model = models["correct"]
        for variant, bundle in (
            ("correct_model_zero_input", current_bundles["query_only"]),
            ("correct_model_shuffled_input", current_bundles["shuffled"]),
            ("correct_model_oracle_input", oracle_bundle(joint, batch, future_frames)),
        ):
            corrected, _, _ = correct_model(source, hidden, *bundle)
            predictions[variant] = core.decode_frozen(backbone, corrected, skip)

        utilization["correct_vs_baseline_l1"].append(
            (predictions["correct"] - baseline).abs().mean().item()
        )
        utilization["correct_vs_shuffled_input_l1"].append(
            (
                predictions["correct"]
                - predictions["correct_model_shuffled_input"]
            ).abs().mean().item()
        )
        utilization["correct_support"].append(
            correct_support
        )
        utilization["correct_gate"].append(
            correct_gate
        )

        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > 0.05
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[variant]["tp"] += (
                predicted_motion & target_motion
            ).sum().item()
            motion_counts[variant]["fp"] += (
                predicted_motion & ~target_motion
            ).sum().item()
            motion_counts[variant]["fn"] += (
                ~predicted_motion & target_motion
            ).sum().item()
            for index, sequence_name in enumerate(batch["name"]):
                records.append(
                    {
                        "name": sequence_name,
                        "variant": variant,
                        **core.sequence_metrics(
                            prediction[index:index + 1],
                            target[index:index + 1],
                            last[index:index + 1],
                        ),
                    }
                )

    metrics = {name: acc.compute() for name, acc in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)

    baseline_bootstrap = {
        candidate: core.paired_bootstrap(
            records, "frozen_tau", candidate, seed
        )
        for candidate in variants if candidate != "frozen_tau"
    }
    causal_bootstrap = {
        control: core.paired_bootstrap(records, control, "correct", seed)
        for control in (
            "query_only",
            "shuffled",
            "correct_model_zero_input",
            "correct_model_shuffled_input",
        )
    }
    oracle_headroom = core.paired_bootstrap(
        records, "correct", "correct_model_oracle_input", seed
    )
    return (
        metrics,
        records,
        baseline_bootstrap,
        causal_bootstrap,
        oracle_headroom,
        {key: float(np.mean(values)) for key, values in utilization.items()},
    )


def main():
    args = arguments()
    seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import (
        ImplicitConditionedJointForecaster,
        VelocityResidualImplicitForecaster,
    )
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata, splits = make_loaders(args)
    (output / "split_paths.json").write_text(
        json.dumps(
            {
                "split_seed": args.split_seed,
                "paths": {
                    split: [str(path) for path in paths]
                    for split, paths in splits.items()
                },
            },
            indent=2,
        )
    )

    checkpoint = torch.load(
        args.backbone_checkpoint, map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("This experiment requires a TAU checkpoint")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError(
            f"checkpoint image size {checkpoint['image_size']} != {args.image_size}"
        )
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])

    trajectory_checkpoint = torch.load(
        args.trajectory_checkpoint, map_location="cpu", weights_only=False
    )
    trajectory_class = (
        VelocityResidualImplicitForecaster
        if trajectory_checkpoint.get("model_variant") == "velocity_residual"
        else ImplicitConditionedJointForecaster
    )
    joint = trajectory_class(
        trajectory_checkpoint["future_frames"],
        trajectory_checkpoint["hidden_dim"],
    ).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"])
    joint.eval()
    for parameter in joint.parameters():
        parameter.requires_grad_(False)

    channels = int(backbone.backbone_config["hid_s"])
    prototype = CausalTAUUVWarp(
        channels,
        future_frames,
        args.flow_kernel,
        args.feature_scale,
        args.min_point_motion,
        args.full_point_motion,
        args.edge_margin,
        input_mode="geometry",
    ).to(device)
    models = {
        "correct": prototype,
        "query_only": copy.deepcopy(prototype),
        "shuffled": copy.deepcopy(prototype),
    }
    models["query_only"].input_mode = "query_only"

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    if args.eval_only:
        for name, model in models.items():
            state = torch.load(
                output / f"{name}_best.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(state["model"])
    else:
        train(
            models,
            loaders["train"],
            loaders["val"],
            backbone,
            joint,
            lpips_model,
            device,
            args,
            output,
            future_frames,
        )

    (
        metrics,
        records,
        baseline_bootstrap,
        causal_bootstrap,
        oracle_headroom,
        utilization,
    ) = evaluate(
        models,
        loaders["test"],
        backbone,
        joint,
        device,
        future_frames,
        args.seed,
    )
    summary = {
        "metrics": metrics,
        "paired_bootstrap_vs_frozen_tau": baseline_bootstrap,
        "correct_vs_controls_paired_bootstrap": causal_bootstrap,
        "oracle_headroom_paired_bootstrap": oracle_headroom,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args),
            "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "trainable_params_each": sum(
                p.numel() for p in prototype.parameters()
            ),
            "frozen_backbone_params": sum(
                p.numel() for p in backbone.parameters()
            ),
            "frozen_trajectory_params": sum(
                p.numel() for p in joint.parameters()
            ),
            "architecture": (
                "D4RT future UV/XYZ -> sparse-to-dense flow -> TAU feature warp; "
                "horizon-conditioned proposal; visibility/motion permission gate"
            ),
            "fair_control": (
                "query_only uses the same initialized and active proposal/gate "
                "capacity but receives no geometry, flow, or geometry mask"
            ),
            "oracle_policy": "future teacher tracks are evaluation-only",
            "leakage_policy": "training/inference geometry is predicted from history",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
