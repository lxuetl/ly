#!/usr/bin/env python3
"""TAU-specific D4RT injection at the temporal-output feature boundary.

The frozen TAU predictor remains the conservative path.  D4RT predicted
geometry may only add a bounded residual where motion, confidence and image
visibility jointly grant permission.  Zero geometry is therefore an exact
identity operation.  Cross-video shuffled geometry is used as an explicit
negative during training rather than only as an evaluation control.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--sample-manifest", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
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
    p.add_argument("--rank-weight", type=float, default=0.50)
    p.add_argument("--rank-margin", type=float, default=0.001)
    p.add_argument("--selection-dynamic-weight", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TAUTemporalGeometryInjector(nn.Module):
    """Horizon-conditioned propagation with strict geometric permission."""

    def __init__(self, channels, future_frames, flow_kernel=17, scale=0.10,
                 min_motion=0.015, full_motion=0.060, edge_margin=0.025):
        super().__init__()
        from video_prediction_model import SpatialGeometryAdapter

        self.raster = core.SparseDenseFlow(flow_kernel)
        self.geometry = SpatialGeometryAdapter(channels)
        self.scale = float(scale)
        self.min_motion = float(min_motion)
        self.motion_span = max(float(full_motion) - float(min_motion), 1e-4)
        self.edge_margin = max(float(edge_margin), 1e-4)
        self.horizon = nn.Parameter(torch.zeros(1, future_frames, channels, 1, 1))

        # hidden + physically warped residual + geometry + horizon + UV flow/support
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
        motion = ((displacement - self.min_motion) / self.motion_span).clamp(0, 1)
        edge_distance = torch.minimum(
            torch.minimum(future_uv[..., 0], 1 - future_uv[..., 0]),
            torch.minimum(future_uv[..., 1], 1 - future_uv[..., 1]),
        )
        visible = (edge_distance / self.edge_margin).clamp(0, 1)
        return reliability * motion * visible

    def forward(self, source, hidden, geometry, future_uv, reliability, anchor_uv):
        _, _, _, height, width = hidden.shape
        permission = self._permission(future_uv, reliability, anchor_uv)
        flow, dense_support = self.raster(
            anchor_uv, future_uv, permission, height, width)
        geometry_map, sparse_support = self.geometry(
            geometry, future_uv, permission, height, width, return_support=True)
        support = torch.maximum(dense_support, sparse_support).clamp(0, 1)
        warped = core.backward_warp(source, flow)
        warped_residual = warped - hidden
        horizon = self.horizon.expand(hidden.shape[0], -1, -1, height, width)
        features = torch.cat(
            (hidden, warped_residual, geometry_map, horizon, flow, support), dim=2,
        ).permute(0, 2, 1, 3, 4)
        proposal = self.proposal(features).permute(0, 2, 1, 3, 4)
        learned_gate = torch.sigmoid(
            self.risk_gate(features).permute(0, 2, 1, 3, 4))
        gate = learned_gate * support
        correction = self.scale * torch.tanh(proposal) * gate
        corrected = hidden + correction
        self.last = {
            "flow": flow.detach(),
            "support": support.detach(),
            "gate": gate.detach(),
            "learned_gate": learned_gate.detach(),
            "permission": permission.detach(),
            "geometry_delta": correction,
            "warped": warped.detach(),
        }
        # Signature remains compatible with the common loss/evaluation code.
        return corrected, torch.zeros_like(correction), correction


def dynamic_l1(prediction, target, last, threshold):
    mask = ((target - last[:, None]).abs().mean(2) > threshold).float()[:, :, None]
    denominator = (mask.sum() * prediction.shape[2]).clamp_min(1)
    return ((prediction - target).abs() * mask).sum() / denominator


@torch.no_grad()
def validate(model, loader, backbone, joint, device, future_frames, args):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw_batch in loader:
        batch = fair.move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"])
        baseline = core.decode_frozen(backbone, hidden, skip)
        correct_bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        shuffled_bundle = core.geometry_bundle("shuffled", joint, batch, future_frames)
        correct_hidden, _, _ = model(source, hidden, *correct_bundle)
        correct = core.decode_frozen(backbone, correct_hidden, skip)
        correct_support = model.last["support"].mean().item()
        correct_gate = model.last["gate"].mean().item()
        shuffled_hidden, _, _ = model(source, hidden, *shuffled_bundle)
        shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
        zero_bundle = core.geometry_bundle("zero", joint, batch, future_frames)
        zero_hidden, _, _ = model(source, hidden, *zero_bundle)
        zero = core.decode_frozen(backbone, zero_hidden, skip)
        error = correct - batch["future_rgb"]
        dyn_correct = dynamic_l1(
            correct, batch["future_rgb"], batch["context_rgb"][:, -1],
            args.dynamic_threshold)
        dyn_shuffled = dynamic_l1(
            shuffled, batch["future_rgb"], batch["context_rgb"][:, -1],
            args.dynamic_threshold)
        totals["MSE"] += error.square().mean().item()
        totals["MAE"] += error.abs().mean().item()
        totals["dynamic_MAE"] += dyn_correct.item()
        totals["rank_success"] += float(dyn_correct + args.rank_margin < dyn_shuffled)
        totals["support"] += correct_support
        totals["gate"] += correct_gate
        totals["zero_identity_max"] = max(
            totals["zero_identity_max"], (zero - baseline).abs().max().item())
        batches += 1
    result = {key: value / max(batches, 1) for key, value in totals.items()
              if key != "zero_identity_max"}
    result["zero_identity_max"] = totals["zero_identity_max"]
    return result


def train(model, train_loader, val_loader, backbone, joint, lpips_model,
          device, args, output, future_frames):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw_batch in train_loader:
            batch = fair.move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(
                    backbone, batch["context_rgb"])
                baseline = core.decode_frozen(backbone, hidden, skip)
                correct_bundle = core.geometry_bundle(
                    "correct", joint, batch, future_frames)
                shuffled_bundle = core.geometry_bundle(
                    "shuffled", joint, batch, future_frames)

            correct_hidden, app_delta, geo_delta = model(
                source, hidden, *correct_bundle)
            correct = core.decode_frozen(backbone, correct_hidden, skip)
            reconstruction, pieces = core.objective(
                correct, batch["future_rgb"], batch["context_rgb"][:, -1],
                baseline, app_delta, geo_delta, batch, lpips_model, args)

            shuffled_hidden, _, _ = model(source, hidden, *shuffled_bundle)
            shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
            dyn_correct = dynamic_l1(
                correct, batch["future_rgb"], batch["context_rgb"][:, -1],
                args.dynamic_threshold)
            dyn_shuffled = dynamic_l1(
                shuffled, batch["future_rgb"], batch["context_rgb"][:, -1],
                args.dynamic_threshold)
            rank = F.relu(args.rank_margin + dyn_correct - dyn_shuffled)
            loss = reconstruction + args.rank_weight * rank

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["rank"] += rank.item()
            totals["dynamic_correct"] += dyn_correct.item()
            totals["dynamic_shuffled"] += dyn_shuffled.item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            batches += 1

        metrics = validate(
            model, val_loader, backbone, joint, device, future_frames, args)
        score = metrics["MSE"] + args.selection_dynamic_weight * metrics["dynamic_MAE"]
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - started,
            "selection_score": score,
            **{f"train_{key}": value / max(batches, 1)
               for key, value in totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch,
                 "validation": metrics, "selection_score": score,
                 "config": vars(args)}
        torch.save(state, output / "last.pt")
        if score < best:
            best = score
            torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "tau_internal_geometry_v2", **row}), flush=True)

    model.load_state_dict(
        torch.load(output / "best.pt", map_location=device)["model"])


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
    backbone_name = str(checkpoint.get("backbone", "simvp")).lower()
    if backbone_name != "tau":
        raise ValueError(f"This experiment requires a TAU checkpoint, got {backbone_name}")
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError(
            f"Backbone is {checkpoint['image_size']}px, requested {args.image_size}px")
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

    channels = int(backbone.backbone_config["hid_s"])
    model = TAUTemporalGeometryInjector(
        channels, future_frames, args.flow_kernel, args.feature_scale,
        args.min_point_motion, args.full_point_motion, args.edge_margin).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    train(model, loaders["train"], loaders["val"], backbone, joint,
          lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = fair.evaluate(
        model, loaders["test"], backbone, joint, device, future_frames,
        args.seed, args.image_size, "tau_internal_v2")
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args),
            "split": split_metadata,
            "used": {key: len(value) for key, value in datasets.items()},
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "frozen_backbone_params": sum(p.numel() for p in backbone.parameters()),
            "frozen_trajectory_params": sum(p.numel() for p in joint.parameters()),
            "backbone": backbone_name,
            "feature_channels": channels,
            "injection": "TAU temporal-output latent, before frozen decoder",
            "identity_control": "zero geometry has exact zero support and exactly preserves TAU latent",
            "negative_control": "cross-video shuffled geometry participates in dynamic ranking loss",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
