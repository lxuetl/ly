#!/usr/bin/env python3
"""Stage-2 D4RT adapter training on top of the strong partial-TAU control.

The TAU tail is initialized from the full-data no-geometry control and frozen.
Only the internal geometry adapter is trained.  Losses explicitly preserve the
control prediction outside reliable dynamic support, demand an advantage inside
that support, and make mismatched geometry fall back toward the control.
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
import torch.nn.functional as F

import train_simvp_feature_warp as core
import train_simvp_feature_warp_128 as fair
import train_tau_last_block_geometry_v3 as v3


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", default="")
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--control-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
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
    p.add_argument("--static-weight", type=float, default=0.50)
    p.add_argument("--advantage-weight", type=float, default=0.50)
    p.add_argument("--wrong-rank-weight", type=float, default=0.50)
    p.add_argument("--wrong-reject-weight", type=float, default=0.10)
    p.add_argument("--advantage-margin", type=float, default=0.001)
    p.add_argument("--wrong-margin", type=float, default=0.001)
    p.add_argument("--selection-dynamic-weight", type=float, default=0.10)
    p.add_argument("--selection-regression-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_adapter_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.injector.parameters():
        parameter.requires_grad_(True)


def train_mode(model):
    model.eval()
    model.injector.train()


def dynamic_permission(model, target, last):
    support = model.injector.last["support"]
    b, k, _, h, w = support.shape
    height, width = target.shape[-2:]
    support_rgb = F.interpolate(
        support.reshape(b * k, 1, h, w), size=(height, width),
        mode="bilinear", align_corners=False,
    ).reshape(b, k, 1, height, width).clamp(0, 1)
    target_motion = (
        (target - last[:, None]).abs().mean(2, keepdim=True) > 0.05
    ).float()
    return (support_rgb * target_motion).detach()


def masked_channel_mean(value, mask):
    channels = value.shape[2]
    return (value * mask).sum() / (mask.sum() * channels).clamp_min(1)


def comparative_losses(prediction, wrong_prediction, control, target,
                       permission, last, args):
    static_mask = 1 - permission
    static_preserve = masked_channel_mean(
        (prediction - control).abs(), static_mask)

    geometry_error = (prediction - target).abs().mean(2, keepdim=True)
    control_error = (control - target).abs().mean(2, keepdim=True)
    advantage = (
        F.relu(geometry_error - control_error + args.advantage_margin)
        * permission
    ).sum() / permission.sum().clamp_min(1)

    correct_dynamic = v3.dynamic_l1(
        prediction, target, last, args.dynamic_threshold)
    wrong_dynamic = v3.dynamic_l1(
        wrong_prediction, target, last, args.dynamic_threshold)
    wrong_rank = F.relu(args.wrong_margin + correct_dynamic - wrong_dynamic)
    wrong_reject = (wrong_prediction - control).abs().mean()
    return {
        "static_preserve": static_preserve,
        "dynamic_advantage": advantage,
        "wrong_rank": wrong_rank,
        "wrong_reject": wrong_reject,
        "dynamic_correct": correct_dynamic,
        "dynamic_wrong": wrong_dynamic,
    }


@torch.no_grad()
def validate(model, control, loader, backbone, joint, device, future_frames, args):
    model.eval()
    control.eval()
    totals = {name: defaultdict(float) for name in ("control", "correct", "zero", "shuffled")}
    totals["diagnostics"] = defaultdict(float)
    batches = 0
    for raw_batch in loader:
        batch = fair.move(raw_batch, device)
        prefix, skip = v3.frozen_prefix(backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        predictions = {
            "control": control(prefix, skip, None, False)[0],
            "correct": model(prefix, skip, bundle, True)[0],
            "zero": model(prefix, skip, v3.zero_bundle(bundle), True)[0],
            "shuffled": model(prefix, skip, v3.shuffled_bundle(bundle), True)[0],
        }
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        for name, prediction in predictions.items():
            error = prediction - target
            totals[name]["MSE"] += error.square().mean().item()
            totals[name]["MAE"] += error.abs().mean().item()
            totals[name]["dynamic_MAE"] += v3.dynamic_l1(
                prediction, target, last, args.dynamic_threshold).item()
        totals["diagnostics"]["zero_identity_max"] = max(
            totals["diagnostics"]["zero_identity_max"],
            (predictions["zero"] - predictions["control"]).abs().max().item())
        totals["diagnostics"]["correct_vs_control_l1"] += (
            predictions["correct"] - predictions["control"]).abs().mean().item()
        totals["diagnostics"]["correct_vs_shuffled_l1"] += (
            predictions["correct"] - predictions["shuffled"]).abs().mean().item()
        batches += 1
    result = {
        name: {key: value / max(batches, 1) for key, value in values.items()}
        for name, values in totals.items()
    }
    result["diagnostics"]["zero_identity_max"] = totals["diagnostics"][
        "zero_identity_max"]
    return result


def train(model, control, train_loader, val_loader, backbone, joint,
          lpips_model, device, args, output, future_frames):
    optimizer = torch.optim.AdamW(
        [p for p in model.injector.parameters() if p.requires_grad],
        lr=args.adapter_lr, weight_decay=1e-4)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_mode(model)
        control.eval()
        totals = defaultdict(float)
        batches = 0
        for raw_batch in train_loader:
            batch = fair.move(raw_batch, device)
            with torch.no_grad():
                prefix, skip = v3.frozen_prefix(backbone, batch["context_rgb"])
                control_prediction = control(prefix, skip, None, False)[0]
                bundle = core.geometry_bundle("correct", joint, batch, future_frames)

            prediction, correction = model(prefix, skip, bundle, True)
            permission = dynamic_permission(
                model, batch["future_rgb"], batch["context_rgb"][:, -1])
            wrong_prediction, _ = model(
                prefix, skip, v3.shuffled_bundle(bundle), True)

            reconstruction, pieces = core.objective(
                prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                control_prediction, torch.zeros_like(correction), correction,
                batch, lpips_model, args)
            comparative = comparative_losses(
                prediction, wrong_prediction, control_prediction,
                batch["future_rgb"], permission,
                batch["context_rgb"][:, -1], args)
            loss = (
                reconstruction
                + args.static_weight * comparative["static_preserve"]
                + args.advantage_weight * comparative["dynamic_advantage"]
                + args.wrong_rank_weight * comparative["wrong_rank"]
                + args.wrong_reject_weight * comparative["wrong_reject"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.injector.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            for key, value in pieces.items():
                totals[key] += float(value.detach())
            for key, value in comparative.items():
                totals[key] += float(value.detach())
            totals["dynamic_permission"] += permission.mean().item()
            batches += 1

        metrics = validate(
            model, control, val_loader, backbone, joint, device,
            future_frames, args)
        regression = max(0.0, metrics["correct"]["MSE"] - metrics["control"]["MSE"])
        score = (
            metrics["correct"]["MSE"]
            + args.selection_dynamic_weight * metrics["correct"]["dynamic_MAE"]
            + args.selection_regression_weight * regression
        )
        row = {
            "epoch": epoch, "epoch_seconds": time.time() - started,
            "selection_score": score, "validation_regression": regression,
            **{f"train_{key}": value / max(batches, 1)
               for key, value in totals.items()},
        }
        for variant, values in metrics.items():
            for key, value in values.items():
                row[f"val_{variant}_{key}"] = value
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch,
                 "validation": metrics, "selection_score": score,
                 "config": vars(args)}
        torch.save(state, output / "last.pt")
        if score < best:
            best = score
            torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "tau_geometry_stage2_advantage", **row}), flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device)["model"])


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

    control_state = torch.load(args.control_checkpoint, map_location="cpu")
    control = v3.PartialTAUTail(backbone, future_frames, args).to(device)
    control.load_state_dict(control_state["model"])
    control.eval()
    for parameter in control.parameters():
        parameter.requires_grad_(False)
    model = copy.deepcopy(control)
    configure_adapter_only(model)

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    train(model, control, loaders["train"], loaders["val"], backbone,
          joint, lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = v3.evaluate(
        {"geometry": model, "control": control}, loaders["test"], backbone,
        joint, device, future_frames, args)
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "meta": {
            "config": vars(args), "split": split_metadata,
            "used": {key: len(value) for key, value in datasets.items()},
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            "control_epoch": control_state.get("epoch"),
            "architecture": "geometry between TAU temporal blocks 5/6",
            "stage2": "frozen strong control tail; adapter-only comparative training",
            "losses": "static preservation + dynamic advantage + wrong rank/rejection",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
