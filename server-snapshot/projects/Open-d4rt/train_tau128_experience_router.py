#!/usr/bin/env python3
"""128px TAU/D4RT geometry-acceptance pilot with history replay.

Future RGB is used to label training patches only. Inference features comprise
past RGB, history replay error, frozen TAU/D4RT proposals, and forecast horizon.
The second training phase upweights hard false accepts/rejects (experience
replay), without adding generated frames as training targets.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_simvp_feature_warp as core
import train_tau128_scaled_cap as prior


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--geometry-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--replay-start", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights-dir", default="")
    p.add_argument("--oracle-eval", action="store_true")
    p.add_argument("--extended-eval", action="store_true")
    return p.parse_args()


class PatchRouter(nn.Module):
    def __init__(self, channels=17, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -0.5)

    def forward(self, feature):
        b, k, c, h, w = feature.shape
        return self.net(feature.reshape(b * k, c, h, w)).reshape(b, k, 1, h, w)


def average_patches(value, patch):
    b, k, c, h, w = value.shape
    return F.avg_pool2d(value.reshape(b * k, c, h, w), patch).reshape(
        b, k, c, h // patch, w // patch,
    )


@torch.no_grad()
def history_replay(batch, raster):
    """Observed t-2 -> t-1 RGB replay; no future frames enter this feature."""
    context = batch["context_rgb"]
    prev = torch.nan_to_num(batch["past_uv"][:, :, -2].float()).clamp(0, 1)
    curr = torch.nan_to_num(batch["past_uv"][:, :, -1].float()).clamp(0, 1)
    valid = (batch["past_valid"][:, :, -2] & batch["past_valid"][:, :, -1]).float()
    h, w = context.shape[-2:]
    flow, support = raster(prev, curr[:, :, None], valid[:, :, None], h, w)
    warped = core.backward_warp(context[:, -2], flow)[:, 0]
    error = (warped - context[:, -1]).abs().mean(1, keepdim=True)
    # Unsupported pixels get a neutral value, not a fabricated zero error.
    return (error * support[:, 0] + 0.1 * (1 - support[:, 0])).clamp(0, 1)


@torch.no_grad()
def candidates(batch, backbone, joint, adapter, replay_raster, patch):
    source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
    tau = core.decode_frozen(backbone, hidden, skip)
    bundle = core.geometry_bundle("correct", joint, batch, tau.shape[1])
    corrected, _, _ = adapter(source, hidden, bundle, bundle)
    geometry = core.decode_frozen(backbone, corrected, skip)
    geometry = prior.match_pixel_budget(geometry, tau, 0.002, "fixed", 0.08)
    b, k, _, h, w = tau.shape
    support = F.interpolate(
        adapter.last["support"].flatten(0, 1), (h, w), mode="bilinear", align_corners=False,
    ).reshape(b, k, 1, h, w)
    flow = F.interpolate(
        adapter.last["flow"].flatten(0, 1), (h, w), mode="bilinear", align_corners=False,
    ).reshape(b, k, 2, h, w)
    delta = (geometry - tau) * ((support - 0.01) / 0.05).clamp(0, 1)
    last = batch["context_rgb"][:, -1]
    last_all = last[:, None].expand(-1, k, -1, -1, -1)
    warped_diff = core.backward_warp(last, flow) - last_all
    replay = history_replay(batch, replay_raster)[:, None].expand(-1, k, -1, -1, -1)
    horizon = torch.linspace(0, 1, k, device=tau.device, dtype=tau.dtype)
    horizon = horizon[None, :, None, None, None].expand(b, -1, 1, h, w)
    shared = (tau, delta, last_all, support, replay, horizon)
    geo_feature = torch.cat((tau, delta, last_all, warped_diff, support, flow, replay, horizon), 2)
    rgb_feature = torch.cat((tau, delta, last_all, torch.zeros_like(warped_diff), support, torch.zeros_like(flow), replay, horizon), 2)
    return tau, delta, average_patches(geo_feature, patch), average_patches(rgb_feature, patch), support


def experience_loss(logits, tau, delta, target, patch, replay_phase):
    full = (tau + delta).clamp(0, 1)
    advantage = average_patches(
        (tau - target).square().mean(2, keepdim=True)
        - (full - target).square().mean(2, keepdim=True), patch,
    )
    correction = average_patches(delta.abs().mean(2, keepdim=True), patch)
    active = (correction > 0.0001).float()
    label = (advantage > 0).float()
    importance = (advantage.abs() / advantage.abs().mean().clamp_min(1e-7)).sqrt().clamp(0.5, 3)
    weight = active * importance
    if replay_phase:
        predicted_positive = (torch.sigmoid(logits.detach()) > 0.5).float()
        mistakes = (predicted_positive != label).float()
        false_accept = predicted_positive * (1 - label)
        weight = weight * (1 + mistakes + false_accept)
    loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
    loss = (loss * weight).sum() / weight.sum().clamp_min(1)
    stats = {
        "positive_fraction": (label * active).sum().div(active.sum().clamp_min(1)).item(),
        "active_fraction": active.mean().item(),
        "false_accept_fraction": (((torch.sigmoid(logits.detach()) > 0.5).float() * (1 - label) * active).sum() / active.sum().clamp_min(1)).item(),
    }
    return loss, stats


def routed(tau, delta, logits, threshold):
    patch_gate = (torch.sigmoid(logits) >= threshold).float()
    b, k, _, h, w = tau.shape
    gate = F.interpolate(
        patch_gate.flatten(0, 1), (h, w), mode="bilinear", align_corners=False,
    ).reshape(b, k, 1, h, w)
    return (tau + gate * delta).clamp(0, 1), gate


@torch.no_grad()
def evaluate(models, loader, backbone, joint, adapter, raster, args, thresholds, max_batches=0, records=False, perceptual=None):
    names = ["tau", "all_geometry", "fixed_0.25", "fixed_0.50", "fixed_0.75"]
    names += [f"{kind}_{thr:.2f}" for kind in models for thr in thresholds]
    names += [f"{kind}_soft" for kind in models]
    if args.oracle_eval:
        names.append("oracle_patch_not_deployable")
    totals = {name: defaultdict(float) for name in names}
    extended_names = {"tau", "all_geometry", "geometry_0.50", "rgb_0.50", "geometry_soft", "oracle_patch_not_deployable"}
    rows = []
    batch_count = 0
    for raw in loader:
        batch = core.move(raw, next(backbone.parameters()).device)
        tau, delta, gf, rf, support = candidates(batch, backbone, joint, adapter, raster, args.patch_size)
        logits = {"geometry": models["geometry"](gf), "rgb": models["rgb"](rf)}
        predictions = {"tau": tau, "all_geometry": (tau + delta).clamp(0, 1)}
        for fraction in (0.25, 0.50, 0.75):
            predictions[f"fixed_{fraction:.2f}"] = (tau + fraction * delta).clamp(0, 1)
        for kind in models:
            soft = F.interpolate(
                torch.sigmoid(logits[kind]).flatten(0, 1), tau.shape[-2:],
                mode="bilinear", align_corners=False,
            ).reshape(tau.shape[0], tau.shape[1], 1, *tau.shape[-2:])
            predictions[f"{kind}_soft"] = (tau + soft * delta).clamp(0, 1)
        target = batch["future_rgb"]
        if args.oracle_eval:
            advantage = average_patches(
                (tau - target).square().mean(2, keepdim=True)
                - (predictions["all_geometry"] - target).square().mean(2, keepdim=True),
                args.patch_size,
            )
            oracle = F.interpolate(
                (advantage > 0).float().flatten(0, 1), tau.shape[-2:],
                mode="bilinear", align_corners=False,
            ).reshape(tau.shape[0], tau.shape[1], 1, *tau.shape[-2:])
            predictions["oracle_patch_not_deployable"] = (tau + oracle * delta).clamp(0, 1)
        gates = {}
        for kind in models:
            for threshold in thresholds:
                name = f"{kind}_{threshold:.2f}"
                predictions[name], gates[name] = routed(tau, delta, logits[kind], threshold)
        dynamic_mask = None
        if perceptual is not None:
            dynamic_mask, _ = prior.camera_compensated_dynamic_mask(
                batch, tau.shape[-2], tau.shape[-1], 4, 6,
            )
        for name, pred in predictions.items():
            error = pred - target
            lpips_values = None
            if perceptual is not None and name in extended_names:
                lpips_values = perceptual(
                    pred.flatten(0, 1) * 2 - 1,
                    target.flatten(0, 1) * 2 - 1,
                ).reshape(pred.shape[0], pred.shape[1], -1).mean((1, 2))
            for bi in range(pred.shape[0]):
                row = {
                    "name": batch["name"][bi], "variant": name,
                    "MSE": error[bi].square().mean().item(),
                    "MAE": error[bi].abs().mean().item(),
                    "pixel_delta": (pred[bi] - tau[bi]).abs().mean().item(),
                }
                if records:
                    rows.append(row)
                for key in ("MSE", "MAE", "pixel_delta"):
                    totals[name][key] += row[key]
                totals[name]["count"] += 1
                if lpips_values is not None:
                    row["LPIPS"] = lpips_values[bi].item()
                    row["temporal_MAE"] = (
                        (pred[bi, 1:] - pred[bi, :-1])
                        - (target[bi, 1:] - target[bi, :-1])
                    ).abs().mean().item()
                    selected = dynamic_mask[bi, :, None].expand_as(error[bi])
                    row["dynamic_MSE"] = error[bi].square()[selected].mean().item() if selected.any() else float("nan")
                    for key in ("LPIPS", "temporal_MAE", "dynamic_MSE"):
                        if math.isfinite(row[key]):
                            totals[name][key + "_sum"] += row[key]
                            totals[name][key + "_count"] += 1
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
    metrics = {
        name: {key: value / max(item["count"], 1) for key, value in item.items() if key != "count"}
        for name, item in totals.items()
    }
    for name, item in totals.items():
        for key in ("LPIPS", "temporal_MAE", "dynamic_MSE"):
            if item[key + "_count"]:
                metrics[name][key] = item[key + "_sum"] / item[key + "_count"]
            metrics[name].pop(key + "_sum", None)
            metrics[name].pop(key + "_count", None)
    return metrics, rows


def load_frozen(args, device):
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster

    backbone_state = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=False)
    assert backbone_state["backbone"].lower() == "tau"
    assert int(backbone_state["image_size"]) == args.image_size
    backbone = build_video_predictor_from_checkpoint(backbone_state).to(device)
    backbone.load_state_dict(backbone_state["model"])
    trajectory_state = torch.load(args.trajectory_checkpoint, map_location="cpu", weights_only=False)
    trajectory_class = VelocityResidualImplicitForecaster if trajectory_state.get("model_variant") == "velocity_residual" else ImplicitConditionedJointForecaster
    joint = trajectory_class(trajectory_state["future_frames"], trajectory_state["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_state["model"])
    adapter = prior.RejectableTransportTAUAdapter(
        int(backbone.backbone_config["hid_s"]), int(backbone_state["future_frames"]),
        flow_kernel=33, max_transport_gate=0.0, mode="correct",
    ).to(device)
    adapter.load_state_dict(torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=False)["model"])
    adapter.coverage_scale = 4.0
    adapter.feature_cap = 0.12
    adapter.feature_budget = 0.006
    replay_raster = core.SparseDenseFlow(17).to(device)
    for model in (backbone, joint, adapter, replay_raster):
        model.eval().requires_grad_(False)
    return backbone, joint, adapter, replay_raster


def main():
    args = arguments()
    if args.image_size % args.patch_size:
        raise ValueError("image-size must be divisible by patch-size")
    prior.seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, metadata, splits = prior.make_loaders(args)
    (output / "split_paths.json").write_text(json.dumps({"paths": {k: [str(p) for p in v] for k, v in splits.items()}}, indent=2))
    backbone, joint, adapter, replay_raster = load_frozen(args, device)
    initial = PatchRouter().to(device)
    models = {"geometry": initial, "rgb": copy.deepcopy(initial)}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) for name, model in models.items()}
    print(json.dumps({"stage": "start", "counts": {k: len(v) for k, v in datasets.items()}, "router_params_each": sum(p.numel() for p in initial.parameters()), "config": vars(args)}), flush=True)
    thresholds = (0.25, 0.50, 0.75)
    if args.eval_only:
        weight_dir = Path(args.weights_dir or args.output_dir)
        for name, model in models.items():
            saved = torch.load(weight_dir / f"{name}_best.pt", map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            model.eval()
        val, _ = evaluate(models, loaders["val"], backbone, joint, adapter, replay_raster, args, thresholds)
        deployable = [name for name in val if name != "oracle_patch_not_deployable"]
        selected = min(deployable, key=lambda name: val[name]["MSE"])
        perceptual = None
        if args.extended_eval:
            import lpips
            perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        test, rows = evaluate(models, loaders["test"], backbone, joint, adapter, replay_raster, args, thresholds, records=True, perceptual=perceptual)
        summary = {"selected_on_full_validation": selected, "val": val, "test": test, "split": metadata, "weights_dir": str(weight_dir), "oracle_uses_test_future": args.oracle_eval}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        (output / "test_per_clip.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (output / "COMPLETE").write_text("complete\n")
        print(json.dumps({"stage": "complete", "test": test}), flush=True)
        return
    history = []
    best_score = float("inf")
    best_epoch = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        totals = {name: defaultdict(float) for name in models}
        batches = 0
        for raw in loaders["train"]:
            batch = core.move(raw, device)
            with torch.no_grad():
                tau, delta, gf, rf, _ = candidates(batch, backbone, joint, adapter, replay_raster, args.patch_size)
            for name, feature in (("geometry", gf), ("rgb", rf)):
                models[name].train()
                logits = models[name](feature)
                loss, stats = experience_loss(logits, tau, delta, batch["future_rgb"], args.patch_size, epoch >= args.replay_start)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(models[name].parameters(), 1.0)
                optimizers[name].step()
                totals[name]["loss"] += loss.item()
                for key, value in stats.items():
                    totals[name][key] += value
            batches += 1
            if batches % 100 == 0:
                print(json.dumps({"stage": "progress", "epoch": epoch, "batches": batches, "elapsed_seconds": round(time.time() - start, 1)}), flush=True)
        for model in models.values():
            model.eval()
        validation, _ = evaluate(models, loaders["val"], backbone, joint, adapter, replay_raster, args, thresholds, math.ceil(args.val_samples / args.batch_size))
        best_name = min((name for name in validation if name.startswith("geometry_")), key=lambda n: validation[n]["MSE"])
        row = {"epoch": epoch, "seconds_total": time.time() - start, "train_batches": batches, "training": {name: {key: value / max(batches, 1) for key, value in data.items()} for name, data in totals.items()}, "val": validation, "best_geometry_variant": best_name}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        for name, model in models.items():
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": vars(args)}, output / f"{name}_last.pt")
        if validation[best_name]["MSE"] < best_score:
            best_score = validation[best_name]["MSE"]
            best_epoch = epoch
            for name, model in models.items():
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": vars(args)}, output / f"{name}_best.pt")
        print(json.dumps({"stage": "epoch", **row}), flush=True)
    for name, model in models.items():
        model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device, weights_only=False)["model"])
        model.eval()
    selected_name = history[best_epoch - 1]["best_geometry_variant"]
    selected_threshold = float(selected_name.rsplit("_", 1)[1])
    test, rows = evaluate(models, loaders["test"], backbone, joint, adapter, replay_raster, args, thresholds, records=True)
    summary = {"selected_by_validation": selected_name, "test": test, "best_epoch": best_epoch, "history": history, "split": metadata, "elapsed_seconds": time.time() - start}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "test_per_clip.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps({"stage": "complete", "best_epoch": best_epoch, "selected_by_validation": selected_name, "test_selected": test[selected_name], "test_tau": test["tau"], "test_all_geometry": test["all_geometry"]}), flush=True)


if __name__ == "__main__":
    main()
