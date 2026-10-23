#!/usr/bin/env python3
"""Frozen TAU + frozen D4RT propagation + masked residual diffusion pilot.

The diffusion models see the same TAU/geometry base and the same vacated-region
mask. Only their conditioning differs (correct D4RT warp versus RGB-only).
Outside the geometry-derived support the output is exactly frozen TAU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

import train_simvp_feature_warp as core
import train_tau128_scaled_cap as previous


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
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=48)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sample-steps", type=int, default=8)
    p.add_argument("--residual-scale", type=float, default=0.20)
    p.add_argument("--local-cap", type=float, default=0.08)
    p.add_argument("--diffusion-budget", type=float, default=0.0005)
    p.add_argument("--visible-gate", type=float, default=0.8)
    p.add_argument("--mask-min-mean", type=float, default=0.00005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights-dir", default="")
    p.add_argument("--lpips-eval", action="store_true")
    return p.parse_args()


class Denoiser(nn.Module):
    """Small framewise U-Net; one shared model processes all forecast horizons."""

    def __init__(self, width=48):
        super().__init__()
        # noisy RGB(3), base(3), last RGB(3), warped RGB(3), flow(2),
        # vacated mask(1), horizon(1), diffusion time(1)
        self.inp = nn.Sequential(nn.Conv2d(17, width, 3, padding=1), nn.SiLU())
        self.enc = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, noisy, base, last, warped, flow, mask, t):
        b, k, _, h, w = noisy.shape
        frame_index = torch.linspace(0, 1, k, device=noisy.device, dtype=noisy.dtype)
        horizon = frame_index[None, :, None, None, None].expand(b, k, 1, h, w)
        step = t.reshape(b, 1, 1, 1, 1).expand(b, k, 1, h, w)
        last = last[:, None].expand(-1, k, -1, -1, -1)
        x = torch.cat((noisy, base, last, warped, flow, mask, horizon, step), 2)
        x = x.reshape(b * k, 17, h, w)
        skip = self.inp(x)
        low = self.mid(self.enc(skip))
        up = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(torch.cat((skip, up), 1)).reshape(b, k, 3, h, w)


class Schedule(nn.Module):
    def __init__(self, steps):
        super().__init__()
        beta = torch.linspace(0.0001, 0.02, steps)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0))


def visibility_masks(adapter, bundle, support, h, w):
    """Approximate vacated motion footprints using source and target track density."""
    _, future_uv, reliability, anchor = bundle
    b, n, k, _ = future_uv.shape
    permission = adapter._permission(future_uv, reliability, anchor)
    anchor_repeated = anchor[:, :, None].expand(-1, -1, k, -1)
    fh, fw = support.shape[-2:]
    _, old = adapter.raster(anchor, anchor_repeated, permission, fh, fw)
    old = 1 - (1 - old).clamp_min(1e-6).pow(adapter.coverage_scale)
    old = F.interpolate(old.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False).reshape(b, k, 1, h, w)
    new = F.interpolate(support.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False).reshape(b, k, 1, h, w)
    # old - new is a geometry-only, future-predicted disocclusion proxy.
    hole = (old - new).relu() * 4
    hole = F.max_pool2d(hole.flatten(0, 1), 7, stride=1, padding=3)
    hole = F.avg_pool2d(hole, 5, stride=1, padding=2)
    hole = hole.reshape(b, k, 1, h, w).clamp(0, 1).detach()
    visible_gate = ((new - 0.02) / 0.06).clamp(0, 1)
    return hole, visible_gate.detach()


@torch.no_grad()
def frozen_inputs(batch, backbone, joint, adapter, args):
    source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
    tau = core.decode_frozen(backbone, hidden, skip)
    bundle = core.geometry_bundle("correct", joint, batch, tau.shape[1])
    corrected, _, _ = adapter(source, hidden, bundle, bundle)
    geometry = core.decode_frozen(backbone, corrected, skip)
    geometry = previous.match_pixel_budget(
        geometry, tau, 0.002, "fixed", 0.08,
    )
    h, w = tau.shape[-2:]
    hole, visible = visibility_masks(adapter, bundle, adapter.last["support"], h, w)
    # Hard identity outside geometry permission; no global appearance rewrite.
    base = (tau + args.visible_gate * visible * (geometry - tau)).clamp(0, 1)
    flow = F.interpolate(adapter.last["flow"].flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False)
    flow = flow.reshape(tau.shape[0], tau.shape[1], 2, h, w)
    last = batch["context_rgb"][:, -1]
    warped = core.backward_warp(last, flow)
    return tau, base, warped, flow, hole, visible


def conditions(kind, last, warped, flow):
    if kind == "geometry":
        return warped, flow
    return last[:, None].expand_as(warped), torch.zeros_like(flow)


def train_batch(model, schedule, kind, base, last, warped, flow, hole, target, args, noise_state):
    cond_warp, cond_flow = conditions(kind, last, warped, flow)
    x0 = ((target - base) / args.residual_scale).clamp(-1, 1).detach()
    b = x0.shape[0]
    index, noise = noise_state
    alpha = schedule.alpha_bar[index].reshape(b, 1, 1, 1, 1)
    noisy = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
    t = index.float() / max(len(schedule.alpha_bar) - 1, 1)
    epsilon = model(noisy, base, last, cond_warp, cond_flow, hole, t)
    weight = hole + 0.02
    loss = ((epsilon - noise).square() * weight).sum() / (weight.sum() * 3).clamp_min(1)
    return loss


@torch.no_grad()
def sample(model, schedule, kind, base, last, warped, flow, hole, args, seed):
    cond_warp, cond_flow = conditions(kind, last, warped, flow)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    shared = torch.randn(base[:, :1].shape, device=base.device, generator=generator)
    x = 0.8 * shared + 0.6 * torch.randn(base.shape, device=base.device, generator=generator)
    times = torch.linspace(len(schedule.alpha_bar) - 1, 0, args.sample_steps, device=base.device).long().tolist()
    for j, step in enumerate(times):
        alpha = schedule.alpha_bar[step]
        t = torch.full((base.shape[0],), step / max(len(schedule.alpha_bar) - 1, 1), device=base.device)
        epsilon = model(x, base, last, cond_warp, cond_flow, hole, t)
        clean = ((x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        if j + 1 < len(times):
            following = schedule.alpha_bar[times[j + 1]]
            x = following.sqrt() * clean + (1 - following).sqrt() * epsilon
        else:
            x = clean
    residual = (args.residual_scale * x).clamp(-args.local_cap, args.local_cap)
    correction = hole * residual
    norm = correction.abs().mean((1, 2, 3, 4), keepdim=True)
    scale = (args.diffusion_budget / norm.clamp_min(1e-8)).clamp(max=1.0)
    return (base + scale * correction).clamp(0, 1)


@torch.no_grad()
def validate(models, schedule, loader, backbone, joint, adapter, args, max_batches):
    totals = {name: {"mse": 0.0, "mae": 0.0, "hole_mse": 0.0, "temporal_mae": 0.0, "pixel_delta_from_tau": 0.0} for name in ("tau", "base", *models)}
    perceptual = None
    if args.lpips_eval:
        import lpips
        perceptual = lpips.LPIPS(net="alex", verbose=False).to(next(backbone.parameters()).device).eval()
    batches = 0
    coverage = 0.0
    for raw_batch in loader:
        batch = core.move(raw_batch, next(backbone.parameters()).device)
        tau, base, warped, flow, hole, _ = frozen_inputs(batch, backbone, joint, adapter, args)
        output = {"tau": tau, "base": base}
        for name, model in models.items():
            model.eval()
            output[name] = sample(model, schedule, name, base, batch["context_rgb"][:, -1], warped, flow, hole, args, args.seed + batches)
        for name, pred in output.items():
            error = pred - batch["future_rgb"]
            totals[name]["mse"] += error.square().mean().item()
            totals[name]["mae"] += error.abs().mean().item()
            totals[name]["hole_mse"] += (error.square() * hole).sum().div((hole.sum() * 3).clamp_min(1)).item()
            totals[name]["temporal_mae"] += (pred[:, 1:] - pred[:, :-1] - batch["future_rgb"][:, 1:] + batch["future_rgb"][:, :-1]).abs().mean().item()
            totals[name]["pixel_delta_from_tau"] += (pred - tau).abs().mean().item()
            if perceptual is not None:
                pred_flat = pred.flatten(0, 1) * 2 - 1
                target_flat = batch["future_rgb"].flatten(0, 1) * 2 - 1
                totals[name].setdefault("lpips", 0.0)
                totals[name]["lpips"] += perceptual(pred_flat, target_flat).mean().item()
        coverage += hole.mean().item()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    return {"metrics": {name: {metric: value / max(batches, 1) for metric, value in row.items()} for name, row in totals.items()}, "hole_mean": coverage / max(batches, 1), "batches": batches}


def main():
    args = arguments()
    previous.seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, metadata, splits = previous.make_loaders(args)
    (output / "split_paths.json").write_text(json.dumps({"paths": {k: [str(p) for p in v] for k, v in splits.items()}}, indent=2))
    backbone_state = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=False)
    assert backbone_state["backbone"].lower() == "tau"
    assert int(backbone_state["image_size"]) == args.image_size
    backbone = build_video_predictor_from_checkpoint(backbone_state).to(device)
    backbone.load_state_dict(backbone_state["model"])
    backbone.eval().requires_grad_(False)
    trajectory_state = torch.load(args.trajectory_checkpoint, map_location="cpu", weights_only=False)
    trajectory_class = VelocityResidualImplicitForecaster if trajectory_state.get("model_variant") == "velocity_residual" else ImplicitConditionedJointForecaster
    joint = trajectory_class(trajectory_state["future_frames"], trajectory_state["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_state["model"])
    joint.eval().requires_grad_(False)
    adapter = previous.RejectableTransportTAUAdapter(
        int(backbone.backbone_config["hid_s"]), int(backbone_state["future_frames"]),
        flow_kernel=33, max_transport_gate=0.0, mode="correct",
    ).to(device)
    adapter.load_state_dict(torch.load(args.geometry_checkpoint, map_location="cpu", weights_only=False)["model"])
    adapter.coverage_scale = 4.0
    adapter.feature_cap = 0.12
    adapter.feature_budget = 0.006
    adapter.eval().requires_grad_(False)
    schedule = Schedule(args.steps).to(device)
    prototype = Denoiser().to(device)
    models = {"geometry": prototype, "rgb": copy.deepcopy(prototype)}
    opts = {k: torch.optim.AdamW(v.parameters(), lr=args.lr, weight_decay=1e-4) for k, v in models.items()}
    report = {"config": vars(args), "split": metadata, "counts": {k: len(v) for k, v in datasets.items()}, "trainable_params_each": sum(p.numel() for p in prototype.parameters())}
    (output / "config.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": "start", **report}), flush=True)
    if args.eval_only:
        weight_dir = Path(args.weights_dir or args.output_dir)
        for name, model in models.items():
            saved = torch.load(weight_dir / f"{name}_last.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
        test = validate(models, schedule, loaders["test"], backbone, joint, adapter, args, 0)
        (output / "summary.json").write_text(json.dumps({"test": test, "config": report}, indent=2))
        (output / "COMPLETE").write_text("complete\n")
        print(json.dumps({"stage": "complete", "test": test}), flush=True)
        return
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        total = {k: 0.0 for k in models}
        mask_total = 0.0
        batches = 0
        for raw_batch in loaders["train"]:
            batch = core.move(raw_batch, device)
            with torch.no_grad():
                _, base, warped, flow, hole, _ = frozen_inputs(batch, backbone, joint, adapter, args)
            if hole.mean().item() < args.mask_min_mean:
                continue
            last = batch["context_rgb"][:, -1]
            index = torch.randint(len(schedule.alpha_bar), (base.shape[0],), device=device)
            shared_noise = torch.randn_like(base[:, :1])
            noise = 0.8 * shared_noise + 0.6 * torch.randn_like(base)
            noise_state = (index, noise)
            for name, model in models.items():
                model.train()
                loss = train_batch(model, schedule, name, base, last, warped, flow, hole, batch["future_rgb"], args, noise_state)
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opts[name].step()
                total[name] += loss.item()
            mask_total += hole.mean().item()
            batches += 1
            if batches % 100 == 0:
                print(json.dumps({"stage": "train_progress", "epoch": epoch, "batches": batches, "seconds": round(time.time() - started, 1)}), flush=True)
        val = validate(models, schedule, loaders["val"], backbone, joint, adapter, args, math.ceil(args.val_samples / args.batch_size))
        row = {"epoch": epoch, "seconds": time.time() - started, "train_batches": batches, "train_loss": {k: v / max(batches, 1) for k, v in total.items()}, "train_hole_mean": mask_total / max(batches, 1), "val": val}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        for name, model in models.items():
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": vars(args)}, output / f"{name}_last.pt")
        print(json.dumps({"stage": "epoch", **row}), flush=True)
    test = validate(models, schedule, loaders["test"], backbone, joint, adapter, args, 0)
    (output / "summary.json").write_text(json.dumps({"test": test, "history": history, "config": report}, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps({"stage": "complete", "test": test}), flush=True)


if __name__ == "__main__":
    main()
