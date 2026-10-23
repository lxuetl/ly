#!/usr/bin/env python3
"""Curriculum stage-2 fine-tuning of the v1 model without changing its architecture."""

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
from torch.utils.data import DataLoader

import train_simvp_feature_warp as core


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--sample-manifest", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--v1-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
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
    p.add_argument("--teacher-weight", type=float, default=0.20)
    p.add_argument("--harm-output-weight", type=float, default=0.50)
    p.add_argument("--harm-gate-weight", type=float, default=0.02)
    p.add_argument("--harm-margin", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def zero_bundle(bundle):
    raw, uv, reliability, anchor = bundle
    return torch.zeros_like(raw), uv, torch.zeros_like(reliability), anchor


def shuffled_bundle(bundle):
    raw, uv, reliability, anchor = bundle
    if raw.shape[0] > 1:
        order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
        return raw[order], uv[order], reliability[order], anchor[order]
    order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
    return raw[:, order], uv[:, order], reliability[:, order], anchor[:, order]


def build_v1(checkpoint_path, device, trainable):
    model = core.DualPathFeatureWarp(16, 9, 0.15).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if trainable:
        for module in (model.geometry, model.propagation, model.gate):
            for parameter in module.parameters(): parameter.requires_grad_(True)
    return model


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate
    splits, metadata = core.select_paths(args)
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
    return loaders, datasets, metadata


def selective_harm_loss(wrong_prediction, teacher_prediction, target, wrong_geo_delta, margin):
    with torch.no_grad():
        teacher_error = (teacher_prediction - target).abs().mean(2, keepdim=True)
        wrong_error = (wrong_prediction - target).abs().mean(2, keepdim=True)
        harmful = (wrong_error > teacher_error + margin).float()
        harmful = F.max_pool2d(harmful.flatten(0, 1), 3, stride=1, padding=1).reshape_as(harmful)
    if harmful.sum() < 1:
        zero = wrong_prediction.sum() * 0
        return zero, zero, harmful.mean()
    output = ((wrong_prediction - teacher_prediction).abs().mean(2, keepdim=True) * harmful).sum()
    output = output / harmful.sum().clamp_min(1)
    harmful_feature = F.adaptive_max_pool2d(
        harmful.flatten(0, 1), wrong_geo_delta.shape[-2:]
    ).reshape(wrong_geo_delta.shape[0], wrong_geo_delta.shape[1], 1,
              *wrong_geo_delta.shape[-2:])
    # Penalize the correction that actually reaches the decoder.  The v1 model
    # stores its diagnostic gate detached, so penalizing that tensor would not
    # train either the gate or propagation branch.
    correction = (wrong_geo_delta.abs().mean(2, keepdim=True) * harmful_feature).sum()
    correction = correction / harmful_feature.sum().clamp_min(1)
    return output, correction, harmful.mean()


@torch.no_grad()
def validate(student, teacher, loader, backbone, joint, device, future_frames):
    totals = defaultdict(float); batches = 0
    student.eval(); teacher.eval()
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        teacher_hidden, _, _ = teacher(source, hidden, *bundle)
        student_hidden, _, _ = student(source, hidden, *bundle)
        wrong_hidden, _, _ = student(source, hidden, *shuffled_bundle(bundle))
        teacher_prediction = core.decode_frozen(backbone, teacher_hidden, skip)
        student_prediction = core.decode_frozen(backbone, student_hidden, skip)
        wrong_prediction = core.decode_frozen(backbone, wrong_hidden, skip)
        target = batch["future_rgb"]
        totals["MSE"] += (student_prediction - target).square().mean().item()
        totals["MAE"] += (student_prediction - target).abs().mean().item()
        totals["teacher_MSE"] += (teacher_prediction - target).square().mean().item()
        totals["teacher_MAE"] += (teacher_prediction - target).abs().mean().item()
        totals["wrong_MSE"] += (wrong_prediction - target).square().mean().item()
        totals["wrong_effect"] += (wrong_prediction - teacher_prediction).abs().mean().item()
        totals["correct_effect"] += (student_prediction - teacher_prediction).abs().mean().item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(student, teacher, train_loader, val_loader, backbone, joint, lpips_model,
          device, args, output, future_frames):
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    history = []; best = float("inf")
    for epoch in range(1, args.epochs + 1):
        student.train(); teacher.eval(); totals = defaultdict(float); batches = 0; started = time.time()
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
                bundle = core.geometry_bundle("correct", joint, batch, future_frames)
                teacher_hidden, _, _ = teacher(source, hidden, *bundle)
                teacher_prediction = core.decode_frozen(backbone, teacher_hidden, skip)
            student_hidden, app_delta, geo_delta = student(source, hidden, *bundle)
            support = student.last["support"].detach()
            student_prediction = core.decode_frozen(backbone, student_hidden, skip)
            loss, pieces = core.objective(
                student_prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                teacher_prediction, torch.zeros_like(app_delta), geo_delta,
                batch, lpips_model, args,
            )
            # Preserve the v1 teacher outside reliable geometry support.
            static = 1 - F.interpolate(
                support.flatten(0, 1), size=student_prediction.shape[-2:], mode="bilinear", align_corners=False
            ).reshape(support.shape[0], support.shape[1], 1, *student_prediction.shape[-2:]).clamp(0, 1)
            teacher_distill = ((student_prediction - teacher_prediction).abs().mean(2, keepdim=True) * static).mean()
            wrong_hidden, _, wrong_geo_delta = student(source, hidden, *shuffled_bundle(bundle))
            wrong_prediction = core.decode_frozen(backbone, wrong_hidden, skip)
            harm_output, harm_correction, harmful_fraction = selective_harm_loss(
                wrong_prediction, teacher_prediction, batch["future_rgb"], wrong_geo_delta, args.harm_margin,
            )
            loss = (loss + args.teacher_weight * teacher_distill
                    + args.harm_output_weight * harm_output
                    + args.harm_gate_weight * harm_correction)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            totals["loss"] += loss.item(); totals["teacher_distill"] += teacher_distill.item()
            totals["harm_output"] += harm_output.item()
            totals["harm_correction"] += harm_correction.item()
            totals["harmful_fraction"] += harmful_fraction.item()
            for key, value in pieces.items(): totals[key] += float(value.detach())
            batches += 1
        metrics = validate(student, teacher, val_loader, backbone, joint, device, future_frames)
        # MSE is primary; MAE may not regress beyond the frozen v1 teacher.
        score = metrics["MSE"] + 0.1 * max(metrics["MAE"] - metrics["teacher_MAE"], 0)
        row = {"epoch": epoch, "epoch_seconds": time.time() - started, "score": score,
               **{f"train_{k}": v / max(batches, 1) for k, v in totals.items()},
               **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row); (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": student.state_dict(), "epoch": epoch, "score": score,
                 "validation": metrics, "config": vars(args)}
        torch.save(state, output / "last.pt"); torch.save(state, output / f"epoch_{epoch:02d}.pt")
        if score < best:
            best = score; torch.save(state, output / "best.pt")
        print(json.dumps({"stage": "v1_stage2", **row}), flush=True)
    student.load_state_dict(torch.load(output / "best.pt", map_location=device)["model"])
    return history


@torch.no_grad()
def evaluate(student, teacher, loader, backbone, joint, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator
    variants = ("frozen_simvp", "v1_teacher", "stage2_correct", "stage2_zero", "stage2_shuffled")
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in variants}
    motion_counts = {name: defaultdict(float) for name in variants}; records = []
    utilization = defaultdict(list)
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        bundle = core.geometry_bundle("correct", joint, batch, future_frames)
        predictions = {"frozen_simvp": core.decode_frozen(backbone, hidden, skip)}
        for name, model_bundle, model in (
            ("v1_teacher", bundle, teacher), ("stage2_correct", bundle, student),
            ("stage2_zero", zero_bundle(bundle), student),
            ("stage2_shuffled", shuffled_bundle(bundle), student),
        ):
            corrected, _, _ = model(source, hidden, *model_bundle)
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
        utilization["correct_vs_zero_l1"].append(
            (predictions["stage2_correct"] - predictions["stage2_zero"]).abs().mean().item())
        utilization["correct_vs_shuffled_l1"].append(
            (predictions["stage2_correct"] - predictions["stage2_shuffled"]).abs().mean().item())
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > 0.05
        for variant, prediction in predictions.items():
            accumulators[variant].update(prediction, target, last)
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[variant]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[variant]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[variant]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, sequence_name in enumerate(batch["name"]):
                records.append({"name": sequence_name, "variant": variant,
                                **core.sequence_metrics(prediction[index:index + 1], target[index:index + 1],
                                                        last[index:index + 1])})
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    bootstrap = {reference: core.paired_bootstrap(records, reference, "stage2_correct", seed)
                 for reference in ("frozen_simvp", "v1_teacher", "stage2_zero", "stage2_shuffled")}
    return metrics, records, bootstrap, {key: float(np.mean(value)) for key, value in utilization.items()}


def main():
    args = arguments(); seed_all(args.seed)
    root = Path(args.d4rt_root); sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = make_loaders(args)
    backbone_checkpoint = torch.load(args.backbone_checkpoint, map_location="cpu")
    backbone = build_video_predictor_from_checkpoint(backbone_checkpoint).to(device)
    backbone.load_state_dict(backbone_checkpoint["model"]); backbone.eval()
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    future_frames = int(backbone_checkpoint["future_frames"])
    trajectory_checkpoint = torch.load(args.trajectory_checkpoint, map_location="cpu")
    cls = VelocityResidualImplicitForecaster if trajectory_checkpoint.get("model_variant") == "velocity_residual" else ImplicitConditionedJointForecaster
    joint = cls(trajectory_checkpoint["future_frames"], trajectory_checkpoint["hidden_dim"]).to(device)
    joint.load_state_dict(trajectory_checkpoint["model"]); joint.eval()
    for parameter in joint.parameters(): parameter.requires_grad_(False)
    student = build_v1(args.v1_checkpoint, device, True)
    teacher = build_v1(args.v1_checkpoint, device, False).eval()
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters(): parameter.requires_grad_(False)
    train(student, teacher, loaders["train"], loaders["val"], backbone, joint,
          lpips_model, device, args, output, future_frames)
    metrics, records, bootstrap, utilization = evaluate(
        student, teacher, loaders["test"], backbone, joint, device, future_frames, args.seed,
    )
    summary = {"metrics": metrics, "paired_bootstrap": bootstrap,
               "geometry_utilization": utilization,
               "meta": {"config": vars(args), "split": split_metadata,
                        "used": {key: len(value) for key, value in datasets.items()},
                        "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad),
                        "stage2": "v1 architecture initialized from v1 best; geometry/propagation/gate only",
                        "negative": "reject shuffled geometry only where it is demonstrably harmful"}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
