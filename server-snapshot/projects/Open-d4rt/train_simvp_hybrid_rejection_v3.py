#!/usr/bin/env python3
"""Frozen v1 global path plus v2 local geometry and mismatched-geometry rejection."""

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

import train_simvp_feature_warp as core

V1Model = core.DualPathFeatureWarp
v1_geometry_bundle = core.geometry_bundle
import train_simvp_visibility_warp_v2 as v2


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
    p.add_argument("--v2-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--rejection-output-weight", type=float, default=1.0)
    p.add_argument("--rejection-gate-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def shuffled_bundle(bundle):
    raw, future_uv, reliability, anchor = bundle
    if raw.shape[0] > 1:
        order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
        return raw[order], future_uv[order], reliability[order], anchor[order]
    order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
    return raw[:, order], future_uv[:, order], reliability[:, order], anchor[:, order]


def zero_bundle(bundle):
    raw, future_uv, reliability, anchor = bundle
    return torch.zeros_like(raw), future_uv, torch.zeros_like(reliability), anchor


class HybridV1V2(nn.Module):
    """Frozen v1 appearance prior with a trainable v2 geometry-only overlay."""

    def __init__(self, v1_checkpoint, v2_checkpoint, channels=16, flow_kernel=9, scale=0.15):
        super().__init__()
        self.global_path = V1Model(channels, flow_kernel, scale)
        self.global_path.load_state_dict(torch.load(v1_checkpoint, map_location="cpu")["model"])
        self.local_path = v2.VisibilityDualPathFeatureWarp(channels, flow_kernel, scale)
        self.local_path.load_state_dict(torch.load(v2_checkpoint, map_location="cpu")["model"])
        for parameter in self.global_path.parameters():
            parameter.requires_grad_(False)
        for parameter in self.local_path.appearance.parameters():
            parameter.requires_grad_(False)
        self.global_path.eval(); self.local_path.appearance.eval()
        self.last = {}

    def train(self, mode=True):
        super().train(mode)
        self.global_path.eval()
        self.local_path.train(mode)
        self.local_path.appearance.eval()
        return self

    def global_hidden(self, source, hidden, bundle):
        with torch.no_grad():
            output, app_delta, _ = self.global_path(source, hidden, *zero_bundle(bundle))
        return output.detach(), app_delta.detach()

    def forward_from_global(self, source, global_hidden, bundle):
        _, _, geometry_delta = self.local_path(source, global_hidden, *bundle)
        corrected = global_hidden + geometry_delta
        self.last = {key: value for key, value in self.local_path.last.items()}
        self.last["geometry_delta"] = geometry_delta
        return corrected, geometry_delta

    def forward(self, source, hidden, bundle):
        global_hidden, app_delta = self.global_hidden(source, hidden, bundle)
        corrected, geometry_delta = self.forward_from_global(source, global_hidden, bundle)
        return corrected, global_hidden, app_delta, geometry_delta

    def auxiliary_loss(self, target, last):
        return self.local_path.auxiliary_loss(target, last)


def make_loaders(args):
    from video_forecast_dataset import VideoForecastDataset, video_pad_collate
    splits, split_metadata = core.select_paths(args)
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
    return loaders, datasets, split_metadata


@torch.no_grad()
def validate(model, loader, backbone, joint, device, future_frames):
    totals = defaultdict(float); batches = 0
    model.eval()
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        bundle = v2.geometry_bundle_v2("correct", joint, batch, future_frames)
        corrected, global_hidden, _, _ = model(source, hidden, bundle)
        correct = core.decode_frozen(backbone, corrected, skip)
        global_prediction = core.decode_frozen(backbone, global_hidden, skip)
        wrong_hidden, _ = model.forward_from_global(source, global_hidden, shuffled_bundle(bundle))
        wrong = core.decode_frozen(backbone, wrong_hidden, skip)
        target = batch["future_rgb"]
        totals["correct_MSE"] += (correct - target).square().mean().item()
        totals["correct_MAE"] += (correct - target).abs().mean().item()
        totals["global_MSE"] += (global_prediction - target).square().mean().item()
        totals["wrong_MSE"] += (wrong - target).square().mean().item()
        totals["wrong_effect"] += (wrong - global_prediction).abs().mean().item()
        totals["correct_effect"] += (correct - global_prediction).abs().mean().item()
        totals["wrong_gate"] += model.last["gate"].mean().item()
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_variant(name, model, use_rejection, train_loader, val_loader, backbone, joint,
                  lpips_model, device, args, output, future_frames):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = defaultdict(float); batches = 0; started = time.time()
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            with torch.no_grad():
                source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
            bundle = v2.geometry_bundle_v2("correct", joint, batch, future_frames)
            corrected, global_hidden, _, geometry_delta = model(source, hidden, bundle)
            correct = core.decode_frozen(backbone, corrected, skip)
            with torch.no_grad():
                global_prediction = core.decode_frozen(backbone, global_hidden, skip)
            positive, pieces = core.objective(
                correct, batch["future_rgb"], batch["context_rgb"][:, -1], global_prediction,
                torch.zeros_like(geometry_delta), geometry_delta, batch, lpips_model, args,
            )
            auxiliary, auxiliary_pieces = model.auxiliary_loss(
                batch["future_rgb"], batch["context_rgb"][:, -1],
            )
            loss = positive + auxiliary
            rejection_output = correct.new_zeros(()); rejection_gate = correct.new_zeros(())
            if use_rejection:
                wrong_hidden, wrong_delta = model.forward_from_global(
                    source, global_hidden, shuffled_bundle(bundle),
                )
                wrong = core.decode_frozen(backbone, wrong_hidden, skip)
                rejection_output = (wrong - global_prediction).abs().mean()
                rejection_gate = model.last["gate"].mean()
                loss = (loss + args.rejection_output_weight * rejection_output
                        + args.rejection_gate_weight * rejection_gate)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["rejection_output"] += rejection_output.item()
            totals["rejection_gate"] += rejection_gate.item()
            for key, value in {**pieces, **auxiliary_pieces}.items():
                totals[key] += float(value.detach())
            batches += 1
        metrics = validate(model, val_loader, backbone, joint, device, future_frames)
        score = metrics["correct_MSE"] + 0.1 * metrics["wrong_effect"]
        row = {"variant": name, "epoch": epoch, "epoch_seconds": time.time() - started,
               "score": score, **{f"train_{k}": v / max(batches, 1) for k, v in totals.items()},
               **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row)
        (output / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "epoch": epoch, "validation": metrics,
                 "score": score, "config": vars(args), "use_rejection": use_rejection}
        torch.save(state, output / f"{name}_last.pt")
        if score < best:
            best = score; torch.save(state, output / f"{name}_best.pt")
        print(json.dumps({"stage": "hybrid_v3", **row}), flush=True)
    model.load_state_dict(torch.load(output / f"{name}_best.pt", map_location=device)["model"])
    return history


@torch.no_grad()
def evaluate(models, loader, backbone, joint, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator
    names = ["frozen_simvp", "v1_global", "v1_full_correct"]
    for model_name in models:
        names += [f"{model_name}_correct", f"{model_name}_zero", f"{model_name}_shuffled"]
    accumulators = {name: VideoMetricAccumulator(device, future_frames) for name in names}
    motion_counts = {name: defaultdict(float) for name in names}
    records = []
    utilization = {name: defaultdict(list) for name in models}
    for raw_batch in loader:
        batch = move(raw_batch, device)
        source, hidden, skip = core.extract_frozen_features(backbone, batch["context_rgb"])
        baseline = core.decode_frozen(backbone, hidden, skip)
        bundle = v2.geometry_bundle_v2("correct", joint, batch, future_frames)
        v1_bundle = v1_geometry_bundle("correct", joint, batch, future_frames)
        first_model = next(iter(models.values()))
        global_hidden, _ = first_model.global_hidden(source, hidden, bundle)
        v1_global = core.decode_frozen(backbone, global_hidden, skip)
        v1_full_hidden, _, _ = first_model.global_path(source, hidden, *v1_bundle)
        predictions = {"frozen_simvp": baseline, "v1_global": v1_global,
                       "v1_full_correct": core.decode_frozen(backbone, v1_full_hidden, skip)}
        for model_name, model in models.items():
            model_global, _ = model.global_hidden(source, hidden, bundle)
            outputs = {}
            for mode, mode_bundle in (("correct", bundle), ("zero", zero_bundle(bundle)),
                                      ("shuffled", shuffled_bundle(bundle))):
                corrected, _ = model.forward_from_global(source, model_global, mode_bundle)
                outputs[mode] = core.decode_frozen(backbone, corrected, skip)
                predictions[f"{model_name}_{mode}"] = outputs[mode]
            utilization[model_name]["correct_vs_zero_l1"].append(
                (outputs["correct"] - outputs["zero"]).abs().mean().item())
            utilization[model_name]["correct_vs_shuffled_l1"].append(
                (outputs["correct"] - outputs["shuffled"]).abs().mean().item())
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
    for name in metrics:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    comparisons = {}
    for model_name in models:
        candidate = f"{model_name}_correct"
        comparisons[model_name] = {
            reference: core.paired_bootstrap(records, reference, candidate, seed)
            for reference in ("frozen_simvp", "v1_global", "v1_full_correct",
                              f"{model_name}_zero", f"{model_name}_shuffled")
        }
    utilization = {model: {key: float(np.mean(values)) for key, values in fields.items()}
                   for model, fields in utilization.items()}
    return metrics, records, comparisons, utilization


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
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters(): parameter.requires_grad_(False)

    prototype = HybridV1V2(args.v1_checkpoint, args.v2_checkpoint, 16, args.flow_kernel, args.feature_scale).to(device)
    models = {"hybrid_rejection": prototype, "hybrid_no_rejection": copy.deepcopy(prototype)}
    train_variant("hybrid_rejection", models["hybrid_rejection"], True, loaders["train"], loaders["val"],
                  backbone, joint, lpips_model, device, args, output, future_frames)
    train_variant("hybrid_no_rejection", models["hybrid_no_rejection"], False, loaders["train"], loaders["val"],
                  backbone, joint, lpips_model, device, args, output, future_frames)
    metrics, records, comparisons, utilization = evaluate(
        models, loaders["test"], backbone, joint, device, future_frames, args.seed,
    )
    summary = {"metrics": metrics, "paired_bootstrap": comparisons,
               "geometry_utilization": utilization,
               "meta": {"config": vars(args), "split": split_metadata,
                        "used": {key: len(value) for key, value in datasets.items()},
                        "trainable_params": sum(p.numel() for p in prototype.parameters() if p.requires_grad),
                        "design": "frozen v1 zero-geometry global path + v2 local geometry overlay",
                        "negative": "cross-video shuffled geometry distilled to the v1 global output"}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "per_sequence.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
