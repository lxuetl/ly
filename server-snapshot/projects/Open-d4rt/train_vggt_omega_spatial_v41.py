#!/usr/bin/env python3
"""VGGT-Omega spatial v4.1: depth-free causal geometry injection.

The adapter is trained from scratch with the depth residual permanently disabled.
Checkpoint selection minimizes validation MSE subject to a minimum matched-vs-
shuffled confidence gap, rather than trading MSE against an arbitrary score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_simvp_feature_warp as core
import train_vggt_omega_latent_v2 as v2
import train_vggt_omega_semantic_v3 as v3
import train_vggt_omega_spatial_v4 as v4


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--omega-cache", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini"
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--alignment-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--alignment-lr", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--semantic-dim", type=int, default=192)
    parser.add_argument("--feature-scale", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--alignment-weight", type=float, default=0.02)
    parser.add_argument("--dynamic-weight", type=float, default=2.0)
    parser.add_argument("--lpips-weight", type=float, default=0.03)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--static-weight", type=float, default=0.10)
    parser.add_argument("--correction-weight", type=float, default=0.01)
    parser.add_argument("--rank-weight", type=float, default=0.50)
    parser.add_argument("--wrong-reject-weight", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=2e-4)
    parser.add_argument("--dynamic-threshold", type=float, default=0.05)
    parser.add_argument("--min-match-gap", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


class SpatialOmegaAdapterV41(v4.SpatialOmegaAdapterV4):
    """Parameter-matched v4 adapter whose depth path is always inactive."""

    def forward(
        self, hidden, tokens, depth, confidence, pose, patch_tokens,
        use_tokens=True, use_pose=False, use_depth=False,
        use_patch=True, use_global=True,
    ):
        del use_depth
        return super().forward(
            hidden, tokens, depth, confidence, pose, patch_tokens,
            use_tokens=use_tokens,
            use_pose=use_pose,
            use_depth=False,
            use_patch=use_patch,
            use_global=use_global,
        )


@torch.no_grad()
def validate(model, loader, backbone, device):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for raw in loader:
        batch = v2.move(raw, device)
        _, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        baseline = core.decode_frozen(backbone, hidden, skip)
        correct_hidden = model(hidden, *v4.omega_inputs(batch))
        correct_probability = model.last["match_probability"].mean().item()
        correct = core.decode_frozen(backbone, correct_hidden, skip)
        shuffled_hidden = model(
            hidden, *v4.omega_inputs(batch, shuffled=True)
        )
        shuffled_probability = model.last["match_probability"].mean().item()
        shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
        target = batch["future_rgb"]
        totals["MSE"] += (correct - target).square().mean().item()
        totals["MAE"] += (correct - target).abs().mean().item()
        totals["baseline_MSE"] += (baseline - target).square().mean().item()
        totals["shuffled_MSE"] += (shuffled - target).square().mean().item()
        totals["correct_vs_baseline_l1"] += (correct - baseline).abs().mean().item()
        totals["correct_vs_shuffled_l1"] += (correct - shuffled).abs().mean().item()
        totals["correct_match_probability"] += correct_probability
        totals["shuffled_match_probability"] += shuffled_probability
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def run_forecast_stage(model, loaders, backbone, lpips_model, device, args, output):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    best_eligible = float("inf")
    best_fallback = float("inf")
    has_eligible = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        started = time.time()
        for raw in loaders["train"]:
            batch = v2.move(raw, device)
            with torch.no_grad():
                _, hidden, skip = core.extract_frozen_features(
                    backbone, batch["context_rgb"]
                )
                baseline = core.decode_frozen(backbone, hidden, skip)
            correct_hidden = model(hidden, *v4.omega_inputs(batch))
            correct_state = {
                key: value for key, value in model.last.items()
                if not key.endswith("attention") and key != "gate"
            }
            correction = model.last["correction"]
            correct = core.decode_frozen(backbone, correct_hidden, skip)
            shuffled_hidden = model(
                hidden, *v4.omega_inputs(batch, shuffled=True)
            )
            shuffled_state = {
                key: value for key, value in model.last.items()
                if not key.endswith("attention") and key != "gate"
            }
            shuffled = core.decode_frozen(backbone, shuffled_hidden, skip)
            video_loss, pieces = v2.objective(
                correct, shuffled, baseline, batch["future_rgb"],
                batch["context_rgb"][:, -1], correction, lpips_model, args,
            )
            semantic_loss, semantic_pieces = v3.alignment_loss(
                correct_state, shuffled_state, args.temperature
            )
            loss = video_loss + args.alignment_weight * semantic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["video_loss"] += video_loss.item()
            totals["semantic_loss"] += semantic_loss.item()
            for key, value in {**pieces, **semantic_pieces}.items():
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()

        validation = validate(model, loaders["val"], backbone, device)
        match_gap = max(
            validation["correct_match_probability"]
            - validation["shuffled_match_probability"],
            0.0,
        )
        eligible = match_gap >= args.min_match_gap
        row = {
            "stage": "spatial_v41_forecast",
            "epoch": epoch,
            "epoch_seconds": time.time() - started,
            "lr": scheduler.get_last_lr()[0],
            "match_gap": match_gap,
            "eligible": eligible,
            "selection": validation["MSE"],
            **{
                f"train_{key}": value / max(batches, 1)
                for key, value in totals.items()
            },
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "match_gap": match_gap,
            "eligible": eligible,
            "config": vars(args),
        }
        torch.save(state, output / "last.pt")
        save = False
        if eligible and validation["MSE"] < best_eligible:
            best_eligible = validation["MSE"]
            has_eligible = True
            save = True
        elif not has_eligible and validation["MSE"] < best_fallback:
            best_fallback = validation["MSE"]
            save = True
        if save:
            torch.save(state, output / "best.pt")
        print(json.dumps(row), flush=True)

    selected = torch.load(
        output / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(selected["model"])
    return history, int(selected["epoch"]), bool(selected["eligible"])


@torch.no_grad()
def evaluate(model, loader, backbone, device, future_frames, seed):
    from video_metrics import VideoMetricAccumulator

    variants = (
        "tau_baseline", "omega_v41", "omega_v41_shuffled",
        "omega_v41_no_patch", "omega_v41_patch_only", "omega_v41_zero_geometry",
    )
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames) for name in variants
    }
    motion_counts = {name: defaultdict(float) for name in variants}
    records = []
    utilization = defaultdict(list)
    model.eval()
    for raw in loader:
        batch = v2.move(raw, device)
        _, hidden, skip = core.extract_frozen_features(
            backbone, batch["context_rgb"]
        )
        predictions = {
            "tau_baseline": core.decode_frozen(backbone, hidden, skip)
        }
        configurations = {
            "omega_v41": (False, {}),
            "omega_v41_shuffled": (True, {}),
            "omega_v41_no_patch": (False, {"use_patch": False}),
            "omega_v41_patch_only": (
                False, {"use_global": False}
            ),
            "omega_v41_zero_geometry": (
                False, {
                    "use_tokens": False, "use_patch": False, "use_global": False
                },
            ),
        }
        full_gate = None
        for name, (shuffled, options) in configurations.items():
            corrected = model(
                hidden, *v4.omega_inputs(batch, shuffled=shuffled), **options
            )
            predictions[name] = core.decode_frozen(backbone, corrected, skip)
            if name == "omega_v41":
                full_gate = model.last["gate"].mean().item()
        utilization["full_vs_baseline_l1"].append(
            (predictions["omega_v41"] - predictions["tau_baseline"])
            .abs().mean().item()
        )
        utilization["full_vs_shuffled_l1"].append(
            (predictions["omega_v41"] - predictions["omega_v41_shuffled"])
            .abs().mean().item()
        )
        utilization["gate"].append(full_gate)
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1]
        target_motion = (target - last[:, None]).abs().mean(2) > 0.05
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, last)
            predicted_motion = (prediction - last[:, None]).abs().mean(2) > 0.05
            motion_counts[name]["tp"] += (predicted_motion & target_motion).sum().item()
            motion_counts[name]["fp"] += (predicted_motion & ~target_motion).sum().item()
            motion_counts[name]["fn"] += (~predicted_motion & target_motion).sum().item()
            for index, sequence_name in enumerate(batch["name"]):
                records.append({
                    "name": sequence_name,
                    "variant": name,
                    **core.sequence_metrics(
                        prediction[index:index + 1], target[index:index + 1],
                        last[index:index + 1],
                    ),
                })
    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    for name in variants:
        rows = [row for row in records if row["variant"] == name]
        for key in ("dynamic_MSE", "dynamic_MAE", "dynamic_PSNR"):
            metrics[name][key] = float(np.nanmean([row[key] for row in rows]))
        tp, fp, fn = (motion_counts[name][key] for key in ("tp", "fp", "fn"))
        metrics[name]["Motion_IoU"] = tp / max(tp + fp + fn, 1)
        metrics[name]["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    bootstrap = {
        reference: core.paired_bootstrap(records, reference, "omega_v41", seed)
        for reference in (
            "tau_baseline", "omega_v41_shuffled",
            "omega_v41_no_patch", "omega_v41_zero_geometry",
        )
    }
    return metrics, records, bootstrap, {
        key: float(np.mean(values)) for key, values in utilization.items()
    }


def main():
    args = arguments()
    v2.seed_all(args.seed)
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "forecasting"))
    from simvp_geometry_model import build_video_predictor_from_checkpoint
    import lpips

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaders, datasets, split_metadata = v4.make_loaders(args)
    checkpoint = torch.load(
        args.backbone_checkpoint, map_location="cpu", weights_only=False
    )
    if str(checkpoint.get("backbone", "simvp")).lower() != "tau":
        raise ValueError("spatial Omega v4.1 requires a TAU checkpoint")
    backbone = build_video_predictor_from_checkpoint(checkpoint).to(device)
    backbone.load_state_dict(checkpoint["model"])
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    future_frames = int(checkpoint["future_frames"])
    channels = int(backbone.backbone_config["hid_s"])
    model = SpatialOmegaAdapterV41(
        channels, future_frames, args.semantic_dim, args.feature_scale
    ).to(device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    original_inputs = v2.omega_inputs
    v2.omega_inputs = v4.omega_inputs
    try:
        alignment_history = v3.run_alignment_stage(
            model, loaders["train"], backbone, device, args, output
        )
    finally:
        v2.omega_inputs = original_inputs
    forecast_history, best_epoch, selection_eligible = run_forecast_stage(
        model, loaders, backbone, lpips_model, device, args, output
    )
    metrics, records, bootstrap, utilization = evaluate(
        model, loaders["test"], backbone, device, future_frames, args.seed
    )
    semantic_test = validate(model, loaders["test"], backbone, device)
    summary = {
        "metrics": metrics,
        "paired_bootstrap": bootstrap,
        "geometry_utilization": utilization,
        "semantic_alignment": semantic_test,
        "meta": {
            "config": vars(args),
            "split": split_metadata,
            "used": {name: len(dataset) for name, dataset in datasets.items()},
            "best_epoch": best_epoch,
            "selection_eligible": selection_eligible,
            "alignment_epochs": len(alignment_history),
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "active_depth_path": False,
            "architecture": (
                "depth-free TAU spatial queries over VGGT patch-token time memory; "
                "global camera/register context; spatial mismatch gate; "
                "validation-MSE selection subject to minimum match-gap constraint"
            ),
            "leakage_policy": "VGGT-Omega sees observed context frames only",
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
