"""Evaluate CaPSR path ablations with one frozen checkpoint and test protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_model import CaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


MODES = (
    "no_geometry",
    "full",
    "transport_only",
    "synthesis_only",
    "no_decomposition",
    "no_state_generation",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--capsr-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def render_components(model, base_prediction, last_rgb, propagated, support, dense_state):
    b, k, _, h, w = base_prediction.shape
    last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
    base_motion = (base_prediction - last).abs().mean(2, keepdim=True)
    disocclusion_prior = (1 - support) * (base_motion / 0.08).clamp(0, 1)
    inputs = torch.cat((
        base_prediction, last, propagated, (propagated - base_prediction).abs(),
        dense_state, support, base_motion, disocclusion_prior,
    ), 2)
    hidden = model.renderer(inputs.reshape(b * k, -1, h, w))
    transport_gate = torch.sigmoid(model.transport_head(hidden)).reshape(b, k, 1, h, w) * support
    synthesis_gate = torch.sigmoid(model.synthesis_gate_head(hidden)).reshape(b, k, 1, h, w)
    synthesis_gate = synthesis_gate * disocclusion_prior * (1 - transport_gate.detach())
    synthesis_residual = model.max_synthesis_residual * torch.tanh(
        model.synthesis_head(hidden)
    ).reshape(b, k, 3, h, w)
    transport_delta = propagated - base_prediction
    outputs = {
        "full": (base_prediction + transport_gate * transport_delta
                 + synthesis_gate * synthesis_residual).clamp(0, 1),
        "transport_only": (base_prediction + transport_gate * transport_delta).clamp(0, 1),
        "synthesis_only": (base_prediction + synthesis_gate * synthesis_residual).clamp(0, 1),
    }
    diagnostics = {
        "transport_gate": transport_gate,
        "synthesis_gate": synthesis_gate,
        "support": support,
    }
    return outputs, diagnostics


def no_decomposition_state(model, original, predicted_xyz, past_valid, past_confidence):
    total = original["total_flow"]
    camera = torch.zeros_like(total)
    object_flow = total
    logits = model.visibility(
        camera, object_flow, total, predicted_xyz, past_valid, past_confidence
    )
    return {
        "anchor_uv": original["anchor_uv"],
        "camera_flow": camera,
        "object_flow": object_flow,
        "total_flow": total,
        "camera_coefficients": torch.zeros_like(original["camera_coefficients"]),
        "visibility_logits": logits,
        "visibility_probabilities": logits.softmax(-1),
    }


@torch.no_grad()
def predict_modes(model, base, trajectory, batch):
    base_prediction, predicted_xyz, predicted_uv = frozen_prediction(base, trajectory, batch)
    full_prediction, full_state = model(
        base_prediction, batch["context_rgb"][:, -1], predicted_xyz, predicted_uv,
        batch["past_xyz"].float(), batch["past_uv"].float(), batch["past_valid"],
        batch["past_confidence"].float(),
    )
    full_outputs = {
        "full": full_prediction,
        "transport_only": (
            base_prediction + full_state["transport_gate"] * full_state["transport_delta"]
        ).clamp(0, 1),
        "synthesis_only": (
            base_prediction + full_state["synthesis_gate"] * full_state["synthesis_residual"]
        ).clamp(0, 1),
    }

    confidence = torch.nan_to_num(batch["past_confidence"].float()).amax(2).clamp(0, 1)
    point_valid = batch["past_valid"].any(2)
    no_decomp_state = no_decomposition_state(
        model, full_state, predicted_xyz, batch["past_valid"], batch["past_confidence"].float()
    )
    propagated, support, dense = model._softsplat(
        batch["context_rgb"][:, -1], predicted_uv, predicted_xyz,
        confidence, no_decomp_state, point_valid,
    )
    no_decomp_outputs, no_decomp_diag = render_components(
        model, base_prediction, batch["context_rgb"][:, -1], propagated, support, dense
    )

    b, k, _, h, w = base_prediction.shape
    zero_rgb = torch.zeros_like(base_prediction)
    zero_support = base_prediction.new_zeros(b, k, 1, h, w)
    zero_state = base_prediction.new_zeros(b, k, model.dense_state_channels, h, w)
    no_state_outputs, no_state_diag = render_components(
        model, base_prediction, batch["context_rgb"][:, -1],
        zero_rgb, zero_support, zero_state,
    )
    outputs = {
        "no_geometry": base_prediction,
        **full_outputs,
        "no_decomposition": no_decomp_outputs["full"],
        "no_state_generation": no_state_outputs["synthesis_only"],
    }
    diagnostics = {
        "full": {
            "transport_gate": full_state["transport_gate"],
            "synthesis_gate": full_state["synthesis_gate"],
            "support": full_state["propagation_support"],
        },
        "no_decomposition": no_decomp_diag,
        "no_state_generation": no_state_diag,
    }
    return outputs, diagnostics


def metric_extras():
    return {name: {
        "dyn_abs": 0.0, "dyn_sq": 0.0, "dyn_count": 0.0,
        "intersection": 0.0, "union": 0.0, "pred_pos": 0.0, "gt_pos": 0.0,
    } for name in MODES}


def finalize(accumulators, extras):
    results = {}
    for name in MODES:
        metrics = accumulators[name].compute()
        stat = extras[name]
        dynamic_mse = stat["dyn_sq"] / max(stat["dyn_count"], 1)
        precision = stat["intersection"] / max(stat["pred_pos"], 1)
        recall = stat["intersection"] / max(stat["gt_pos"], 1)
        metrics.update({
            "dynamic_MAE": stat["dyn_abs"] / max(stat["dyn_count"], 1),
            "dynamic_MSE": dynamic_mse,
            "dynamic_PSNR": -10 * math.log10(max(dynamic_mse, 1e-12)),
            "motion_mask_IoU": stat["intersection"] / max(stat["union"], 1),
            "motion_mask_F1": 2 * precision * recall / max(precision + recall, 1e-12),
        })
        results[name] = metrics
    return results


def main():
    a = parse_args()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(a)
    dataset = VideoForecastDataset(
        a.cache_root, "test", 64, paths=splits["test"], cache_items=False
    )
    if a.max_test > 0:
        dataset = Subset(dataset, range(min(a.max_test, len(dataset))))
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, _, trajectory_checkpoint = load_frozen_models(a, device)
    checkpoint = torch.load(a.capsr_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model = CaPSR(
        float(config.get("sigma", 0.035)),
        float(config.get("depth_temperature", 1.0)),
        int(config.get("hidden", 48)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    import lpips
    shared_lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for parameter in shared_lpips.parameters():
        parameter.requires_grad_(False)
    future_frames = int(trajectory_checkpoint["future_frames"])
    accumulators = {
        name: VideoMetricAccumulator(device, future_frames, use_lpips=False) for name in MODES
    }
    for accumulator in accumulators.values():
        accumulator.lpips_model = shared_lpips
    extras = metric_extras()
    gate_sums = {name: {"transport": 0.0, "synthesis": 0.0, "support": 0.0, "pixels": 0.0}
                 for name in ("full", "no_decomposition", "no_state_generation")}

    for index, batch in enumerate(loader, 1):
        batch = move(batch, device)
        predictions, diagnostics = predict_modes(model, base, trajectory, batch)
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        gt_motion = (target - last).abs().mean(2) > 0.05
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, target, batch["context_rgb"][:, -1])
            difference = prediction - target
            dynamic = gt_motion[:, :, None].expand_as(difference)
            stat = extras[name]
            stat["dyn_abs"] += difference.abs()[dynamic].sum().item()
            stat["dyn_sq"] += difference.square()[dynamic].sum().item()
            stat["dyn_count"] += dynamic.sum().item()
            pred_motion = (prediction - last).abs().mean(2) > 0.05
            stat["intersection"] += (gt_motion & pred_motion).sum().item()
            stat["union"] += (gt_motion | pred_motion).sum().item()
            stat["pred_pos"] += pred_motion.sum().item()
            stat["gt_pos"] += gt_motion.sum().item()
        for name, values in diagnostics.items():
            row = gate_sums[name]
            row["transport"] += values["transport_gate"].sum().item()
            row["synthesis"] += values["synthesis_gate"].sum().item()
            row["support"] += values["support"].sum().item()
            row["pixels"] += values["support"].numel()
        if index % 20 == 0 or index == len(loader):
            print(json.dumps({"processed": index, "total": len(loader)}), flush=True)

    metrics = finalize(accumulators, extras)
    baseline = metrics["no_geometry"]
    deltas = {}
    for name, values in metrics.items():
        deltas[name] = {
            "MSE_percent": 100 * (values["MSE"] / baseline["MSE"] - 1),
            "dynamic_MSE_percent": 100 * (values["dynamic_MSE"] / baseline["dynamic_MSE"] - 1),
            "LPIPS_percent": 100 * (values["LPIPS"] / baseline["LPIPS"] - 1),
            "temporal_MAE_percent": 100 * (
                values["temporal_delta_MAE"] / baseline["temporal_delta_MAE"] - 1
            ),
        }
    gate_fractions = {
        name: {key + "_fraction": row[key] / max(row["pixels"], 1)
               for key in ("transport", "synthesis", "support")}
        for name, row in gate_sums.items()
    }
    summary = {
        "metrics": metrics,
        "relative_to_no_geometry": deltas,
        "gate_fractions": gate_fractions,
        "meta": {
            "num_test_clips": len(dataset),
            "split": split_meta,
            "capsr_checkpoint": a.capsr_checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "training": False,
            "note": (
                "no_state_generation is an inference intervention on the trained checkpoint; "
                "it is not a retrained equal-capacity no-geometry control"
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
