"""Small counterfactual risk-router proof of concept for calibrated CaPSR.

The video backbone, trajectory forecaster, and camera renderer are frozen.  A
small convolutional router learns when to accept a camera-warp candidate and
when to fall back to the strong video-prediction baseline.  Training mixes
correct and deliberately corrupted motion conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from capsr_calibrated_model import StructuredCaPSR
from train_capsr import frozen_prediction, load_frozen_models
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


CONDITIONS = (
    "correct", "reverse_direction", "amplify_3x", "horizon_mismatch",
    "spatial_shift", "local_patch_flip", "cross_video",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--correction-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-train", type=int, default=128)
    p.add_argument("--max-val", type=int, default=32)
    p.add_argument("--max-test", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class CounterfactualRiskRouter(nn.Module):
    """Two-level conservative routing: horizon rejection then local permission."""

    def __init__(self, hidden=24):
        super().__init__()
        # baseline/candidate/last/correction RGB (12), correction magnitude (1),
        # camera weight (1), flow xy (2), valid mask (1), future horizon (1).
        self.local_net = nn.Sequential(
            nn.Conv2d(18, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # Spatial mean and standard deviation of all 18 feature channels.
        self.horizon_net = nn.Sequential(
            nn.Linear(36, 32), nn.GELU(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.local_net[-1].weight)
        nn.init.constant_(self.local_net[-1].bias, 0.0)
        nn.init.zeros_(self.horizon_net[-1].weight)
        nn.init.constant_(self.horizon_net[-1].bias, -1.0)

    def forward(self, baseline, candidate, last_rgb, state):
        b, k, _, h, w = baseline.shape
        last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
        correction = candidate - baseline
        magnitude = correction.abs().mean(2, keepdim=True)
        correction_gray = correction.mean(2, keepdim=True)
        gradient_x = F.pad(
            correction_gray[..., 1:] - correction_gray[..., :-1], (0, 1, 0, 0),
        )
        gradient_y = F.pad(
            correction_gray[..., 1:, :] - correction_gray[..., :-1, :], (0, 0, 0, 1),
        )
        horizon = torch.linspace(0, 1, k, device=baseline.device, dtype=baseline.dtype)
        horizon = horizon.view(1, k, 1, 1, 1).expand(b, -1, -1, h, w)
        features = torch.cat((
            baseline, candidate, last, correction, magnitude,
            state["camera_weight"], gradient_x, gradient_y,
            state["camera_valid"], horizon,
        ), dim=2)
        flat = features.flatten(0, 1)
        local_gate = torch.sigmoid(self.local_net(flat)).view(b, k, 1, h, w)
        pooled = torch.cat((flat.mean((-2, -1)), flat.std((-2, -1))), -1)
        horizon_gate = torch.sigmoid(self.horizon_net(pooled)).view(b, k, 1, 1, 1)
        gate = local_gate * horizon_gate
        prediction = baseline + gate * correction
        return prediction.clamp(0, 1), gate, horizon_gate, local_gate


def limited(paths, maximum):
    paths = list(paths)
    return paths if maximum <= 0 else paths[:maximum]


def camera_candidate(model, baseline, batch, xyz, uv, past=None):
    source = batch if past is None else past
    _, state = model(
        baseline, batch["context_rgb"][:, -1], xyz, uv,
        source["past_xyz"].float(), source["past_uv"].float(),
        source["past_valid"], source["past_confidence"].float(),
        mode="structured",
    )
    candidate = (
        baseline + state["camera_weight"] * (state["camera_rgb"] - baseline)
    ).clamp(0, 1)
    return candidate, state


def condition_candidate(condition, camera_model, baseline, batch, xyz, uv, seed):
    correct, state = camera_candidate(camera_model, baseline, batch, xyz, uv)
    correction = correct - baseline
    if condition == "correct":
        return correct, state
    if condition == "reverse_direction":
        return (baseline - 1.5 * correction).clamp(0, 1), state
    if condition == "amplify_3x":
        return (baseline + 3.0 * correction).clamp(0, 1), state
    if condition == "horizon_mismatch":
        return (baseline + correction.flip(1)).clamp(0, 1), state
    if condition == "spatial_shift":
        shifted = torch.roll(correction, shifts=(9, -7), dims=(-2, -1))
        return (baseline + shifted).clamp(0, 1), state
    if condition == "local_patch_flip":
        generator = torch.Generator(device=uv.device).manual_seed(seed)
        b, k, _, h, w = correction.shape
        mask = torch.zeros((b, k, 1, h, w), device=uv.device, dtype=uv.dtype)
        for batch_index in range(b):
            top = int(torch.randint(0, max(h // 2, 1), (), generator=generator, device=uv.device))
            left = int(torch.randint(0, max(w // 2, 1), (), generator=generator, device=uv.device))
            mask[batch_index, :, :, top:top + h // 2, left:left + w // 2] = 1
        corrupted = correction * (1 - mask) - 2.0 * correction * mask
        return (baseline + corrupted).clamp(0, 1), state
    if condition == "cross_video" and xyz.shape[0] > 1:
        rolled = {
            key: torch.roll(batch[key], 1, 0)
            for key in ("past_xyz", "past_uv", "past_valid", "past_confidence")
        }
        return camera_candidate(
            camera_model, baseline, batch, torch.roll(xyz, 1, 0),
            torch.roll(uv, 1, 0), rolled,
        )
    return (baseline - 1.5 * correction).clamp(0, 1), state


def routing_loss(prediction, gate, horizon_gate, local_gate, baseline, candidate, target, last):
    base_error = (baseline - target).abs().mean(2, keepdim=True)
    candidate_error = (candidate - target).abs().mean(2, keepdim=True)
    routed_error = (prediction - target).abs().mean(2, keepdim=True)
    dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).float()

    reconstruction = (routed_error * (1 + 2 * dynamic)).mean()
    pixel_oracle = torch.sigmoid((base_error - candidate_error) / 0.002).detach()
    pixel_calibration = F.binary_cross_entropy(
        local_gate.clamp(1e-5, 1 - 1e-5), pixel_oracle,
    )
    horizon_regret = (candidate_error - base_error).mean((-2, -1), keepdim=True)
    horizon_oracle = torch.sigmoid(-horizon_regret / 0.0015).detach()
    horizon_calibration = F.binary_cross_entropy(
        horizon_gate.clamp(1e-5, 1 - 1e-5), horizon_oracle,
    )
    harmful = F.relu(candidate_error - base_error).detach()
    beneficial = F.relu(base_error - candidate_error).detach()
    reject_harm = (gate * harmful).mean()
    retain_benefit = ((1 - gate) * beneficial).mean()
    static_protection = (
        gate * (candidate - baseline).abs().mean(2, keepdim=True) * (1 - dynamic)
    ).mean()
    temporal = (gate[:, 1:] - gate[:, :-1]).abs().mean()
    horizon_reject = (horizon_gate * F.relu(horizon_regret).detach()).mean()
    horizon_retain = ((1 - horizon_gate) * F.relu(-horizon_regret).detach()).mean()
    loss = (
        reconstruction + 0.12 * pixel_calibration + 0.30 * horizon_calibration
        + 1.5 * reject_harm + 0.50 * retain_benefit
        + 3.0 * horizon_reject + 0.75 * horizon_retain
        + 0.05 * static_protection + 0.01 * temporal
    )
    pieces = {
        "loss": loss, "reconstruction": reconstruction,
        "pixel_calibration": pixel_calibration, "horizon_calibration": horizon_calibration,
        "reject_harm": reject_harm, "retain_benefit": retain_benefit,
        "horizon_reject": horizon_reject, "horizon_retain": horizon_retain,
        "static_protection": static_protection, "gate_temporal": temporal,
        "gate": gate.mean(), "horizon_gate": horizon_gate.mean(), "local_gate": local_gate.mean(),
        "gate_harmful": (gate * (harmful > 0).float()).sum() / (harmful > 0).sum().clamp_min(1),
        "gate_beneficial": (gate * (beneficial > 0).float()).sum() / (beneficial > 0).sum().clamp_min(1),
    }
    return loss, pieces


class MetricAccumulator:
    def __init__(self):
        self.values = defaultdict(float)

    def update(self, prediction, target, last, gate=None):
        difference = prediction - target
        dynamic = ((target - last[:, None]).abs().mean(2, keepdim=True) > 0.05).expand_as(difference)
        previous_prediction = torch.cat((last[:, None], prediction[:, :-1]), 1)
        previous_target = torch.cat((last[:, None], target[:, :-1]), 1)
        self.values["abs"] += difference.abs().sum().item()
        self.values["sq"] += difference.square().sum().item()
        self.values["count"] += difference.numel()
        self.values["dynamic_sq"] += difference.square()[dynamic].sum().item()
        self.values["dynamic_count"] += dynamic.sum().item()
        self.values["temporal_abs"] += (
            (prediction - previous_prediction) - (target - previous_target)
        ).abs().sum().item()
        if gate is not None:
            base = self.values
            base["gate_sum"] += gate.sum().item()
            base["gate_count"] += gate.numel()

    def result(self):
        v = self.values
        mse = v["sq"] / max(v["count"], 1)
        return {
            "MAE": v["abs"] / max(v["count"], 1),
            "MSE": mse,
            "PSNR": -10 * math.log10(max(mse, 1e-12)),
            "temporal_delta_MAE": v["temporal_abs"] / max(v["count"], 1),
            "dynamic_MSE": v["dynamic_sq"] / max(v["dynamic_count"], 1),
            "gate_mean": v["gate_sum"] / max(v["gate_count"], 1),
        }


@torch.no_grad()
def evaluate(router, base, trajectory, camera_model, loader, device):
    router.eval()
    accumulators = {"base": MetricAccumulator()}
    for condition in CONDITIONS:
        accumulators[f"raw_{condition}"] = MetricAccumulator()
        accumulators[f"routed_{condition}"] = MetricAccumulator()
    condition_gate = defaultdict(lambda: defaultdict(float))
    for batch_index, batch in enumerate(loader):
        batch = move(batch, device)
        baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
        target, last = batch["future_rgb"], batch["context_rgb"][:, -1]
        accumulators["base"].update(baseline, target, last)
        for condition_index, condition in enumerate(CONDITIONS):
            candidate, state = condition_candidate(
                condition, camera_model, baseline, batch, xyz, uv,
                100000 + batch_index * 10 + condition_index,
            )
            routed, gate, horizon_gate, local_gate = router(baseline, candidate, last, state)
            accumulators[f"raw_{condition}"].update(candidate, target, last)
            accumulators[f"routed_{condition}"].update(routed, target, last, gate)
            base_error = (baseline - target).abs().mean(2, keepdim=True)
            candidate_error = (candidate - target).abs().mean(2, keepdim=True)
            beneficial = candidate_error < base_error
            harmful = candidate_error > base_error
            for name, mask in (("beneficial", beneficial), ("harmful", harmful)):
                condition_gate[condition][f"{name}_sum"] += gate[mask].sum().item()
                condition_gate[condition][f"{name}_count"] += mask.sum().item()
            condition_gate[condition]["horizon_sum"] += horizon_gate.sum().item()
            condition_gate[condition]["horizon_count"] += horizon_gate.numel()
    result = {name: accumulator.result() for name, accumulator in accumulators.items()}
    result["conditional_gate"] = {
        condition: {
            **{
                name: values[f"{name}_sum"] / max(values[f"{name}_count"], 1)
                for name in ("beneficial", "harmful")
            },
            "horizon": values["horizon_sum"] / max(values["horizon_count"], 1),
        }
        for condition, values in condition_gate.items()
    }
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    splits, split_meta = combined_splits(args)
    datasets = {
        split: VideoForecastDataset(
            args.cache_root, split, 64, paths=limited(splits[split], maximum), cache_items=False,
        )
        for split, maximum in (("train", args.max_train), ("val", args.max_val), ("test", args.max_test))
    }
    train_loader = DataLoader(
        datasets["train"], args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=video_pad_collate, num_workers=0,
    )
    val_loader = DataLoader(datasets["val"], args.batch_size, collate_fn=video_pad_collate, num_workers=0)
    # Keep at least two clips per batch so the cross-video condition has a real
    # donor instead of falling back to temporal reversal.
    test_loader = DataLoader(
        datasets["test"], args.batch_size, collate_fn=video_pad_collate, num_workers=0,
    )

    base, trajectory, _, _ = load_frozen_models(args, device)
    checkpoint = torch.load(args.correction_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    camera_model = StructuredCaPSR(
        config.get("sigma", 0.035), config.get("hidden", 48),
        num_motion_clusters=config.get("num_motion_clusters", 4),
    ).to(device)
    camera_model.load_state_dict(checkpoint["model"])
    camera_model.eval()
    for model in (base, trajectory, camera_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    router = CounterfactualRiskRouter().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(parameter.numel() for parameter in router.parameters())
    history = []
    start = time.time()
    print(json.dumps({
        "trainable_params": trainable,
        "used": {key: len(value) for key, value in datasets.items()},
        "conditions": CONDITIONS, "split": split_meta,
    }), flush=True)

    for epoch in range(1, args.epochs + 1):
        router.train()
        totals = defaultdict(float)
        for batch_index, batch in enumerate(train_loader):
            batch = move(batch, device)
            condition = CONDITIONS[(batch_index + epoch - 1) % len(CONDITIONS)]
            with torch.no_grad():
                baseline, xyz, uv = frozen_prediction(base, trajectory, batch)
                candidate, state = condition_candidate(
                    condition, camera_model, baseline, batch, xyz, uv,
                    args.seed + epoch * 10000 + batch_index,
                )
            prediction, gate, horizon_gate, local_gate = router(
                baseline.detach(), candidate.detach(), batch["context_rgb"][:, -1], state,
            )
            loss, pieces = routing_loss(
                prediction, gate, horizon_gate, local_gate, baseline, candidate,
                batch["future_rgb"], batch["context_rgb"][:, -1],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            for name, value in pieces.items():
                totals[name] += float(value.detach())
        # A small, all-condition validation is deliberately used for the PoC.
        validation = evaluate(router, base, trajectory, camera_model, val_loader, device)
        row = {
            "epoch": epoch, "elapsed_seconds": time.time() - start,
            **{f"train_{name}": value / len(train_loader) for name, value in totals.items()},
            "val_base_MSE": validation["base"]["MSE"],
            "val_correct_MSE": validation["routed_correct"]["MSE"],
            "val_cross_video_MSE": validation["routed_cross_video"]["MSE"],
            "val_correct_gate": validation["routed_correct"]["gate_mean"],
            "val_cross_video_gate": validation["routed_cross_video"]["gate_mean"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"model": router.state_dict(), "epoch": epoch, "config": vars(args)}, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))

    test = evaluate(router, base, trajectory, camera_model, test_loader, device)
    result = {
        "test": test,
        "meta": {
            "trainable_params": trainable, "conditions": CONDITIONS,
            "used": {key: len(value) for key, value in datasets.items()},
            "elapsed_seconds": time.time() - start, "split": split_meta,
            "proof_of_concept": True,
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
