"""Train CaPSR on frozen D4RT trajectories and a frozen video backbone."""

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

from capsr_model import CaPSR, visibility_targets
from simvp_geometry_model import build_video_predictor_from_checkpoint
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True)
    p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dynamic-weight", type=float, default=3.0)
    p.add_argument("--visibility-weight", type=float, default=0.15)
    p.add_argument("--temporal-weight", type=float, default=0.10)
    p.add_argument("--lpips-weight", type=float, default=0.03)
    p.add_argument("--gradient-weight", type=float, default=0.05)
    p.add_argument("--static-identity-weight", type=float, default=0.10)
    p.add_argument("--correction-weight", type=float, default=0.005)
    p.add_argument("--sigma", type=float, default=0.035)
    p.add_argument("--depth-temperature", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def limited(dataset, maximum):
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, range(maximum))


def load_frozen_models(a, device):
    backbone_checkpoint = torch.load(a.backbone_checkpoint, map_location="cpu")
    base = build_video_predictor_from_checkpoint(backbone_checkpoint)
    incompatible = base.load_state_dict(backbone_checkpoint["model"], strict=False)
    allowed_missing = {
        "latent_gate_logit", "skip_gate_logit", "latent_film.weight", "latent_film.bias",
        "skip_film.weight", "skip_film.bias",
    }
    bad_missing = set(incompatible.missing_keys) - allowed_missing
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible backbone: missing={sorted(bad_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    base.to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    trajectory_checkpoint = torch.load(a.trajectory_checkpoint, map_location="cpu")
    trajectory_class = (
        VelocityResidualImplicitForecaster
        if trajectory_checkpoint.get("model_variant") == "velocity_residual"
        else ImplicitConditionedJointForecaster
    )
    trajectory = trajectory_class(
        trajectory_checkpoint["future_frames"], trajectory_checkpoint["hidden_dim"]
    )
    trajectory.load_state_dict(trajectory_checkpoint["model"])
    trajectory.to(device).eval()
    for parameter in trajectory.parameters():
        parameter.requires_grad_(False)
    return base, trajectory, backbone_checkpoint, trajectory_checkpoint


@torch.no_grad()
def frozen_prediction(base, trajectory, batch):
    base_prediction = base(batch["context_rgb"])
    xyz, uv = trajectory(
        batch["past_xyz"].float(), batch["past_uv"].float(), batch["past_valid"],
        batch["past_confidence"].float(),
    )
    return base_prediction, xyz, uv


def forward_capsr(model, base, trajectory, batch):
    base_prediction, xyz, uv = frozen_prediction(base, trajectory, batch)
    prediction, state = model(
        base_prediction, batch["context_rgb"][:, -1], xyz, uv,
        batch["past_xyz"].float(), batch["past_uv"].float(), batch["past_valid"],
        batch["past_confidence"].float(),
    )
    return prediction, base_prediction, state


def training_loss(prediction, base_prediction, state, batch, a, lpips_model=None):
    target = batch["future_rgb"]
    last = batch["context_rgb"][:, -1:]
    dynamic = ((target - last).abs().mean(2, keepdim=True) > 0.05).float()
    error = (prediction - target).abs() + 0.1 * (prediction - target).square()
    reconstruction = (error * (1 + a.dynamic_weight * dynamic)).mean()

    labels, visibility_mask = visibility_targets(batch["past_valid"], batch["future_valid"])
    logits = state["visibility_logits"]
    counts = torch.stack([((labels == index) & visibility_mask).sum() for index in range(3)]).float()
    class_weights = torch.sqrt(visibility_mask.sum().float() / counts.clamp_min(1))
    class_weights = class_weights.clamp(0.5, 10.0)
    probabilities = logits.softmax(-1)
    target_probability = probabilities.gather(-1, labels[..., None]).squeeze(-1).clamp_min(1e-6)
    visibility_raw = (
        -class_weights[labels] * (1 - target_probability).square() * target_probability.log()
    )
    visibility = (visibility_raw * visibility_mask).sum() / visibility_mask.sum().clamp_min(1)

    prediction_previous = torch.cat((last, prediction[:, :-1]), 1)
    target_previous = torch.cat((last, target[:, :-1]), 1)
    temporal = ((prediction - prediction_previous) - (target - target_previous)).abs().mean()

    lpips_loss = prediction.new_zeros(())
    if lpips_model is not None and a.lpips_weight > 0:
        b, k, c, h, w = prediction.shape
        lpips_loss = lpips_model(
            prediction.reshape(b * k, c, h, w) * 2 - 1,
            target.reshape(b * k, c, h, w) * 2 - 1,
        ).mean()
    pred_x, target_x = prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1]
    pred_y, target_y = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    gradient = 0.5 * ((pred_x - target_x).abs().mean() + (pred_y - target_y).abs().mean())

    static = (1 - dynamic)
    correction = prediction - base_prediction
    static_identity = (correction.abs().mean(2, keepdim=True) * static).sum() / static.sum().clamp_min(1)
    correction_size = (
        state["transport_gate"] * state["transport_delta"].abs().mean(2, keepdim=True)
        + state["synthesis_gate"] * state["synthesis_residual"].abs().mean(2, keepdim=True)
    ).mean()
    total = (
        reconstruction + a.visibility_weight * visibility + a.temporal_weight * temporal
        + a.lpips_weight * lpips_loss + a.gradient_weight * gradient
        + a.static_identity_weight * static_identity + a.correction_weight * correction_size
    )
    return total, {
        "reconstruction": reconstruction.detach(),
        "visibility": visibility.detach(),
        "temporal": temporal.detach(),
        "lpips": lpips_loss.detach(),
        "gradient": gradient.detach(),
        "static_identity": static_identity.detach(),
        "correction": correction_size.detach(),
    }


@torch.no_grad()
def validation(model, base, trajectory, loader, device):
    model.eval()
    mse_sum = pixel_count = dynamic_abs = dynamic_count = temporal_sum = batches = 0.0
    visible_correct = visible_count = 0.0
    for batch in loader:
        batch = move(batch, device)
        prediction, _, state = forward_capsr(model, base, trajectory, batch)
        target = batch["future_rgb"]
        difference = prediction - target
        mse_sum += difference.square().sum().item()
        pixel_count += target.numel()
        last = batch["context_rgb"][:, -1:]
        dynamic = ((target - last).abs().mean(2, keepdim=True) > 0.05).expand_as(target)
        dynamic_abs += difference.abs()[dynamic].sum().item()
        dynamic_count += dynamic.sum().item()
        pred_previous = torch.cat((last, prediction[:, :-1]), 1)
        target_previous = torch.cat((last, target[:, :-1]), 1)
        temporal_sum += ((prediction - pred_previous) - (target - target_previous)).abs().mean().item()
        labels, mask = visibility_targets(batch["past_valid"], batch["future_valid"])
        visible_correct += ((state["visibility_logits"].argmax(-1) == labels) & mask).sum().item()
        visible_count += mask.sum().item()
        batches += 1
    return {
        "MSE": mse_sum / max(pixel_count, 1),
        "dynamic_MAE": dynamic_abs / max(dynamic_count, 1),
        "temporal_MAE": temporal_sum / max(batches, 1),
        "visibility_accuracy": visible_correct / max(visible_count, 1),
    }


@torch.no_grad()
def full_evaluation(model, base, trajectory, loader, device, future_frames):
    model.eval()
    accumulators = {
        "no_geometry": VideoMetricAccumulator(device, future_frames),
        "capsr": VideoMetricAccumulator(device, future_frames),
    }
    extra = {
        key: {"dyn_abs": 0.0, "dyn_sq": 0.0, "dyn_count": 0.0,
              "intersection": 0.0, "union": 0.0, "pred_pos": 0.0, "gt_pos": 0.0}
        for key in accumulators
    }
    visibility_confusion = torch.zeros(3, 3, dtype=torch.float64)
    transport_sum = synthesis_sum = support_sum = camera_sum = object_sum = points = pixels = 0.0
    for batch in loader:
        batch = move(batch, device)
        prediction, baseline, state = forward_capsr(model, base, trajectory, batch)
        target = batch["future_rgb"]
        last = batch["context_rgb"][:, -1:]
        gt_motion = (target - last).abs().mean(2) > 0.05
        for name, value in (("no_geometry", baseline), ("capsr", prediction)):
            accumulators[name].update(value, target, batch["context_rgb"][:, -1])
            difference = value - target
            dynamic = gt_motion[:, :, None].expand_as(difference)
            stat = extra[name]
            stat["dyn_abs"] += difference.abs()[dynamic].sum().item()
            stat["dyn_sq"] += difference.square()[dynamic].sum().item()
            stat["dyn_count"] += dynamic.sum().item()
            pred_motion = (value - last).abs().mean(2) > 0.05
            stat["intersection"] += (gt_motion & pred_motion).sum().item()
            stat["union"] += (gt_motion | pred_motion).sum().item()
            stat["pred_pos"] += pred_motion.sum().item()
            stat["gt_pos"] += gt_motion.sum().item()
        labels, mask = visibility_targets(batch["past_valid"], batch["future_valid"])
        predicted_labels = state["visibility_logits"].argmax(-1)
        for truth in range(3):
            for predicted in range(3):
                visibility_confusion[truth, predicted] += (
                    (labels == truth) & (predicted_labels == predicted) & mask
                ).sum().cpu()
        transport_sum += state["transport_gate"].sum().item()
        synthesis_sum += state["synthesis_gate"].sum().item()
        support_sum += state["propagation_support"].sum().item()
        pixels += state["transport_gate"].numel()
        valid = batch["past_valid"].any(2)[:, :, None].expand_as(state["camera_flow"][..., 0])
        camera_sum += torch.linalg.vector_norm(state["camera_flow"], dim=-1)[valid].sum().item()
        object_sum += torch.linalg.vector_norm(state["object_flow"], dim=-1)[valid].sum().item()
        points += valid.sum().item()
    results = {}
    for name, accumulator in accumulators.items():
        metrics = accumulator.compute()
        stat = extra[name]
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
    row_sum = visibility_confusion.sum(1, keepdim=True).clamp_min(1)
    results["state"] = {
        "visibility_confusion": visibility_confusion.tolist(),
        "visibility_recall_per_class": (visibility_confusion.diag() / row_sum[:, 0]).tolist(),
        "transport_gate_fraction": transport_sum / max(pixels, 1),
        "synthesis_gate_fraction": synthesis_sum / max(pixels, 1),
        "propagation_support_fraction": support_sum / max(pixels, 1),
        "camera_flow_mean": camera_sum / max(points, 1),
        "object_flow_mean": object_sum / max(points, 1),
    }
    return results


def main():
    a = parse_args()
    seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits, split_meta = combined_splits(a)
    make_dataset = lambda split: VideoForecastDataset(
        a.cache_root, split, 64, paths=splits[split], cache_items=False
    )
    train_dataset = limited(make_dataset("train"), a.max_train)
    val_dataset = limited(make_dataset("val"), a.max_val)
    test_dataset = limited(make_dataset("test"), a.max_test)
    train_loader = DataLoader(
        train_dataset, a.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(a.seed), collate_fn=video_pad_collate,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, a.batch_size, collate_fn=video_pad_collate, num_workers=0)
    test_loader = DataLoader(test_dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    base, trajectory, backbone_checkpoint, trajectory_checkpoint = load_frozen_models(a, device)
    model = CaPSR(a.sigma, a.depth_temperature, a.hidden).to(device)
    lpips_model = None
    if a.lpips_weight > 0:
        import lpips
        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    history = []
    best = float("inf")
    start_epoch = 1
    last_path, best_path = output / "last.pt", output / "best.pt"
    if a.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best", float("inf")))
        if (output / "history.json").exists():
            history = json.loads((output / "history.json").read_text())
    print(json.dumps({
        "split": split_meta,
        "used": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_epoch": backbone_checkpoint.get("epoch"),
        "trajectory_epoch": trajectory_checkpoint.get("epoch"),
    }), flush=True)
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        sums = {name: 0.0 for name in (
            "loss", "reconstruction", "visibility", "temporal", "lpips", "gradient",
            "static_identity", "correction"
        )}
        for batch in train_loader:
            batch = move(batch, device)
            prediction, baseline, state = forward_capsr(model, base, trajectory, batch)
            loss, pieces = training_loss(prediction, baseline, state, batch, a, lpips_model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["loss"] += loss.item()
            for name, value in pieces.items():
                sums[name] += value.item()
        val = validation(model, base, trajectory, val_loader, device)
        score = val["MSE"] + 0.01 * val["dynamic_MAE"] + 0.05 * val["temporal_MAE"]
        row = {"epoch": epoch, **{f"train_{k}": v / len(train_loader) for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items()}, "score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        state_dict = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": min(best, score), "config": vars(a),
        }
        torch.save(state_dict, last_path)
        if score < best:
            best = score
            torch.save(state_dict, best_path)
        (output / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    metrics = full_evaluation(model, base, trajectory, test_loader, device,
                              trajectory_checkpoint["future_frames"])
    summary = {
        "metrics": metrics,
        "meta": {
            "split": split_meta,
            "used": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
            "best_epoch": torch.load(best_path, map_location="cpu")["epoch"],
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "backbone_checkpoint": a.backbone_checkpoint,
            "trajectory_checkpoint": a.trajectory_checkpoint,
            "architecture": "depth-affine egomotion + object residual + 3-state visibility + softsplat/synthesis",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
