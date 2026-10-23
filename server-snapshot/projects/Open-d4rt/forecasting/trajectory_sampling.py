"""Curriculum sampling utilities for motion-imbalanced UV trajectories."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from trajectory_metrics import last_uv
from video_forecast_dataset import video_pad_collate


MOTION_EDGES = (0.0, 0.02, 0.05, 0.10, float("inf"))
START_MIX = (0.55, 0.25, 0.15, 0.05)
MID_MIX = (0.35, 0.25, 0.25, 0.15)
FINAL_MIX = (0.25, 0.25, 0.25, 0.25)


def curriculum_mix(epoch: int, epochs: int) -> torch.Tensor:
    """Linearly interpolate start -> middle -> balanced bucket ratios."""
    progress = 1.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
    if progress <= 0.5:
        alpha = progress / 0.5
        a, b = START_MIX, MID_MIX
    else:
        alpha = (progress - 0.5) / 0.5
        a, b = MID_MIX, FINAL_MIX
    return torch.tensor([(1 - alpha) * x + alpha * y for x, y in zip(a, b)], dtype=torch.float32)


def point_buckets(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final-displacement bucket ids and points having a valid future."""
    anchor = last_uv(batch)
    target = torch.nan_to_num(batch["future_uv_teacher"].float())
    amp = torch.linalg.vector_norm(target[:, :, -1] - anchor, dim=-1)
    valid = batch["future_valid"].bool().any(-1) & torch.isfinite(
        batch["future_uv_teacher"].float()
    ).all(-1).any(-1)
    bucket = torch.bucketize(amp, torch.tensor(MOTION_EDGES[1:-1], device=amp.device))
    return bucket, valid


def curriculum_losses(model, batch, epoch, epochs, uv_weight=5.0, mild_weights=(1.0, 1.2, 1.5, 2.0)):
    """Use all XYZ targets but resample UV tracks according to the curriculum."""
    xyz, uv = model(batch["past_xyz"].float(), batch["past_uv"].float(),
                    batch["past_valid"], batch["past_confidence"].float())
    tx = batch["future_xyz_teacher"].float()
    tu = batch["future_uv_teacher"].float()
    mask = (batch["future_valid"].bool() & torch.isfinite(tx).all(-1)
            & torch.isfinite(tu).all(-1))
    lxyz = F.smooth_l1_loss(xyz, torch.nan_to_num(tx), reduction="none").mean(-1)[mask].mean()

    raw_uv = F.smooth_l1_loss(uv, torch.nan_to_num(tu), reduction="none").mean(-1)
    bucket, point_valid = point_buckets(batch)
    mix = curriculum_mix(epoch, epochs).to(uv.device)
    available = torch.tensor([((bucket == i) & point_valid).sum() for i in range(4)], device=uv.device)
    nonempty = available > 0
    effective_mix = mix * nonempty
    effective_mix = effective_mix / effective_mix.sum().clamp_min(1e-8)
    # Keep the number of selected tracks close to the number of valid tracks.
    total_points = int(point_valid.sum().item())
    selected = []
    counts = []
    for i in range(4):
        candidates = ((bucket == i) & point_valid).nonzero(as_tuple=False)
        quota = int(round(total_points * float(effective_mix[i])))
        if quota <= 0 or candidates.numel() == 0:
            counts.append(0)
            continue
        choice = torch.randint(candidates.shape[0], (quota,), device=uv.device)
        selected.append(candidates[choice])
        counts.append(quota)
    if selected:
        indices = torch.cat(selected, 0)
        point_loss = []
        point_weight = []
        for bi, ni in indices:
            valid_t = mask[bi, ni]
            if valid_t.any():
                point_loss.append(raw_uv[bi, ni][valid_t].mean())
                point_weight.append(mild_weights[int(bucket[bi, ni])])
        if point_loss:
            values = torch.stack(point_loss)
            weights = values.new_tensor(point_weight)
            luv = (values * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            luv = raw_uv[mask].mean()
    else:
        luv = raw_uv[mask].mean()
    kinematic = uv.new_zeros(())
    if getattr(model, "uses_velocity_residual", False):
        anchor = last_uv(batch)
        pred_path = torch.cat((anchor[:, :, None], uv), 2)
        target_path = torch.cat((anchor[:, :, None], torch.nan_to_num(tu)), 2)
        pred_velocity = pred_path[:, :, 1:] - pred_path[:, :, :-1]
        target_velocity = target_path[:, :, 1:] - target_path[:, :, :-1]
        velocity_error = F.smooth_l1_loss(pred_velocity, target_velocity, reduction="none").mean(-1)
        velocity_loss = velocity_error[mask].mean()
        final_mask = mask[:, :, -1]
        endpoint_loss = (raw_uv[:, :, -1][final_mask].mean()
                         if final_mask.any() else uv.new_zeros(()))
        pred_acceleration = pred_velocity[:, :, 1:] - pred_velocity[:, :, :-1]
        target_acceleration = target_velocity[:, :, 1:] - target_velocity[:, :, :-1]
        acceleration_mask = mask[:, :, 1:] & mask[:, :, :-1]
        acceleration_error = F.smooth_l1_loss(
            pred_acceleration, target_acceleration, reduction="none").mean(-1)
        acceleration_loss = (acceleration_error[acceleration_mask].mean()
                             if acceleration_mask.any() else uv.new_zeros(()))
        kinematic = 0.75 * velocity_loss + 1.5 * endpoint_loss + 0.15 * acceleration_loss
    return lxyz + uv_weight * (luv + kinematic), lxyz, luv, xyz, uv, mask, {
        "target_mix": mix.tolist(), "sampled_counts": counts,
        "available_counts": available.tolist(),
        "kinematic_loss": float(kinematic.detach()),
    }


def clip_histograms(dataset) -> torch.Tensor:
    """Count motion buckets in every clip once (dataset caching makes this cheap later)."""
    rows = []
    for i in range(len(dataset)):
        sample = video_pad_collate([dataset[i]])
        bucket, valid = point_buckets(sample)
        rows.append([(bucket[valid] == j).sum().item() for j in range(4)])
    return torch.tensor(rows, dtype=torch.float64)


def curriculum_loader(dataset, histograms, epoch, epochs, batch_size, seed):
    """Oversample clips containing buckets that are rare relative to the target mix."""
    target = curriculum_mix(epoch, epochs).double()
    global_freq = histograms.sum(0)
    global_freq = global_freq / global_freq.sum().clamp_min(1)
    rarity = target / global_freq.clamp_min(1e-8)
    proportions = histograms / histograms.sum(1, keepdim=True).clamp_min(1)
    weights = (proportions * rarity).sum(1).clamp_min(1e-6)
    generator = torch.Generator().manual_seed(seed + epoch)
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size, sampler=sampler, collate_fn=video_pad_collate,
                      num_workers=0), global_freq.float().tolist()
