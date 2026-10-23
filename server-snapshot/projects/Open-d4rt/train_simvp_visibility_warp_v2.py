#!/usr/bin/env python3
"""Stage-2 visibility-, boundary-, and object-motion-aware feature warp."""

from __future__ import annotations

import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_simvp_feature_warp as base


def _last_and_velocity(values, valid, recent_pairs=4):
    values = torch.nan_to_num(values.float())
    pair = valid[:, :, 1:] & valid[:, :, :-1]
    if pair.shape[2] > recent_pairs:
        pair[:, :, :-recent_pairs] = False
    delta = values[:, :, 1:] - values[:, :, :-1]
    velocity = torch.nanmedian(delta.masked_fill(~pair[..., None], float("nan")), dim=2).values
    velocity = torch.nan_to_num(velocity).clamp(-0.08, 0.08)
    steps = values.shape[2]
    index = (valid.long() * torch.arange(1, steps + 1, device=values.device)).argmax(2)
    last = values.gather(2, index[..., None, None].expand(-1, -1, 1, values.shape[-1])).squeeze(2)
    return last, velocity


def zbuffer_visibility(xyz, uv, valid, grid_size=32, temperature=0.12):
    """Suppress farther points when several trajectories project to one cell."""
    b, n, k, _ = xyz.shape
    log_depth = torch.log1p(torch.linalg.vector_norm(torch.nan_to_num(xyz.float()), dim=-1))
    x = (uv[..., 0].clamp(0, 1) * (grid_size - 1)).round().long()
    y = (uv[..., 1].clamp(0, 1) * (grid_size - 1)).round().long()
    visibility = torch.zeros_like(log_depth)
    for batch_index in range(b):
        for step in range(k):
            active = valid[batch_index, :, step].bool()
            if not active.any():
                continue
            index = y[batch_index, :, step] * grid_size + x[batch_index, :, step]
            minimum = log_depth.new_full((grid_size * grid_size,), float("inf"))
            minimum.scatter_reduce_(0, index[active], log_depth[batch_index, active, step],
                                    reduce="amin", include_self=True)
            gap = (log_depth[batch_index, :, step] - minimum[index]).clamp_min(0)
            visibility[batch_index, :, step] = torch.exp(-gap / temperature) * active.float()
    return visibility


def geometry_bundle_v2(mode, joint, batch, future_frames):
    """History-only D4RT geometry with camera/object and visibility calibration."""
    from trajectory_model import GeometryConditionEncoder

    valid = batch["past_valid"].bool()
    point_ok = valid.any(2)
    anchor_uv, historical_velocity = _last_and_velocity(batch["past_uv"], valid)
    with torch.no_grad():
        xyz, future_uv = joint(
            batch["past_xyz"].float(), batch["past_uv"].float(), valid,
            batch["past_confidence"].float(),
        )
    confidence = torch.nan_to_num(batch["past_confidence"].float()).amax(2)
    confidence = confidence[:, :, None].expand(-1, -1, future_frames).clamp(0, 1)
    displacement = future_uv - anchor_uv[:, :, None]

    # Robust median image motion is treated as the camera/common component.
    masked = displacement.masked_fill(~point_ok[:, :, None, None], float("nan"))
    camera_flow = torch.nanmedian(masked, dim=1).values
    camera_flow = torch.nan_to_num(camera_flow)
    object_flow = displacement - camera_flow[:, None]
    object_motion = torch.linalg.vector_norm(object_flow, dim=-1)
    dynamic = ((object_motion - 0.008) / 0.035).clamp(0, 1)

    velocity = torch.cat((displacement[:, :, :1], future_uv[:, :, 1:] - future_uv[:, :, :-1]), 2)
    acceleration = torch.zeros_like(confidence)
    acceleration[:, :, 1:] = torch.linalg.vector_norm(velocity[:, :, 1:] - velocity[:, :, :-1], dim=-1)
    temporal_consistency = torch.exp(-acceleration / 0.05)
    history_consistency = torch.exp(
        -torch.linalg.vector_norm(displacement[:, :, 0] - historical_velocity, dim=-1) / 0.04
    )[:, :, None]
    visibility = zbuffer_visibility(
        xyz, future_uv, point_ok[:, :, None].expand(-1, -1, future_frames),
    )
    reliability = (point_ok[:, :, None].float() * confidence * dynamic
                   * temporal_consistency * history_consistency * visibility)
    raw = GeometryConditionEncoder.raw_features(xyz, future_uv, confidence)
    if mode == "zero":
        return torch.zeros_like(raw), future_uv, torch.zeros_like(reliability), anchor_uv
    if mode == "shuffled":
        if raw.shape[0] > 1:
            order = torch.roll(torch.arange(raw.shape[0], device=raw.device), 1)
            return raw[order], future_uv[order], reliability[order], anchor_uv[order]
        order = torch.arange(raw.shape[1] - 1, -1, -1, device=raw.device)
        return raw[:, order], future_uv[:, order], reliability[:, order], anchor_uv[:, order]
    if mode != "correct":
        raise ValueError(mode)
    return raw, future_uv, reliability, anchor_uv


class CoherentSparseDenseFlow(base.SparseDenseFlow):
    """Dense flow with a local motion-variance confidence."""

    def forward(self, anchor_uv, future_uv, reliability, height, width):
        displacement = future_uv - anchor_uv[:, :, None]
        numerator, denominator = self._splat(displacement, future_uv, reliability, height, width)
        square_numerator, _ = self._splat(displacement.square(), future_uv, reliability, height, width)
        padding = self.kernel.shape[-1] // 2
        channels = numerator.shape[1]
        kernel = self.kernel.expand(channels, 1, -1, -1)
        numerator = F.conv2d(numerator, kernel, padding=padding, groups=channels)
        square_numerator = F.conv2d(square_numerator, kernel, padding=padding, groups=channels)
        denominator = F.conv2d(denominator, self.kernel, padding=padding)
        flow = numerator / denominator.clamp_min(1e-5)
        second = square_numerator / denominator.clamp_min(1e-5)
        variance = (second - flow.square()).clamp_min(0).sum(1, keepdim=True).sqrt()
        coherence = torch.exp(-variance / 0.025)
        support = (1 - torch.exp(-4 * denominator)).clamp(0, 1) * coherence
        b, k = anchor_uv.shape[0], future_uv.shape[2]
        return (flow.reshape(b, k, 2, height, width),
                support.reshape(b, k, 1, height, width),
                coherence.reshape(b, k, 1, height, width))


class VisibilityDualPathFeatureWarp(base.DualPathFeatureWarp):
    """Object propagation with one confidence application and explicit synthesis permission."""

    appearance_warmup_epochs = 3
    gate_loss_weight = 0.02

    def __init__(self, channels=16, flow_kernel=9, scale=0.15):
        super().__init__(channels, flow_kernel, scale)
        self.raster = CoherentSparseDenseFlow(flow_kernel)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        train_appearance = epoch > self.appearance_warmup_epochs
        for parameter in self.appearance.parameters():
            parameter.requires_grad_(train_appearance)

    @staticmethod
    def edge_permission(hidden):
        feature = hidden.mean(2, keepdim=True)
        gx = F.pad((feature[..., :, 1:] - feature[..., :, :-1]).abs(), (0, 1, 0, 0))
        gy = F.pad((feature[..., 1:, :] - feature[..., :-1, :]).abs(), (0, 0, 0, 1))
        gradient = gx + gy
        scale = gradient.mean((-2, -1), keepdim=True).clamp_min(1e-4)
        return torch.exp(-1.5 * gradient / scale)

    @staticmethod
    def visual_motion_permission(source, hidden):
        change = (hidden - source[:, None]).abs().mean(2, keepdim=True)
        mean = change.mean((-2, -1), keepdim=True)
        std = change.std((-2, -1), keepdim=True).clamp_min(1e-4)
        permission = (change > mean + 0.35 * std).float()
        flat = F.max_pool2d(permission.flatten(0, 1), 3, stride=1, padding=1)
        return flat.reshape_as(permission)

    def forward(self, source, hidden, geometry, future_uv, reliability, anchor_uv):
        b, k, c, h, w = hidden.shape
        flow, dense_support, coherence = self.raster(anchor_uv, future_uv, reliability, h, w)
        geometry_map, sparse_support = self.geometry(
            geometry, future_uv, reliability, h, w, return_support=True,
        )
        edges = self.edge_permission(hidden)
        support = torch.maximum(sparse_support, dense_support * edges).clamp(0, 1)
        warped = base.backward_warp(source, flow)
        geo_input = torch.cat((hidden, warped, geometry_map, flow, support), 2).permute(0, 2, 1, 3, 4)
        raw_app = self.appearance(hidden.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        raw_geo = self.propagation(geo_input).permute(0, 2, 1, 3, 4)
        gate_probability = torch.sigmoid(self.gate(geo_input).permute(0, 2, 1, 3, 4))
        gate = gate_probability * support

        expanded = F.max_pool2d(support.flatten(0, 1), 5, stride=1, padding=2).reshape_as(support)
        disocclusion = (expanded - support).clamp(0, 1)
        visual_motion = self.visual_motion_permission(source, hidden)
        synthesis_permission = torch.maximum(disocclusion, visual_motion * 0.35).clamp(0, 1)
        app_delta = self.scale * torch.tanh(raw_app) * synthesis_permission
        geo_delta = self.scale * torch.tanh(raw_geo) * gate
        corrected = hidden + app_delta + geo_delta
        self.last = {
            "flow": flow.detach(), "support": support.detach(), "coherence": coherence.detach(),
            "gate": gate.detach(), "gate_probability": gate_probability,
            "appearance_permission": synthesis_permission.detach(), "disocclusion": disocclusion.detach(),
            "appearance_delta": app_delta, "geometry_delta": geo_delta, "warped": warped.detach(),
        }
        return corrected, app_delta, geo_delta

    def auxiliary_loss(self, future_rgb, last_rgb):
        gate = self.last["gate_probability"]
        support = self.last["support"].detach()
        b, k, _, h, w = gate.shape
        target = ((future_rgb - last_rgb[:, None]).abs().mean(2, keepdim=True) > 0.05).float()
        target = F.adaptive_max_pool2d(target.flatten(0, 1), (h, w)).reshape(b, k, 1, h, w)
        active = F.max_pool2d(support.flatten(0, 1), 3, stride=1, padding=1).reshape_as(support)
        if active.sum() < 1:
            value = gate.sum() * 0
        else:
            bce = F.binary_cross_entropy(gate, target, reduction="none")
            bce = (bce * active).sum() / active.sum().clamp_min(1)
            prediction = gate * active
            truth = target * active
            dice = 1 - (2 * (prediction * truth).sum() + 1) / (prediction.sum() + truth.sum() + 1)
            value = self.gate_loss_weight * (bce + dice)
        return value, {"motion_gate_loss": value.detach()}


# Reuse the rigorously tested training/evaluation pipeline while replacing only
# the stage-2 causal geometry interface and correction module.
base.DualPathFeatureWarp = VisibilityDualPathFeatureWarp
base.geometry_bundle = geometry_bundle_v2


if __name__ == "__main__":
    base.main()
