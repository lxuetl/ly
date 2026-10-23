"""Camera-canonical predictive state and dual-path rendering (CaPSR).

The module keeps D4RT and the video backbone frozen.  It estimates a robust,
depth-conditioned global egomotion field from historical tracks, separates the
future UV displacement into egomotion and object residual components, predicts
explicit track visibility transitions, then uses two complementary paths:

* observable path: depth-aware point soft-splat from the last observed RGB;
* unobservable path: a small synthesis residual on top of the frozen backbone.

Only the lightweight visibility and rendering heads are trainable.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _last_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    steps = values.shape[2]
    index = (valid.long() * torch.arange(1, steps + 1, device=values.device)).argmax(2)
    gathered = values.gather(
        2, index[..., None, None].expand(-1, -1, 1, values.shape[-1])
    ).squeeze(2)
    return torch.where(valid.any(2)[..., None], torch.nan_to_num(gathered), torch.zeros_like(gathered))


class RobustDepthAffineEgomotion(nn.Module):
    """Fit dominant image motion from history without dataset/camera IDs.

    The flow basis [1, u, v, normalized inverse depth] captures translation,
    rotation/zoom-like affine motion, and first-order depth parallax.  Two IRLS
    iterations reduce contamination by independently moving objects.
    """

    def __init__(self, recent_pairs: int = 4, ridge: float = 1e-3, huber_scale: float = 1.5):
        super().__init__()
        self.recent_pairs = int(recent_pairs)
        self.ridge = float(ridge)
        self.huber_scale = float(huber_scale)

    @staticmethod
    def _basis(uv, xyz, valid):
        depth = torch.linalg.vector_norm(torch.nan_to_num(xyz), dim=-1).clamp_min(1e-3)
        inverse = depth.reciprocal()
        masked = inverse.masked_fill(~valid, float("nan"))
        median = torch.nanmedian(masked.reshape(masked.shape[0], -1), dim=1).values
        median = torch.nan_to_num(median, nan=1.0).clamp_min(1e-3)
        inverse = (inverse / median[:, None, None]).clamp(0, 8)
        return torch.stack((torch.ones_like(inverse), uv[..., 0], uv[..., 1], inverse), -1)

    @torch.no_grad()
    def forward(self, past_uv, past_xyz, past_valid, past_confidence, future_frames):
        uv = torch.nan_to_num(past_uv.float())
        xyz = torch.nan_to_num(past_xyz.float())
        valid = past_valid.bool()
        pair = valid[:, :, 1:] & valid[:, :, :-1]
        if pair.shape[2] > self.recent_pairs:
            pair[:, :, :-self.recent_pairs] = False
        confidence = torch.minimum(
            torch.nan_to_num(past_confidence[:, :, 1:].float()),
            torch.nan_to_num(past_confidence[:, :, :-1].float()),
        ).clamp(0, 1)
        design = self._basis(uv[:, :, :-1], xyz[:, :, :-1], pair)
        target = uv[:, :, 1:] - uv[:, :, :-1]
        coefficients = []
        eye = torch.eye(4, device=uv.device, dtype=uv.dtype)
        for batch_index in range(uv.shape[0]):
            x = design[batch_index].reshape(-1, 4)
            y = target[batch_index].reshape(-1, 2)
            weight = (pair[batch_index].float() * confidence[batch_index]).reshape(-1)
            if int((weight > 0).sum()) < 8:
                coefficients.append(torch.zeros(4, 2, device=uv.device, dtype=uv.dtype))
                continue
            beta = torch.zeros(4, 2, device=uv.device, dtype=uv.dtype)
            robust = torch.ones_like(weight)
            for _ in range(3):
                w = weight * robust
                lhs = x.T @ (w[:, None] * x) + self.ridge * eye
                rhs = x.T @ (w[:, None] * y)
                beta = torch.linalg.solve(lhs, rhs)
                error = torch.linalg.vector_norm(x @ beta - y, dim=-1)
                scale = torch.median(error[weight > 0]).clamp_min(1e-4)
                robust = torch.clamp(self.huber_scale * scale / error.clamp_min(1e-6), max=1.0)
            coefficients.append(beta)
        beta = torch.stack(coefficients)
        anchor_uv = _last_valid(uv, valid)
        anchor_xyz = _last_valid(xyz, valid)
        anchor_basis = self._basis(anchor_uv[:, :, None], anchor_xyz[:, :, None], valid.any(2)[:, :, None])[:, :, 0]
        one_step = torch.einsum("bnc,bcd->bnd", anchor_basis, beta)
        steps = torch.arange(1, future_frames + 1, device=uv.device, dtype=uv.dtype).view(1, 1, -1, 1)
        return one_step[:, :, None] * steps, beta


class PredictiveVisibilityHead(nn.Module):
    """Predict visible, occluded, and reappeared track states."""

    feature_dim = 12

    def __init__(self, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor((1.5, -0.5, -2.0)))

    def forward(self, camera_flow, object_flow, total_flow, future_xyz,
                past_valid, past_confidence):
        b, n, k, _ = total_flow.shape
        depth = torch.log1p(torch.linalg.vector_norm(torch.nan_to_num(future_xyz), dim=-1))[..., None]
        confidence = torch.nan_to_num(past_confidence.float()).amax(2)[:, :, None, None].expand(-1, -1, k, -1)
        last_visible = past_valid[:, :, -1].float()[:, :, None, None].expand(-1, -1, k, -1)
        visible_fraction = past_valid.float().mean(2)[:, :, None, None].expand(-1, -1, k, -1)
        time_since_seen = self._time_since_seen(past_valid)[:, :, None, None].expand(-1, -1, k, -1)
        horizon = torch.linspace(0, 1, k, device=total_flow.device, dtype=total_flow.dtype)
        horizon = horizon.view(1, 1, k, 1).expand(b, n, -1, -1)
        features = torch.cat((camera_flow, object_flow, total_flow, depth, confidence,
                              last_visible, visible_fraction, time_since_seen, horizon), -1)
        return self.net(features)

    @staticmethod
    def _time_since_seen(valid):
        t = valid.shape[2]
        index = (valid.long() * torch.arange(1, t + 1, device=valid.device)).amax(2)
        return (t - index).float() / max(t, 1)


class CaPSR(nn.Module):
    """Lightweight camera/object state decomposition and dual renderer."""

    dense_state_channels = 7  # camera flow 2, object flow 2, visibility probabilities 3

    def __init__(self, sigma: float = 0.035, depth_temperature: float = 1.0,
                 hidden: int = 48, max_synthesis_residual: float = 0.12):
        super().__init__()
        self.sigma = float(sigma)
        self.depth_temperature = float(depth_temperature)
        self.max_synthesis_residual = float(max_synthesis_residual)
        self.egomotion = RobustDepthAffineEgomotion()
        self.visibility = PredictiveVisibilityHead(hidden)
        # base RGB, last RGB, propagated RGB, disagreement RGB, dense state,
        # propagation support, base motion, and disocclusion prior.
        channels = 3 + 3 + 3 + 3 + self.dense_state_channels + 3
        self.renderer = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.transport_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.synthesis_head = nn.Conv2d(hidden, 3, 3, padding=1)
        self.synthesis_gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.synthesis_head.weight)
        nn.init.zeros_(self.synthesis_head.bias)
        nn.init.zeros_(self.transport_head.weight)
        nn.init.constant_(self.transport_head.bias, -1.5)
        nn.init.zeros_(self.synthesis_gate_head.weight)
        nn.init.constant_(self.synthesis_gate_head.bias, -2.0)

    def decompose(self, predicted_xyz, predicted_uv, past_xyz, past_uv,
                  past_valid, past_confidence):
        k = predicted_uv.shape[2]
        anchor_uv = _last_valid(past_uv, past_valid)
        camera_flow, camera_coefficients = self.egomotion(
            past_uv, past_xyz, past_valid, past_confidence, k
        )
        total_flow = predicted_uv - anchor_uv[:, :, None]
        object_flow = total_flow - camera_flow
        visibility_logits = self.visibility(
            camera_flow, object_flow, total_flow, predicted_xyz,
            past_valid, past_confidence,
        )
        return {
            "anchor_uv": anchor_uv,
            "camera_flow": camera_flow,
            "object_flow": object_flow,
            "total_flow": total_flow,
            "camera_coefficients": camera_coefficients,
            "visibility_logits": visibility_logits,
            "visibility_probabilities": visibility_logits.softmax(-1),
        }

    def _softsplat(self, last_rgb, predicted_uv, predicted_xyz, confidence,
                   state, point_valid):
        b, n, k, _ = predicted_uv.shape
        h, w = last_rgb.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=last_rgb.device, dtype=last_rgb.dtype),
            torch.linspace(0, 1, w, device=last_rgb.device, dtype=last_rgb.dtype),
            indexing="ij",
        )
        source_grid = state["anchor_uv"].mul(2).sub(1).view(b, n, 1, 2)
        source_color = F.grid_sample(
            last_rgb, source_grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(-1).permute(0, 2, 1)
        probabilities = state["visibility_probabilities"]
        outputs, supports, dense_states = [], [], []
        for step in range(k):
            uv = torch.nan_to_num(predicted_uv[:, :, step]).clamp(0, 1)
            distance = (xx[None, None] - uv[..., 0, None, None]).square()
            distance = distance + (yy[None, None] - uv[..., 1, None, None]).square()
            gaussian = torch.exp(-distance / (2 * self.sigma ** 2))
            depth = torch.linalg.vector_norm(torch.nan_to_num(predicted_xyz[:, :, step]), dim=-1).clamp_min(1e-3)
            valid_depth = depth.masked_fill(~point_valid, float("nan"))
            median_depth = torch.nanmedian(valid_depth, dim=1).values
            median_depth = torch.nan_to_num(median_depth, nan=1.0).clamp_min(1e-3)
            depth_weight = torch.exp(-depth / (median_depth[:, None] * self.depth_temperature))
            visible = probabilities[:, :, step, 0]
            authority = point_valid.float() * confidence * visible * depth_weight
            weight = gaussian * authority[..., None, None]
            denominator = weight.sum(1, keepdim=True)
            propagated = torch.einsum("bnhw,bnc->bchw", weight, source_color) / denominator.clamp_min(1e-6)
            support = (1 - torch.exp(-denominator)).clamp(0, 1)
            point_state = torch.cat((
                state["camera_flow"][:, :, step],
                state["object_flow"][:, :, step],
                probabilities[:, :, step],
            ), -1)
            dense = torch.einsum("bnhw,bnc->bchw", weight, point_state) / denominator.clamp_min(1e-6)
            outputs.append(propagated)
            supports.append(support)
            dense_states.append(dense)
        return torch.stack(outputs, 1), torch.stack(supports, 1), torch.stack(dense_states, 1)

    def forward(self, base_prediction, last_rgb, predicted_xyz, predicted_uv,
                past_xyz, past_uv, past_valid, past_confidence):
        b, k, _, h, w = base_prediction.shape
        state = self.decompose(
            predicted_xyz, predicted_uv, past_xyz, past_uv, past_valid, past_confidence
        )
        confidence = torch.nan_to_num(past_confidence.float()).amax(2).clamp(0, 1)
        point_valid = past_valid.any(2)
        propagated, support, dense_state = self._softsplat(
            last_rgb, predicted_uv, predicted_xyz, confidence, state, point_valid
        )
        last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
        base_motion = (base_prediction - last).abs().mean(2, keepdim=True)
        disocclusion_prior = (1 - support) * (base_motion / 0.08).clamp(0, 1)
        inputs = torch.cat((
            base_prediction, last, propagated, (propagated - base_prediction).abs(),
            dense_state, support, base_motion, disocclusion_prior,
        ), 2)
        hidden = self.renderer(inputs.reshape(b * k, -1, h, w))
        transport_gate = torch.sigmoid(self.transport_head(hidden)).reshape(b, k, 1, h, w) * support
        synthesis_gate = torch.sigmoid(self.synthesis_gate_head(hidden)).reshape(b, k, 1, h, w)
        synthesis_gate = synthesis_gate * disocclusion_prior * (1 - transport_gate.detach())
        synthesis_residual = self.max_synthesis_residual * torch.tanh(
            self.synthesis_head(hidden)
        ).reshape(b, k, 3, h, w)
        transport_delta = propagated - base_prediction
        prediction = (
            base_prediction + transport_gate * transport_delta + synthesis_gate * synthesis_residual
        ).clamp(0, 1)
        state.update({
            "propagated_rgb": propagated,
            "propagation_support": support,
            "transport_gate": transport_gate,
            "synthesis_gate": synthesis_gate,
            "synthesis_residual": synthesis_residual,
            "transport_delta": transport_delta,
            "disocclusion_prior": disocclusion_prior,
        })
        return prediction, state


def visibility_targets(past_valid, future_valid):
    """Return three-state labels and a padding-safe supervision mask."""
    ever_visible = past_valid.any(2)
    last_visible = past_valid[:, :, -1]
    future_visible = future_valid.bool()
    labels = torch.ones_like(future_visible, dtype=torch.long)  # occluded
    labels[future_visible & last_visible[:, :, None]] = 0       # remains visible
    labels[future_visible & ~last_visible[:, :, None] & ever_visible[:, :, None]] = 2
    mask = ever_visible[:, :, None].expand_as(future_visible)
    return labels, mask
