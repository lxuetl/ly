"""CaPSR with a unified spatiotemporal multi-path router.

The rendering order is causal and cannot collapse back to a single total-UV
warp:

1. dominant camera flow globally canonicalizes the last observed RGB;
2. object residual flow locally transports visible appearance in that frame;
3. synthesis is permitted only on explicit disocclusion support.

The same module also exposes an RGB-only capacity-control mode.  It has exactly
the same parameters, receives no geometric state, and may use its synthesis
path globally.  This is a deliberately strong equal-capacity baseline.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from capsr_model import PredictiveVisibilityHead, RobustDepthAffineEgomotion, _last_valid


class StructuredCaPSR(nn.Module):
    renderer_channels = 24

    def __init__(self, sigma: float = 0.035, hidden: int = 48,
                 max_synthesis_residual: float = 0.12,
                 num_motion_clusters: int = 4):
        super().__init__()
        self.sigma = float(sigma)
        self.max_synthesis_residual = float(max_synthesis_residual)
        self.num_motion_clusters = int(num_motion_clusters)
        self.cluster_temperature = 0.18
        self.cycle_radius = 0.09
        self.egomotion = RobustDepthAffineEgomotion()
        self.visibility = PredictiveVisibilityHead(hidden)
        self.renderer = nn.Sequential(
            nn.Conv2d(self.renderer_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.camera_gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.object_gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.synthesis_gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.synthesis_head = nn.Conv2d(hidden, 3, 3, padding=1)
        # Residual logits over Base/Camera/Object/Synthesis.  The temporal 3-D
        # receptive field lets adjacent horizons share routing evidence, while
        # the zero final layer initializes the model at the frozen v3 routing.
        self.multipath_router = nn.Sequential(
            nn.Conv3d(hidden + 4, 24, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv3d(24, 4, kernel_size=3, padding=1),
        )
        for head, bias in (
            (self.camera_gate_head, -2.0),
            (self.object_gate_head, -0.5),
            (self.synthesis_gate_head, -2.0),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, bias)
        nn.init.zeros_(self.synthesis_head.weight)
        nn.init.zeros_(self.synthesis_head.bias)
        nn.init.zeros_(self.multipath_router[-1].weight)
        nn.init.zeros_(self.multipath_router[-1].bias)

    @staticmethod
    def _image_grid(batch, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=device, dtype=dtype),
            torch.linspace(0, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        normalized = torch.stack((xx * 2 - 1, yy * 2 - 1), -1)
        return xx, yy, normalized[None].expand(batch, -1, -1, -1)

    def _dense_camera_flow(self, coefficients, future_frames, height, width):
        b = coefficients.shape[0]
        xx, yy, _ = self._image_grid(
            b, height, width, coefficients.device, coefficients.dtype
        )
        basis = torch.stack((torch.ones_like(xx), xx, yy, torch.ones_like(xx)), -1)
        one_step = torch.einsum("hwc,bcd->bdhw", basis, coefficients)
        steps = torch.arange(
            1, future_frames + 1, device=coefficients.device, dtype=coefficients.dtype
        ).view(1, -1, 1, 1, 1)
        return one_step[:, None] * steps

    def _camera_warp(self, last_rgb, dense_camera_flow):
        b, k, _, h, w = dense_camera_flow.shape
        _, _, base_grid = self._image_grid(b, h, w, last_rgb.device, last_rgb.dtype)
        # Camera flow is forward motion. Backward sampling approximately queries
        # the source at destination minus the forward displacement.
        grid = base_grid[:, None].expand(-1, k, -1, -1, -1).clone()
        grid = grid - 2 * dense_camera_flow.permute(0, 1, 3, 4, 2)
        source = last_rgb[:, None].expand(-1, k, -1, -1, -1).reshape(b * k, 3, h, w)
        warped = F.grid_sample(
            source, grid.reshape(b * k, h, w, 2), mode="bilinear",
            padding_mode="border", align_corners=True,
        ).reshape(b, k, 3, h, w)
        valid = ((grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)).float()[:, :, None]
        return warped, valid, grid

    def _motion_clusters(self, source, flow, valid):
        """Differentiation-free soft k-means over position and residual motion."""
        feature = torch.cat((1.5 * source, flow / 0.04), -1)
        feature = torch.nan_to_num(feature)
        valid_f = valid.float()
        first = (feature * valid_f[..., None]).sum(1) / valid_f.sum(1, keepdim=True).clamp_min(1)
        centers = [first]
        closest = (feature - first[:, None]).square().sum(-1)
        for _ in range(1, self.num_motion_clusters):
            index = closest.masked_fill(~valid, -1).argmax(1)
            center = feature.gather(1, index[:, None, None].expand(-1, 1, feature.shape[-1])).squeeze(1)
            centers.append(center)
            closest = torch.minimum(closest, (feature - center[:, None]).square().sum(-1))
        centers = torch.stack(centers, 1)
        for _ in range(3):
            distance = (feature[:, :, None] - centers[:, None]).square().sum(-1)
            assignment = torch.softmax(-distance / self.cluster_temperature, -1) * valid_f[..., None]
            centers = torch.einsum("bnm,bnd->bmd", assignment, feature)
            centers = centers / assignment.sum(1)[..., None].clamp_min(1e-6)
        return assignment

    def _transport_confidence(self, source, destination, flow, depth,
                              assignment, valid):
        """Independent neighbour round-trip and soft z-buffer confidence (v3)."""
        b, n, _ = source.shape
        eye = torch.eye(n, device=source.device, dtype=torch.bool)[None]
        cluster_similarity = torch.einsum("bnm,bjm->bnj", assignment, assignment)
        valid_pair = valid[:, :, None] & valid[:, None, :] & ~eye
        source_distance = (source[:, :, None] - source[:, None, :]).square().sum(-1)
        destination_distance = (
            destination[:, :, None] - destination[:, None, :]
        ).square().sum(-1)
        source_weight = torch.exp(
            -source_distance / (2 * self.cycle_radius ** 2)
        ) * cluster_similarity * valid_pair.float()
        destination_weight = torch.exp(
            -destination_distance / (2 * self.cycle_radius ** 2)
        ) * cluster_similarity * valid_pair.float()
        forward_neighbour = torch.einsum("bij,bjc->bic", source_weight, flow)
        forward_neighbour = forward_neighbour / source_weight.sum(-1, keepdim=True).clamp_min(1e-6)
        backward_neighbour = torch.einsum("bij,bjc->bic", destination_weight, -flow)
        backward_neighbour = backward_neighbour / destination_weight.sum(-1, keepdim=True).clamp_min(1e-6)
        cycle_error = torch.linalg.vector_norm(flow - forward_neighbour, dim=-1)
        cycle_error += torch.linalg.vector_norm(flow + backward_neighbour, dim=-1)
        has_neighbour = (source_weight.sum(-1) > 1e-4) & (destination_weight.sum(-1) > 1e-4)
        cycle = torch.exp(-cycle_error / 0.025)
        cycle = torch.where(has_neighbour, cycle, torch.full_like(cycle, 0.75))

        depth = torch.nan_to_num(depth).abs()
        farther = torch.sigmoid((depth[:, :, None] - depth[:, None, :]) / 0.04)
        collision = torch.exp(-destination_distance / (2 * 0.025 ** 2))
        collision = collision * cluster_similarity * valid_pair.float()
        occlusion_score = (collision * farther).sum(-1) / collision.sum(-1).clamp_min(1e-6)
        non_occluded = torch.exp(-2.0 * occlusion_score)
        return cycle * valid.float(), non_occluded * valid.float()

    def _object_transport(self, camera_rgb, camera_uv, future_uv, predicted_xyz,
                          confidence, probabilities, point_valid, object_flow):
        b, n, k, _ = future_uv.shape
        h, w = camera_rgb.shape[-2:]
        xx, yy, _ = self._image_grid(b, h, w, camera_rgb.device, camera_rgb.dtype)
        propagated, destination_support, source_support = [], [], []
        raw_destination_support = []
        dense_object, dense_visibility = [], []
        reappear_support, occlusion_support = [], []
        dense_cycle_confidence, dense_occlusion_confidence = [], []
        cluster_balance = []
        for step in range(k):
            source_grid = camera_uv[:, :, step].mul(2).sub(1).view(b, n, 1, 2)
            source_color = F.grid_sample(
                camera_rgb[:, step], source_grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            ).squeeze(-1).permute(0, 2, 1)
            destination = torch.nan_to_num(future_uv[:, :, step]).clamp(0, 1)
            canonical_source = torch.nan_to_num(camera_uv[:, :, step]).clamp(0, 1)
            dest_distance = (xx[None, None] - destination[..., 0, None, None]).square()
            dest_distance += (yy[None, None] - destination[..., 1, None, None]).square()
            source_distance = (xx[None, None] - canonical_source[..., 0, None, None]).square()
            source_distance += (yy[None, None] - canonical_source[..., 1, None, None]).square()
            destination_gaussian = torch.exp(-dest_distance / (2 * self.sigma ** 2))
            source_gaussian = torch.exp(-source_distance / (2 * self.sigma ** 2))
            object_magnitude = torch.linalg.vector_norm(object_flow[:, :, step], dim=-1)
            object_authority = torch.sigmoid((object_magnitude - 0.004) / 0.003)
            visible = probabilities[:, :, step, 0]
            assignment = self._motion_clusters(canonical_source, object_flow[:, :, step], point_valid)
            cycle, non_occluded = self._transport_confidence(
                canonical_source, destination, object_flow[:, :, step],
                predicted_xyz[:, :, step, 2], assignment, point_valid,
            )
            authority = (
                point_valid.float() * confidence * visible * object_authority
                * cycle * non_occluded
            )
            cluster_colors, cluster_flows, cluster_visibilities, cluster_denominators = [], [], [], []
            for cluster in range(self.num_motion_clusters):
                cluster_authority = authority * assignment[:, :, cluster]
                cluster_weight = destination_gaussian * cluster_authority[..., None, None]
                cluster_denominator = cluster_weight.sum(1, keepdim=True)
                cluster_colors.append(
                    torch.einsum("bnhw,bnc->bchw", cluster_weight, source_color)
                    / cluster_denominator.clamp_min(1e-6)
                )
                cluster_flows.append(
                    torch.einsum("bnhw,bnc->bchw", cluster_weight, object_flow[:, :, step])
                    / cluster_denominator.clamp_min(1e-6)
                )
                cluster_visibilities.append(
                    torch.einsum("bnhw,bnc->bchw", cluster_weight, probabilities[:, :, step])
                    / cluster_denominator.clamp_min(1e-6)
                )
                cluster_denominators.append(cluster_denominator)
            cluster_denominator = torch.stack(cluster_denominators, 1)
            cluster_support = 1 - torch.exp(-cluster_denominator)
            cluster_priority = torch.softmax(cluster_denominator / 0.35, 1) * cluster_support
            priority_sum = cluster_priority.sum(1).clamp_min(1e-6)
            color = (
                torch.stack(cluster_colors, 1) * cluster_priority
            ).sum(1) / priority_sum
            dense_flow = (
                torch.stack(cluster_flows, 1) * cluster_priority
            ).sum(1) / priority_sum
            dense_probabilities = (
                torch.stack(cluster_visibilities, 1) * cluster_priority
            ).sum(1) / priority_sum
            denominator = cluster_denominator.sum(1)
            destination_mask = (1 - torch.exp(-denominator)).clamp(0, 1)
            source_weight = source_gaussian * authority[..., None, None]
            source_mask = (1 - torch.exp(-source_weight.sum(1, keepdim=True))).clamp(0, 1)
            combined_weight = destination_gaussian * authority[..., None, None]
            dense_cycle = torch.einsum("bnhw,bn->bhw", combined_weight, cycle)[:, None]
            dense_cycle = dense_cycle / denominator.clamp_min(1e-6)
            dense_non_occluded = torch.einsum(
                "bnhw,bn->bhw", combined_weight, non_occluded
            )[:, None] / denominator.clamp_min(1e-6)
            reliable_mask = destination_mask * dense_cycle * dense_non_occluded
            reappear_weight = destination_gaussian * (
                point_valid.float() * confidence * probabilities[:, :, step, 2]
            )[..., None, None]
            occlusion_weight = source_gaussian * (
                point_valid.float() * confidence * probabilities[:, :, step, 1]
            )[..., None, None]
            propagated.append(color)
            destination_support.append(reliable_mask)
            raw_destination_support.append(destination_mask)
            source_support.append(source_mask)
            dense_object.append(dense_flow)
            dense_visibility.append(dense_probabilities)
            reappear_support.append((1 - torch.exp(-reappear_weight.sum(1, keepdim=True))).clamp(0, 1))
            occlusion_support.append((1 - torch.exp(-occlusion_weight.sum(1, keepdim=True))).clamp(0, 1))
            dense_cycle_confidence.append(dense_cycle)
            dense_occlusion_confidence.append(dense_non_occluded)
            cluster_balance.append(
                -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(-1).mean(1)
            )
        return tuple(torch.stack(values, 1) for values in (
            propagated, destination_support, source_support, dense_object,
            dense_visibility, reappear_support, occlusion_support,
            dense_cycle_confidence, dense_occlusion_confidence,
            raw_destination_support,
        )) + (torch.stack(cluster_balance, 1),)

    @staticmethod
    def _rgb_statistics(base_prediction, last_rgb):
        b, k, _, _, _ = base_prediction.shape
        last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
        base_mean = base_prediction.mean((-2, -1))
        last_mean = last.mean((-2, -1))
        difference_mean = (base_prediction - last).abs().mean((-2, -1))
        motion = (base_prediction - last).abs().mean((2, 3, 4), keepdim=False)[..., None]
        horizon = torch.linspace(0, 1, k, device=base_prediction.device, dtype=base_prediction.dtype)
        horizon = horizon.view(1, k, 1).expand(b, -1, -1)
        constant = torch.ones_like(horizon)
        return torch.cat((base_mean, last_mean, difference_mean, motion, horizon, constant), -1)

    def _render(self, base_prediction, last_rgb, camera_rgb, object_rgb,
                dense_camera_flow, dense_object_flow, dense_visibility,
                camera_valid, object_support, disocclusion, mode,
                cycle_confidence=None, occlusion_confidence=None):
        b, k, _, h, w = base_prediction.shape
        last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
        camera_disagreement = (camera_rgb - base_prediction).abs().mean(2, keepdim=True)
        object_disagreement = (object_rgb - base_prediction).abs().mean(2, keepdim=True)
        inputs = torch.cat((
            base_prediction, last, camera_rgb, object_rgb,
            camera_disagreement, object_disagreement,
            dense_camera_flow, dense_object_flow, dense_visibility,
            camera_valid, object_support, disocclusion,
        ), 2)
        hidden = self.renderer(inputs.reshape(b * k, -1, h, w))
        if mode == "structured":
            object_magnitude = torch.linalg.vector_norm(dense_object_flow, dim=2, keepdim=True)
            # A hard geometric floor strictly prevents the residual path from
            # changing points whose canonical object motion is effectively static.
            moving = (object_magnitude > 0.006).to(object_magnitude.dtype)
            motion_ramp = torch.sigmoid((object_magnitude - 0.008) / 0.0025)
            object_permission = object_support * moving * motion_ramp * (1 - disocclusion)
            camera_permission = camera_valid * (1 - object_permission) * (1 - disocclusion)
            camera_weight = camera_permission * torch.sigmoid(
                self.camera_gate_head(hidden)
            ).reshape(b, k, 1, h, w)
            object_gate = torch.sigmoid(
                self.object_gate_head(hidden).reshape(b, k, 1, h, w) + 0.75
            )
            object_weight = object_permission * (1 - camera_weight) * object_gate
            synthesis_weight = disocclusion * (1 - camera_weight - object_weight).clamp_min(0) * torch.sigmoid(
                self.synthesis_gate_head(hidden)
            ).reshape(b, k, 1, h, w)
        elif mode == "no_geometry":
            camera_weight = base_prediction.new_zeros(b, k, 1, h, w)
            object_weight = base_prediction.new_zeros(b, k, 1, h, w)
            object_permission = base_prediction.new_zeros(b, k, 1, h, w)
            object_gate = base_prediction.new_zeros(b, k, 1, h, w)
            # The visibility MLP is reused as an RGB gate-bias network so the
            # equal-parameter control does not contain dead trainable modules.
            rgb_logits = self.visibility.net(self._rgb_statistics(base_prediction, last_rgb))
            rgb_bias = rgb_logits[:, :, 0].view(b, k, 1, 1, 1)
            synthesis_weight = torch.sigmoid(
                self.synthesis_gate_head(hidden).reshape(b, k, 1, h, w) + rgb_bias
            )
        else:
            raise ValueError(mode)
        synthesis_residual = self.max_synthesis_residual * torch.tanh(
            self.synthesis_head(hidden)
        ).reshape(b, k, 3, h, w)
        synthesis_candidate = (base_prediction + synthesis_residual).clamp(0, 1)
        remaining = (1 - camera_weight - object_weight - synthesis_weight).clamp_min(0)
        prior_weights = torch.cat((
            remaining, camera_weight, object_weight, synthesis_weight,
        ), 2).clamp_min(0)
        candidate_stack = torch.stack((
            base_prediction, camera_rgb, object_rgb, synthesis_candidate,
        ), 2)
        if mode == "structured":
            horizon = torch.linspace(
                0, 1, k, device=base_prediction.device, dtype=base_prediction.dtype
            )
            horizon = torch.stack((
                horizon, horizon.square(), torch.sin(torch.pi * horizon),
                torch.cos(torch.pi * horizon),
            ), 0)[None, :, :, None, None].expand(b, -1, -1, h, w)
            hidden_3d = hidden.reshape(b, k, -1, h, w).permute(0, 2, 1, 3, 4)
            residual_logits = self.multipath_router(
                torch.cat((hidden_3d, horizon), 1)
            ).permute(0, 2, 1, 3, 4)
            router_logits = prior_weights.clamp_min(1e-6).log() + residual_logits
            availability = torch.cat((
                torch.ones_like(remaining), camera_permission,
                object_permission, disocclusion,
            ), 2) > 1e-6
            router_logits = router_logits.masked_fill(~availability, -30.0)
            router_probabilities = router_logits.softmax(2)
            prediction = (
                router_probabilities[:, :, :, None] * candidate_stack
            ).sum(2).clamp(0, 1)
            remaining = router_probabilities[:, :, 0:1]
            camera_weight = router_probabilities[:, :, 1:2]
            object_weight = router_probabilities[:, :, 2:3]
            synthesis_weight = router_probabilities[:, :, 3:4]
        else:
            router_logits = prior_weights.clamp_min(1e-6).log()
            availability = prior_weights > 1e-6
            router_probabilities = prior_weights / prior_weights.sum(2, keepdim=True).clamp_min(1e-6)
            prediction = (
                remaining * base_prediction + camera_weight * camera_rgb
                + object_weight * object_rgb + synthesis_weight * synthesis_candidate
            ).clamp(0, 1)
        return prediction, {
            "camera_weight": camera_weight,
            "object_weight": object_weight,
            "synthesis_weight": synthesis_weight,
            "synthesis_residual": synthesis_residual,
            "disocclusion": disocclusion,
            "camera_valid": camera_valid,
            "object_support": object_support,
            "dynamic_object_permission": object_permission,
            "object_gate": object_gate,
            "router_logits": router_logits,
            "router_probabilities": router_probabilities,
            "prior_weights": prior_weights,
            "candidate_stack": candidate_stack,
            "path_availability": availability,
            "synthesis_candidate": synthesis_candidate,
            "cycle_confidence": cycle_confidence if cycle_confidence is not None else object_support,
            "occlusion_confidence": occlusion_confidence if occlusion_confidence is not None else object_support,
        }

    def forward(self, base_prediction, last_rgb, predicted_xyz, predicted_uv,
                past_xyz, past_uv, past_valid, past_confidence,
                mode: str = "structured"):
        b, k, _, h, w = base_prediction.shape
        if mode == "no_geometry":
            zeros2 = base_prediction.new_zeros(b, k, 2, h, w)
            zeros3 = base_prediction.new_zeros(b, k, 3, h, w)
            zeros1 = base_prediction.new_zeros(b, k, 1, h, w)
            ones1 = torch.ones_like(zeros1)
            last = last_rgb[:, None].expand(-1, k, -1, -1, -1)
            return self._render(
                base_prediction, last_rgb, last, last, zeros2, zeros2, zeros3,
                zeros1, zeros1, ones1, mode,
            )

        anchor_uv = _last_valid(past_uv, past_valid)
        camera_point_flow, coefficients = self.egomotion(
            past_uv, past_xyz, past_valid, past_confidence, k
        )
        total_flow = predicted_uv - anchor_uv[:, :, None]
        object_flow = total_flow - camera_point_flow
        camera_uv = (anchor_uv[:, :, None] + camera_point_flow).clamp(0, 1)
        visibility_logits = self.visibility(
            camera_point_flow, object_flow, total_flow, predicted_xyz,
            past_valid, past_confidence,
        )
        probabilities = visibility_logits.softmax(-1)
        dense_camera_flow = self._dense_camera_flow(coefficients, k, h, w)
        camera_rgb, camera_valid, _ = self._camera_warp(last_rgb, dense_camera_flow)
        confidence = torch.nan_to_num(past_confidence.float()).amax(2).clamp(0, 1)
        point_valid = past_valid.any(2)
        (object_rgb, object_support, source_support, dense_object_flow,
         dense_visibility, reappear_support, occlusion_support,
         dense_cycle_confidence, dense_occlusion_confidence,
         raw_object_support, cluster_entropy) = self._object_transport(
            camera_rgb, camera_uv, predicted_uv, predicted_xyz, confidence,
            probabilities, point_valid, object_flow,
        )
        camera_hole = 1 - camera_valid
        vacated = (source_support - object_support).clamp_min(0)
        disocclusion = (
            camera_hole + vacated + reappear_support + occlusion_support
        ).clamp(0, 1)
        disocclusion = F.max_pool3d(
            disocclusion.permute(0, 2, 1, 3, 4), kernel_size=(1, 3, 3),
            stride=1, padding=(0, 1, 1),
        ).permute(0, 2, 1, 3, 4)
        prediction, state = self._render(
            base_prediction, last_rgb, camera_rgb, object_rgb,
            dense_camera_flow, dense_object_flow, dense_visibility,
            camera_valid, object_support, disocclusion, mode,
            dense_cycle_confidence, dense_occlusion_confidence,
        )
        state.update({
            "visibility_logits": visibility_logits,
            "visibility_probabilities": probabilities,
            "camera_point_flow": camera_point_flow,
            "object_point_flow": object_flow,
            "total_flow": total_flow,
            "camera_rgb": camera_rgb,
            "object_rgb": object_rgb,
            "raw_object_support": raw_object_support,
            "cluster_entropy": cluster_entropy,
        })
        return prediction, state
