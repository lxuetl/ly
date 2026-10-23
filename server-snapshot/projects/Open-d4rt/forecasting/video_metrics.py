"""Shared full-reference metrics for video prediction evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


class VideoMetricAccumulator:
    """Accumulate pixel, perceptual, temporal, and per-horizon metrics."""

    def __init__(self, device: torch.device, future_frames: int, use_lpips: bool = True):
        self.k = int(future_frames)
        self.abs_sum = self.sq_sum = self.count = 0.0
        self.ssim_sum = self.ssim_count = 0.0
        self.lpips_sum = self.lpips_count = 0.0
        self.temporal_abs_sum = self.temporal_sq_sum = self.temporal_count = 0.0
        self.h_abs = np.zeros(self.k, dtype=np.float64)
        self.h_sq = np.zeros(self.k, dtype=np.float64)
        self.h_count = np.zeros(self.k, dtype=np.float64)
        self.h_ssim = np.zeros(self.k, dtype=np.float64)
        self.h_ssim_count = np.zeros(self.k, dtype=np.float64)
        self.h_lpips = np.zeros(self.k, dtype=np.float64)
        self.h_lpips_count = np.zeros(self.k, dtype=np.float64)
        self.lpips_model = None
        if use_lpips:
            import lpips
            self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            for parameter in self.lpips_model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, last_context: torch.Tensor):
        """Inputs are [B,K,3,H,W] in [0,1]; last_context is [B,3,H,W]."""
        diff = prediction - target
        self.abs_sum += diff.abs().sum().item()
        self.sq_sum += diff.square().sum().item()
        self.count += target.numel()

        previous_prediction = torch.cat((last_context[:, None], prediction[:, :-1]), dim=1)
        previous_target = torch.cat((last_context[:, None], target[:, :-1]), dim=1)
        temporal_error = (prediction - previous_prediction) - (target - previous_target)
        self.temporal_abs_sum += temporal_error.abs().sum().item()
        self.temporal_sq_sum += temporal_error.square().sum().item()
        self.temporal_count += temporal_error.numel()

        pn = prediction.detach().cpu().numpy()
        tn = target.detach().cpu().numpy()
        for step in range(self.k):
            step_diff = diff[:, step]
            self.h_abs[step] += step_diff.abs().sum().item()
            self.h_sq[step] += step_diff.square().sum().item()
            self.h_count[step] += target[:, step].numel()
            for batch_index in range(pn.shape[0]):
                value = structural_similarity(
                    tn[batch_index, step].transpose(1, 2, 0),
                    pn[batch_index, step].transpose(1, 2, 0),
                    channel_axis=-1,
                    data_range=1.0,
                )
                self.ssim_sum += value
                self.ssim_count += 1
                self.h_ssim[step] += value
                self.h_ssim_count[step] += 1

        if self.lpips_model is not None:
            batch, steps, channels, height, width = prediction.shape
            pred_flat = prediction.reshape(batch * steps, channels, height, width) * 2 - 1
            target_flat = target.reshape(batch * steps, channels, height, width) * 2 - 1
            values = self.lpips_model(pred_flat, target_flat).reshape(batch, steps)
            self.lpips_sum += values.sum().item()
            self.lpips_count += values.numel()
            step_values = values.sum(0).detach().cpu().numpy()
            self.h_lpips += step_values
            self.h_lpips_count += batch

    def compute(self) -> dict:
        mse = self.sq_sum / max(self.count, 1)
        temporal_mse = self.temporal_sq_sum / max(self.temporal_count, 1)
        horizons = {}
        for step in range(self.k):
            step_mse = self.h_sq[step] / max(self.h_count[step], 1)
            horizons[f"t+{step + 1}"] = {
                "MAE": float(self.h_abs[step] / max(self.h_count[step], 1)),
                "MSE": float(step_mse),
                "PSNR": float(-10 * math.log10(max(step_mse, 1e-12))),
                "SSIM": float(self.h_ssim[step] / max(self.h_ssim_count[step], 1)),
                "LPIPS": float(self.h_lpips[step] / max(self.h_lpips_count[step], 1)),
            }
        return {
            "MAE": float(self.abs_sum / max(self.count, 1)),
            "MSE": float(mse),
            "RMSE": float(math.sqrt(mse)),
            "PSNR": float(-10 * math.log10(max(mse, 1e-12))),
            "SSIM": float(self.ssim_sum / max(self.ssim_count, 1)),
            "LPIPS": float(self.lpips_sum / max(self.lpips_count, 1)),
            "temporal_delta_MAE": float(self.temporal_abs_sum / max(self.temporal_count, 1)),
            "temporal_delta_MSE": float(temporal_mse),
            "per_horizon": horizons,
        }
