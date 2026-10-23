"""Geometry-conditioned lightweight video-prediction baseline."""

from __future__ import annotations

import torch
from torch import nn


class SpatialGeometryAdapter(nn.Module):
    """Rasterize point-wise 7D future geometry onto latent feature maps."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(7, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, geometry_7d, query_uv, point_valid, height: int, width: int,
                return_support: bool = False):
        b, n, k, _ = geometry_7d.shape
        tokens = self.proj(geometry_7d)
        uv = query_uv[:, :, None].expand(-1, -1, k, -1) if query_uv.ndim == 3 else query_uv
        xf = uv[..., 0].clamp(0, 1) * (width - 1)
        yf = uv[..., 1].clamp(0, 1) * (height - 1)
        x0, y0 = xf.floor().long(), yf.floor().long()
        x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
        dx, dy = xf - x0.float(), yf - y0.float()
        maps = geometry_7d.new_zeros((b, k, height * width, tokens.shape[-1]))
        counts = geometry_7d.new_zeros((b, k, height * width, 1))
        src = tokens.permute(0, 2, 1, 3)
        if point_valid.ndim == 2:
            reliability = point_valid[:, :, None].expand(-1, -1, k).float()
        elif point_valid.ndim == 3:
            reliability = point_valid.float()
        else:
            raise ValueError(f"point_valid/reliability must be [B,N] or [B,N,K], got {point_valid.shape}")
        active = reliability.permute(0, 2, 1)[..., None]
        # Bilinear four-neighbour splatting avoids jumps caused by rounding UV.
        for xx, yy, weight in (
            (x0, y0, (1.0 - dx) * (1.0 - dy)),
            (x1, y0, dx * (1.0 - dy)),
            (x0, y1, (1.0 - dx) * dy),
            (x1, y1, dx * dy),
        ):
            wgt = weight.permute(0, 2, 1)[..., None] * active
            index = (yy * width + xx).permute(0, 2, 1)[..., None]
            maps.scatter_add_(2, index.expand_as(src), src * wgt)
            counts.scatter_add_(2, index, wgt)
        maps = maps / counts.clamp_min(1.0)
        maps = maps.permute(0, 1, 3, 2).reshape(b, k, -1, height, width)
        support = counts.clamp(0, 1).permute(0, 1, 3, 2).reshape(b, k, 1, height, width)
        return (maps, support) if return_support else maps


class GeometryConditionedVideoPredictor(nn.Module):
    """Encode context frames once, then decode K frames with spatial geometry."""
    def __init__(self, context_frames=16, future_frames=8, latent_dim=96):
        super().__init__()
        self.context_frames, self.future_frames = context_frames, future_frames
        self.encoder = nn.Sequential(
            nn.Conv2d(context_frames * 3, 48, 5, 2, 2), nn.GELU(),
            nn.Conv2d(48, latent_dim, 3, 2, 1), nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 2, 1), nn.GELU(),
        )
        self.geometry = SpatialGeometryAdapter(latent_dim)
        self.time_embed = nn.Embedding(future_frames, latent_dim)
        self.fuse = nn.Sequential(nn.Conv2d(latent_dim * 3, latent_dim, 3, 1, 1), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, context_rgb, geometry_7d, query_uv, point_valid, use_geometry=True):
        b, c, _, h0, w0 = context_rgb.shape
        visual = self.encoder(context_rgb.reshape(b, c * 3, h0, w0))
        _, d, h, w = visual.shape
        if use_geometry:
            geom = self.geometry(geometry_7d, query_uv, point_valid, h, w)
        else:
            geom = visual.new_zeros((b, self.future_frames, d, h, w))
        time = self.time_embed.weight[None, :, :, None, None].expand(b, -1, -1, h, w)
        visual = visual[:, None].expand(-1, self.future_frames, -1, -1, -1)
        fused = self.fuse(torch.cat((visual, geom, time), 2).reshape(b * self.future_frames, d * 3, h, w))
        return self.decoder(fused).reshape(b, self.future_frames, 3, h0, w0)
