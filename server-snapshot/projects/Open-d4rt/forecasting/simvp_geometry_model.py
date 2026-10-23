"""SimVP with leakage-safe spatial D4RT geometry injection."""

from __future__ import annotations

import torch
from torch import nn

from openstl.models import SimVP_Model
from video_prediction_model import SpatialGeometryAdapter


class GeometryVideoPredictor(nn.Module):
    """Unified geometry-aware interface for OpenSTL recurrent-free backbones."""
    def __init__(self, context_frames=16, future_frames=8, image_size=64,
                 hid_s=16, hid_t=64, n_s=2, n_t=2, injection_type="add",
                 backbone="simvp", drop_path=0.0):
        super().__init__()
        self.context_frames, self.future_frames = int(context_frames), int(future_frames)
        self.injection_type = injection_type
        self.backbone_name = backbone.lower()
        if self.backbone_name not in ("simvp", "tau"):
            raise ValueError(f"Unsupported video backbone: {backbone}")
        model_type = "gSTA" if self.backbone_name == "simvp" else "tau"
        self.backbone_config = dict(hid_s=int(hid_s), hid_t=int(hid_t), n_s=int(n_s),
                                    n_t=int(n_t), drop_path=float(drop_path))
        self.backbone = SimVP_Model(
            (context_frames, 3, image_size, image_size), hid_S=hid_s, hid_T=hid_t,
            N_S=n_s, N_T=n_t, model_type=model_type, drop_path=drop_path)
        # Project context-indexed latent/skip sequences to K future slots.
        self.latent_time = nn.Conv3d(context_frames, future_frames, 1)
        self.skip_time = nn.Conv3d(context_frames, future_frames, 1)
        self.geometry = SpatialGeometryAdapter(hid_s)
        self.geometry_fuse = nn.Sequential(
            nn.Conv2d(hid_s * 2, hid_s, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid_s, hid_s, 3, padding=1),
        )
        self.latent_film = nn.Conv2d(hid_s, hid_s * 2, 3, padding=1)
        self.skip_film = nn.Conv2d(hid_s, hid_s * 2, 3, padding=1)
        self.latent_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.skip_gate_logit = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _film(feature, geometry, support, layer, gate_logit):
        b,k,c,h,w=feature.shape
        gamma,beta=layer(geometry.reshape(b*k,c,h,w)).chunk(2,1)
        gamma=gamma.reshape(b,k,c,h,w); beta=beta.reshape(b,k,c,h,w)
        gate=torch.sigmoid(gate_logit)
        return feature + support*gate*(feature*torch.tanh(gamma)+beta)

    def forward(self, context_rgb, geometry_7d=None, query_uv=None, point_valid=None):
        b, t, c, h0, w0 = context_rgb.shape
        embed, skip = self.backbone.enc(context_rgb.reshape(b * t, c, h0, w0))
        ch, h, w = embed.shape[1:]
        hidden = self.backbone.hid(embed.reshape(b, t, ch, h, w))
        hidden = self.latent_time(hidden)
        skip = self.skip_time(skip.reshape(b, t, *skip.shape[1:]))
        if geometry_7d is not None:
            geom,support = self.geometry(geometry_7d, query_uv, point_valid, h, w, return_support=True)
            if self.injection_type == "film":
                hidden=self._film(hidden,geom,support,self.latent_film,self.latent_gate_logit)
                hs,ws=skip.shape[-2:]
                geom_skip,support_skip=self.geometry(geometry_7d,query_uv,point_valid,hs,ws,return_support=True)
                skip=self._film(skip,geom_skip,support_skip,self.skip_film,self.skip_gate_logit)
            else:
                flat = torch.cat((hidden, geom), 2).reshape(b * self.future_frames, ch * 2, h, w)
                hidden = hidden + support*self.geometry_fuse(flat).reshape_as(hidden)
        decoded = self.backbone.dec(
            hidden.reshape(b * self.future_frames, ch, h, w),
            skip.reshape(b * self.future_frames, *skip.shape[2:]),
        )
        return torch.sigmoid(decoded.reshape(b, self.future_frames, 3, h0, w0))


class GeometrySimVP(GeometryVideoPredictor):
    """Backward-compatible name for existing SimVP checkpoints."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backbone", "simvp")
        super().__init__(*args, **kwargs)


def build_video_predictor(context_frames, future_frames, image_size, *, backbone="simvp",
                          injection_type="add", hid_s=None, hid_t=None, n_s=2, n_t=None,
                          drop_path=None):
    """Build a backbone with explicit, checkpointable defaults."""
    is_tau = backbone.lower() == "tau"
    return GeometryVideoPredictor(
        context_frames, future_frames, image_size,
        hid_s=(64 if is_tau else 16) if hid_s is None else hid_s,
        hid_t=(256 if is_tau else 64) if hid_t is None else hid_t,
        n_s=n_s,
        n_t=(6 if is_tau else 2) if n_t is None else n_t,
        injection_type=injection_type,
        backbone=backbone,
        drop_path=(0.1 if is_tau else 0.0) if drop_path is None else drop_path,
    )


def build_video_predictor_from_checkpoint(checkpoint):
    cfg = checkpoint.get("backbone_config", {})
    return build_video_predictor(
        checkpoint["context_frames"], checkpoint["future_frames"], checkpoint["image_size"],
        backbone=checkpoint.get("backbone", "simvp"),
        injection_type=checkpoint.get("injection", "add"), **cfg)
