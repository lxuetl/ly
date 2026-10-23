"""Lightweight future-trajectory model and video-conditioning adapter."""

from __future__ import annotations

import torch
from torch import nn


def last_valid_xyz(xyz: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return [B,N,3] last valid point, or zero when a track has no valid point."""
    t = xyz.shape[2]
    idx = (valid.long() * torch.arange(1, t + 1, device=xyz.device)).argmax(2)
    out = xyz.gather(2, idx[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    return torch.where(valid.any(2)[..., None], out, torch.zeros_like(out))


class LiteTrajectoryForecaster(nn.Module):
    def __init__(self, future_frames: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.future_frames = int(future_frames)
        # relative xyz(3), uv(2), valid(1), confidence(1), normalized time(1)
        self.temporal = nn.GRU(8, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.future_frames * 3),
        )

    def forward(self, past_xyz, past_uv, past_valid, past_confidence):
        b, n, t, _ = past_xyz.shape
        anchor = last_valid_xyz(past_xyz, past_valid)
        rel = torch.nan_to_num(past_xyz - anchor[:, :, None], nan=0.0)
        uv = torch.nan_to_num(past_uv, nan=0.0)
        valid = past_valid.float()[..., None]
        conf = torch.nan_to_num(past_confidence, nan=0.0)[..., None] * valid
        time = torch.linspace(0, 1, t, device=past_xyz.device).view(1, 1, t, 1).expand(b, n, -1, -1)
        feat = torch.cat((rel, uv, valid, conf, time), dim=-1).reshape(b * n, t, 8)
        _, h = self.temporal(feat)
        point_h = h[-1].reshape(b, n, -1)
        track_ok = past_valid.any(2).float()[..., None]
        scene_h = (point_h * track_ok).sum(1) / track_ok.sum(1).clamp_min(1.0)
        scene_h = scene_h[:, None].expand(-1, n, -1)
        delta = self.head(torch.cat((point_h, scene_h), -1)).reshape(b, n, self.future_frames, 3)
        return anchor[:, :, None] + delta


class JointTrajectoryForecaster(nn.Module):
    """Shared temporal encoder with consistent future XYZ and image-plane UV heads."""
    def __init__(self, future_frames: int = 8, hidden_dim: int = 128):
        super().__init__(); self.future_frames = int(future_frames)
        self.temporal = nn.GRU(8, hidden_dim, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.xyz_head = nn.Linear(hidden_dim, self.future_frames * 3)
        self.uv_head = nn.Linear(hidden_dim, self.future_frames * 2)

    def forward(self, past_xyz, past_uv, past_valid, past_confidence):
        b,n,t,_=past_xyz.shape
        anchor_xyz=last_valid_xyz(past_xyz,past_valid)
        idx=(past_valid.long()*torch.arange(1,t+1,device=past_valid.device)).argmax(2)
        anchor_uv=torch.nan_to_num(past_uv).gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        rel=torch.nan_to_num(past_xyz-anchor_xyz[:,:,None],nan=0.0)
        uv=torch.nan_to_num(past_uv,nan=0.0); valid=past_valid.float()[...,None]
        conf=torch.nan_to_num(past_confidence,nan=0.0)[...,None]*valid
        time=torch.linspace(0,1,t,device=past_xyz.device).view(1,1,t,1).expand(b,n,-1,-1)
        feat=torch.cat((rel,uv,valid,conf,time),-1).reshape(b*n,t,8)
        _,h=self.temporal(feat); point=h[-1].reshape(b,n,-1)
        ok=past_valid.any(2).float()[...,None]; scene=(point*ok).sum(1)/ok.sum(1).clamp_min(1)
        fused=self.trunk(torch.cat((point,scene[:,None].expand(-1,n,-1)),-1))
        xyz=anchor_xyz[:,:,None]+self.xyz_head(fused).reshape(b,n,self.future_frames,3)
        # Residual image-plane motion, bounded to 25% of image size per horizon.
        uv=(anchor_uv[:,:,None]+0.25*torch.tanh(self.uv_head(fused).reshape(b,n,self.future_frames,2))).clamp(0,1)
        return xyz,uv


class ImplicitConditionedJointForecaster(JointTrajectoryForecaster):
    """Joint forecaster conditioned only on statistics observable at inference."""
    stats_dim = 21

    def __init__(self, future_frames: int = 8, hidden_dim: int = 128):
        super().__init__(future_frames, hidden_dim)
        self.uv_trunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.uv_trunk.load_state_dict(self.trunk.state_dict())
        self.condition = nn.Sequential(nn.Linear(self.stats_dim, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, hidden_dim * 2))
        nn.init.zeros_(self.condition[-1].weight); nn.init.zeros_(self.condition[-1].bias)

    @staticmethod
    def _moments(x, weight, dims):
        while weight.ndim < x.ndim: weight = weight[..., None]
        denom = weight.sum(dims).clamp_min(1.0)
        mean = (x * weight).sum(dims) / denom
        view = mean
        for dim in sorted(dims): view = view.unsqueeze(dim)
        var = ((x - view).square() * weight).sum(dims) / denom
        return mean, torch.sqrt(var + 1e-6)

    def scene_statistics(self, xyz, uv, valid, confidence):
        v=valid.float(); conf=torch.nan_to_num(confidence)*v
        conf_m,conf_s=self._moments(conf,v,(1,2))
        anchor=last_valid_xyz(xyz,valid); ok=valid.any(2).float(); norm=torch.linalg.vector_norm(anchor,dim=-1)
        norm_m,norm_s=self._moments(torch.log1p(norm),ok,(1,)); z=torch.sign(anchor[...,2])*torch.log1p(anchor[...,2].abs());z_m,z_s=self._moments(z,ok,(1,))
        pair=valid[:,:,1:]&valid[:,:,:-1]; uvv=torch.nan_to_num(uv[:,:,1:]-uv[:,:,:-1]); uv_m,uv_s=self._moments(uvv,pair.float(),(1,2))
        xyzv=torch.nan_to_num(xyz[:,:,1:]-xyz[:,:,:-1]); xyzv=torch.sign(xyzv)*torch.log1p(xyzv.abs()); xv_m,_=self._moments(xyzv,pair.float(),(1,2)); speed=torch.linalg.vector_norm(xyzv,dim=-1);sp_m,sp_s=self._moments(speed,pair.float(),(1,2))
        pair2=pair[:,:,1:]&pair[:,:,:-1]; uva=uvv[:,:,1:]-uvv[:,:,:-1]; ua_m,_=self._moments(uva,pair2.float(),(1,2));uan=torch.linalg.vector_norm(uva,dim=-1);_,ua_s=self._moments(uan,pair2.float(),(1,2))
        xyza=xyzv[:,:,1:]-xyzv[:,:,:-1]; xan=torch.linalg.vector_norm(xyza,dim=-1);xa_m,xa_s=self._moments(xan,pair2.float(),(1,2))
        return torch.cat((v.mean((1,2))[:,None],conf_m[:,None],conf_s[:,None],norm_m[:,None],norm_s[:,None],z_m[:,None],z_s[:,None],uv_m,uv_s,xv_m,sp_m[:,None],sp_s[:,None],ua_m,ua_s[:,None],xa_m[:,None],xa_s[:,None]),1)

    def forward(self,past_xyz,past_uv,past_valid,past_confidence):
        b,n,t,_=past_xyz.shape;anchor_xyz=last_valid_xyz(past_xyz,past_valid);idx=(past_valid.long()*torch.arange(1,t+1,device=past_valid.device)).argmax(2);anchor_uv=torch.nan_to_num(past_uv).gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        rel=torch.nan_to_num(past_xyz-anchor_xyz[:,:,None]);uv=torch.nan_to_num(past_uv);valid=past_valid.float()[...,None];conf=torch.nan_to_num(past_confidence)[...,None]*valid;time=torch.linspace(0,1,t,device=past_xyz.device).view(1,1,t,1).expand(b,n,-1,-1);feat=torch.cat((rel,uv,valid,conf,time),-1).reshape(b*n,t,8)
        _,h=self.temporal(feat);point=h[-1].reshape(b,n,-1);ok=past_valid.any(2).float()[...,None];scene=(point*ok).sum(1)/ok.sum(1).clamp_min(1)
        xyz_fused=self.trunk(torch.cat((point,scene[:,None].expand(-1,n,-1)),-1))
        # XYZ and UV share only the temporal encoder. The UV-specific branch sees
        # implicit scene conditions before its own point/scene feature fusion.
        gamma,beta=self.condition(self.scene_statistics(past_xyz,past_uv,past_valid,past_confidence)).chunk(2,-1)
        uv_point=point*(1+torch.tanh(gamma)[:,None])+beta[:,None]
        uv_scene=(uv_point*ok).sum(1)/ok.sum(1).clamp_min(1)
        uv_fused=self.uv_trunk(torch.cat((uv_point,uv_scene[:,None].expand(-1,n,-1)),-1))
        xyz=anchor_xyz[:,:,None]+self.xyz_head(xyz_fused).reshape(b,n,self.future_frames,3)
        uv=(anchor_uv[:,:,None]+.25*torch.tanh(self.uv_head(uv_fused).reshape(b,n,self.future_frames,2))).clamp(0,1)
        return xyz,uv


class VelocityResidualImplicitForecaster(ImplicitConditionedJointForecaster):
    """Predict UV as a bounded residual over robust historical velocity."""

    uses_velocity_residual = True

    def __init__(self, future_frames: int = 8, hidden_dim: int = 128,
                 residual_scale: float = 0.08):
        super().__init__(future_frames, hidden_dim)
        self.residual_scale = float(residual_scale)
        self.uv_residual_head = nn.Linear(hidden_dim, self.future_frames * 2)
        # The warm-started model initially behaves exactly like the robust
        # constant-velocity baseline and only learns corrections supported by data.
        nn.init.zeros_(self.uv_residual_head.weight)
        nn.init.zeros_(self.uv_residual_head.bias)

    @staticmethod
    def robust_uv_velocity(past_uv, past_valid, recent_pairs=4):
        uv = torch.nan_to_num(past_uv.float())
        pair = past_valid.bool()[:, :, 1:] & past_valid.bool()[:, :, :-1]
        if pair.shape[2] > recent_pairs:
            pair[:, :, :-recent_pairs] = False
        delta = uv[:, :, 1:] - uv[:, :, :-1]
        masked = delta.masked_fill(~pair[..., None], float("nan"))
        velocity = torch.nanmedian(masked, dim=2).values
        return torch.nan_to_num(velocity).clamp(-0.08, 0.08)

    def forward(self,past_xyz,past_uv,past_valid,past_confidence):
        b,n,t,_=past_xyz.shape;anchor_xyz=last_valid_xyz(past_xyz,past_valid);idx=(past_valid.long()*torch.arange(1,t+1,device=past_valid.device)).argmax(2);anchor_uv=torch.nan_to_num(past_uv).gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        rel=torch.nan_to_num(past_xyz-anchor_xyz[:,:,None]);uv=torch.nan_to_num(past_uv);valid=past_valid.float()[...,None];conf=torch.nan_to_num(past_confidence)[...,None]*valid;time=torch.linspace(0,1,t,device=past_xyz.device).view(1,1,t,1).expand(b,n,-1,-1);feat=torch.cat((rel,uv,valid,conf,time),-1).reshape(b*n,t,8)
        _,h=self.temporal(feat);point=h[-1].reshape(b,n,-1);ok=past_valid.any(2).float()[...,None];scene=(point*ok).sum(1)/ok.sum(1).clamp_min(1)
        xyz_fused=self.trunk(torch.cat((point,scene[:,None].expand(-1,n,-1)),-1))
        gamma,beta=self.condition(self.scene_statistics(past_xyz,past_uv,past_valid,past_confidence)).chunk(2,-1)
        uv_point=point*(1+torch.tanh(gamma)[:,None])+beta[:,None];uv_scene=(uv_point*ok).sum(1)/ok.sum(1).clamp_min(1)
        uv_fused=self.uv_trunk(torch.cat((uv_point,uv_scene[:,None].expand(-1,n,-1)),-1))
        xyz=anchor_xyz[:,:,None]+self.xyz_head(xyz_fused).reshape(b,n,self.future_frames,3)
        steps=torch.arange(1,self.future_frames+1,device=past_uv.device,dtype=past_uv.dtype).view(1,1,-1,1)
        baseline=anchor_uv[:,:,None]+self.robust_uv_velocity(past_uv,past_valid)[:,:,None]*steps
        residual=self.residual_scale*torch.tanh(self.uv_residual_head(uv_fused).reshape(b,n,self.future_frames,2))
        return xyz,(baseline+residual).clamp(0,1)


class GeometryConditionEncoder(nn.Module):
    """Convert predicted 3D trajectories into tokens for video-model cross-attention."""
    def __init__(self, token_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(7, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))

    @staticmethod
    def raw_features(future_xyz: torch.Tensor, query_uv: torch.Tensor, confidence: torch.Tensor | None = None):
        b, n, k, _ = future_xyz.shape
        origin = future_xyz[:, :, :1]
        rel = future_xyz - origin
        uv = query_uv[:, :, None].expand(-1, -1, k, -1) if query_uv.ndim == 3 else query_uv
        if confidence is None:
            confidence = torch.ones((b, n, k, 1), device=future_xyz.device, dtype=future_xyz.dtype)
        elif confidence.ndim == 3:
            confidence = confidence[..., None]
        time = torch.linspace(0, 1, k, device=future_xyz.device, dtype=future_xyz.dtype)
        time = time.view(1, 1, k, 1).expand(b, n, -1, -1)
        return torch.cat((rel, uv, confidence, time), -1)

    def forward(self, future_xyz: torch.Tensor, query_uv: torch.Tensor, confidence: torch.Tensor | None = None):
        features = self.raw_features(future_xyz, query_uv, confidence)
        return self.proj(features).reshape(features.shape[0], -1, self.proj[-1].out_features)
