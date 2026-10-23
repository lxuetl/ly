"""One temporal cross-attention layer per track, historical appearance values.

Only UV and observed validity/confidence are used in the first prototype.
The oracle arm replaces FUTURE UV ONLY; no future visibility/RGB enters tokens.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from visibility_interface import future_gate


def sample(feature, uv):
    # feature [B,T,C,H,W], uv [B,N,T,2] -> [B,N,T,C].
    b,t,c,h,w=feature.shape;n=uv.shape[1]
    finite=torch.isfinite(uv).all(-1)
    safe=torch.where(finite[...,None],uv,torch.full_like(uv,2.))
    grid=safe.permute(0,2,1,3).reshape(b*t,n,1,2)*2-1
    z=F.grid_sample(feature.reshape(b*t,c,h,w),grid,align_corners=True,padding_mode='zeros')
    return z[...,0].reshape(b,t,c,n).permute(0,3,1,2)*finite[...,None]


def splat(values, uv, weight, h, w):
    # [B,N,K,C], [B,N,K,2], [B,N,K]. No per-sample amplification to force budget.
    b,n,k,c=values.shape
    inside=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
    inside=inside&torch.isfinite(values).all(-1)&torch.isfinite(weight)
    values=torch.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
    weight=torch.nan_to_num(weight,nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    uv=torch.nan_to_num(uv).clamp(0,1);xf=uv[...,0]*(w-1);yf=uv[...,1]*(h-1)
    x0=xf.floor().long();y0=yf.floor().long();dx=xf-x0;dy=yf-y0
    num=values.new_zeros(b,k,h*w,c);den=values.new_zeros(b,k,h*w,1)
    for xx,yy,a in [(x0,y0,(1-dx)*(1-dy)),((x0+1).clamp_max(w-1),y0,dx*(1-dy)),(x0,(y0+1).clamp_max(h-1),(1-dx)*dy),((x0+1).clamp_max(w-1),(y0+1).clamp_max(h-1),dx*dy)]:
        ww=(a*weight*inside).permute(0,2,1)[...,None]
        idx=(yy*w+xx).permute(0,2,1)[...,None]
        num.scatter_add_(2,idx.expand(-1,-1,-1,c),values.permute(0,2,1,3)*ww)
        den.scatter_add_(2,idx,ww)
    num=num.permute(0,1,3,2).reshape(b*k,c,h,w);den=den.permute(0,1,3,2).reshape(b*k,1,h,w)
    num=F.avg_pool2d(num,5,1,2);den=F.avg_pool2d(den,5,1,2)
    # Clamp support instead of confidence cancellation through normalization.
    support=(den*25).clamp(0,1)
    out=num/den.clamp_min(1e-6)*support
    return out.reshape(b,k,c,h,w),support.reshape(b,k,1,h,w)


class TrackCrossAttention(nn.Module):
    def __init__(self,channels,dim=64,heads=4):
        super().__init__();self.dim=dim;self.heads=heads
        self.q=nn.Linear(channels+3,dim)
        self.k=nn.Linear(channels+4,dim)
        self.v=nn.Linear(channels,dim)
        self.out=nn.Linear(dim,channels)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,hidden,track_values,history_uv,valid,confidence,future_uv,uniform=False,horizon_values=None,
                future_visibility=None,visibility_source='unknown',oracle_diagnostic=False):
        b,n,t,c=track_values.shape;k=future_uv.shape[2]
        query=sample(hidden,future_uv)
        gate=future_gate(future_uv,future_visibility,source=visibility_source,oracle_diagnostic=oracle_diagnostic)
        # Keep finite arithmetic even for rejected points; raw UV is never clamped to edges.
        future_uv=torch.nan_to_num(future_uv,nan=2.,posinf=2.,neginf=-1.)
        finite_values=torch.isfinite(track_values).all(-1)
        finite_uv=torch.isfinite(history_uv).all(-1)
        track_values=torch.nan_to_num(track_values,nan=0.,posinf=0.,neginf=0.)
        history_uv=torch.nan_to_num(history_uv,nan=2.,posinf=2.,neginf=-1.)
        valid=valid.bool()&finite_values&finite_uv
        confidence=torch.nan_to_num(confidence,nan=0.,posinf=0.,neginf=0.).clamp(0,1)
        horizon=(torch.linspace(0,1,k,device=hidden.device).view(1,1,k,1).expand(b,n,-1,-1)
                 if horizon_values is None else horizon_values)
        q=self.q(torch.cat([query,future_uv,horizon],-1))
        ht=torch.linspace(0,1,t,device=hidden.device).view(1,1,t,1).expand(b,n,-1,-1)
        # history_uv may be [B,N,K,T,2] in the fixed-position control.
        if track_values.ndim!=4:raise ValueError('use forward_fixed for fixed-position memories')
        rel=history_uv-history_uv[:,:,-1:]
        kk=self.k(torch.cat([track_values,rel,ht,confidence[...,None]],-1))
        vv=self.v(track_values)
        d=self.dim//self.heads
        q=q.view(b,n,k,self.heads,d).permute(0,1,3,2,4)
        kk=kk.view(b,n,t,self.heads,d).permute(0,1,3,2,4)
        vv=vv.view(b,n,t,self.heads,d).permute(0,1,3,2,4)
        logits=(q@kk.transpose(-1,-2))/math.sqrt(d)
        if uniform:logits=logits*0
        good=valid.bool()&torch.isfinite(confidence)&(confidence>0)
        logits=logits+confidence.clamp_min(1e-6).log()[:,:,None,None,:]
        logits=logits.masked_fill(~good[:,:,None,None,:],-1e4)
        a=logits.softmax(-1)*good[:,:,None,None,:]
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
        z=(a@vv).permute(0,1,3,2,4).reshape(b,n,k,self.dim)
        z=self.out(z)
        rel_weight=(confidence*good).sum(-1)/good.sum(-1).clamp_min(1)
        # Historical reliability is only a transport weight, not future visibility.
        weight=rel_weight.clamp(0,1)[:,:,None].expand(-1,-1,k)*gate
        delta,support=splat(z,future_uv,weight,*hidden.shape[-2:])
        # A fixed maximum feature correction; never boost tiny proposals.
        delta=.05*torch.tanh(delta)
        # Per-horizon cap is invariant to fixed-control's B*K vectorization.
        norm=delta.abs().mean((2,3,4),keepdim=True)
        delta=delta*(.002/norm.clamp_min(1e-9)).clamp(max=1)
        return hidden+delta,support

    def forward_fixed(self,hidden,memories,future_uv,valid,confidence):
        # Same [point, future horizon] endpoint is held fixed across history time.
        # The endpoint is PREDICTED, never oracle in this deployable control.
        b,n,k,t,c=memories.shape
        h,w=hidden.shape[-2:]
        # Vectorize K as batch; the main layer sees a single future horizon.
        hh=hidden.permute(0,1,2,3,4).reshape(b*k,1,c,h,w)
        vals=memories.permute(0,2,1,3,4).reshape(b*k,n,t,c)
        uv=future_uv.permute(0,2,1,3).reshape(b*k,n,1,2)
        hist=uv.expand(-1,-1,t,-1)
        inside=(hist>=0).all(-1)&(hist<=1).all(-1)
        good=valid[:,None].expand(-1,k,-1,-1).reshape(b*k,n,t)&inside
        conf=confidence[:,None].expand(-1,k,-1,-1).reshape(b*k,n,t)
        ht=torch.linspace(0,1,k,device=hidden.device)[None,:,None,None,None].expand(b,-1,n,1,1).reshape(b*k,n,1,1)
        result,sup=self.forward(hh,vals,hist,good,conf,uv,horizon_values=ht)
        return result.reshape(b,k,c,h,w),sup.reshape(b,k,1,h,w)
