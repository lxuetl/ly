"""Frozen SimVP with trainable geometry-guided decoder feature warping."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class GeometryFeatureWarpSimVP(nn.Module):
    def __init__(self, base, consistency=False, visibility=False, dense_field=False):
        super().__init__(); self.base=base; self.consistency=consistency; self.visibility=visibility; self.dense_field=dense_field
        for p in base.parameters(): p.requires_grad_(False)
        ch=base.geometry.proj[-1].out_features
        self.latent_delta=nn.Sequential(nn.Conv2d(ch*2,ch,3,padding=1),nn.GELU(),nn.Conv2d(ch,ch,3,padding=1))
        self.skip_delta=nn.Sequential(nn.Conv2d(ch*2,ch,3,padding=1),nn.GELU(),nn.Conv2d(ch,ch,3,padding=1))
        self.latent_gate=nn.Conv2d(ch*2,1,3,padding=1); self.skip_gate=nn.Conv2d(ch*2,1,3,padding=1)
        nn.init.constant_(self.latent_gate.bias,-2.5); nn.init.constant_(self.skip_gate.bias,-2.5)

    @staticmethod
    def _last_uv(past_uv,past_valid):
        idx=(past_valid.long()*torch.arange(1,past_valid.shape[2]+1,device=past_uv.device)).argmax(2)
        return torch.nan_to_num(past_uv).gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)

    def point_reliability(self,future_uv,future_xyz,confidence,last_uv,past_uv,past_valid):
        rel=confidence.clamp(0,1)
        if self.consistency:
            velocity=future_uv[:,:,1:]-future_uv[:,:,:-1]
            mean_v=velocity.mean(2); reconstructed=future_uv[:,:,0]-mean_v
            cycle=torch.linalg.vector_norm(reconstructed-last_uv,dim=-1)
            hist_ok=past_valid[:,:,-1]&past_valid[:,:,-2]
            hist_v=torch.nan_to_num(past_uv[:,:,-1]-past_uv[:,:,-2])
            vel_error=torch.linalg.vector_norm((future_uv[:,:,0]-last_uv)-hist_v,dim=-1)
            score=torch.exp(-cycle/0.04)*torch.where(hist_ok,torch.exp(-vel_error/0.08),torch.ones_like(cycle))
            rel=rel*score[:,:,None]
        if self.visibility:
            edge=torch.minimum(torch.minimum(future_uv[...,0],1-future_uv[...,0]),torch.minimum(future_uv[...,1],1-future_uv[...,1]))
            rel=rel*(edge/0.04).clamp(0,1)
            # Soft z-buffer inside an 8x8 image-plane grid: retain the front-most trajectory.
            b,n,k,_=future_uv.shape; cell=(future_uv[...,1].clamp(0,1)*7).long()*8+(future_uv[...,0].clamp(0,1)*7).long()
            z=future_xyz[...,2]; nearest=torch.full((b,k,64),float('inf'),device=z.device,dtype=z.dtype)
            nearest.scatter_reduce_(2,cell.permute(0,2,1),z.permute(0,2,1),reduce='amin',include_self=True)
            zmin=nearest.gather(2,cell.permute(0,2,1)).permute(0,2,1)
            rel=rel*torch.exp(-(z-zmin).clamp_min(0)/0.20)
        return rel

    def flow_field(self,future_uv,future_xyz,confidence,last_uv,past_uv,past_valid,valid,h,w):
        b,n,k,_=future_uv.shape; yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=future_uv.device),torch.linspace(0,1,w,device=future_uv.device),indexing='ij')
        reliability=self.point_reliability(future_uv,future_xyz,confidence,last_uv,past_uv,past_valid)*valid[:,:,None]
        sigma=0.075 if self.dense_field else 0.030; flows=[]; masks=[]
        for t in range(k):
            uv=future_uv[:,:,t].clamp(0,1); d2=(xx[None,None]-uv[...,0,None,None]).square()+(yy[None,None]-uv[...,1,None,None]).square()
            kernel=torch.exp(-d2/(2*sigma*sigma)); motion=torch.linalg.vector_norm(uv-last_uv,dim=-1)
            weight=kernel*reliability[:,:,t,None,None]*(motion/0.04).clamp(0,1)[...,None,None]
            denom=weight.sum(1,keepdim=True).clamp_min(1e-6); backward=last_uv-uv
            flow=torch.einsum('bnhw,bnc->bchw',weight,backward)/denom
            mask=weight.amax(1,keepdim=True).clamp(0,1)
            if self.dense_field: mask=F.max_pool2d(mask,5,1,2)
            flows.append(flow); masks.append(mask)
        return torch.stack(flows,1),torch.stack(masks,1)

    @staticmethod
    def warp(source,flow):
        b,k,c,h,w=source.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=source.device),torch.linspace(-1,1,w,device=source.device),indexing='ij')
        grid=torch.stack((xx,yy),-1)[None,None].expand(b,k,-1,-1,-1).clone(); grid[...,0]+=2*flow[:,:,0]; grid[...,1]+=2*flow[:,:,1]
        return F.grid_sample(source.reshape(b*k,c,h,w),grid.reshape(b*k,h,w,2),padding_mode='border',align_corners=True).reshape_as(source)

    def _fuse(self,base_feature,warped,mask,delta_net,gate_net):
        b,k,c,h,w=base_feature.shape; pair=torch.cat((base_feature,warped),2).reshape(b*k,c*2,h,w)
        delta=delta_net(pair).reshape(b,k,c,h,w); gate=torch.sigmoid(gate_net(pair)).reshape(b,k,1,h,w)*mask
        return base_feature+gate*delta,gate

    def forward(self,context,future_uv,future_xyz,confidence,past_uv,past_valid):
        b,t,c,h0,w0=context.shape; embed,skip=self.base.backbone.enc(context.reshape(b*t,c,h0,w0)); ch,h,w=embed.shape[1:]; k=future_uv.shape[2]
        hidden=self.base.latent_time(self.base.backbone.hid(embed.reshape(b,t,ch,h,w)))
        skip_base=self.base.skip_time(skip.reshape(b,t,*skip.shape[1:])); last_uv=self._last_uv(past_uv,past_valid); point_valid=past_valid.any(2)
        flow,mask=self.flow_field(future_uv,future_xyz,confidence,last_uv,past_uv,past_valid,point_valid,h,w)
        last_embed=embed.reshape(b,t,ch,h,w)[:,-1:,].expand(-1,k,-1,-1,-1); warped=self.warp(last_embed,flow)
        hidden,gate1=self._fuse(hidden,warped,mask,self.latent_delta,self.latent_gate)
        hs,ws=skip_base.shape[-2:]; flow_s=F.interpolate(flow.reshape(b*k,2,h,w),size=(hs,ws),mode='bilinear',align_corners=False).reshape(b,k,2,hs,ws)
        mask_s=F.interpolate(mask.reshape(b*k,1,h,w),size=(hs,ws),mode='bilinear',align_corners=False).reshape(b,k,1,hs,ws)
        last_skip=skip.reshape(b,t,*skip.shape[1:])[:,-1:,].expand(-1,k,-1,-1,-1); warped_skip=self.warp(last_skip,flow_s)
        skip_base,gate2=self._fuse(skip_base,warped_skip,mask_s,self.skip_delta,self.skip_gate)
        decoded=self.base.backbone.dec(hidden.reshape(b*k,ch,h,w),skip_base.reshape(b*k,*skip_base.shape[2:]))
        return torch.sigmoid(decoded.reshape(b,k,3,h0,w0)),(gate1.mean()+gate2.mean())/2
