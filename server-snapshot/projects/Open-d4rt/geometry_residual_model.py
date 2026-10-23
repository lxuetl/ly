"""A lightweight geometry-masked residual corrector for frozen video predictions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GeometryResidualCorrector(nn.Module):
    def __init__(self, geometry_channels=7, hidden=32, max_residual=0.25, sigma=0.045,
                 dilation_kernel=7):
        super().__init__()
        self.max_residual = float(max_residual)
        self.sigma = float(sigma)
        self.dilation_kernel = int(dilation_kernel)
        channels = 3 + 3 + geometry_channels + 1
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.residual_head = nn.Conv2d(hidden, 3, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 3, padding=1)
        # Start close to the frozen baseline; training must earn the right to edit.
        nn.init.constant_(self.gate_head.bias, -2.5)

    def set_gate_only(self, enabled=True):
        for parameter in self.trunk.parameters(): parameter.requires_grad_(not enabled)
        for parameter in self.residual_head.parameters(): parameter.requires_grad_(not enabled)
        for parameter in self.gate_head.parameters(): parameter.requires_grad_(True)

    def rasterize(self, geometry, future_uv, point_valid, last_uv, height, width):
        """Rasterize sparse points and return dense geometry plus a motion-only soft mask."""
        b, n, k, channels = geometry.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=geometry.device, dtype=geometry.dtype),
            torch.linspace(0, 1, width, device=geometry.device, dtype=geometry.dtype), indexing="ij")
        grids = []; masks = []
        if point_valid.ndim == 3:
            reliability = point_valid.float()
            valid = reliability.any(-1) & torch.isfinite(last_uv).all(-1)
        else:
            reliability = point_valid.float()[:, :, None].expand(-1, -1, k)
            valid = point_valid.bool() & torch.isfinite(last_uv).all(-1)
        for step in range(k):
            uv = torch.nan_to_num(future_uv[:, :, step]).clamp(0, 1)
            finite = valid & torch.isfinite(future_uv[:, :, step]).all(-1)
            dist2 = (xx[None, None] - uv[..., 0, None, None]).square()
            dist2 = dist2 + (yy[None, None] - uv[..., 1, None, None]).square()
            gaussian = torch.exp(-dist2 / (2 * self.sigma ** 2)) * finite[..., None, None]
            motion = torch.linalg.vector_norm(uv - torch.nan_to_num(last_uv), dim=-1)
            # raw_features layout: [dXYZ(3), UV(2), confidence, time].
            confidence = torch.nan_to_num(geometry[:, :, step, 5]).clamp(0, 1)
            moving = (motion / 0.04).clamp(0, 1) * confidence * reliability[:,:,step] * finite
            weighted = gaussian * moving[..., None, None]
            mask = weighted.amax(1, keepdim=True).clamp(0, 1)
            # A small dilation gives the point trajectory authority over the object around it.
            if self.dilation_kernel > 1:
                mask = F.max_pool2d(mask, kernel_size=self.dilation_kernel, stride=1,
                                    padding=self.dilation_kernel // 2)
            denom = weighted.sum(1, keepdim=True).clamp_min(1e-6)
            dense = torch.einsum("bnhw,bnc->bchw", weighted, torch.nan_to_num(geometry[:, :, step])) / denom
            grids.append(dense); masks.append(mask)
        return torch.stack(grids, 1), torch.stack(masks, 1)

    def forward(self, base_prediction, last_context, geometry, future_uv, point_valid, last_uv):
        b, k, _, h, w = base_prediction.shape
        dense, mask = self.rasterize(geometry, future_uv, point_valid, last_uv, h, w)
        last = last_context[:, None].expand(-1, k, -1, -1, -1)
        features = torch.cat((base_prediction, last, dense, mask), dim=2)
        hidden = self.trunk(features.reshape(b*k, -1, h, w))
        residual = self.max_residual * torch.tanh(self.residual_head(hidden)).reshape(b, k, 3, h, w)
        learned_gate = torch.sigmoid(self.gate_head(hidden)).reshape(b, k, 1, h, w)
        effective_mask = mask * learned_gate
        prediction = (base_prediction + effective_mask * residual).clamp(0, 1)
        return prediction, residual, effective_mask


class GeometryMotionCompensator(GeometryResidualCorrector):
    """Warp the last observed RGB locally along predicted UV trajectories."""
    def __init__(self, geometry_channels=7, hidden=32, sigma=0.03, dilation_kernel=3,
                 motion_consistency=False, min_motion=0.0, min_reliability=0.0,
                 motion_ramp=0.04, mask_power=1.0, base_motion_threshold=0.0,
                 base_motion_ramp=0.0, flow_coherence_scale=0.0,
                 first_step_scale=1.0, horizon_ramp_steps=1):
        super().__init__(geometry_channels, hidden, 1.0, sigma, dilation_kernel)
        self.motion_consistency = bool(motion_consistency)
        self.min_motion = float(min_motion)
        self.min_reliability = float(min_reliability)
        self.motion_ramp = float(motion_ramp)
        self.mask_power = float(mask_power)
        self.base_motion_threshold = float(base_motion_threshold)
        self.base_motion_ramp = float(base_motion_ramp)
        self.flow_coherence_scale = float(flow_coherence_scale)
        self.first_step_scale = float(first_step_scale)
        self.horizon_ramp_steps = int(horizon_ramp_steps)
        # No RGB residual head: the only candidate correction is motion-aligned appearance.
        self.residual_head = nn.Identity()

    def rasterize_motion(self, geometry, future_uv, point_valid, last_uv, height, width):
        b,n,k,channels=geometry.shape
        yy,xx=torch.meshgrid(torch.linspace(0,1,height,device=geometry.device,dtype=geometry.dtype),
                             torch.linspace(0,1,width,device=geometry.device,dtype=geometry.dtype),indexing='ij')
        if point_valid.ndim==3:
            reliability=point_valid.float();valid=reliability.any(-1)&torch.isfinite(last_uv).all(-1)
        else:
            reliability=point_valid.float()[:,:,None].expand(-1,-1,k);valid=point_valid.bool()&torch.isfinite(last_uv).all(-1)
        dense_all=[]; flow_all=[]; mask_all=[]
        for step in range(k):
            uv=torch.nan_to_num(future_uv[:,:,step]).clamp(0,1); finite=valid&torch.isfinite(future_uv[:,:,step]).all(-1)
            dist2=(xx[None,None]-uv[...,0,None,None]).square()+(yy[None,None]-uv[...,1,None,None]).square()
            gaussian=torch.exp(-dist2/(2*self.sigma**2))*finite[...,None,None]
            motion=torch.linalg.vector_norm(uv-torch.nan_to_num(last_uv),dim=-1)
            # raw_features layout: [dXYZ(3), UV(2), confidence, time].
            confidence=torch.nan_to_num(geometry[:,:,step,5]).clamp(0,1)
            rel = reliability[:,:,step]
            # A trajectory earns spatial edit authority only after clearing both
            # motion and reliability floors.  The ramp avoids a hard visual edge,
            # while the final power suppresses weak tails around static regions.
            motion_authority=((motion-self.min_motion)/max(self.motion_ramp-self.min_motion,1e-6)).clamp(0,1)
            allowed=(motion>=self.min_motion)&(rel>=self.min_reliability)&finite
            moving=motion_authority*confidence*rel*allowed
            if self.motion_consistency:
                velocity=future_uv[:,:,1:]-future_uv[:,:,:-1]
                mean_velocity=velocity.mean(2)
                reconstructed=future_uv[:,:,0]-mean_velocity
                cycle_error=torch.linalg.vector_norm(reconstructed-last_uv,dim=-1)
                score=torch.exp(-cycle_error/0.04)
                # Soft filtering: retain at least half the contribution so that
                # genuine acceleration and turns are not treated as bad tracks.
                moving=moving*(0.5+0.5*score)
            weighted=gaussian*moving[...,None,None]
            mask=weighted.amax(1,keepdim=True).clamp(0,1).pow(self.mask_power)
            if self.dilation_kernel>1: mask=F.max_pool2d(mask,self.dilation_kernel,1,self.dilation_kernel//2)
            denom=weighted.sum(1,keepdim=True).clamp_min(1e-6)
            dense=torch.einsum('bnhw,bnc->bchw',weighted,torch.nan_to_num(geometry[:,:,step]))/denom
            # Backward sampling flow: destination future UV samples appearance at last UV.
            backward=torch.nan_to_num(last_uv)-uv
            flow=torch.einsum('bnhw,bnc->bchw',weighted,backward)/denom
            if self.flow_coherence_scale > 0:
                second=torch.einsum('bnhw,bnc->bchw',weighted,backward.square())/denom
                flow_variance=(second-flow.square()).clamp_min(0).sum(1,keepdim=True)
                # Suppress destinations receiving mutually inconsistent tracks.
                coherence=torch.exp(-flow_variance/(self.flow_coherence_scale**2))
                mask=mask*coherence
            dense_all.append(dense); flow_all.append(flow); mask_all.append(mask)
        return torch.stack(dense_all,1),torch.stack(flow_all,1),torch.stack(mask_all,1)

    def forward(self,base_prediction,last_context,geometry,future_uv,point_valid,last_uv):
        b,k,_,h,w=base_prediction.shape
        dense,flow,mask=self.rasterize_motion(geometry,future_uv,point_valid,last_uv,h,w)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=base_prediction.device,dtype=base_prediction.dtype),
                             torch.linspace(-1,1,w,device=base_prediction.device,dtype=base_prediction.dtype),indexing='ij')
        grid=torch.stack((xx,yy),-1)[None,None].expand(b,k,-1,-1,-1).clone()
        grid[...,0]+=2*flow[:,:,0]; grid[...,1]+=2*flow[:,:,1]
        source=last_context[:,None].expand(-1,k,-1,-1,-1).reshape(b*k,3,h,w)
        warped=F.grid_sample(source,grid.reshape(b*k,h,w,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(b,k,3,h,w)
        last=last_context[:,None].expand(-1,k,-1,-1,-1)
        if self.base_motion_ramp > self.base_motion_threshold:
            # Geometry is allowed to edit only where the frozen video backbone
            # independently predicts appearance motion. This protects static
            # areas from erroneous sparse UV trajectories at inference time.
            base_motion=(base_prediction-last).abs().mean(2,keepdim=True)
            base_authority=((base_motion-self.base_motion_threshold)/
                            (self.base_motion_ramp-self.base_motion_threshold)).clamp(0,1)
            mask=mask*base_authority
        if self.first_step_scale < 1.0:
            denom=max(self.horizon_ramp_steps-1,1)
            progress=(torch.arange(k,device=mask.device,dtype=mask.dtype)/denom).clamp(0,1)
            horizon_scale=self.first_step_scale+(1-self.first_step_scale)*progress
            mask=mask*horizon_scale.view(1,k,1,1,1)
        features=torch.cat((base_prediction,last,dense,mask),2)
        hidden=self.trunk(features.reshape(b*k,-1,h,w)); gate=torch.sigmoid(self.gate_head(hidden)).reshape(b,k,1,h,w)
        effective=mask*gate; delta=warped-base_prediction
        return (base_prediction+effective*delta).clamp(0,1),delta,effective


class GeometryFeatureProposalCorrector(GeometryMotionCompensator):
    """Fuse a geometry-warped appearance proposal with a frozen video forecast.

    The frozen predictor remains the generation/inpainting path. Geometry only
    proposes propagated appearance. A post-warp network observes both feature
    streams, their disagreement, and geometry confidence before deciding where
    the proposal is beneficial. This avoids committing uncertain UV directly to
    RGB, especially in occluded or newly revealed regions.
    """
    def __init__(self, geometry_channels=7, feature_channels=16, hidden=48,
                 proposal_residual_scale=0.08, benefit_gate_bias=-0.5,
                 benefit_topk_fraction=0.0, benefit_temporal_align_weight=0.0,
                 benefit_temporal_ema_alpha=1.0, benefit_rank_margin=0.0,
                 dual_benefit_head=False,
                 benefit_min_joint_probability=0.0,
                 temporal_risk_max_probability=1.0,
                 **kwargs):
        super().__init__(geometry_channels=geometry_channels, hidden=hidden, **kwargs)
        self.feature_channels=int(feature_channels)
        self.proposal_residual_scale=float(proposal_residual_scale)
        self.benefit_gate_bias=float(benefit_gate_bias)
        self.benefit_topk_fraction=float(benefit_topk_fraction)
        self.benefit_temporal_align_weight=float(benefit_temporal_align_weight)
        self.benefit_temporal_ema_alpha=float(benefit_temporal_ema_alpha)
        self.benefit_rank_margin=float(benefit_rank_margin)
        self.dual_benefit_head=bool(dual_benefit_head)
        self.benefit_min_joint_probability=float(benefit_min_joint_probability)
        self.temporal_risk_max_probability=float(temporal_risk_max_probability)
        self.appearance_encoder=nn.Sequential(
            nn.Conv2d(3,self.feature_channels,3,padding=1),nn.GELU(),
            nn.Conv2d(self.feature_channels,self.feature_channels,3,padding=1),nn.GELU())
        fusion_channels=2*self.feature_channels+geometry_channels+3
        self.fusion_trunk=nn.Sequential(
            nn.Conv2d(fusion_channels,hidden,3,padding=1),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU())
        self.proposal_head=nn.Conv2d(hidden,3,3,padding=1)
        self.benefit_head=nn.Conv2d(hidden,1,3,padding=1)
        # Temporal risk is inherently a relation between forecast steps.  A
        # frame-wise Conv2d cannot observe flicker or inconsistent motion.
        self.temporal_risk_head=(nn.Conv3d(hidden,1,(3,3,3),padding=(1,1,1))
                                 if self.dual_benefit_head else None)
        nn.init.zeros_(self.proposal_head.weight);nn.init.zeros_(self.proposal_head.bias)
        nn.init.zeros_(self.benefit_head.weight);nn.init.constant_(self.benefit_head.bias,self.benefit_gate_bias)
        if self.temporal_risk_head is not None:
            nn.init.zeros_(self.temporal_risk_head.weight);nn.init.zeros_(self.temporal_risk_head.bias)
        # The inherited RGB trunk/head are intentionally unused.
        del self.trunk;del self.gate_head
        self.last_gate_logits=None;self.last_gate_selection=None
        self.last_support=None;self.last_proposal=None
        self.last_base_prediction=None
        self.last_temporal_gate_consistency=None
        self.last_temporal_selection_consistency=None
        self.last_rank_boundary_loss=None
        self.last_correction_temporal_consistency=None
        self.last_temporal_risk_logits=None

    def set_gate_only(self, enabled=True):
        for parameter in self.parameters():parameter.requires_grad_(not enabled)
        # The benefit head cannot learn a useful classifier from a frozen,
        # randomly initialized fusion representation.  During warm-up, learn
        # the representation and classifier together while keeping the RGB
        # proposal path frozen.
        for parameter in self.fusion_trunk.parameters():parameter.requires_grad_(True)
        for parameter in self.benefit_head.parameters():parameter.requires_grad_(True)
        if self.temporal_risk_head is not None:
            for parameter in self.temporal_risk_head.parameters():parameter.requires_grad_(True)

    def forward(self,base_prediction,last_context,geometry,future_uv,point_valid,last_uv):
        b,k,_,h,w=base_prediction.shape
        dense,flow,mask=self.rasterize_motion(geometry,future_uv,point_valid,last_uv,h,w)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=base_prediction.device,dtype=base_prediction.dtype),
                             torch.linspace(-1,1,w,device=base_prediction.device,dtype=base_prediction.dtype),indexing='ij')
        base_grid=torch.stack((xx,yy),-1)[None,None].expand(b,k,-1,-1,-1)
        grid=base_grid.clone()
        grid[...,0]+=2*flow[:,:,0];grid[...,1]+=2*flow[:,:,1]
        flat_grid=grid.reshape(b*k,h,w,2)
        last=last_context[:,None].expand(-1,k,-1,-1,-1)
        source=last.reshape(b*k,3,h,w)
        warped_rgb=F.grid_sample(source,flat_grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(b,k,3,h,w)
        if self.base_motion_ramp>self.base_motion_threshold:
            base_motion=(base_prediction-last).abs().mean(2,keepdim=True)
            authority=((base_motion-self.base_motion_threshold)/
                       (self.base_motion_ramp-self.base_motion_threshold)).clamp(0,1)
            mask=mask*authority
        else:base_motion=(base_prediction-last).abs().mean(2,keepdim=True)
        if self.first_step_scale<1.0:
            denom=max(self.horizon_ramp_steps-1,1)
            progress=(torch.arange(k,device=mask.device,dtype=mask.dtype)/denom).clamp(0,1)
            scale=self.first_step_scale+(1-self.first_step_scale)*progress
            mask=mask*scale.view(1,k,1,1,1)
        source_feature=self.appearance_encoder(last_context)
        source_feature=source_feature[:,None].expand(-1,k,-1,-1,-1).reshape(b*k,self.feature_channels,h,w)
        warped_feature=F.grid_sample(source_feature,flat_grid,mode='bilinear',padding_mode='border',align_corners=True)
        base_feature=self.appearance_encoder(base_prediction.reshape(b*k,3,h,w))
        disagreement=(warped_rgb-base_prediction).abs().mean(2,keepdim=True)
        inputs=torch.cat((base_feature,warped_feature,dense.reshape(b*k,-1,h,w),
                          mask.reshape(b*k,1,h,w),disagreement.reshape(b*k,1,h,w),
                          base_motion.reshape(b*k,1,h,w)),1)
        hidden=self.fusion_trunk(inputs)
        refinement=self.proposal_residual_scale*torch.tanh(self.proposal_head(hidden)).reshape(b,k,3,h,w)
        proposal=(warped_rgb+refinement).clamp(0,1)
        logits=self.benefit_head(hidden).reshape(b,k,1,h,w)
        temporal_risk_logits=None
        if self.temporal_risk_head is not None:
            temporal_hidden=hidden.reshape(b,k,-1,h,w).permute(0,2,1,3,4)
            temporal_risk_logits=self.temporal_risk_head(temporal_hidden).permute(0,2,1,3,4)
        if temporal_risk_logits is not None:
            joint_probability=(torch.sigmoid(logits)*torch.sigmoid(-temporal_risk_logits)).clamp(1e-5,1-1e-5)
            rank_logits=torch.logit(joint_probability)
        else:
            rank_logits=logits
        temporal_consistency=logits.new_zeros(())
        if self.benefit_temporal_align_weight>0 and k>1:
            weight=self.benefit_temporal_align_weight
            aligned=[];consistency_sum=logits.new_zeros(());consistency_count=logits.new_zeros(())
            current_support=(mask>.01).to(logits.dtype)
            # Align the joint spatial-benefit/temporal-safety score.  Aligning
            # raw spatial logits here would silently discard the risk head.
            rank_source=rank_logits
            gate_probability=torch.sigmoid(rank_source)
            for step in range(k):
                numerator=rank_source[:,step];denominator=torch.ones_like(numerator)
                for neighbour in (step-1,step+1):
                    if neighbour<0 or neighbour>=k:continue
                    relative=(flow[:,step]-flow[:,neighbour]).permute(0,2,3,1)
                    relative_grid=base_grid[:,step]+2*relative
                    neighbour_logits=F.grid_sample(rank_source[:,neighbour],relative_grid,
                        mode='bilinear',padding_mode='border',align_corners=True)
                    neighbour_support=F.grid_sample(mask[:,neighbour],relative_grid,
                        mode='bilinear',padding_mode='zeros',align_corners=True)
                    valid=(neighbour_support>.01).to(logits.dtype)*current_support[:,step]
                    numerator=numerator+weight*valid*neighbour_logits
                    denominator=denominator+weight*valid
                    neighbour_probability=F.grid_sample(gate_probability[:,neighbour],relative_grid,
                        mode='bilinear',padding_mode='zeros',align_corners=True)
                    consistency_sum=consistency_sum+((gate_probability[:,step]-neighbour_probability).abs()*valid).sum()
                    consistency_count=consistency_count+valid.sum()
                aligned.append(numerator/denominator)
            rank_logits=torch.stack(aligned,1)
            temporal_consistency=consistency_sum/consistency_count.clamp_min(1)
        if self.benefit_temporal_ema_alpha<1 and k>1:
            alpha=self.benefit_temporal_ema_alpha
            propagated=[rank_logits[:,0]]
            current_support=(mask>.01).to(logits.dtype)
            for step in range(1,k):
                relative=(flow[:,step]-flow[:,step-1]).permute(0,2,3,1)
                relative_grid=base_grid[:,step]+2*relative
                previous_rank=F.grid_sample(propagated[-1],relative_grid,
                    mode='bilinear',padding_mode='border',align_corners=True)
                previous_support=F.grid_sample(mask[:,step-1],relative_grid,
                    mode='bilinear',padding_mode='zeros',align_corners=True)
                valid=(previous_support>.01).to(logits.dtype)*current_support[:,step]
                fused=alpha*rank_logits[:,step]+(1-alpha)*previous_rank
                propagated.append(valid*fused+(1-valid)*rank_logits[:,step])
            rank_logits=torch.stack(propagated,1)
        soft_gate=torch.sigmoid(rank_logits)
        boundary_losses=[]
        if 0<self.benefit_topk_fraction<1:
            flat_logits=rank_logits.reshape(b*k,-1);flat_support=(mask>.01).reshape(b*k,-1)
            hard_gate=torch.zeros_like(flat_logits)
            for row in range(b*k):
                valid=torch.nonzero(flat_support[row],as_tuple=False).flatten()
                if valid.numel()==0:continue
                keep=max(1,int(round(self.benefit_topk_fraction*valid.numel())))
                valid_scores=flat_logits[row,valid]
                # Top-K is a maximum edit budget, not a quota.  Never force a
                # low-confidence or temporally risky pixel into the correction.
                eligible=torch.ones_like(valid_scores,dtype=torch.bool)
                if self.benefit_min_joint_probability>0:
                    eligible&=torch.sigmoid(valid_scores)>=self.benefit_min_joint_probability
                if temporal_risk_logits is not None and self.temporal_risk_max_probability<1:
                    risk_probability=torch.sigmoid(temporal_risk_logits.reshape(b*k,-1)[row,valid])
                    eligible&=risk_probability<=self.temporal_risk_max_probability
                valid=valid[eligible];valid_scores=valid_scores[eligible]
                if valid.numel()==0:continue
                ranked=torch.topk(valid_scores,min(keep+1,valid.numel())).values
                chosen=valid[torch.topk(valid_scores,min(keep,valid.numel())).indices]
                hard_gate[row,chosen]=1
                if self.benefit_rank_margin>0 and keep<valid.numel():
                    boundary_losses.append(torch.relu(
                        flat_logits.new_tensor(self.benefit_rank_margin)-ranked[keep-1]+ranked[keep]))
            hard_gate=hard_gate.reshape_as(logits)
            gate=hard_gate+soft_gate-soft_gate.detach() if self.training else hard_gate
        else:
            gate=soft_gate;hard_gate=(soft_gate>.5).to(soft_gate.dtype)
        rank_boundary_loss=(torch.stack(boundary_losses).mean() if boundary_losses
                            else logits.new_zeros(()))
        selection_sum=logits.new_zeros(());selection_count=logits.new_zeros(())
        if k>1:
            current_support=(mask>.01).to(logits.dtype)
            for step in range(1,k):
                relative=(flow[:,step]-flow[:,step-1]).permute(0,2,3,1)
                relative_grid=base_grid[:,step]+2*relative
                previous_gate=F.grid_sample(gate[:,step-1],relative_grid,
                    mode='bilinear',padding_mode='zeros',align_corners=True)
                previous_support=F.grid_sample(mask[:,step-1],relative_grid,
                    mode='bilinear',padding_mode='zeros',align_corners=True)
                valid=(previous_support>.01).to(logits.dtype)*current_support[:,step]
                selection_sum=selection_sum+((gate[:,step]-previous_gate).abs()*valid).sum()
                selection_count=selection_count+valid.sum()
        selection_consistency=selection_sum/selection_count.clamp_min(1)
        effective=mask*gate
        delta=proposal-base_prediction
        correction=effective*delta
        correction_sum=logits.new_zeros(());correction_count=logits.new_zeros(())
        if k>1:
            for step in range(1,k):
                relative=(flow[:,step]-flow[:,step-1]).permute(0,2,3,1)
                relative_grid=base_grid[:,step]+2*relative
                previous_correction=F.grid_sample(correction[:,step-1],relative_grid,
                    mode='bilinear',padding_mode='zeros',align_corners=True)
                previous_confidence=F.grid_sample(mask[:,step-1],relative_grid,
                    mode='bilinear',padding_mode='zeros',align_corners=True)
                # Continuous overlap excludes occlusions/newly revealed pixels and
                # weights the constraint by geometric reliability at both frames.
                overlap=(mask[:,step]*previous_confidence).detach()
                correction_sum=correction_sum+((correction[:,step]-previous_correction).abs()*overlap).sum()
                correction_count=correction_count+overlap.sum()*correction.shape[2]
        correction_consistency=correction_sum/correction_count.clamp_min(1)
        prediction=(base_prediction+correction).clamp(0,1)
        self.last_gate_logits=logits;self.last_gate_rank_logits=rank_logits
        self.last_temporal_risk_logits=temporal_risk_logits
        self.last_gate_selection=hard_gate.detach()
        self.last_support=mask;self.last_proposal=proposal
        self.last_base_prediction=base_prediction
        self.last_context=last_context
        self.last_flow=flow.detach()
        self.last_temporal_gate_consistency=temporal_consistency
        self.last_temporal_selection_consistency=selection_consistency
        self.last_rank_boundary_loss=rank_boundary_loss
        self.last_correction_temporal_consistency=correction_consistency
        return prediction,delta,effective
