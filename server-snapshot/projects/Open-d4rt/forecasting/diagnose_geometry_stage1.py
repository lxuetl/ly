"""Stage-1 diagnostics for geometry-guided DMVFN video prediction.

Measures where a trained geometry correction helps or hurts without retraining.
Occlusion/disocclusion regions are explicitly labelled as sparse-track proxies;
they are diagnostics, not ground-truth dense occlusion masks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from dmvfn_adapter import DMVFNVideoPredictor
from geometry_residual_model import GeometryFeatureProposalCorrector,GeometryMotionCompensator
from train_geometry_residual import latest_uv
from train_pointodyssey_dmvfn_injection import combined_splits, geometry_for_source
from train_simvp_ablation import move, seed_all
from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True); p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--trajectory-checkpoint", required=True); p.add_argument("--backbone-checkpoint", required=True)
    p.add_argument("--corrector-checkpoint", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--visualize", type=int, default=4)
    p.add_argument("--mask-threshold", type=float, default=.01)
    p.add_argument("--support-radius", type=int, default=2)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--lpips-regions", default="modified,boundary,co_visible_proxy,occluded_proxy,uncovered_dynamic_proxy")
    p.add_argument("--base-motion-threshold", type=float, default=0.0)
    p.add_argument("--base-motion-ramp", type=float, default=0.0)
    p.add_argument("--flow-coherence-scale", type=float, default=0.0)
    p.add_argument("--first-step-scale", type=float, default=1.0)
    p.add_argument("--horizon-ramp-steps", type=int, default=1)
    p.add_argument("--fusion-type", choices=("auto","rgb_gate","feature_proposal"), default="auto")
    p.add_argument("--feature-channels", type=int, default=16)
    p.add_argument("--proposal-residual-scale", type=float, default=0.08)
    p.add_argument("--benefit-gate-bias", type=float, default=-0.5)
    return p.parse_args()


class PixelStats:
    def __init__(self):
        self.rows = defaultdict(lambda: defaultdict(float))

    def add(self, method, region, prediction, target, mask, base_error):
        error = (prediction - target).abs().mean(2)
        squared = (prediction - target).square().mean(2)
        selected = mask.bool(); n = selected.sum().item()
        if n == 0: return
        row = self.rows[(method, region)]
        row["pixels"] += n; row["abs"] += error[selected].sum().item(); row["sq"] += squared[selected].sum().item()
        row["wins_vs_base"] += (error[selected] < base_error[selected]).sum().item()
        row["ties_vs_base"] += torch.isclose(error[selected], base_error[selected], atol=1e-8).sum().item()

    def result(self):
        out = {}
        for (method, region), row in self.rows.items():
            n = max(row["pixels"], 1); mse = row["sq"] / n
            out.setdefault(method, {})[region] = {
                "pixels": int(row["pixels"]), "MAE": row["abs"] / n, "MSE": mse,
                "RMSE": math.sqrt(mse), "PSNR": -10 * math.log10(max(mse, 1e-12)),
                "win_rate_vs_base": row["wins_vs_base"] / n,
                "tie_rate_vs_base": row["ties_vs_base"] / n,
            }
        return out


def morphology(mask, radius=2):
    b,k,h,w = mask.shape; x = mask.float().reshape(b*k,1,h,w); kernel = 2*radius+1
    dilated = F.max_pool2d(x, kernel, 1, radius) > 0
    eroded = 1 - F.max_pool2d(1-x, kernel, 1, radius); eroded = eroded > .5
    return dilated.reshape(b,k,h,w), eroded.reshape(b,k,h,w)


def point_support(uv, selector, height, width, radius):
    """Nearest-pixel sparse support, expanded only for diagnostic readability."""
    b,n,k,_ = uv.shape; out = torch.zeros((b,k,height,width), dtype=torch.bool, device=uv.device)
    x = (torch.nan_to_num(uv[...,0]).clamp(0,1)*(width-1)).round().long()
    y = (torch.nan_to_num(uv[...,1]).clamp(0,1)*(height-1)).round().long()
    for bi in range(b):
        for ti in range(k):
            sel = selector[bi,:,ti].bool()
            if sel.any(): out[bi,ti,y[bi,sel,ti],x[bi,sel,ti]] = True
    if radius:
        out = F.max_pool2d(out.float().reshape(b*k,1,height,width),2*radius+1,1,radius).reshape(b,k,height,width)>0
    return out


def collision_support(uv, selector, height, width, radius):
    b,n,k,_=uv.shape;out=torch.zeros((b,k,height,width),dtype=torch.bool,device=uv.device)
    x=(torch.nan_to_num(uv[...,0]).clamp(0,1)*(width-1)).round().long();y=(torch.nan_to_num(uv[...,1]).clamp(0,1)*(height-1)).round().long()
    for bi in range(b):
        for ti in range(k):
            sel=selector[bi,:,ti].bool();flat=y[bi,sel,ti]*width+x[bi,sel,ti]
            if flat.numel():
                count=torch.bincount(flat,minlength=height*width).reshape(height,width);out[bi,ti]=count>1
    if radius:out=F.max_pool2d(out.float().reshape(b*k,1,height,width),2*radius+1,1,radius).reshape(b,k,height,width)>0
    return out


@torch.no_grad()
def lpips_values(model, prediction, target, mask=None, chunk=32):
    if mask is not None:
        prediction = target + mask[:,:,None].float() * (prediction-target)
    b,k,c,h,w=prediction.shape; a=(prediction.reshape(b*k,c,h,w)*2-1);t=(target.reshape(b*k,c,h,w)*2-1)
    values=[]
    for start in range(0,a.shape[0],chunk): values.append(model(a[start:start+chunk],t[start:start+chunk]).flatten())
    return torch.cat(values).reshape(b,k)


def add_lpips(store, method, region, values):
    for step in range(values.shape[1]):
        key=(method,region,step);store[key][0]+=values[:,step].sum().item();store[key][1]+=values.shape[0]


def save_visual(out, index, context, target, base, warp, final, mask, benefit, boundary, uncovered):
    step=target.shape[0]-1; items=[target[step],base[step],warp[step],final[step],mask[step],benefit[step],boundary[step],uncovered[step]]
    titles=["target","base","raw warp","final","effective mask","final benefit","boundary","uncovered dyn proxy"]
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,img,title in zip(axes.flat,items,titles):
        array=img.detach().cpu()
        if array.ndim==3: ax.imshow(array.permute(1,2,0).clamp(0,1))
        else: ax.imshow(array,cmap="magma",vmin=0,vmax=1)
        ax.set_title(title);ax.axis("off")
    fig.tight_layout();fig.savefig(out/f"diagnostic_{index:03d}.png",dpi=140);plt.close(fig)


def main():
    a=parse_args();seed_all(a.seed)
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device))
    # The third-party DMVFN implementation creates two zero tensors with a
    # hard-coded Tensor.cuda() call.  On a CPU-only diagnostic node, make that
    # tensor-local transfer a no-op; model parameters and all inputs are already
    # placed explicitly on ``device`` below.  Training code/checkpoints are not
    # modified by this compatibility shim.
    if device.type == "cpu":
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    splits,manifest=combined_splits(a);dataset=VideoForecastDataset(a.cache_root,"test",64,paths=splits["test"],cache_items=False)
    if a.limit>0: dataset=Subset(dataset,range(min(a.limit,len(dataset))))
    loader=DataLoader(dataset,1,collate_fn=video_pad_collate,num_workers=0)
    sample=dataset[0];c=sample["context_rgb"].shape[0];k=sample["future_rgb"].shape[0]
    bck=torch.load(a.backbone_checkpoint,map_location="cpu");base=DMVFNVideoPredictor(c,k);base.load_state_dict(bck["model"]);base.to(device).eval()
    jck=torch.load(a.trajectory_checkpoint,map_location="cpu");joint_cls=VelocityResidualImplicitForecaster if jck.get("model_variant")=="velocity_residual" else ImplicitConditionedJointForecaster
    joint=joint_cls(jck["future_frames"],jck["hidden_dim"]);joint.load_state_dict(jck["model"]);joint.to(device).eval()
    corrector_checkpoint=torch.load(a.corrector_checkpoint,map_location=device)
    fusion_type=corrector_checkpoint.get("fusion_type","rgb_gate") if a.fusion_type=="auto" else a.fusion_type
    corrector_cls=GeometryFeatureProposalCorrector if fusion_type=="feature_proposal" else GeometryMotionCompensator
    feature_channels=int(corrector_checkpoint.get("feature_channels",a.feature_channels))
    proposal_scale=float(corrector_checkpoint.get("proposal_residual_scale",a.proposal_residual_scale))
    benefit_gate_bias=float(corrector_checkpoint.get("benefit_gate_bias",a.benefit_gate_bias))
    benefit_topk_fraction=float(corrector_checkpoint.get("benefit_topk_fraction",0.0))
    benefit_temporal_align_weight=float(corrector_checkpoint.get("benefit_temporal_align_weight",0.0))
    extra={"feature_channels":feature_channels,"proposal_residual_scale":proposal_scale,
           "benefit_gate_bias":benefit_gate_bias,
           "benefit_topk_fraction":benefit_topk_fraction,
           "benefit_temporal_align_weight":benefit_temporal_align_weight} if fusion_type=="feature_proposal" else {}
    corrector=corrector_cls(
        sigma=.03,dilation_kernel=1,min_motion=.025,min_reliability=.30,mask_power=1.5,
        base_motion_threshold=a.base_motion_threshold,base_motion_ramp=a.base_motion_ramp,
        flow_coherence_scale=a.flow_coherence_scale,first_step_scale=a.first_step_scale,
        horizon_ramp_steps=a.horizon_ramp_steps,**extra).to(device).eval()
    corrector.load_state_dict(corrector_checkpoint["model"])
    lpips_model=None
    if not a.no_lpips:
        import lpips
        lpips_model=lpips.LPIPS(net="alex",verbose=False).to(device).eval()
    stats=PixelStats();temporal_stats=PixelStats();lpips_stats=defaultdict(lambda:[0.,0.]);coverage=defaultdict(float);horizon=defaultdict(lambda:defaultdict(float));temporal_horizon=defaultdict(lambda:defaultdict(float))
    requested_lpips={x.strip() for x in a.lpips_regions.split(",") if x.strip()}
    for index,batch in enumerate(loader):
        batch=move(batch,device);context=batch["context_rgb"];target=batch["future_rgb"];h,w=target.shape[-2:]
        base_pred=base(context);geometry,reliability,uv=geometry_for_source("predicted",joint,batch,k,"uv_only",True,True)
        final,delta,effective=corrector(base_pred,context[:,-1],geometry,uv,reliability,latest_uv(batch));warp=(base_pred+delta).clamp(0,1)
        hard=effective[:,:,0]>a.mask_threshold;dilated,core=morphology(hard,2);boundary=dilated&~core
        past_ok=batch["past_valid"].bool().any(2)[:,:,None].expand(-1,-1,k)
        future_ok=batch["future_valid"].bool()&torch.isfinite(batch["future_uv_teacher"]).all(-1)
        visible=point_support(uv,past_ok&future_ok,h,w,a.support_radius);occluded=point_support(uv,past_ok&~future_ok,h,w,a.support_radius)
        covered=point_support(uv,past_ok,h,w,a.support_radius);collision=collision_support(uv,past_ok,h,w,a.support_radius)
        gt_motion=(target-context[:,-1:]).abs().mean(2)>.05;uncovered=gt_motion&~covered
        regions={"all":torch.ones_like(hard),"modified":hard,"unmodified":~hard,"core":core,"boundary":boundary,
                 "co_visible_proxy":visible,"occluded_proxy":occluded,"collision_proxy":collision,
                 "uncovered_dynamic_proxy":uncovered,"gt_dynamic":gt_motion,"gt_static":~gt_motion}
        base_error=(base_pred-target).abs().mean(2)
        target_previous=torch.cat((context[:,-1:],target[:,:-1]),dim=1);target_delta=target-target_previous
        temporal_deltas={}
        for method,pred in (("base",base_pred),("warp",warp),("final",final)):
            pred_previous=torch.cat((context[:,-1:],pred[:,:-1]),dim=1)
            temporal_deltas[method]=pred-pred_previous
        base_temporal_error=(temporal_deltas["base"]-target_delta).abs().mean(2)
        for name,region in regions.items():
            coverage[name]+=region.sum().item();coverage[name+"_denom"]+=region.numel()
            for method,pred in (("base",base_pred),("warp",warp),("final",final)):
                stats.add(method,name,pred,target,region,base_error)
                temporal_stats.add(method,name,temporal_deltas[method],target_delta,region,base_temporal_error)
        for step in range(k):
            for method,pred in (("base",base_pred),("warp",warp),("final",final)):
                diff=pred[:,step]-target[:,step];n=diff.numel();row=horizon[(method,step)];row["abs"]+=diff.abs().sum().item();row["sq"]+=diff.square().sum().item();row["n"]+=n
                temporal_diff=temporal_deltas[method][:,step]-target_delta[:,step];temporal_row=temporal_horizon[(method,step)]
                temporal_row["abs"]+=temporal_diff.abs().sum().item();temporal_row["sq"]+=temporal_diff.square().sum().item();temporal_row["n"]+=temporal_diff.numel()
        if lpips_model is not None:
            for method,pred in (("base",base_pred),("warp",warp),("final",final)): add_lpips(lpips_stats,method,"all",lpips_values(lpips_model,pred,target))
            for region_name in requested_lpips:
                region=regions[region_name]
                add_lpips(lpips_stats,"base",region_name,lpips_values(lpips_model,base_pred,target,region))
                add_lpips(lpips_stats,"final",region_name,lpips_values(lpips_model,final,target,region))
        if index<a.visualize:
            benefit=((base_pred-target).abs().mean(2)-(final-target).abs().mean(2)>0)[0]
            save_visual(out,index,context[0],target[0],base_pred[0],warp[0],final[0],effective[0, :,0],benefit,boundary[0],uncovered[0])
        if (index+1)%10==0 or index+1==len(loader): print("processed",index+1,"/",len(loader),flush=True)
    result={"meta":{**manifest,"evaluated_clips":len(dataset),"device":str(device),"mask_threshold":a.mask_threshold,"support_radius":a.support_radius,
                    "base_motion_threshold":a.base_motion_threshold,"base_motion_ramp":a.base_motion_ramp,
                    "flow_coherence_scale":a.flow_coherence_scale,"first_step_scale":a.first_step_scale,
                    "horizon_ramp_steps":a.horizon_ramp_steps,
                    "fusion_type":fusion_type,"feature_channels":feature_channels,
                    "proposal_residual_scale":proposal_scale,"benefit_gate_bias":benefit_gate_bias,
                    "benefit_topk_fraction":benefit_topk_fraction,
                    "benefit_temporal_align_weight":benefit_temporal_align_weight,
                    "proxy_warning":"co-visible/occluded/uncovered are sparse-track support proxies, not dense ground-truth occlusion labels"},
            "region_coverage":{k:coverage[k]/max(coverage[k+"_denom"],1) for k in regions},"regions":stats.result(),
            "temporal_regions":temporal_stats.result(),"per_horizon":{},"temporal_per_horizon":{}}
    for (method,step),row in horizon.items():
        mse=row["sq"]/row["n"];entry={"MAE":row["abs"]/row["n"],"MSE":mse,"PSNR":-10*math.log10(max(mse,1e-12))}
        lp=lpips_stats.get((method,"all",step));entry["LPIPS"]=lp[0]/lp[1] if lp and lp[1] else None
        result["per_horizon"].setdefault(method,{})[f"t+{step+1}"]=entry
    for (method,step),row in temporal_horizon.items():
        mse=row["sq"]/row["n"]
        result["temporal_per_horizon"].setdefault(method,{})[f"t+{step+1}"]={"temporal_delta_MAE":row["abs"]/row["n"],"temporal_delta_MSE":mse}
    result["regional_LPIPS"]={}
    for (method,region,step),(total,count) in lpips_stats.items():
        if region=="all":continue
        result["regional_LPIPS"].setdefault(method,{}).setdefault(region,{})[f"t+{step+1}"]=total/max(count,1)
    (out/"stage1_diagnostics.json").write_text(json.dumps(result,indent=2))
    with (out/"region_metrics.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["method","region","pixels","MAE","MSE","RMSE","PSNR","win_rate_vs_base","tie_rate_vs_base"])
        for method,regions_out in result["regions"].items():
            for region,row in regions_out.items():writer.writerow([method,region,*[row[k] for k in ("pixels","MAE","MSE","RMSE","PSNR","win_rate_vs_base","tie_rate_vs_base")]])
    with (out/"temporal_region_metrics.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["method","region","pixels","temporal_delta_MAE","temporal_delta_MSE","temporal_delta_RMSE","win_rate_vs_base","tie_rate_vs_base"])
        for method,regions_out in result["temporal_regions"].items():
            for region,row in regions_out.items():writer.writerow([method,region,row["pixels"],row["MAE"],row["MSE"],row["RMSE"],row["win_rate_vs_base"],row["tie_rate_vs_base"]])
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
