"""Train a geometry-masked residual branch on top of a frozen mixed-domain SimVP."""

from __future__ import annotations

import argparse, json, math, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent))
from geometry_residual_model import GeometryMotionCompensator, GeometryResidualCorrector
from simvp_geometry_model import GeometrySimVP
from trajectory_model import JointTrajectoryForecaster
from train_mixed_adt_ds import make_splits
from train_simvp_ablation import make_geometry, move, seed_all
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from video_metrics import VideoMetricAccumulator


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',default='data/d4rt_forecasting_cache')
    p.add_argument('--mixed-root',default='outputs/mixed_adt_ds')
    p.add_argument('--output-dir',default='outputs/mixed_adt_ds/video/geometry_residual')
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-size',type=int,default=64); p.add_argument('--device',default='cuda')
    p.add_argument('--dynamic-weight',type=float,default=3.0); p.add_argument('--residual-weight',type=float,default=0.01)
    p.add_argument('--sigma',type=float,default=0.045); p.add_argument('--dilation-kernel',type=int,default=7)
    p.add_argument('--max-residual',type=float,default=0.25)
    p.add_argument('--gate-weight',type=float,default=0.0)
    p.add_argument('--gate-only-epochs',type=int,default=0)
    p.add_argument('--dynamic-select-weight',type=float,default=0.0)
    p.add_argument('--init-checkpoint',default='')
    p.add_argument('--model-type',choices=('residual','motion_comp'),default='residual')
    p.add_argument('--motion-consistency',action='store_true')
    p.add_argument('--eval-only',action='store_true')
    return p.parse_args()


def latest_uv(batch):
    valid=batch['past_valid'].bool(); uv=torch.nan_to_num(batch['past_uv'].float())
    idx=(valid.long()*torch.arange(1,valid.shape[2]+1,device=valid.device)).argmax(2)
    return uv.gather(2,idx[...,None,None].expand(-1,-1,1,2)).squeeze(2)


def forward(base,corrector,joint,batch,k):
    with torch.no_grad():
        base_pred=base(batch['context_rgb'])
        geometry,ok,future_uv=make_geometry('joint_future_geometry',None,joint,batch,k)
    return corrector(base_pred,batch['context_rgb'][:,-1],geometry,future_uv,ok,latest_uv(batch))


def loss_fn(pred,target,last,residual,mask,dynamic_weight,residual_weight,gate_weight):
    error=(pred-target).abs()+0.1*(pred-target).square()
    dynamic=((target-last[:,None]).abs().mean(2,keepdim=True)>0.05).float()
    reconstruction=(error*(1+dynamic_weight*dynamic)).mean()
    return reconstruction+residual_weight*(mask*residual).abs().mean()+gate_weight*mask.mean()


@torch.no_grad()
def validation_metrics(base,corrector,joint,loader,device,k):
    corrector.eval(); total=count=dyn_abs=dyn_count=0.0
    for batch in loader:
        batch=move(batch,device); pred,_,_=forward(base,corrector,joint,batch,k)
        total+=F.mse_loss(pred,batch['future_rgb'],reduction='sum').item(); count+=batch['future_rgb'].numel()
        dynamic=((batch['future_rgb']-batch['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)>0.05).expand_as(pred)
        dyn_abs+=(pred-batch['future_rgb']).abs()[dynamic].sum().item(); dyn_count+=dynamic.sum().item()
    return {'val_MSE':total/count,'val_dynamic_MAE':dyn_abs/max(dyn_count,1)}


def load_legacy_gated(model,path,device):
    """Load the previous shared-head gated checkpoint into split residual/gate heads."""
    state=torch.load(path,map_location=device)['model']
    if any(key.startswith('trunk.') for key in state):
        missing,unexpected=model.load_state_dict(state,strict=False)
        if unexpected or missing: print('checkpoint load',{'missing':missing,'unexpected':unexpected},flush=True)
        return
    converted={}
    for key,value in state.items():
        if key.startswith('net.0.'): converted[key.replace('net.0.','trunk.0.')]=value
        elif key.startswith('net.2.'): converted[key.replace('net.2.','trunk.2.')]=value
        elif key.startswith('net.4.'): converted[key.replace('net.4.','trunk.4.')]=value
        elif key=='net.6.weight':
            converted['residual_head.weight']=value[:3]; converted['gate_head.weight']=value[3:4]
        elif key=='net.6.bias':
            converted['residual_head.bias']=value[:3]; converted['gate_head.bias']=value[3:4]
    missing,unexpected=model.load_state_dict(converted,strict=False)
    if unexpected or missing: print('checkpoint conversion',{'missing':missing,'unexpected':unexpected},flush=True)


@torch.no_grad()
def evaluate(base,corrector,joint,loader,device,k,out,motion_threshold=0.05):
    corrector.eval(); metrics_accumulator=VideoMetricAccumulator(device,k)
    dyn_abs=dyn_sq=dyn_count=inter=union=pred_pos=gt_pos=mask_sum=0.0; first=None
    for batch in loader:
        batch=move(batch,device); pred,residual,mask=forward(base,corrector,joint,batch,k); target=batch['future_rgb']; diff=pred-target
        metrics_accumulator.update(pred,target,batch['context_rgb'][:,-1]); mask_sum+=mask.sum().item()
        last=batch['context_rgb'][:,-1:]; gt=(target-last).abs().mean(2)>motion_threshold; pm=(pred-last).abs().mean(2)>motion_threshold
        dyn3=gt[:,:,None].expand_as(diff); dyn_abs+=diff.abs()[dyn3].sum().item(); dyn_sq+=diff.square()[dyn3].sum().item(); dyn_count+=dyn3.sum().item()
        inter+=(gt&pm).sum().item(); union+=(gt|pm).sum().item(); pred_pos+=pm.sum().item(); gt_pos+=gt.sum().item()
        if first is None: first=(batch['context_rgb'][0].cpu(),target[0].cpu(),pred[0].cpu(),mask[0].cpu())
    dmse=dyn_sq/max(dyn_count,1); precision=inter/max(pred_pos,1); recall=inter/max(gt_pos,1)
    metrics=metrics_accumulator.compute();metrics.update({'dynamic_MAE':dyn_abs/max(dyn_count,1),'dynamic_MSE':float(dmse),'dynamic_PSNR':-10*math.log10(max(dmse,1e-12)),
             'motion_mask_IoU':inter/max(union,1),'motion_mask_F1':2*precision*recall/max(precision+recall,1e-12),
             'motion_mask_precision':float(precision),'motion_mask_recall':float(recall),'geometry_mask_fraction':float(mask_sum/(metrics_accumulator.count/3))})
    context,target,pred,mask=first; ids=[-1,0,1,k//2,k-1]; fig,axes=plt.subplots(3,5,figsize=(15,8))
    for col,idx in enumerate(ids):
        images=(context[-1],context[-1],torch.zeros_like(mask[0])) if idx==-1 else (target[idx],pred[idx],mask[idx])
        for row,img in enumerate(images): axes[row,col].imshow(img.permute(1,2,0).squeeze().clamp(0,1),cmap='magma' if row==2 else None); axes[row,col].axis('off')
    axes[0,0].set_ylabel('target'); axes[1,0].set_ylabel('prediction'); axes[2,0].set_ylabel('geometry mask')
    fig.tight_layout(); fig.savefig(out/'test_montage.png',dpi=140); plt.close(fig); return metrics


def main():
    a=args(); seed_all(a.seed); device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    splits=make_splits(a.cache_root,a.seed)
    train_ds=VideoForecastDataset(a.cache_root,'train',a.image_size,paths=splits['train'],cache_items=True)
    val_ds=VideoForecastDataset(a.cache_root,'val',a.image_size,paths=splits['val'],cache_items=True)
    test_ds=VideoForecastDataset(a.cache_root,'test',a.image_size,paths=splits['test'],cache_items=True)
    train=DataLoader(train_ds,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate)
    val=DataLoader(val_ds,a.batch_size,collate_fn=video_pad_collate); test=DataLoader(test_ds,1,collate_fn=video_pad_collate)
    bck=torch.load(Path(a.mixed_root)/'video/baseline_no_geometry/best.pt',map_location='cpu')
    base=GeometrySimVP(bck['context_frames'],bck['future_frames'],bck['image_size'],injection_type=bck['injection']); base.load_state_dict(bck['model']); base.to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)
    jck=torch.load(Path(a.mixed_root)/'joint_trajectory/best.pt',map_location='cpu'); joint=JointTrajectoryForecaster(jck['future_frames'],jck['hidden_dim']); joint.load_state_dict(jck['model']); joint.to(device).eval()
    for p in joint.parameters(): p.requires_grad_(False)
    k=bck['future_frames']
    if a.model_type=='motion_comp':
        corrector=GeometryMotionCompensator(sigma=a.sigma,dilation_kernel=a.dilation_kernel,
                                            motion_consistency=a.motion_consistency).to(device)
    else:
        corrector=GeometryResidualCorrector(sigma=a.sigma,dilation_kernel=a.dilation_kernel,max_residual=a.max_residual).to(device)
    if a.init_checkpoint: load_legacy_gated(corrector,a.init_checkpoint,device)
    opt=torch.optim.AdamW(corrector.parameters(),lr=a.lr,weight_decay=1e-4); best=float('inf'); history=[]
    if a.eval_only:
        corrector.load_state_dict(torch.load(out/'best.pt',map_location=device)['model'])
        metrics=evaluate(base,corrector,joint,test,device,k,out); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2)); return
    for epoch in range(1,a.epochs+1):
        corrector.set_gate_only(epoch<=a.gate_only_epochs)
        corrector.train(); total=0.0
        for batch in train:
            batch=move(batch,device); pred,residual,mask=forward(base,corrector,joint,batch,k)
            loss=loss_fn(pred,batch['future_rgb'],batch['context_rgb'][:,-1],residual,mask,a.dynamic_weight,a.residual_weight,a.gate_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(corrector.parameters(),1); opt.step(); total+=loss.item()
        vm=validation_metrics(base,corrector,joint,val,device,k); score=vm['val_MSE']+a.dynamic_select_weight*vm['val_dynamic_MAE']
        history.append({'epoch':epoch,'stage':'gate_only' if epoch<=a.gate_only_epochs else 'joint','train_loss':total/len(train),**vm,'val_score':score})
        if score<best: best=score; torch.save({'model':corrector.state_dict(),'epoch':epoch,
            'sigma':a.sigma,'dilation_kernel':a.dilation_kernel,'max_residual':a.max_residual},out/'best.pt')
        print(f"[residual] epoch={epoch:03d} stage={history[-1]['stage']} train={total/len(train):.6f} val_mse={vm['val_MSE']:.6f} val_dyn_mae={vm['val_dynamic_MAE']:.6f} score={score:.6f}",flush=True)
    (out/'history.json').write_text(json.dumps(history,indent=2)); corrector.load_state_dict(torch.load(out/'best.pt',map_location=device)['model'])
    metrics=evaluate(base,corrector,joint,test,device,k,out); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
