"""Train backbone-agnostic geometry injection with the PointOdyssey forecaster."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from dmvfn_adapter import DMVFNVideoPredictor
from simvp_geometry_model import build_video_predictor_from_checkpoint
from geometry_residual_model import GeometryFeatureProposalCorrector,GeometryMotionCompensator
from train_all_datasets import make_splits
from train_dmvfn_ablation import motion_val,reliable_forward
from train_geometry_residual import evaluate,latest_uv,loss_fn
from train_simvp_ablation import make_geometry,move,seed_all,test_metrics
from trajectory_model import (GeometryConditionEncoder,ImplicitConditionedJointForecaster,
                              VelocityResidualImplicitForecaster)
from video_forecast_dataset import VideoForecastDataset,video_pad_collate


def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache-root',default='data/d4rt_forecasting_cache')
 p.add_argument('--po-cache',required=True);p.add_argument('--po-split-manifest',required=True)
 p.add_argument('--trajectory-checkpoint',required=True);p.add_argument('--backbone-checkpoint',required=True)
 p.add_argument('--backbone',choices=('dmvfn','simvp','tau'),default='dmvfn')
 p.add_argument('--init-corrector',default='');p.add_argument('--output-dir',required=True)
 p.add_argument('--resume',action='store_true')
 p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=2)
 p.add_argument('--min-motion',type=float,default=0.0);p.add_argument('--min-reliability',type=float,default=0.0)
 p.add_argument('--mask-power',type=float,default=1.0);p.add_argument('--dilation-kernel',type=int,default=3)
 p.add_argument('--base-motion-threshold',type=float,default=0.0)
 p.add_argument('--base-motion-ramp',type=float,default=0.0)
 p.add_argument('--flow-coherence-scale',type=float,default=0.0)
 p.add_argument('--first-step-scale',type=float,default=1.0)
 p.add_argument('--horizon-ramp-steps',type=int,default=1)
 p.add_argument('--fusion-type',choices=('rgb_gate','feature_proposal'),default='rgb_gate')
 p.add_argument('--feature-channels',type=int,default=16)
 p.add_argument('--proposal-residual-scale',type=float,default=0.08)
 p.add_argument('--benefit-gate-bias',type=float,default=-0.5)
 p.add_argument('--benefit-loss-weight',type=float,default=0.0)
 p.add_argument('--benefit-margin',type=float,default=0.002)
 p.add_argument('--benefit-negative-weight',type=float,default=1.5)
 p.add_argument('--benefit-temperature',type=float,default=0.01)
 p.add_argument('--benefit-gate-warmup-epochs',type=int,default=0)
 p.add_argument('--benefit-positive-prob-floor',type=float,default=0.55)
 p.add_argument('--benefit-recall-loss-weight',type=float,default=1.0)
 p.add_argument('--benefit-patch-kernel',type=int,default=1)
 p.add_argument('--benefit-budget-weight',type=float,default=0.0)
 p.add_argument('--benefit-topk-fraction',type=float,default=0.0)
 p.add_argument('--benefit-temporal-align-weight',type=float,default=0.0)
 p.add_argument('--gate-temporal-consistency-weight',type=float,default=0.0)
 p.add_argument('--benefit-temporal-ema-alpha',type=float,default=1.0)
 p.add_argument('--benefit-rank-margin',type=float,default=0.0)
 p.add_argument('--gate-selection-consistency-weight',type=float,default=0.0)
 p.add_argument('--correction-temporal-consistency-weight',type=float,default=0.0)
 p.add_argument('--dual-benefit-head',action='store_true')
 p.add_argument('--temporal-risk-margin',type=float,default=0.0005)
 p.add_argument('--temporal-risk-loss-weight',type=float,default=1.0)
 p.add_argument('--joint-benefit-loss-weight',type=float,default=0.5)
 p.add_argument('--temporal-risk-patch-kernel',type=int,default=1)
 p.add_argument('--temporal-risk-positive-weight',type=float,default=1.0)
 p.add_argument('--temporal-risk-prob-floor',type=float,default=0.5)
 p.add_argument('--temporal-risk-recall-loss-weight',type=float,default=0.0)
 p.add_argument('--temporal-risk-recall-floor',type=float,default=0.0)
 p.add_argument('--temporal-risk-recall-penalty',type=float,default=0.0)
 p.add_argument('--benefit-min-joint-probability',type=float,default=0.0)
 p.add_argument('--temporal-risk-max-probability',type=float,default=1.0)
 p.add_argument('--benefit-recall-floor',type=float,default=0.0)
 p.add_argument('--benefit-recall-penalty',type=float,default=0.0)
 p.add_argument('--geometry-source',choices=('predicted','history','oracle'),default='predicted')
 p.add_argument('--geometry-features',choices=('xyz_uv','uv_only'),default='xyz_uv')
 p.add_argument('--disable-reliability-filter',action='store_true')
 p.add_argument('--disable-dynamic-constraint',action='store_true')
 p.add_argument('--lpips-loss-weight',type=float,default=0.0)
 p.add_argument('--temporal-loss-weight',type=float,default=0.0)
 p.add_argument('--gradient-loss-weight',type=float,default=0.0)
 p.add_argument('--temporal-select-weight',type=float,default=0.0)
 p.add_argument('--static-identity-weight',type=float,default=0.0)
 p.add_argument('--static-select-weight',type=float,default=0.0)
 p.add_argument('--mae-select-weight',type=float,default=0.0)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda')
 p.add_argument('--legacy-subsets',default='adt_mini,ds_mini,po_mini,pstudio_mini')
 return p.parse_args()


def _last_and_velocity(values,valid,recent_pairs=4):
 """Return the latest observation and a recent constant-velocity estimate."""
 values=torch.nan_to_num(values.float());valid=valid.bool();steps=values.shape[2]
 pair=valid[:,:,1:]&valid[:,:,:-1]
 if pair.shape[2]>recent_pairs:pair[:,:,:-recent_pairs]=False
 delta=values[:,:,1:]-values[:,:,:-1]
 velocity=(delta*pair[...,None]).sum(2)/pair.sum(2).clamp_min(1)[...,None]
 last_idx=(valid.long()*torch.arange(1,steps+1,device=values.device)).argmax(2)
 last=values.gather(2,last_idx[...,None,None].expand(-1,-1,1,values.shape[-1])).squeeze(2)
 return last,velocity


def geometry_for_source(source,joint,batch,k,geometry_features='xyz_uv',
                        use_reliability=True,use_dynamic=True):
 """Build deployment, history-only, or teacher-forced geometry consistently."""
 valid=batch['past_valid'].bool();point_ok=valid.any(2)
 last_uv,_= _last_and_velocity(batch['past_uv'],valid)
 steps=torch.arange(1,k+1,device=valid.device,dtype=torch.float32).view(1,1,k,1)
 if source=='predicted':
  with torch.no_grad():
   xyz,future_uv=joint(batch['past_xyz'].float(),batch['past_uv'].float(),valid,
                       batch['past_confidence'].float())
  confidence=torch.nan_to_num(batch['past_confidence'].float()).amax(2)[:,:,None].expand(-1,-1,k).clamp(0,1)
  velocity=torch.cat((future_uv[:,:,:1]-last_uv[:,:,None],future_uv[:,:,1:]-future_uv[:,:,:-1]),2)
  acceleration=torch.zeros_like(confidence);acceleration[:,:,1:]=torch.linalg.vector_norm(velocity[:,:,1:]-velocity[:,:,:-1],dim=-1)
  consistency=torch.exp(-acceleration/.05)
  future_valid=point_ok[:,:,None].float()
 elif source=='history':
  last_xyz,xyz_velocity=_last_and_velocity(batch['past_xyz'],valid)
  _,uv_velocity=_last_and_velocity(batch['past_uv'],valid)
  xyz=last_xyz[:,:,None]+xyz_velocity[:,:,None]*steps
  future_uv=(last_uv[:,:,None]+uv_velocity[:,:,None]*steps).clamp(0,1)
  confidence=torch.nan_to_num(batch['past_confidence'].float()).amax(2)[:,:,None].expand(-1,-1,k)
  displacement=torch.linalg.vector_norm(future_uv-last_uv[:,:,None],dim=-1)
  dynamic=((displacement-.015)/.035).clamp(0,1)
  consistency=torch.ones_like(confidence);future_valid=point_ok[:,:,None].float()
 elif source=='oracle':
  xyz=torch.nan_to_num(batch['future_xyz_teacher'].float())
  future_uv=torch.nan_to_num(batch['future_uv_teacher'].float()).clamp(0,1)
  confidence=torch.nan_to_num(batch['future_confidence'].float()).clamp(0,1)
  future_valid=batch['future_valid'].bool()&torch.isfinite(batch['future_uv_teacher']).all(-1)
  displacement=torch.linalg.vector_norm(future_uv-last_uv[:,:,None],dim=-1)
  dynamic=((displacement-.015)/.035).clamp(0,1)
  consistency=torch.ones_like(confidence);future_valid=future_valid.float()
 else:raise ValueError(source)
 displacement=torch.linalg.vector_norm(future_uv-last_uv[:,:,None],dim=-1)
 dynamic=((displacement-.015)/.035).clamp(0,1)
 reliability=future_valid
 if use_reliability:reliability=reliability*confidence*consistency
 if use_dynamic:reliability=reliability*dynamic
 raw=GeometryConditionEncoder.raw_features(xyz,future_uv,confidence)
 if geometry_features=='uv_only':
  raw=raw.clone();raw[...,:3]=0
 return raw,reliability,future_uv


def source_forward(base,corrector,joint,batch,k,source,geometry_features='xyz_uv',
                   use_reliability=True,use_dynamic=True):
 with torch.no_grad():
  base_pred=base(batch['context_rgb']);geometry,weight,uv=geometry_for_source(
   source,joint,batch,k,geometry_features,use_reliability,use_dynamic)
 return corrector(base_pred,batch['context_rgb'][:,-1],geometry,uv,weight,latest_uv(batch))


def perceptual_consistency_loss(pred,target,last,lpips_model,lpips_weight,
                                temporal_weight,gradient_weight):
 loss=pred.new_zeros(());parts={}
 if lpips_weight>0:
  b,k,c,h,w=pred.shape
  value=lpips_model((pred.reshape(b*k,c,h,w)*2-1),(target.reshape(b*k,c,h,w)*2-1)).mean()
  loss=loss+lpips_weight*value;parts['lpips']=float(value.detach())
 if temporal_weight>0:
  pred_prev=torch.cat((last[:,None],pred[:,:-1]),1);target_prev=torch.cat((last[:,None],target[:,:-1]),1)
  value=((pred-pred_prev)-(target-target_prev)).abs().mean()
  loss=loss+temporal_weight*value;parts['temporal']=float(value.detach())
 if gradient_weight>0:
  px=pred[...,1:]-pred[...,:-1];tx=target[...,1:]-target[...,:-1]
  py=pred[...,1:,:]-pred[...,:-1,:];ty=target[...,1:,:]-target[...,:-1,:]
  value=.5*((px-tx).abs().mean()+(py-ty).abs().mean())
  loss=loss+gradient_weight*value;parts['gradient']=float(value.detach())
 return loss,parts


def benefit_gate_loss(corrector,target,margin=0.002,negative_weight=1.0,temperature=0.01,
                      positive_prob_floor=0.55,recall_loss_weight=1.0,
                      patch_kernel=1,budget_weight=0.0,temporal_risk_margin=0.0005,
                      temporal_risk_loss_weight=1.0,joint_benefit_loss_weight=0.5,
                      temporal_risk_patch_kernel=1,temporal_risk_positive_weight=1.0,
                      temporal_risk_prob_floor=0.5,temporal_risk_recall_loss_weight=0.0):
 """Teach the inference-time gate whether geometry beats the frozen forecast."""
 logits=getattr(corrector,'last_gate_logits',None)
 support=getattr(corrector,'last_support',None)
 proposal=getattr(corrector,'last_proposal',None);base=getattr(corrector,'last_base_prediction',None)
 if logits is None:return target.new_zeros(())
 with torch.no_grad():
  base_error=(base-target).abs().mean(2,keepdim=True)
  proposal_error=(proposal-target).abs().mean(2,keepdim=True)
  advantage=base_error-proposal_error-margin
  risk_logits=getattr(corrector,'last_temporal_risk_logits',None)
  if risk_logits is not None:
   last=getattr(corrector,'last_context')[:,None]
   base_prev=torch.cat((last,base[:,:-1]),1)
   proposal_prev=torch.cat((last,proposal[:,:-1]),1)
   target_prev=torch.cat((last,target[:,:-1]),1)
   target_change=target-target_prev
   base_temporal=((base-base_prev)-target_change).abs().mean(2,keepdim=True)
   proposal_temporal=((proposal-proposal_prev)-target_change).abs().mean(2,keepdim=True)
   temporal_advantage=base_temporal-proposal_temporal
 active=support.detach().clamp(0,1)
 if patch_kernel>1:
  kernel=int(patch_kernel)//2*2+1;pad=kernel//2
  flat_adv=(advantage*active).flatten(0,1);flat_active=active.flatten(0,1)
  region_num=torch.nn.functional.avg_pool2d(flat_adv,kernel,1,pad)
  region_den=torch.nn.functional.avg_pool2d(flat_active,kernel,1,pad)
  advantage=(region_num/region_den.clamp_min(1e-4)).reshape_as(advantage)
 if risk_logits is not None and temporal_risk_patch_kernel>1:
  risk_kernel=int(temporal_risk_patch_kernel)//2*2+1;risk_pad=risk_kernel//2
  flat_temporal=(temporal_advantage*active).flatten(0,1);flat_active=active.flatten(0,1)
  temporal_num=torch.nn.functional.avg_pool2d(flat_temporal,risk_kernel,1,risk_pad)
  temporal_den=torch.nn.functional.avg_pool2d(flat_active,risk_kernel,1,risk_pad)
  temporal_advantage=(temporal_num/temporal_den.clamp_min(1e-4)).reshape_as(temporal_advantage)
 label=(advantage>0).float()
 positive=active*label;negative=active*(1-label)
 # Normalize the two classes independently.  A pixel-dominated BCE otherwise
 # learns the safe all-closed solution even when many supported edits are useful.
 positive_loss=(torch.nn.functional.softplus(-logits)*positive).sum()/positive.sum().clamp_min(1)
 negative_loss=(torch.nn.functional.softplus(logits)*negative).sum()/negative.sum().clamp_min(1)
 loss=positive_loss+negative_weight*negative_loss
 # Hard recall is measured at p=0.5.  Keep the mean positive probability on
 # the open side of that boundary instead of using recall only for model selection.
 positive_prob=(torch.sigmoid(logits)*positive).sum()/positive.sum().clamp_min(1)
 recall_loss=torch.relu(logits.new_tensor(positive_prob_floor)-positive_prob).square()
 predicted_budget=(torch.sigmoid(logits)*active).sum()/active.sum().clamp_min(1)
 target_budget=(label*active).sum()/active.sum().clamp_min(1)
 budget_loss=(predicted_budget-target_budget.detach()).square()
 if risk_logits is not None:
  risk_label=(temporal_advantage < -temporal_risk_margin).float()
  risky=active*risk_label;safe=active*(1-risk_label)
  risk_loss=temporal_risk_positive_weight*(torch.nn.functional.softplus(-risk_logits)*risky).sum()/risky.sum().clamp_min(1)
  risk_loss=risk_loss+(torch.nn.functional.softplus(risk_logits)*safe).sum()/safe.sum().clamp_min(1)
  risk_positive_prob=(torch.sigmoid(risk_logits)*risky).sum()/risky.sum().clamp_min(1)
  risk_recall_loss=torch.relu(risk_logits.new_tensor(temporal_risk_prob_floor)-risk_positive_prob).square()
  joint_probability=(torch.sigmoid(logits)*torch.sigmoid(-risk_logits)).clamp(1e-5,1-1e-5)
  joint_logit=torch.logit(joint_probability)
  joint_label=label*(1-risk_label)
  joint_positive=active*joint_label;joint_negative=active*(1-joint_label)
  joint_loss=(torch.nn.functional.softplus(-joint_logit)*joint_positive).sum()/joint_positive.sum().clamp_min(1)
  joint_loss=joint_loss+(torch.nn.functional.softplus(joint_logit)*joint_negative).sum()/joint_negative.sum().clamp_min(1)
  loss=loss+temporal_risk_loss_weight*risk_loss+joint_benefit_loss_weight*joint_loss
  loss=loss+temporal_risk_recall_loss_weight*risk_recall_loss
 return loss+recall_loss_weight*recall_loss+budget_weight*budget_loss


@torch.no_grad()
def benefit_gate_metrics(base,corrector,joint,loader,device,k,source,geometry_features,
                         use_reliability,use_dynamic,margin,temporal_risk_margin=0.0005):
 """Measure whether the learned gate selects pixels where its proposal wins."""
 if not isinstance(corrector,GeometryFeatureProposalCorrector):return {}
 tp=predicted=positive=active_count=total=correction_consistency=batches=0.
 risk_tp=risk_predicted=risk_positive=joint_tp=joint_positive=0.
 corrector.eval()
 for batch in loader:
  batch=move(batch,device);source_forward(base,corrector,joint,batch,k,source,
   geometry_features,use_reliability,use_dynamic)
  support=corrector.last_support>.01
  gate=getattr(corrector,'last_gate_selection',None)
  gate=(gate>.5) if gate is not None else (torch.sigmoid(corrector.last_gate_logits)>.5)
  base_error=(corrector.last_base_prediction-batch['future_rgb']).abs().mean(2,keepdim=True)
  proposal_error=(corrector.last_proposal-batch['future_rgb']).abs().mean(2,keepdim=True)
  label=proposal_error+margin<base_error;selected=gate&support
  tp+=(selected&label).sum().item();predicted+=selected.sum().item()
  positive+=(label&support).sum().item();active_count+=support.sum().item();total+=support.numel()
  risk_logits=getattr(corrector,'last_temporal_risk_logits',None)
  if risk_logits is not None:
   last=batch['context_rgb'][:,-1:]
   base_prev=torch.cat((last,corrector.last_base_prediction[:,:-1]),1)
   proposal_prev=torch.cat((last,corrector.last_proposal[:,:-1]),1)
   target_prev=torch.cat((last,batch['future_rgb'][:,:-1]),1)
   target_change=batch['future_rgb']-target_prev
   base_temporal=((corrector.last_base_prediction-base_prev)-target_change).abs().mean(2,keepdim=True)
   proposal_temporal=((corrector.last_proposal-proposal_prev)-target_change).abs().mean(2,keepdim=True)
   risk=(proposal_temporal>base_temporal+temporal_risk_margin)&support
   predicted_risk=(torch.sigmoid(risk_logits)>.5)&support
   joint_label_mask=label&~risk&support
   risk_tp+=(predicted_risk&risk).sum().item();risk_predicted+=predicted_risk.sum().item()
   risk_positive+=risk.sum().item();joint_tp+=(selected&joint_label_mask).sum().item();joint_positive+=joint_label_mask.sum().item()
  value=getattr(corrector,'last_correction_temporal_consistency',None)
  if value is not None:correction_consistency+=float(value);batches+=1
 precision=tp/max(predicted,1);recall=tp/max(positive,1)
 result={'benefit_gate_precision':precision,'benefit_gate_recall':recall,
         'benefit_gate_F1':2*precision*recall/max(precision+recall,1e-12),
         'benefit_oracle_positive_fraction':positive/max(active_count,1),
         'benefit_selected_fraction':predicted/max(active_count,1),
         'benefit_support_fraction':active_count/max(total,1),
         'correction_temporal_consistency':correction_consistency/max(batches,1)}
 if risk_positive>0:
  risk_precision=risk_tp/max(risk_predicted,1);risk_recall=risk_tp/max(risk_positive,1)
  joint_precision=joint_tp/max(predicted,1);joint_recall=joint_tp/max(joint_positive,1)
  result.update({'temporal_risk_precision':risk_precision,'temporal_risk_recall':risk_recall,
   'temporal_risk_F1':2*risk_precision*risk_recall/max(risk_precision+risk_recall,1e-12),
   'temporal_risk_fraction':risk_positive/max(active_count,1),
   'joint_benefit_precision':joint_precision,'joint_benefit_recall':joint_recall,
   'joint_benefit_F1':2*joint_precision*joint_recall/max(joint_precision+joint_recall,1e-12),
   'joint_oracle_positive_fraction':joint_positive/max(active_count,1)})
 return result


@torch.no_grad()
def source_val(base,corrector,joint,loader,device,k,source,geometry_features='xyz_uv',
               use_reliability=True,use_dynamic=True,benefit_margin=0.002,
               temporal_risk_margin=0.0005):
 corrector.eval();total=count=abs_total=dyn_abs=dyn_count=static_abs=static_count=temporal_abs=temporal_count=0.
 gate_tp=gate_selected=gate_positive=gate_active=0.
 risk_tp=risk_selected=risk_positive=0.
 for batch in loader:
  batch=move(batch,device);pred,_,_=source_forward(base,corrector,joint,batch,k,source,
   geometry_features,use_reliability,use_dynamic)
  total+=(pred-batch['future_rgb']).square().sum().item();abs_total+=(pred-batch['future_rgb']).abs().sum().item();count+=batch['future_rgb'].numel()
  dynamic=((batch['future_rgb']-batch['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)>.05).expand_as(pred)
  dyn_abs+=(pred-batch['future_rgb']).abs()[dynamic].sum().item();dyn_count+=dynamic.sum().item()
  static=~dynamic;static_abs+=(pred-batch['future_rgb']).abs()[static].sum().item();static_count+=static.sum().item()
  last=batch['context_rgb'][:,-1];pp=torch.cat((last[:,None],pred[:,:-1]),1);tp=torch.cat((last[:,None],batch['future_rgb'][:,:-1]),1)
  td=(pred-pp)-(batch['future_rgb']-tp);temporal_abs+=td.abs().sum().item();temporal_count+=td.numel()
  if isinstance(corrector,GeometryFeatureProposalCorrector):
   support=corrector.last_support>.01
   gate=getattr(corrector,'last_gate_selection',None)
   gate=(gate>.5) if gate is not None else (torch.sigmoid(corrector.last_gate_logits)>.5)
   selected=gate&support
   base_error=(corrector.last_base_prediction-batch['future_rgb']).abs().mean(2,keepdim=True)
   proposal_error=(corrector.last_proposal-batch['future_rgb']).abs().mean(2,keepdim=True)
   positive=(proposal_error+benefit_margin<base_error)&support
   gate_tp+=(selected&positive).sum().item();gate_selected+=selected.sum().item()
   gate_positive+=positive.sum().item();gate_active+=support.sum().item()
   risk_logits=getattr(corrector,'last_temporal_risk_logits',None)
   if risk_logits is not None:
    base_prev=torch.cat((batch['context_rgb'][:,-1:],corrector.last_base_prediction[:,:-1]),1)
    proposal_prev=torch.cat((batch['context_rgb'][:,-1:],corrector.last_proposal[:,:-1]),1)
    target_prev=torch.cat((batch['context_rgb'][:,-1:],batch['future_rgb'][:,:-1]),1)
    target_change=batch['future_rgb']-target_prev
    base_temporal=((corrector.last_base_prediction-base_prev)-target_change).abs().mean(2,keepdim=True)
    proposal_temporal=((corrector.last_proposal-proposal_prev)-target_change).abs().mean(2,keepdim=True)
    risk=(proposal_temporal>base_temporal+temporal_risk_margin)&support
    risk_prediction=(torch.sigmoid(risk_logits)>.5)&support
    risk_tp+=(risk_prediction&risk).sum().item();risk_selected+=risk_prediction.sum().item()
    risk_positive+=risk.sum().item()
 result={'val_MSE':total/count,'val_MAE':abs_total/count,'val_dynamic_MAE':dyn_abs/max(dyn_count,1),
         'val_static_MAE':static_abs/max(static_count,1),
         'val_temporal_delta_MAE':temporal_abs/max(temporal_count,1)}
 if isinstance(corrector,GeometryFeatureProposalCorrector):
  precision=gate_tp/max(gate_selected,1);recall=gate_tp/max(gate_positive,1)
  result.update({'val_benefit_gate_precision':precision,'val_benefit_gate_recall':recall,
   'val_benefit_gate_F1':2*precision*recall/max(precision+recall,1e-12),
   'val_benefit_selected_fraction':gate_selected/max(gate_active,1)})
  if risk_positive>0:
   risk_precision=risk_tp/max(risk_selected,1);risk_recall=risk_tp/max(risk_positive,1)
   result.update({'val_temporal_risk_precision':risk_precision,
    'val_temporal_risk_recall':risk_recall,
    'val_temporal_risk_F1':2*risk_precision*risk_recall/max(risk_precision+risk_recall,1e-12)})
 return result


def combined_splits(a):
 subsets=[x.strip() for x in a.legacy_subsets.split(',') if x.strip()]
 splits,by=make_splits(a.cache_root,subsets,a.seed)
 spec=json.loads(Path(a.po_split_manifest).read_text());root=Path(a.po_cache)/'po50';counts={}
 for split in ('train','val','test'):
  usable=[]
  for name in spec['clips'][split]:
   p=root/name
   if not p.exists():raise FileNotFoundError(p)
   with np.load(p) as z:
    if np.asarray(z['future_valid']).any():usable.append(p)
  splits[split]+=usable;counts[split]=len(usable)
 return splits,{'legacy_subsets':subsets,'pointodyssey_usable':counts,
                'totals':{s:len(v) for s,v in splits.items()}}


def main():
 a=args();seed_all(a.seed);device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 use_reliability=not a.disable_reliability_filter;use_dynamic=not a.disable_dynamic_constraint
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);splits,manifest=combined_splits(a)
 (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
 mk=lambda s:VideoForecastDataset(a.cache_root,s,64,paths=splits[s],cache_items=False)
 tr,va,te=mk('train'),mk('val'),mk('test')
 train=DataLoader(tr,a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed),collate_fn=video_pad_collate,num_workers=0)
 val=DataLoader(va,a.batch_size,collate_fn=video_pad_collate,num_workers=0)
 test=DataLoader(te,1,collate_fn=video_pad_collate,num_workers=0)
 bck=torch.load(a.backbone_checkpoint,map_location='cpu');c,k=tr[0]['context_rgb'].shape[0],tr[0]['future_rgb'].shape[0]
 if a.backbone=='dmvfn':
  base=DMVFNVideoPredictor(c,k)
 else:
  base=build_video_predictor_from_checkpoint(bck)
  checkpoint_backbone=str(bck.get('backbone','simvp')).lower()
  if checkpoint_backbone!=a.backbone:
   raise ValueError(f'checkpoint backbone {checkpoint_backbone!r} does not match --backbone {a.backbone!r}')
 if int(getattr(base,'context_frames',c))!=c or int(getattr(base,'future_frames',k))!=k:
  raise ValueError(f'backbone frame contract mismatch: model={getattr(base,"context_frames",None)}->{getattr(base,"future_frames",None)}, data={c}->{k}')
 if a.backbone=='simvp':
  incompatible=base.load_state_dict(bck['model'],strict=False)
  allowed_missing={'latent_gate_logit','skip_gate_logit','latent_film.weight','latent_film.bias',
                   'skip_film.weight','skip_film.bias'}
  bad_missing=set(incompatible.missing_keys)-allowed_missing
  if bad_missing or incompatible.unexpected_keys:
   raise RuntimeError(f'incompatible SimVP checkpoint: missing={sorted(bad_missing)}, unexpected={incompatible.unexpected_keys}')
  print('simvp_legacy_missing_keys',sorted(incompatible.missing_keys),flush=True)
 else:
  base.load_state_dict(bck['model'])
 base.to(device).eval()
 for p in base.parameters():p.requires_grad_(False)
 jck=torch.load(a.trajectory_checkpoint,map_location='cpu')
 joint_cls=VelocityResidualImplicitForecaster if jck.get('model_variant')=='velocity_residual' else ImplicitConditionedJointForecaster
 joint=joint_cls(jck['future_frames'],jck['hidden_dim']);joint.load_state_dict(jck['model']);joint.to(device).eval()
 for p in joint.parameters():p.requires_grad_(False)
 lpips_model=None
 if a.lpips_loss_weight>0:
  import lpips
  lpips_model=lpips.LPIPS(net='alex',verbose=False).to(device).eval()
  for p in lpips_model.parameters():p.requires_grad_(False)
 print('split',manifest,'backbone',a.backbone,'backbone_epoch',bck.get('epoch'),
       'trajectory_epoch',jck.get('epoch'),'backbone_params',sum(p.numel() for p in base.parameters()),flush=True)
 bd=out/'no_geometry';bd.mkdir(exist_ok=True)
 torch.manual_seed(a.seed);baseline=test_metrics(base,None,None,test,'no_geometry',device,k,bd)
 (bd/'metrics.json').write_text(json.dumps(baseline,indent=2));print('baseline',baseline,flush=True)
 corrector_cls=GeometryFeatureProposalCorrector if a.fusion_type=='feature_proposal' else GeometryMotionCompensator
 extra_corrector={}
 if a.fusion_type=='feature_proposal':extra_corrector={'feature_channels':a.feature_channels,
  'proposal_residual_scale':a.proposal_residual_scale,'benefit_gate_bias':a.benefit_gate_bias,
  'benefit_topk_fraction':a.benefit_topk_fraction,
  'benefit_temporal_align_weight':a.benefit_temporal_align_weight,
  'benefit_temporal_ema_alpha':a.benefit_temporal_ema_alpha,
  'benefit_rank_margin':a.benefit_rank_margin,
   'dual_benefit_head':a.dual_benefit_head,
   'benefit_min_joint_probability':a.benefit_min_joint_probability,
   'temporal_risk_max_probability':a.temporal_risk_max_probability}
 corrector=corrector_cls(sigma=.03,dilation_kernel=a.dilation_kernel,
  min_motion=a.min_motion,min_reliability=a.min_reliability,mask_power=a.mask_power,
  base_motion_threshold=a.base_motion_threshold,base_motion_ramp=a.base_motion_ramp,
  flow_coherence_scale=a.flow_coherence_scale,first_step_scale=a.first_step_scale,
  horizon_ramp_steps=a.horizon_ramp_steps,**extra_corrector).to(device)
 if a.init_corrector:
  init=torch.load(a.init_corrector,map_location=device)
  current=corrector.state_dict();compatible={name:value for name,value in init['model'].items()
   if name in current and current[name].shape==value.shape}
  skipped=sorted(set(init['model'])-set(compatible))
  corrector.load_state_dict(compatible,strict=False)
  print('warm_start',a.init_corrector,'loaded',len(compatible),'skipped',skipped,flush=True)
 optimizer=torch.optim.AdamW(corrector.parameters(),lr=3e-4,weight_decay=1e-4)
 best=float('inf');history=[];start_epoch=1;md=out/'reliable_geometry_motion_comp';md.mkdir(exist_ok=True)
 if a.resume and (md/'last.pt').exists():
  ck=torch.load(md/'last.pt',map_location=device);corrector.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
  start_epoch=int(ck['epoch'])+1;hp=md/'history.json'
  if hp.exists():history=[x for x in json.loads(hp.read_text()) if int(x['epoch'])<start_epoch]
  best=min([float(x['score']) for x in history]+[float(ck.get('score',float('inf')))])
  print('resumed',md/'last.pt','start_epoch',start_epoch,'best',best,flush=True)
 for epoch in range(start_epoch,a.epochs+1):
  if isinstance(corrector,GeometryFeatureProposalCorrector):
   corrector.set_gate_only(epoch<=a.benefit_gate_warmup_epochs)
  corrector.train();total=0.;selection_total=0.;boundary_total=0.;correction_total=0.
  for batch in train:
   batch=move(batch,device);pred,residual,mask=source_forward(base,corrector,joint,batch,k,
    a.geometry_source,a.geometry_features,use_reliability,use_dynamic)
   loss=loss_fn(pred,batch['future_rgb'],batch['context_rgb'][:,-1],residual,mask,3.,.01,.01)
   if a.static_identity_weight>0:
    static=((batch['future_rgb']-batch['context_rgb'][:,-1:]).abs().mean(2,keepdim=True)<=.05).float()
    edit=(mask*residual).abs().mean(2,keepdim=True)
    loss=loss+a.static_identity_weight*(edit*static).sum()/static.sum().clamp_min(1)
   if a.benefit_loss_weight>0:
    loss=loss+a.benefit_loss_weight*benefit_gate_loss(
     corrector,batch['future_rgb'],a.benefit_margin,a.benefit_negative_weight,a.benefit_temperature,
     a.benefit_positive_prob_floor,a.benefit_recall_loss_weight,
     a.benefit_patch_kernel,a.benefit_budget_weight,a.temporal_risk_margin,
     a.temporal_risk_loss_weight,a.joint_benefit_loss_weight,
     a.temporal_risk_patch_kernel,a.temporal_risk_positive_weight,
     a.temporal_risk_prob_floor,a.temporal_risk_recall_loss_weight)
   if a.gate_temporal_consistency_weight>0:
    consistency=getattr(corrector,'last_temporal_gate_consistency',None)
    if consistency is not None:
     loss=loss+a.gate_temporal_consistency_weight*consistency
   if a.gate_selection_consistency_weight>0:
    selection=getattr(corrector,'last_temporal_selection_consistency',None)
    boundary=getattr(corrector,'last_rank_boundary_loss',None)
    if selection is not None:
     loss=loss+a.gate_selection_consistency_weight*selection
     selection_total+=float(selection.detach())
    if boundary is not None:
     loss=loss+a.gate_selection_consistency_weight*boundary
     boundary_total+=float(boundary.detach())
   if a.correction_temporal_consistency_weight>0:
    correction_consistency=getattr(corrector,'last_correction_temporal_consistency',None)
    if correction_consistency is not None:
     loss=loss+a.correction_temporal_consistency_weight*correction_consistency
     correction_total+=float(correction_consistency.detach())
   extra,_=perceptual_consistency_loss(pred,batch['future_rgb'],batch['context_rgb'][:,-1],
    lpips_model,a.lpips_loss_weight,a.temporal_loss_weight,a.gradient_loss_weight)
   loss=loss+extra
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.);optimizer.step();total+=loss.item()
  vm=source_val(base,corrector,joint,val,device,k,a.geometry_source,a.geometry_features,
   use_reliability,use_dynamic,a.benefit_margin,a.temporal_risk_margin);score=(vm['val_MSE']+.01*vm['val_dynamic_MAE']+
    a.temporal_select_weight*vm['val_temporal_delta_MAE']+
    a.static_select_weight*vm['val_static_MAE']+a.mae_select_weight*vm['val_MAE'])
  recall_penalty=a.benefit_recall_penalty*max(a.benefit_recall_floor-vm.get('val_benefit_gate_recall',1),0)
  risk_recall_penalty=a.temporal_risk_recall_penalty*max(
   a.temporal_risk_recall_floor-vm.get('val_temporal_risk_recall',1),0)
  score+=recall_penalty+risk_recall_penalty
  row={'epoch':epoch,'stage':'gate_warmup' if epoch<=a.benefit_gate_warmup_epochs else 'joint',
       'train_loss':total/len(train),
       'train_selection_consistency':selection_total/len(train),
       'train_rank_boundary_loss':boundary_total/len(train),
       'train_correction_temporal_consistency':correction_total/len(train),
       **vm,'recall_penalty':recall_penalty,
       'risk_recall_penalty':risk_recall_penalty,'score':score};history.append(row)
  (md/'history.json').write_text(json.dumps(history,indent=2));print('injection',row,flush=True)
  state={'model':corrector.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'score':score,
         'fusion_type':a.fusion_type,'feature_channels':a.feature_channels,
         'proposal_residual_scale':a.proposal_residual_scale,'benefit_gate_bias':a.benefit_gate_bias,
         'benefit_topk_fraction':a.benefit_topk_fraction,
         'benefit_temporal_align_weight':a.benefit_temporal_align_weight}
  state['gate_temporal_consistency_weight']=a.gate_temporal_consistency_weight
  state['benefit_temporal_ema_alpha']=a.benefit_temporal_ema_alpha
  state['benefit_rank_margin']=a.benefit_rank_margin
  state['gate_selection_consistency_weight']=a.gate_selection_consistency_weight
  state['correction_temporal_consistency_weight']=a.correction_temporal_consistency_weight
  state['dual_benefit_head']=a.dual_benefit_head
  state['temporal_risk_margin']=a.temporal_risk_margin
  state['temporal_risk_loss_weight']=a.temporal_risk_loss_weight
  state['joint_benefit_loss_weight']=a.joint_benefit_loss_weight
  state['temporal_risk_patch_kernel']=a.temporal_risk_patch_kernel
  state['temporal_risk_positive_weight']=a.temporal_risk_positive_weight
  state['temporal_risk_recall_loss_weight']=a.temporal_risk_recall_loss_weight
  state['benefit_min_joint_probability']=a.benefit_min_joint_probability
  state['temporal_risk_max_probability']=a.temporal_risk_max_probability
  torch.save(state,md/'last.pt')
  if score<best:best=score;torch.save(state,md/'best.pt')
 corrector.load_state_dict(torch.load(md/'best.pt',map_location=device)['model'])
 import train_geometry_residual as residual_module
 original=residual_module.forward
 residual_module.forward=lambda base,corrector,joint,batch,k:source_forward(
  base,corrector,joint,batch,k,a.geometry_source,a.geometry_features,use_reliability,use_dynamic)
 try:metrics=evaluate(base,corrector,joint,test,device,k,md)
 finally:residual_module.forward=original
 metrics.update(benefit_gate_metrics(base,corrector,joint,test,device,k,a.geometry_source,
  a.geometry_features,use_reliability,use_dynamic,a.benefit_margin,a.temporal_risk_margin))
 (md/'metrics.json').write_text(json.dumps(metrics,indent=2))
 summary={'no_geometry':baseline,'reliable_geometry_motion_comp':metrics,
          'meta':{**manifest,'geometry_source':a.geometry_source,
                  'backbone':a.backbone,
                  'backbone_checkpoint':str(Path(a.backbone_checkpoint)),
                  'geometry_features':a.geometry_features,
                  'reliability_filter':use_reliability,
                  'dynamic_constraint':use_dynamic,
                  'lpips_loss_weight':a.lpips_loss_weight,
                  'temporal_loss_weight':a.temporal_loss_weight,
                  'gradient_loss_weight':a.gradient_loss_weight,
                  'temporal_select_weight':a.temporal_select_weight,
                  'static_identity_weight':a.static_identity_weight,
                  'static_select_weight':a.static_select_weight,
                  'mae_select_weight':a.mae_select_weight,
                  'base_motion_threshold':a.base_motion_threshold,
                  'base_motion_ramp':a.base_motion_ramp,
                  'flow_coherence_scale':a.flow_coherence_scale,
                  'first_step_scale':a.first_step_scale,
                  'horizon_ramp_steps':a.horizon_ramp_steps,
                  'fusion_type':a.fusion_type,
                  'feature_channels':a.feature_channels,
                  'proposal_residual_scale':a.proposal_residual_scale,
                  'benefit_gate_bias':a.benefit_gate_bias,
                  'benefit_loss_weight':a.benefit_loss_weight,
                  'benefit_margin':a.benefit_margin,
                  'benefit_negative_weight':a.benefit_negative_weight,
                  'benefit_temperature':a.benefit_temperature,
                  'benefit_gate_warmup_epochs':a.benefit_gate_warmup_epochs,
                  'benefit_positive_prob_floor':a.benefit_positive_prob_floor,
                  'benefit_recall_loss_weight':a.benefit_recall_loss_weight,
                  'benefit_patch_kernel':a.benefit_patch_kernel,
                  'benefit_budget_weight':a.benefit_budget_weight,
                  'benefit_topk_fraction':a.benefit_topk_fraction,
                  'benefit_temporal_align_weight':a.benefit_temporal_align_weight,
                  'gate_temporal_consistency_weight':a.gate_temporal_consistency_weight,
                  'benefit_temporal_ema_alpha':a.benefit_temporal_ema_alpha,
                  'benefit_rank_margin':a.benefit_rank_margin,
                  'gate_selection_consistency_weight':a.gate_selection_consistency_weight,
                  'correction_temporal_consistency_weight':a.correction_temporal_consistency_weight,
                  'dual_benefit_head':a.dual_benefit_head,
                  'temporal_risk_margin':a.temporal_risk_margin,
                  'temporal_risk_loss_weight':a.temporal_risk_loss_weight,
                  'joint_benefit_loss_weight':a.joint_benefit_loss_weight,
                  'temporal_risk_patch_kernel':a.temporal_risk_patch_kernel,
                  'temporal_risk_positive_weight':a.temporal_risk_positive_weight,
                  'temporal_risk_prob_floor':a.temporal_risk_prob_floor,
                  'temporal_risk_recall_loss_weight':a.temporal_risk_recall_loss_weight,
                  'temporal_risk_recall_floor':a.temporal_risk_recall_floor,
                  'temporal_risk_recall_penalty':a.temporal_risk_recall_penalty,
                  'benefit_min_joint_probability':a.benefit_min_joint_probability,
                  'temporal_risk_max_probability':a.temporal_risk_max_probability,
                  'benefit_recall_floor':a.benefit_recall_floor,
                  'benefit_recall_penalty':a.benefit_recall_penalty}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
