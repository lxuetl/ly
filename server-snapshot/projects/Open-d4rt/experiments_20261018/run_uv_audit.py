"""Frozen UV time/normalization/error audit and same-interface RGB contrasts."""
import json,time,traceback
from pathlib import Path
import numpy as np
import torch
import cv2
from common import ROOT,PROJECT,manifest,seed,dump,sha,lpips_model,metrics,pixel_bound,summarize,ci
from run_alignment_v2 import load_models,VideoForecastDataset,load,device,inputs,encode,weights,history_target,make_gradient,bounded_feature,core,selection_cost
from uv_diagnostics import uv_baselines,cached_model_parts,gather_time

DEST=ROOT/'uv_time_coordinate_audit_20261019'
NAMES=['oracle','predicted','hold','cv_model','cv_elapsed','cv_last2','pred_unclipped','pred_reanchor_gt']

def mean(x,mask=None):
    if mask is not None:x=x[mask]
    return float(x.mean()) if x.numel() else None

def selected_indices(x):
    uv=x['past_uv'][None].cuda().float();v=x['past_valid'][None].cuda().bool()
    c=torch.nan_to_num(x['past_confidence'][None].cuda().float()).clamp_min(0);c=c/(1+c)
    g=v&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
    return (g*c).sum(-1).topk(min(128,uv.shape[1]),dim=1).indices[0]

@torch.no_grad()
def uv_row(path,x,d,traj):
    args=[x[n][None].cuda() for n in ['past_xyz','past_uv','past_valid','past_confidence']]
    args=[a.float() if a.is_floating_point() else a.bool() for a in args]
    captured=[];hook=traj.uv_residual_head.register_forward_hook(lambda m,a,out:captured.append(out.detach()))
    _,model_uv=traj(*args);hook.remove()
    unbounded,residual,info=cached_model_parts(traj,captured[0],args[1],args[2])
    assert torch.allclose(unbounded.clamp(0,1),model_uv,atol=2e-6,rtol=0)
    ids=selected_indices(x);fu,fv=inputs(path,x,d)
    assert torch.allclose(model_uv[:,ids],d['pred'],atol=2e-6,rtol=0)
    bases,bi=uv_baselines(args[1],args[2],8)
    assert torch.allclose(bi['velocity'],traj.robust_uv_velocity(args[1],args[2]),atol=1e-7)
    coords={n:v[:,ids] for n,v in bases.items()}
    coords.update(oracle=fu,predicted=model_uv[:,ids],pred_unclipped=unbounded[:,ids])
    gt_history=x['history_uv_gt'].cuda()[ids][None]
    anchor_gt=gather_time(gt_history,info['index'][:,ids]);anchor=info['anchor'][:,ids]
    finite_anchor=torch.isfinite(anchor_gt).all(-1)
    shift=torch.where(finite_anchor[...,None],anchor_gt-anchor,torch.zeros_like(anchor))
    coords['pred_reanchor_gt']=unbounded[:,ids]+shift[:,:,None]
    gt_bases,gt_info=uv_baselines(gt_history,x['history_valid_gt'].cuda()[ids][None],8)
    errors={n:torch.linalg.vector_norm((p-fu)*127,dim=-1) for n,p in coords.items()}
    errors['cv_gt_history']=torch.linalg.vector_norm((gt_bases['cv_last2']-fu)*127,dim=-1)
    errors['hold_gt_history']=torch.linalg.vector_norm((gt_bases['hold']-fu)*127,dim=-1)
    age=info['age'][:,ids];delta=fu-anchor[:,:,None];amp=torch.linalg.vector_norm(delta*127,dim=-1)
    teacher=x['future_valid'].cuda()[ids][None].bool()
    agegroups={'fresh':age==0,'gap1_3':(age>=1)&(age<=3),'gap4plus':age>=4}
    motiongroups={'0_2px':amp<=2,'2_8px':(amp>2)&(amp<=8),'8pxplus':amp>8}
    allvisible=fv.all(-1)
    matrix=[]
    for j in range(8):
        matrix.append([mean(torch.linalg.vector_norm((coords['predicted'][:,:,j]-fu[:,:,k])*127,dim=-1),allvisible) for k in range(8)])
    # Shared inner six target horizons avoid unequal edge replication in lag probes.
    common=fv[:,:,0:6]&fv[:,:,1:7]&fv[:,:,2:8]
    lag={str(offset):mean(torch.linalg.vector_norm((coords['predicted'][:,:,1:7]-fu[:,:,1+offset:7+offset])*127,dim=-1),common) for offset in [-1,0,1]}
    pred_delta=coords['pred_unclipped']-anchor[:,:,None];gt_length=delta.norm(dim=-1)
    moving=fv&(amp>2)
    projection=(pred_delta*delta).sum(-1)/gt_length.square().clamp_min(1e-10)
    anchor_error=torch.linalg.vector_norm((anchor-anchor_gt)*127,dim=-1)
    pv=args[2][:,ids];pu=args[1][:,ids];invalid_hist=pv&(~torch.isfinite(pu).all(-1)|(pu<0).any(-1)|(pu>1).any(-1))
    row={'path':path,'group':d['group'],'UV_ADE_px128':{n:mean(e,fv) for n,e in errors.items()},
      'horizon_EPE_px128':{n:[mean(e[:,:,k],fv[:,:,k]) for k in range(8)] for n,e in errors.items()},
      'visible_points_per_horizon':[int(fv[:,:,k].sum()) for k in range(8)],
      'age_groups':{g:{'tracks':int(mm.sum()),'ADE':{n:mean(e,fv&mm[:,:,None]) for n,e in errors.items()}} for g,mm in agegroups.items()},
      'motion_groups':{g:{'samples':int((fv&mm).sum()),'ADE':{n:mean(e,fv&mm) for n,e in errors.items()}} for g,mm in motiongroups.items()},
      'horizon_confusion_px128':matrix,'horizon_confusion_common_tracks':int(allvisible.sum()),'lag_probe_shared_mask':lag,
      'mechanism':{'last_observation_age_mean':mean(age.float()),'last_observation_stale_fraction':mean((age>0).float()),
        'no_recent_velocity_pairs_fraction':mean((bi['recent_valid_pairs'][:,ids]==0).float()),
        'history_model_valid_but_bad_uv_fraction':mean(invalid_hist.float()),'anchor_error_px128':mean(anchor_error,finite_anchor),
        'residual_abs_px128':mean(residual[:,ids].abs()*127),'residual_worsens_fraction':mean((errors['predicted']>errors['cv_model']).float(),fv),
        'pred_clamped_fraction':mean(((unbounded[:,ids]<0)|(unbounded[:,ids]>1)).any(-1).float()),
        'pred_motion_projection_ratio_moving':mean(projection,moving),'GT_teacher_mask_disagreement':mean((teacher!=fv).float()),
        'pred_ADE_teacher_mask':mean(errors['predicted'],teacher&torch.isfinite(fu).all(-1)),
        'pred_displacement_px128':mean(pred_delta.norm(dim=-1)*127,moving),'GT_displacement_px128':mean(amp,moving)}}
    # Normalization hypotheses are diagnostics only, not evaluation-selected fixes.
    z=np.load(path);meta=json.loads(str(z['metadata_json']));assert meta['context_frames']==16 and meta['future_frames']==8
    source=np.load(meta['source_path'],allow_pickle=True);image=cv2.imdecode(np.frombuffer(source['images_jpeg_bytes'][0],np.uint8),cv2.IMREAD_COLOR);ht,wd=image.shape[:2]
    p=coords['predicted'];converted=p*p.new_tensor([(wd-1)/wd,(ht-1)/ht])
    probes={'as_is':p,'divide_by_W_H_instead_Wm1_Hm1':converted,'swap_xy':p.flip(-1),'assume_minus1_plus1':(p+1)/2,'assume_pixels128':p/127}
    row['normalization']={'source_hw':[ht,wd],'hypothesis_ADE_px128':{n:mean(torch.linalg.vector_norm((v-fu)*127,dim=-1),fv) for n,v in probes.items()},
      'W_vs_Wm1_displacement_px128':mean(torch.linalg.vector_norm((converted-p)*127,dim=-1),fv)}
    if 'tracks_uv_source' in source.files:
        qids=z['query_ids'][ids.cpu().numpy()];source_uv=torch.tensor(np.asarray(source['tracks_uv_source'][:24,qids]).transpose(1,0,2),device='cuda').float()/p.new_tensor([wd-1,ht-1])
        e=torch.linalg.vector_norm((source_uv[:,16:][None]-fu)*127,dim=-1)
        row['normalization']['projected_vs_stored_UV_px128']=mean(e,fv)
    if 'metadata_json' in source.files:
        sm=json.loads(str(source['metadata_json']));row['source_time_metadata']={k:sm[k] for k in ['source_frame_start','source_frame_end_exclusive','frame_stride','fps','source_sequence'] if k in sm}
    return coords,fv,row

def aggregate_uv(rows):
    def avg(values):
        vv=[v for v in values if v is not None];return sum(vv)/len(vv) if vv else None
    names=list(rows[0]['UV_ADE_px128'])
    return {'videos':len(rows),'UV_ADE_px128_video_equal':{n:avg([r['UV_ADE_px128'][n] for r in rows]) for n in names},
      'horizon_EPE_px128_video_equal':{n:[avg([r['horizon_EPE_px128'][n][k] for r in rows]) for k in range(8)] for n in names},
      'mechanism_video_equal':{n:avg([r['mechanism'][n] for r in rows]) for n in rows[0]['mechanism']},
      'normalization_ADE_px128':{n:avg([r['normalization']['hypothesis_ADE_px128'][n] for r in rows]) for n in rows[0]['normalization']['hypothesis_ADE_px128']},
      'lag_probe_shared_mask_video_equal':{n:avg([r['lag_probe_shared_mask'][n] for r in rows]) for n in ['-1','0','1']},
      'age_groups':{g:{'tracks':sum(r['age_groups'][g]['tracks'] for r in rows),'ADE_video_equal_nonempty':{n:avg([r['age_groups'][g]['ADE'][n] for r in rows]) for n in names}} for g in rows[0]['age_groups']},
      'motion_groups':{g:{'samples':sum(r['motion_groups'][g]['samples'] for r in rows),'ADE_video_equal_nonempty':{n:avg([r['motion_groups'][g]['ADE'][n] for r in rows]) for n in names}} for g in rows[0]['motion_groups']},
      'paired_EPE_vs_predicted':{n:ci([r['UV_ADE_px128'][n]-r['UV_ADE_px128']['predicted'] for r in rows if r['UV_ADE_px128'][n] is not None and r['UV_ADE_px128']['predicted'] is not None]) for n in names if n!='predicted'},
      'per_source':{s:{n:avg([r['UV_ADE_px128'][n] for r in rows if r['group'][0]==s]) for n in names} for s in sorted({r['group'][0] for r in rows})}}

def main():
    seed();DEST.mkdir(parents=True,exist_ok=True);m=manifest();start=time.time()
    dump(DEST/'protocol.json',{'formal_training':False,'weights_updated':False,'optimizer':False,'parameter_fitting':False,
      'methods':NAMES,'step':.002,'memory':'same recent exp((t-15)/4) historical confidence weighting as v2',
      'UV_mask':'source GT visible AND finite in-bounds, identical for every UV method; pixels in 128x128 using factor127',
      'RGB_masks':['gtvis (Oracle diagnostic)','unknown (bounds only)'],'image_budget':.002,'image_cap':.03,
      'raw_vs_clipped':'model/cv_model preserve original output clamp; elapsed/last2/unclipped preserve raw UV and reject bounds',
      'RGB_budget_matching':'per-video minimum actual L1 over all methods separately for each visibility interface, scale down only',
      'selection':'dev12 keep-or-fixed-step (.002) by previous composite; eval40 locked; nonzero fixed-step results always saved',
      'labels':'oracle future UV/GT visibility and observed-GT reanchor are explicit label-dependent diagnostics; last2/hold/CV/pred_unknown use history only',
      'paths':{'dev':m['dev'],'evaluation':m['evaluation']},'source_hashes':{n:sha(Path(__file__).with_name(n)) for n in ['run_uv_audit.py','uv_diagnostics.py','run_alignment_v2.py']}})
    net,traj=load_models();assert getattr(traj,'uses_velocity_residual',False)
    before={name:{k:v.cpu().clone() for k,v in model.state_dict().items()} for name,model in [('tau',net),('trajectory',traj)]};lp=lpips_model();selection=None
    for split in ['dev','evaluation']:
      uvrows=[];rgbrows=[];matchedrows=[];ds=VideoForecastDataset('','all',128,paths=m[split],explicit_trajectory_interface=True)
      for i,path in enumerate(m[split]):
        x=ds[i];d=device(load(path));coords,fv,ur=uv_row(path,x,d,traj);uvrows.append(ur)
        with torch.no_grad():
            enc,_=encode(net,x['context_rgb'][None].cuda());scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
            # For recent memory coords are not read: source depends on history only.
            source=history_target(enc,d,d['pred'],'recent');base=d['base'][None];h=d['hidden'];s=d['skip']
            bm=metrics(d['base'],d['target'],d['base'],lp);mets={'tau':bm};matched={'tau':bm};objective={};amounts={};gates={}
        for visname in ['gtvis','unknown']:
          predictions={}
          for name in NAMES:
            co=coords[name];ww=weights(d,co,fv if visname=='gtvis' else None,'oracle_gt' if visname=='gtvis' else 'unknown')
            gh,gs,loss=make_gradient(net,h,s,source,co,ww,scale)
            with torch.no_grad():
                pred=pixel_bound(core.decode_frozen(net,h+bounded_feature(gh,.002,normalize=True),s+bounded_feature(gs,.002,normalize=True)),base)[0]
                predictions[name]=pred;arm=name+'_'+visname
                mets[arm]=metrics(pred,d['target'],d['base'],lp);assert mets[arm]['budget']<=.002001
                assert float((pred-d['base']).abs().max())<=.030001
                fe,_=encode(net,pred[None]);from run_alignment_v2 import weighted_loss
                objective[arm]={'before':loss,'after':float(weighted_loss(fe,source,co,ww,scale)),'active_samples':int((ww>0).sum())}
          with torch.no_grad():
            amount=min(float((p-d['base']).abs().mean()) for p in predictions.values());amounts[visname]=amount
            for name,pred in predictions.items():
                q=pixel_bound(pred[None],base,budget=amount)[0];arm=name+'_'+visname
                matched[arm]=metrics(q,d['target'],d['base'],lp);assert abs(matched[arm]['budget']-amount)<2e-7
        rgbrows.append({'path':path,'group':d['group'],'metrics':mets,'objective':objective})
        matchedrows.append({'path':path,'group':d['group'],'metrics':matched,'matched_amounts':amounts})
        dump(DEST/(split+'_uv_per_video.json'),uvrows);dump(DEST/(split+'_rgb_per_video.json'),rgbrows);dump(DEST/(split+'_matched_per_video.json'),matchedrows)
        dump(DEST/'progress.json',{'stage':split,'done':i+1,'total':len(m[split]),'seconds':time.time()-start});print('UV_RGB_AUDIT',split,i+1,len(m[split]),flush=True)
      dump(DEST/(split+'_uv_summary.json'),aggregate_uv(uvrows));dump(DEST/(split+'_rgb_summary.json'),summarize(rgbrows));dump(DEST/(split+'_matched_summary.json'),summarize(matchedrows))
      if split=='dev':
        selection={a:sum(selection_cost(r['metrics'][a])-selection_cost(r['metrics']['tau']) for r in rgbrows)<0 for a in rgbrows[0]['metrics'] if a!='tau'}
        dump(DEST/'dev_selection.json',{'edit':selection,'rule':'keep or locked step .002 by dev composite, no evaluation tuning'})
      else:
        policy=[{'path':r['path'],'group':r['group'],'metrics':{'tau':r['metrics']['tau'],**{a:r['metrics'][a] if edit else r['metrics']['tau'] for a,edit in selection.items()}}} for r in rgbrows]
        dump(DEST/'evaluation_policy_per_video.json',policy);dump(DEST/'evaluation_policy_summary.json',summarize(policy))
        contrasts={vis:{n:{metric:ci([r['metrics'][n+'_'+vis][metric]-r['metrics']['predicted_'+vis][metric] for r in matchedrows]) for metric in ['mse','lpips','temporal']} for n in NAMES if n!='predicted'} for vis in ['gtvis','unknown']}
        dump(DEST/'evaluation_matched_vs_predicted.json',contrasts)
    for name,model in [('tau',net),('trajectory',traj)]:
        assert all(torch.equal(v.cpu(),before[name][k]) for k,v in model.state_dict().items());assert not any(p.grad is not None for p in model.parameters())
    dump(DEST/'COMPLETE.json',{'seconds':time.time()-start,'weights_updated':False,'frozen_tau_and_trajectory_bitwise_verified':True});print('UV_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as ex:
        dump(DEST/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc()});raise
