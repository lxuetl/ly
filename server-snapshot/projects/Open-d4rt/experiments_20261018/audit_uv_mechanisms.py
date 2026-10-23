"""Post-hoc descriptive UV mechanisms; no fitting, no RGB policy changes."""
import json
import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment
from common import ROOT,manifest,seed,dump,sha,ci
from trajectory_model import VelocityResidualImplicitForecaster
from run_uv_audit import DEST,VideoForecastDataset,load,uv_baselines,mean

@torch.no_grad()
def main():
    seed();m=manifest();paths=m['evaluation'];ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True);rows=[]
    ck=torch.load(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt',map_location='cpu',weights_only=False)
    model=VelocityResidualImplicitForecaster(ck['future_frames'],ck['hidden_dim']).eval().requires_grad_(False);model.load_state_dict(ck['model'])
    before={k:v.clone() for k,v in model.state_dict().items()};skipped=[]
    for i,path in enumerate(paths):
        x=ds[i];d=load(path)
        # Reconstruct selected identity/order from observed UV rather than depend
        # on CPU/GPU top-k tie ordering. No future label chooses these identities.
        original=torch.nan_to_num(x['past_uv'].float())
        distance=(original[:,None]-d['uv'][0][None]).abs().flatten(2).amax(-1)
        cf=torch.nan_to_num(x['past_confidence'].float()).clamp_min(0);cf=cf/(1+cf)
        valid=x['past_valid'].bool()&torch.isfinite(x['past_uv']).all(-1)&(original>=0).all(-1)&(original<=1).all(-1)
        confidence=cf*valid
        distance=distance+(confidence[:,None]-d['confidence'][0][None]).abs().amax(-1)
        distance=distance+(valid[:,None]!=d['valid'][0][None]).float().amax(-1)
        # Distinct source points can have identical D4RT histories. Use the frozen
        # CPU trajectory prediction as another observed-input signature, then a
        # one-to-one assignment. Equal full signatures are permutation-invariant
        # for these baselines when all source tracks were selected.
        _,full_pred=model(*[x[n][None].float() if x[n].is_floating_point() else x[n][None].bool() for n in ['past_xyz','past_uv','past_valid','past_confidence']])
        distance=distance+(full_pred[0][:,None]-d['pred'][0][None]).abs().flatten(2).amax(-1)
        rr,cc=linear_sum_assignment(distance.numpy());ids=torch.empty(d['uv'].shape[1],dtype=torch.long);ids[cc]=torch.tensor(rr)
        best=distance[ids,torch.arange(len(ids))]
        if best.max()>=1e-4:
            # Do not guess an identity match. Main GPU-generated 40-video results
            # remain unchanged; this optional CPU-only descriptive subset is explicit.
            skipped.append({'path':path,'input_signature_match_error':float(best.max()),'reason':'CPU signature cannot reliably recover GPU cache track identity'})
            continue
        ambiguous=(distance<1e-5).sum(0)>1
        excluded=ambiguous if original.shape[0]>len(ids) else torch.zeros_like(ambiguous)
        fu=x['future_uv_raw'][ids][None].float();fv=x['future_visibility_gt'][ids][None].bool()
        fv=fv&torch.isfinite(fu).all(-1)&(fu>=0).all(-1)&(fu<=1).all(-1)
        fv=fv&~excluded[None,:,None]
        uv=x['past_uv'][ids][None].float();v=x['past_valid'][ids][None].bool();base,info=uv_baselines(uv,v)
        p=d['pred'];advanced=torch.cat([p[:,:,1:],p[:,:,-1:]+p[:,:,-1:]-p[:,:,-2:-1]],2).clamp(0,1)
        gh=x['history_uv_gt'][ids][None];gv=x['history_valid_gt'][ids][None]
        current=gh[:,:,-1:];current_good=gv[:,:,-1]
        mask=fv&current_good[...,None]
        amp=torch.linalg.vector_norm((fu-current)*127,dim=-1)
        coords={'predicted':p,**base,'cv_elapsed_clipped':base['cv_elapsed'].clamp(0,1),'advance1_diagnostic':advanced}
        errors={n:torch.linalg.vector_norm((c-fu)*127,dim=-1) for n,c in coords.items()}
        bins={'0_2px':amp<=2,'2_8px':(amp>2)&(amp<=8),'8pxplus':amp>8}
        # Genuine observed-boundary GT displacement, not displacement from a noisy D4RT anchor.
        pred_delta=(p-current)*127;true_delta=(fu-current)*127;moving=mask&(amp>2)
        ratio=(pred_delta*true_delta).sum(-1)/true_delta.square().sum(-1).clamp_min(1e-9)
        allgt=torch.cat([gh[:,:,-1:],fu],2);allvalid=torch.cat([gv[:,:,-1:],fv],2)
        lag={}
        for name,c in coords.items():
            common=allvalid[:,:,0:6]&allvalid[:,:,1:7]&allvalid[:,:,2:8]
            lag[name]={str(off):mean(torch.linalg.vector_norm((c[:,:,0:6]-allgt[:,:,1+off:7+off])*127,dim=-1),common) for off in [-1,0,1]}
        row={'path':path,'group':d['group'],'identity_assignment_max_error':float(best.max()),'duplicate_signatures':int(ambiguous.sum()),'excluded_ambiguous_subsampled_tracks':int(excluded.sum()),'ADE':{n:mean(e,fv) for n,e in errors.items()},
          'actual_GT_boundary_motion_bins':{b:{'samples':int((mask&mm).sum()),'ADE':{n:mean(e,mask&mm) for n,e in errors.items()}} for b,mm in bins.items()},
          'GT_boundary_motion_projection_ratio':mean(ratio,moving),'lag_same_mask':lag,
          'horizon_confusion_px128':[[mean(torch.linalg.vector_norm((p[:,:,j]-fu[:,:,k])*127,dim=-1),fv.all(-1)) for k in range(8)] for j in range(8)]}
        z=np.load(path);meta=json.loads(str(z['metadata_json']));source=np.load(meta['source_path'],allow_pickle=True)
        if 'tracks_uv_source' in source.files:
            image=cv2.imdecode(np.frombuffer(source['images_jpeg_bytes'][0],np.uint8),cv2.IMREAD_COLOR);ht,wd=image.shape[:2]
            qids=z['query_ids'][ids.cpu().numpy()]
            provided=torch.tensor(np.asarray(source['tracks_uv_source'][16:24,qids]).transpose(1,0,2))[None]/fu.new_tensor([wd-1,ht-1])
            e=torch.linalg.vector_norm((provided-fu)*127,dim=-1);ee=e[fv]
            row['projected_vs_stored_UV']={'mean':mean(ee),'median':float(ee.median()),'max':float(ee.max()),'over1px_fraction':mean((ee>1).float()),'by_horizon':[mean(e[:,:,k],fv[:,:,k]) for k in range(8)]}
        rows.append(row)
    def avg(xs):
        vals=[x for x in xs if x is not None];return sum(vals)/len(vals) if vals else None
    summary={'videos':len(rows),'requested_videos':len(paths),'skipped_identity_unresolved':skipped,'scope':'post-hoc reused validation, descriptive only; no selection or deployment change; main40 never filtered',
      'source_sha256':sha(__file__),'ADE':{n:avg([r['ADE'][n] for r in rows]) for n in coords},
      'excluded_ambiguous_subsampled_tracks':sum(r['excluded_ambiguous_subsampled_tracks'] for r in rows),
      'CPU_trajectory_frozen_verified':all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),
      'advance1_vs_original_EPE':ci([r['ADE']['advance1_diagnostic']-r['ADE']['predicted'] for r in rows if r['ADE']['predicted'] is not None]),
      'gap_correction_same_clipping_EPE':ci([r['ADE']['cv_elapsed_clipped']-r['ADE']['cv_model'] for r in rows if r['ADE']['predicted'] is not None]),
      'GT_boundary_motion_projection_ratio':avg([r['GT_boundary_motion_projection_ratio'] for r in rows]),
      'actual_GT_boundary_motion_bins':{b:{'samples':sum(r['actual_GT_boundary_motion_bins'][b]['samples'] for r in rows),'ADE_video_equal_nonempty':{n:avg([r['actual_GT_boundary_motion_bins'][b]['ADE'][n] for r in rows]) for n in coords}} for b in bins},
      'lag_same_mask':{n:{off:avg([r['lag_same_mask'][n][off] for r in rows]) for off in ['-1','0','1']} for n in coords},
      'horizon_confusion_px128_video_equal':[[avg([r['horizon_confusion_px128'][j][k] for r in rows]) for k in range(8)] for j in range(8)]}
    dump(DEST/'mechanisms_per_video.json',rows);dump(DEST/'mechanisms_summary.json',summary);print('UV_MECHANISMS_COMPLETE',flush=True)

if __name__=='__main__':main()
