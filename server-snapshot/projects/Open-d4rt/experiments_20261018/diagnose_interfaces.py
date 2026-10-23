"""No optimizer: paired interface probes on the existing 12 development videos.

Future RGB/UV/visibility are explicitly label-only diagnostic interventions.
They must never be used by a deployed policy or as a training result.
"""
import json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, OUT, PROJECT, manifest, seed, key, dump, sha, pixel_bound
from run_trajectory import load_models, load, device
from trajectory_attention import TrackCrossAttention, sample, splat
from local_risk import RiskScorer
from video_forecast_dataset import VideoForecastDataset
from eval_track3d_in_worldtrack import load_worldtrack_sequence
import train_simvp_feature_warp as core

DEST=ROOT/'interface_diagnostics_20261018'

def stats(x):
    x=torch.as_tensor(x).detach().float().flatten().cpu();x=x[torch.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()), q50=float(x.median()), q90=float(torch.quantile(x,.9)), max=float(x.max())) if len(x) else dict(n=0)

def mse(a,b):return float((a-b).square().mean())

def masked_mse(a,b,mask):
    return float(((a-b).square()*mask).sum()/(mask.sum()*a.shape[2]).clamp_min(1e-9))

@torch.no_grad()
def scorer_audit(m):
    # All predictions are existing models; no threshold/weight is selected here.
    data={sp:[torch.load(OUT/'local_risk/cache'/(key(p)+'.pt'),weights_only=False) for p in m[sp]] for sp in ['fit','dev','calibration','evaluation']}
    result={}
    for version in ['local_risk','local_risk_normalized']:
      result[version]={}
      for kind in ['global','local']:
        model=RiskScorer(kind=='local',version.endswith('normalized')).cuda().eval().requires_grad_(False)
        ck=torch.load(OUT/version/(kind+'_best.pt'),weights_only=False);model.load_state_dict(ck['model'])
        margin=torch.tensor(json.loads((OUT/version/'calibration.json').read_text())[kind]['margins'])
        entry={'epoch':ck['epoch'],'margins':margin.tolist(),'splits':{}}
        cal_extremes=[]
        for split,rows in data.items():
            pp=torch.stack([model(r['global'].cuda(),r['local'].cuda().float()).cpu() for r in rows]);yy=torch.stack([r['labels'] for r in rows])
            gain=pp[:,:,0]-margin[0]>1e-4;rm=pp[:,:,1]+margin[1]<=.01;rt=pp[:,:,2]+margin[2]<=.01
            truth=(yy[:,:,0]>1e-4)&(yy[:,:,1:]<=.01).all(-1)
            calibrated=gain&rm&rt;raw=(pp[:,:,0]>1e-4)&(pp[:,:,1:]<=.01).all(-1)
            # Necessary prediction thresholds, and leave-one-constraint-out counts.
            es={'videos':len(rows),'candidates':pp.shape[0]*4,'true_safe_videos':int(truth.any(-1).sum()),
                'raw_accept_videos':int(raw.any(-1).sum()),'calibrated_accept_videos':int(calibrated.any(-1).sum()),
                'pass_gain_candidates':int(gain.sum()),'pass_mse_candidates':int(rm.sum()),'pass_temporal_candidates':int(rt.sum()),
                'pass_gain_videos':int(gain.any(-1).sum()),'pass_mse_videos':int(rm.any(-1).sum()),'pass_temporal_videos':int(rt.any(-1).sum()),
                'drop_gain_accept_videos':int((rm&rt).any(-1).sum()),'drop_mse_accept_videos':int((gain&rt).any(-1).sum()),'drop_temporal_accept_videos':int((gain&rm).any(-1).sum()),
                'prediction_required':{'gain_gt':float(margin[0]+1e-4),'mse_le':float(.01-margin[1]),'temporal_le':float(.01-margin[2])},
                'head_stats':[],'by_action':[]}
            for j in range(3):
                p=pp[:,:,j].flatten();y=yy[:,:,j].flatten();e=p-y
                corr=float(torch.corrcoef(torch.stack([p,y]))[0,1]) if p.std()>1e-9 and y.std()>1e-9 else None
                es['head_stats'].append({'prediction':stats(p),'truth':stats(y),'bias':float(e.mean()),'mae':float(e.abs().mean()),'rmse':float(e.square().mean().sqrt()),'correlation':corr})
            for j,name in enumerate(['align_rgb','align_texture','velocity_rgb','velocity_texture']):
                es['by_action'].append({'name':name,'pred_mean':pp[:,j].mean(0).tolist(),'true_mean':yy[:,j].mean(0).tolist(),'mae':(pp[:,j]-yy[:,j]).abs().mean(0).tolist(),'safe':int(truth[:,j].sum()),'pass_gain':int(gain[:,j].sum()),'pass_mse':int(rm[:,j].sum()),'pass_temporal':int(rt[:,j].sum())})
            if split=='calibration':
                residual=yy-pp;residual[:,:,0]*=-1
                maxima,which=residual.max(1);recomputed=torch.quantile(maxima,.9,dim=0,interpolation='higher').clamp_min(0)
                assert torch.allclose(margin,recomputed,atol=2e-8)
                for j in range(3):
                    order=maxima[:,j].argsort(descending=True)
                    cal_extremes.append({'head':j,'top':[{'path':rows[int(i)]['path'],'group':rows[int(i)]['group'],'residual':float(maxima[i,j]),'action':int(which[i,j]),'prediction':pp[i,which[i,j]].tolist(),'truth':yy[i,which[i,j]].tolist()} for i in order[:3]]})
                # Descriptive attribution only: do NOT deploy these ablations.
                es['residuals_per_video']=maxima.tolist()
                es['margins_candidate_pooled_q90']=torch.quantile(residual.reshape(-1,3),.9,dim=0,interpolation='higher').clamp_min(0).tolist()
            entry['splits'][split]=es
        entry['calibration_extremes']=cal_extremes;result[version][kind]=entry
        print('SCORER_AUDIT',version,kind,flush=True)
    dump(DEST/'scorer_audit.json',result)

@torch.no_grad()
def trajectory_audit(m):
    net,traj=load_models();del traj
    before={k:v.cpu().clone() for k,v in net.state_dict().items()}
    ds=VideoForecastDataset('','all',128,paths=m['dev']);rows=[]
    for i,path in enumerate(m['dev']):
        d=device(load(path));x=ds[i];z=np.load(path);meta=json.loads(str(z['metadata_json']))
        raw=load_worldtrack_sequence(Path(meta['source_path']),24);ht,wd=raw['video_rgb'].shape[1:3]
        ids0=np.asarray(z['query_ids'],np.int64)
        guv=torch.tensor(raw['tracks_uv'][:,ids0].transpose(1,0,2),device='cuda').float()/torch.tensor([wd-1,ht-1],device='cuda')
        gvis=torch.tensor(raw['visibility'][:,ids0].T,device='cuda').bool()
        # Reproduce historical selection exactly, assert cache identity/order.
        uv=x['past_uv'][None].cuda().float();valid=x['past_valid'][None].cuda().bool()
        cf=torch.nan_to_num(x['past_confidence'][None].cuda().float()).clamp_min(0);cf=cf/(1+cf)
        good=valid&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
        ids=(good.float()*cf).sum(-1).topk(min(128,uv.shape[1]),dim=1).indices[0]
        assert torch.equal(torch.nan_to_num(uv[:,ids]),d['uv'])
        guv=guv[ids][None];gvis=gvis[ids][None]
        finite=torch.isfinite(guv).all(-1);inside=finite&(guv>=0).all(-1)&(guv<=1).all(-1)
        gtgood=gvis&inside
        fu=guv[:,:,16:];fv=gtgood[:,:,16:]
        assert torch.allclose(torch.nan_to_num(fu).clamp(0,1),d['oracle'],atol=2e-6)
        context=x['context_rgb'][None].cuda();target=d['target'][None]
        enc,sk=net.backbone.enc(context.flatten(0,1));enc=enc.reshape(1,16,*enc.shape[1:])
        te,ts=net.backbone.enc(target.flatten(0,1));te=te.reshape(1,8,*te.shape[1:]);ts=ts.reshape(1,8,*ts.shape[1:])
        h=d['hidden'];s=d['skip'];base=d['base'][None]
        parity=core.decode_frozen(net,h,s);assert torch.allclose(parity,base,atol=2e-6,rtol=0)
        models={}
        for arm in ['oracle_attention','predicted_attention']:
            model=TrackCrossAttention(h.shape[2]).cuda().eval().requires_grad_(False)
            model.load_state_dict(torch.load(OUT/'trajectory'/(arm+'_best.pt'),weights_only=False)['model']);models[arm]=model
        ap=sample(enc,torch.nan_to_num(guv[:,:,:16],nan=2.,posinf=2.,neginf=-1.));gtvalid=gtgood[:,:,:16]
        variants={};details={}
        # Same existing oracle model, isolate one interface change at a time.
        for name,vals,hu,va,co,uvf,vis,unif in [
          ('original',d['appearance'],d['uv'],d['valid'],d['confidence'],d['oracle'],None,False),
          ('future_visible',d['appearance'],d['uv'],d['valid'],d['confidence'],fu,fv,False),
          ('gt_history',ap,guv[:,:,:16],gtvalid,gtvalid.float()*.5,d['oracle'],None,False),
          ('gt_history_visible',ap,guv[:,:,:16],gtvalid,gtvalid.float()*.5,fu,fv,False),
          ('uniform_same_weights',d['appearance'],d['uv'],d['valid'],d['confidence'],d['oracle'],None,True),
          ('pred_uv_same_weights',d['appearance'],d['uv'],d['valid'],d['confidence'],d['pred'],None,False)]:
            safeuv=torch.nan_to_num(hu,nan=0.,posinf=2.,neginf=-1.);uvf=torch.nan_to_num(uvf,nan=2.,posinf=2.,neginf=-1.)
            hidden,support=models['oracle_attention'](h,vals,safeuv,va,co,uvf,uniform=unif)
            delta=hidden-h
            if vis is not None:
                # Exact per-endpoint visibility masking before splat via batch-horizon vectorization.
                result=[];sup=[]
                for k in range(8):
                    vk=va&vis[:,:,k,None];ck=co*vis[:,:,k,None]
                    hh,ss=models['oracle_attention'](h[:,k:k+1],vals,safeuv,vk,ck,uvf[:,:,k:k+1],horizon_values=torch.full((1,vals.shape[1],1,1),k/7,device='cuda'))
                    result.append(hh-h[:,k:k+1]);sup.append(ss)
                delta=torch.cat(result,1);support=torch.cat(sup,1)
            pred=core.decode_frozen(net,h+delta,s)
            variants[name]=mse(pixel_bound(pred,base),target)
            details[name]={'feature_abs':float(delta.abs().mean()),'rgb_abs':float((pred-base).abs().mean()),'support_fraction':float((support>1e-6).float().mean()),'support_mean':float(support.mean())}
        # Oracle endpoint feature transport: bypass learned projection/correction.
        w=(d['confidence']*d['valid']).sum(-1)/d['valid'].sum(-1).clamp_min(1)
        v=d['valid'].float()*d['confidence'];mean=(d['appearance']*v[...,None]).sum(2)/v.sum(2).clamp_min(1e-9)[...,None]
        vg=gtvalid.float();mean_gt=(ap*vg[...,None]).sum(2)/vg.sum(2).clamp_min(1)[...,None]
        for name,values in [('history_d4rt',mean[:,:,None].expand(-1,-1,8,-1)),('history_gt',mean_gt[:,:,None].expand(-1,-1,8,-1)),('future_encoder_oracle',sample(te,torch.nan_to_num(fu,nan=2.,posinf=2.,neginf=-1.)))]:
            raster,sup=splat(values,torch.nan_to_num(fu,nan=2.,posinf=2.,neginf=-1.),w[:,:,None]*fv,*h.shape[-2:])
            unit=raster/sup.clamp_min(1e-6)
            # Same sparse support; replace rather than add unaligned encoder feature.
            replacement=h*(1-sup)+unit*sup
            pred=core.decode_frozen(net,replacement,s)
            variants['replace_'+name]=mse(pred,target)
            variants['replace_'+name+'_pixel_bounded']=mse(pixel_bound(pred,base),target)
            details['replace_'+name]={'support_fraction':float((sup>1e-6).float().mean()),'support_mean':float(sup.mean()),'feature_mse_to_true_encoder':masked_mse(unit,te,sup),'baseline_feature_mse_to_true_encoder':masked_mse(h,te,sup),'rgb_abs':float((pred-base).abs().mean())}
        ceilings={'tau':mse(base,target),'future_encoder_only':mse(core.decode_frozen(net,te,s),target),'future_skip_only':mse(core.decode_frozen(net,h,ts),target),'future_both_autoencode':mse(core.decode_frozen(net,te,ts),target),**variants}
        histmask=gtgood[:,:,:16]&d['valid'];e=torch.linalg.vector_norm((d['uv']-guv[:,:,:16])*127,dim=-1)
        fp=torch.linalg.vector_norm((d['pred']-fu)*127,dim=-1)
        # D4RT teacher visibility is not GT physical visibility: explicitly separate them.
        teacher=x['future_valid'].cuda()[ids][None].bool()
        row={'path':path,'group':d['group'],'mse':ceilings,'interventions':details,
          'interfaces':{'history_uv_error_px128':stats(e[histmask]),'query_frame0_error_px128':stats(e[:,:,0]),'pred_future_error_px128':stats(fp[fv]),'future_out_of_bounds_fraction':float((~inside[:,:,16:]).float().mean()),'future_gt_invisible_fraction':float((~gvis[:,:,16:]).float().mean()),'future_visible_inside_fraction':float(fv.float().mean()),'past_d4rt_valid_gt_invalid_fraction':float((d['valid']&~gtgood[:,:,:16]).float().sum()/d['valid'].sum().clamp_min(1)), 'future_teacher_gt_visibility_disagreement':float((teacher!=fv).float().mean()),'invalid_future_old_weight_fraction':float((w[:,:,None]*(~fv)).sum()/(w.sum()*8).clamp_min(1e-9)), 'all_tracks_dead_fraction':float((~d['valid'].any(-1)).float().mean()),'historical_confidence':stats(d['confidence'][d['valid']]),'feature_shape':list(h.shape),'skip_shape':list(s.shape)}}
        rows.append(row);dump(DEST/'trajectory_per_video.json',rows);print('TRAJECTORY_AUDIT',i+1,len(m['dev']),flush=True)
    summary={'videos':len(rows),'split':'existing development, no test/evaluation used for trajectory diagnostics','mse_video_equal':{k:float(np.mean([r['mse'][k] for r in rows])) for k in rows[0]['mse']},'interfaces_video_equal':{k:float(np.mean([r['interfaces'][k] for r in rows])) for k in rows[0]['interfaces'] if isinstance(rows[0]['interfaces'][k],float)},'frozen_backbone_bitwise_verified':all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items())}
    dump(DEST/'trajectory_summary.json',summary)

def main():
    seed();DEST.mkdir(parents=True,exist_ok=True);start=time.time();m=manifest()
    dump(DEST/'protocol.json',{'formal_training':False,'optimizer':False,'weights_updated':False,'trajectory_split':'existing dev12','scorer_splits':'existing fit64/dev12/cal12/eval40; descriptive only','future_labels':'oracle diagnostics only, not deployable','source_sha256':sha(__file__)})
    try:
        scorer_audit(m);trajectory_audit(m)
        dump(DEST/'COMPLETE.json',{'seconds':time.time()-start,'weights_updated':False});print('DIAGNOSTICS_COMPLETE',flush=True)
    except Exception as ex:
        dump(DEST/'FAILED.json',{'type':type(ex).__name__,'error':str(ex)});raise

if __name__=='__main__':main()
