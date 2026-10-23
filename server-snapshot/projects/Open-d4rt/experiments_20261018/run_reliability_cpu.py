"""Bounded frozen diagnostics; strict RGB replay vs explicitly noncausal D4RT slice."""
import argparse,gc,json,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT,manifest,dump,sha,metrics,pixel_bound,key
from run_cpu_history_endpoint import VideoForecastDataset,VelocityResidualImplicitForecaster,build_video_predictor_from_checkpoint,load,encode,sample,avg,make_gradient,bounded_feature,core
from reliability_cpu import refine_history,alternatives,strict_rgb_replay,gate_future,inside

OUT=ROOT/'cpu_reliability_audit_20261019'
ARMS=['raw','history_filter','history_resample','endpoint_gate','joint_gate']


def trajectory():
    ck=torch.load(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt',map_location='cpu',weights_only=False)
    model=VelocityResidualImplicitForecaster(ck['future_frames'],ck['hidden_dim']).eval().requires_grad_(False)
    model.load_state_dict(ck['model']);return model


@torch.no_grad()
def observed(x,traj):
    # Explicit allowlist. No history/future GT coordinate or visibility inputs.
    uv=x['past_uv'].float();valid=x['past_valid'].bool()&inside(uv)
    cf=torch.nan_to_num(x['past_confidence'].float(),nan=0.,posinf=0.,neginf=0.).clamp_min(0);cf=cf/(1+cf)
    ids=torch.argsort((cf*valid).sum(-1),descending=True,stable=True)[:128]
    _,p=traj(x['past_xyz'][None].float(),uv[None],valid[None],x['past_confidence'][None].float())
    refined,mask,info=refine_history(x['context_rgb'],uv[ids],valid[ids])
    replay=strict_rgb_replay(x['context_rgb'])
    simple=alternatives(uv[ids],valid[ids]);end=gate_future(p[0,ids],simple,replay)
    return dict(ids=ids,uv=uv[ids],valid=valid[ids],cf=cf[ids],p=p[:,ids],refined=refined,mask=mask,info=info,replay=replay,end=end)


def audit():
    OUT.mkdir(parents=True,exist_ok=True);paths=manifest()['evaluation'];traj=trajectory();rows=[];start=time.time()
    dump(OUT/'protocol.json',{'formal_training':False,'parameter_fitting':False,'CPU_only':True,
        'history':'last-observed anchor, adjacent LK cycle <=1px, local3x3 NCC improvement>.03; NCC>=.6, displacement to LK<=2px, texture std>.02',
        'endpoint_gate':'prefix RGB replay median error<=2px and prediction/simple-model disagreement<=2px; heuristic not probability',
        'threshold_selection':'fixed before outcomes; not tuned on validation or GT; no claimed calibration',
        'strict_replay':'RGB-only hold/CV; prefix0:8 isolated. Continued LK pseudo-reference on8:16.',
        'predicted_replay':'cached16 D4RT sliced to8 only NONCAUSAL sensitivity; not used in gate',
        'blocked':'strict three-arm D4RT model replay needs independent prefix D4RT XYZ/UV/confidence; unavailable',
        'final_RGB_cases':'first4 evaluation manifest clips, chosen without outcome inspection; reused validation exploratory',
        'arms':ARMS,'step':.002,'image_budget':.002,'image_cap':.03,
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('reliability_cpu.py')]}})
    ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True)
    for i,path in enumerate(paths):
        x=ds[i];h=observed(x,traj);ids=h['ids'];valid=h['valid'];m=h['mask'];uv=h['uv'];r=h['refined']
        # Label-only audit occurs AFTER all decisions have been fixed.
        gt=x['history_uv_gt'][ids];gm=x['history_valid_gt'][ids];common=valid&gm
        rgb=x['context_rgb'][None];a=sample(rgb,uv[None]);b=sample(rgb,r[None]);g=sample(rgb,gt[None])
        vals={'coverage_history':float(m.sum()/valid.sum().clamp_min(1)),
            'coverage_future':float(h['end'].float().mean()),
            'raw_EPE_px':avg((uv-gt).norm(dim=-1)*127,common),
            'raw_selected_EPE_px':avg((uv-gt).norm(dim=-1)*127,common&m),
            'resampled_selected_EPE_px':avg((r-gt).norm(dim=-1)*127,common&m),
            'raw_selected_RGB_MAE':avg((a-g).abs(),(common&m)[None]),
            'resampled_selected_RGB_MAE':avg((b-g).abs(),(common&m)[None]),
            'selected_labeled_samples':int((common&m).sum()),'moved_fraction':float((h['info']['moved']>0).float().mean())}
        # A deliberate leakage-labelled sensitivity, NEVER a causal replay claim.
        with torch.no_grad():
            _,rp=traj(x['past_xyz'][None,:,:8].float(),x['past_uv'][None,:,:8].float(),x['past_valid'][None,:,:8].bool(),x['past_confidence'][None,:,:8].float())
            base=alternatives(x['past_uv'][ids,:8].float(),x['past_valid'][ids,:8].bool())
            base['predicted']=rp[0,ids];target=gt[:,8:16];vm=gm[:,8:16]
            sensitivity={name:avg((p-target).norm(dim=-1)*127,vm) for name,p in base.items()}
        row={'path':path,'metrics':vals,'strict_RGB_replay':h['replay'],
            'NONCAUSAL_cached_D4RT_prefix_sensitivity_EPE_px':sensitivity}
        rows.append(row);torch.save(h,OUT/(key(path)+'_observed.pt'));dump(OUT/'history_replay_per_video.json',rows)
        dump(OUT/'progress.json',{'stage':'history_replay','done':i+1,'total':len(paths),'seconds':time.time()-start})
        print('HISTORY_REPLAY',i+1,len(paths),flush=True)
    dump(OUT/'history_replay_summary.json',{'videos':len(rows),
        'means':{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
        'strict_RGB_replay':{name:float(np.mean([r['strict_RGB_replay']['mean_error_px'][name] for r in rows if r['strict_RGB_replay']['available']])) for name in ['hold','cv']},
        'NONCAUSAL_sensitivity':{name:float(np.mean([r['NONCAUSAL_cached_D4RT_prefix_sensitivity_EPE_px'][name] for r in rows])) for name in ['hold','cv','predicted']}})
    dump(OUT/'HISTORY_COMPLETE.json',{'videos':len(rows),'seconds':time.time()-start})


def run_case(index):
    start=time.time();path=manifest()['evaluation'][index];h=torch.load(OUT/(key(path)+'_observed.pt'),weights_only=False)
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;gc.collect()
    frozen={k:v.clone() for k,v in net.state_dict().items()}
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
    x=VideoForecastDataset('','all',128,paths=[path])[0];d=load(path);hh=d['hidden'];ss=d['skip']
    with torch.no_grad():
        enc,_=encode(net,x['context_rgb'][None]);scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        base=core.decode_frozen(net,hh,ss);assert (base[0]-d['base']).abs().max()<1e-3
        assert torch.allclose(x['future_rgb'],d['target'],atol=2e-6,rtol=0)
    outputs={};support={}
    for arm in ARMS:
        filtered=arm in ['history_filter','history_resample','joint_gate']
        coords=h['refined'] if arm in ['history_resample','joint_gate'] else h['uv']
        mask=h['mask'] if filtered else h['valid'];cf=h['cf'];timew=torch.exp((torch.arange(16)-15)/4)
        w=cf*mask*timew;w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        with torch.no_grad():memory=(sample(enc,coords[None])*w[None,...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
        trackok=mask.sum(-1)>=3
        fw=(cf*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        ww=fw[:,None]*trackok[:,None]*inside(h['p'][0])
        if arm in ['endpoint_gate','joint_gate']:ww=ww*h['end']
        support[arm]={'accepted_writes':int((ww>0).sum()),'possible_writes':ww.numel(),'history_samples':int(mask.sum())}
        if not (ww>0).any():p=base[0].clone()
        else:
            gh,gs,_=make_gradient(net,hh,ss,memory,h['p'],ww[None],scale)
            with torch.no_grad():p=pixel_bound(core.decode_frozen(net,hh+bounded_feature(gh,.002,normalize=True),ss+bounded_feature(gs,.002,normalize=True)),base)[0]
            del gh,gs
        outputs[arm]=p.detach();gc.collect();print('FROZEN_CASE',index,arm,support[arm],flush=True)
    # All inference decisions finished before future labels are evaluated.
    with torch.no_grad():
        raw={'tau':metrics(base[0],d['target'],base[0],lp)}
        raw.update({a:metrics(p,d['target'],base[0],lp) for a,p in outputs.items()})
        positive=[v['budget'] for a,v in raw.items() if a!='tau' and v['budget']>1e-10]
        amount=min(positive) if positive else 0
        matched={'tau':raw['tau']};matched_outputs={}
        for arm,p in outputs.items():
            q=pixel_bound(p[None],base,budget=amount)[0];matched_outputs[arm]=q
            matched[arm]=metrics(q,d['target'],base[0],lp)
        assert all(torch.equal(v,frozen[k]) for k,v in net.state_dict().items())
        assert not any(p.grad is not None for p in net.parameters())
        assert all(v['budget']<=.002001 for v in raw.values())
        assert all(float((p-base[0]).abs().max())<=.030001 for p in outputs.values())
        dump(OUT/f'case_{index}.json',{'path':path,'index':index,'raw_metrics':raw,'matched_metrics':matched,
            'common_positive_budget':amount,'zero_support_keeps_not_amplified':True,'support':support,
            'weights_updated':False,'TAU_bitwise_verified':True,'seconds':time.time()-start})
        torch.save({'base':base[0],'raw':outputs,'matched':matched_outputs},OUT/f'predictions_{index}.pt')


def finalize():
    rows=[json.loads((OUT/f'case_{i}.json').read_text()) for i in range(4)]
    summary={kind:{arm:{metric:float(np.mean([r[kind][arm][metric] for r in rows])) for metric in ['mse','lpips','temporal','budget']} for arm in ['tau']+ARMS} for kind in ['raw_metrics','matched_metrics']}
    summary['per_case']=rows;summary['cases']=4;summary['strict_three_arm_replay_complete']=False
    dump(OUT/'final_summary.json',summary)
    dump(OUT/'COMPLETE.json',{'training':False,'weights_updated':False,'CPU_only':True,'cases':4,
        'strict_three_arm_replay':'BLOCKED: independent D4RT prefix cache absent; no leakage workaround used',
        'scope':'history/replay audit40; frozen RGB4 reused validation; exploratory not confirmatory'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--case',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(20261019)
    try:
        if a.audit:audit()
        elif a.finalize:finalize()
        else:run_case(a.case)
    except Exception as e:
        dump(OUT/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
