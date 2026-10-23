"""Label-dependent frozen CPU factorial: appearance sampling x future endpoint.

All masks/weights and actual image budgets shared. Oracle controls are diagnostics,
not inference procedures, model training, calibration or deployment claims.
"""
import argparse,gc,json,time,traceback
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
from common import ROOT,manifest,key,dump,sha,metrics,pixel_bound
from run_cpu_history_endpoint import VideoForecastDataset,load,encode,sample,make_gradient,bounded_feature,weighted_loss,core,build_video_predictor_from_checkpoint
from cross_objective_cpu import MAIN,ARMS,common_support,factor_effects,direction_audit,mse_decomposition

OUT=ROOT/'cpu_first_query_cross_objective_20261019'
OLD=ROOT/'cpu_reliability_audit_20261019'
FIRST=ROOT/'cpu_first_query_audit_20261019'


def protocol():
    return {'CPU_only':True,'training':False,'parameter_fitting':False,'threshold_search':False,
        'cases':[0,1,2,3],'selection':'same manifest first4, no outcome selection; reused validation exploratory',
        'factorial':'D=original D4RT sampling; L=first-query local<=2px; F=first-query full forward; P=predicted future UV; G=raw GT future UV',
        'main_arms':MAIN,'all_arms':ARMS,
        'controls':{'H_G':'GT historical sampling, historical time/confidence pooling, Oracle endpoint',
            'T_G':'actual future encoder features at Oracle endpoints; label-only target',
            'RGB_GT':'full future RGB MSE descent through same frozen decoder; feasibility/direction reference, NOT fair deployable competitor'},
        'mask':'identical first-query mask intersect GT historical visibility/bounds and bounds of all4 historical coordinates; future GT visibility/bounds and predicted bounds intersect for ALL feature arms',
        'labels':'whole experiment is label-dependent diagnostic, including common masks. No claim of deployable P branch with GT masks.',
        'weights':'same historical confidence/recency weights and same future reliability; at least3 shared history observations',
        'budgets':'one latent step .002 normalized per horizon, element cap .05; image mean L1 ceiling .002, cap .03; match minimum positive actual RGB L1 across ALL9 arms by scaling down only',
        'MSE_bias_test':'feature objective before/after; actual latent-step cosine with true RGB descent; exact pixel deltaMSE=linear+quadratic after image budget matching',
        'no_line_search':True,'no_oracle_based_hyperparameter_selection':True,
        'hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('cross_objective_cpu.py'),Path(__file__).with_name('run_alignment_v2.py')]}}


def rgb_gradient(net,h,s,target):
    hi=h.detach().requires_grad_(True);si=s.detach().requires_grad_(True)
    y=core.decode_frozen(net,hi,si);loss=F.mse_loss(y,target[None])
    g=torch.autograd.grad(loss,(hi,si));return tuple(v.detach() for v in g)


def run_case(index):
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol.json',protocol());start=time.time()
    path=manifest()['evaluation'][index];old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False)
    identity=torch.load(FIRST/(key(path)+'_identity.pt'),weights_only=False)
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;gc.collect()
    frozen={k:v.clone() for k,v in net.state_dict().items()}
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
    x=VideoForecastDataset('','all',128,paths=[path],explicit_trajectory_interface=True)[0];d=load(path);h=d['hidden'];s=d['skip'];ids=old['ids']
    with torch.no_grad():
        context,_=encode(net,x['context_rgb'][None]);true_feature,_=encode(net,x['future_rgb'][None])
        scale=context.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        base=core.decode_frozen(net,h,s);baseline_features,_=encode(net,base)
        assert (base[0]-d['base']).abs().max()<1e-3
        assert torch.allclose(x['future_rgb'],d['target'],atol=2e-6,rtol=0)
        gu=x['history_uv_gt'][ids];g=x['future_uv_raw'][ids][None]
        mask,w,ww=common_support(old,identity,gu,x['history_valid_gt'][ids],g[0],x['future_visibility_gt'][ids]);ww=ww[None]
        assert bool((ww>0).any()),'No shared diagnostic support; do not silently drop or force a nonzero update.'
        memory={}
        for name,uv in [('D',old['uv']),('L',identity['local']),('F',identity['forward']),('H',gu)]:
            memory[name]=(sample(context,uv[None])*w[None,...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1).detach()
        truth=sample(true_feature,g).detach()
        cfg={a:(memory[a[0]],old['p'] if a.endswith('_P') else g) for a in MAIN}
        cfg.update({'H_G':(memory['H'],g),'T_G':(truth,g)})
        target_mismatch={name:float((((mem-truth)/scale).square().mean(-1)*ww).sum()/ww.sum().clamp_min(1e-9)) for name,mem in memory.items()}
        base_true_feature_loss=float(weighted_loss(baseline_features,truth,g,ww,scale))
    rg=rgb_gradient(net,h,s,d['target']);outputs={};diagnostics={}
    for arm in ARMS:
        if arm=='RGB_GT':gh,gs=(-rg[0],-rg[1]);before=float(F.mse_loss(base[0],d['target']))
        else:
            mem,coords=cfg[arm];gh,gs,before=make_gradient(net,h,s,mem,coords,ww,scale)
        with torch.no_grad():
            dh=bounded_feature(gh,.002,normalize=True);ds=bounded_feature(gs,.002,normalize=True)
            direction=direction_audit((dh,ds),rg)
            p=pixel_bound(core.decode_frozen(net,h+dh,s+ds),base)[0]
            assert float((p-base[0]).abs().mean())<=.002001 and float((p-base[0]).abs().max())<=.030001
            outputs[arm]=p.detach();diagnostics[arm]={'own_objective_before':before,**direction}
        del gh,gs,dh,ds;gc.collect()
        dump(OUT/'progress.json',{'case':index,'stage':arm,'seconds':time.time()-start})
        print('CROSS_OBJECTIVE',index,arm,flush=True)
    with torch.no_grad():
        raw={'tau':metrics(base[0],d['target'],base[0],lp)}
        raw.update({a:metrics(p,d['target'],base[0],lp) for a,p in outputs.items()})
        amounts=[raw[a]['budget'] for a in ARMS];assert all(v>1e-10 for v in amounts),'Zero direction: keep must be explicit, do not manufacture a matched nonzero effect'
        amount=min(amounts);matched={'tau':raw['tau']};matched_predictions={}
        for arm,p in outputs.items():
            q=pixel_bound(p[None],base,budget=amount)[0];matched_predictions[arm]=q;matched[arm]=metrics(q,d['target'],base[0],lp)
            assert abs(matched[arm]['budget']-amount)<2e-7
            for kind,pred in [('raw',p),('matched',q)]:
                features,_=encode(net,pred[None])
                own=float(F.mse_loss(pred,d['target'])) if arm=='RGB_GT' else float(weighted_loss(features,*cfg[arm],ww,scale))
                diagnostics[arm][kind]={'own_objective_after':own,
                    'true_future_feature_loss_GT':float(weighted_loss(features,truth,g,ww,scale)),
                    'pixel_MSE_decomposition':mse_decomposition(pred,base[0],d['target'])}
        # Target substitution is label-only: no target, gradient or metric selects an action.
        assert all(torch.equal(v,frozen[k]) for k,v in net.state_dict().items())
        assert not any(p.grad is not None for p in net.parameters())
        support={'history_samples':int(mask.sum()),'observed_first_mask_samples':int(identity['memory_mask'].sum()),
            'future_writes':int((ww>0).sum()),'future_weight_sum':float(ww.sum()),
            'historical_weight_sum':float(w.sum()),'same_support_for_all_feature_arms':True}
        pairs={'L_at_P':('L_P','D_P'),'L_at_G':('L_G','D_G'),
            'F_at_P':('F_P','D_P'),'F_at_G':('F_G','D_G'),
            'endpoint_D':('D_G','D_P'),'endpoint_L':('L_G','L_P'),'endpoint_F':('F_G','F_P'),
            'GT_history_vs_local_at_G':('H_G','L_G'),'true_future_vs_GT_history_at_G':('T_G','H_G')}
        pairwise={name:mse_decomposition(matched_predictions[a],matched_predictions[b],d['target']) for name,(a,b) in pairs.items()}
        result={'index':index,'path':path,'cache_indices':ids.tolist(),'raw_metrics':raw,'matched_metrics':matched,
            'effects_raw':factor_effects(raw),'effects_matched':factor_effects(matched),'diagnostics':diagnostics,
            'history_target_vs_true_future_MSE':target_mismatch,'baseline_true_future_feature_loss_GT':base_true_feature_loss,
            'common_positive_budget':amount,'support':support,'pairwise_matched_MSE_decomposition':pairwise,
            'TAU_bitwise_verified':True,'weights_updated':False,'seconds':time.time()-start}
        dump(OUT/f'case_{index}.json',result)
        torch.save({'base':base[0],'target':d['target'],'raw':outputs,'matched':matched_predictions,
            'history_mask':mask,'history_weights':w,'future_weights':ww,'predicted_uv':old['p'],'oracle_uv':g},OUT/f'predictions_{index}.pt')


def finalize():
    rows=[json.loads((OUT/f'case_{i}.json').read_text()) for i in range(4)]
    result={kind:{a:{k:float(np.mean([r[kind][a][k] for r in rows])) for k in ['mse','lpips','temporal','budget']} for a in ['tau']+ARMS} for kind in ['raw_metrics','matched_metrics']}
    result['effects_matched']={effect:{k:float(np.mean([r['effects_matched'][effect][k] for r in rows])) for k in ['mse','lpips','temporal']} for effect in rows[0]['effects_matched']}
    result['per_case']=rows;dump(OUT/'final_summary.json',result)
    dump(OUT/'COMPLETE.json',{'CPU_only':True,'training':False,'weights_updated':False,'TAU_bitwise_verified':True,'cases':4,'arms_per_case':9,'label_dependent_diagnostic':True})
    print('CROSS_OBJECTIVE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261019)
    try:
        if a.finalize:finalize()
        else:run_case(a.case)
    except Exception as e:
        dump(OUT/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
