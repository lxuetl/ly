"""40-video first-query audit and 4 bounded frozen TAU cases. No training."""
import argparse,gc,json,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT,manifest,dump,sha,key,metrics,pixel_bound
from run_cpu_history_endpoint import VideoForecastDataset,load,encode,sample,avg,make_gradient,bounded_feature,core,build_video_predictor_from_checkpoint
from first_query_cpu import first_query_identity
from reliability_cpu import inside

OLD=ROOT/'cpu_reliability_audit_20261019'
OUT=ROOT/'cpu_first_query_audit_20261019'
ARMS=['raw','last_anchor','identity_filter','first_mask_raw','first_local','first_forward','first_reanchor']


def nullable_avg(value,mask):
    return avg(value,mask) if bool(mask.any()) else None


def mean_valid(values):
    a=[v for v in values if v is not None];return float(np.mean(a)) if a else None


def protocol():
    return {'CPU_only':True,'training':False,'parameter_fitting':False,'threshold_search':False,
        'history_frames':16,'image_size':128,'videos':40,'final_cases':[0,1,2,3],
        'selection':'same manifest40 and first4 as previous audit; reused validation, exploratory',
        'normal_inputs':['context_rgb','query_uv','past_uv','past_valid','past_confidence'],
        'query':'provided frame0 query coordinate; no label-based re-anchoring or query reselection',
        'tracking':'adjacent LK cumulative forward from query; reverse from forward endpoint AND D4RT endpoint',
        'fixed_thresholds':{'adjacent_cycle_px':1,'global_return_px':2,'bidirectional_distance_px':2,'texture_std':.02,'local_max_radius_px':2},
        'masks':'history validity + complete query-forward-roundtrip; identity_filter additionally requires D4RT-tail reverse agrees with query and forward track',
        'limitations':['LK loop self-consistency is not proof of identity','complete-track gate abstains on occlusion/loss','no independent prefix D4RT replay','no confidence or outcome calibration'],
        'arms':ARMS,'future_endpoints':'unchanged frozen prediction for all arms except explicit first_reanchor translation by forward_last-D4RT_last',
        'step':.002,'feature_cap':.05,'image_budget':.002,'image_cap':.03,
        'budget_matching':'per-clip minimum positive actual modification, scale down only; keep stays zero, zero-support not considered matched positive comparison',
        'hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('first_query_cpu.py'),Path(__file__).with_name('reliability_cpu.py')]}}


def audit():
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol.json',protocol());paths=manifest()['evaluation'];rows=[];start=time.time()
    ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True)
    for i,path in enumerate(paths):
        x=ds[i];old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False);ids=old['ids']
        # All normal decisions are finished before consulting label tensors.
        result=first_query_identity(x['context_rgb'],x['query_uv'][ids].float(),old['uv'],old['valid'])
        torch.save(result,OUT/(key(path)+'_identity.pt'))
        uv=old['uv'];f=result['forward'];local=result['local'];m=result['memory_mask'];gt=x['history_uv_gt'][ids];gm=x['history_valid_gt'][ids]
        raw_error=(uv-gt).norm(dim=-1)*127;forward_error=(f-gt).norm(dim=-1)*127
        cm=old['valid']&gm;selected=cm&m;tail=gm[:,-1]&old['valid'][:,-1]
        available=tail&result['proxy_available'];truth=raw_error[:,-1]>2;flag=result['proxy_drift']
        a=sample(x['context_rgb'][None],uv[None]);b=sample(x['context_rgb'][None],f[None]);l=sample(x['context_rgb'][None],local[None]);g=sample(x['context_rgb'][None],gt[None])
        values={'query_to_GT0_EPE_px':nullable_avg((x['query_uv'][ids]-gt[:,0]).norm(dim=-1)*127,gm[:,0]),
            'D4RT_first_EPE_px':nullable_avg(raw_error[:,0],cm[:,0]),
            'D4RT_last_EPE_px':nullable_avg(raw_error[:,-1],tail),
            'D4RT_last_drift_fraction_gt2px':nullable_avg(truth.float(),tail),
            'first_roundtrip_track_coverage':float(m.any(-1).float().mean()),
            'history_sample_coverage':float(m.sum()/old['valid'].sum().clamp_min(1)),
            'identity_filter_history_coverage':float(result['identity_mask'].sum()/old['valid'].sum().clamp_min(1)),
            'D4RT_last_EPE_same_reliable_tracks':nullable_avg(raw_error[:,-1],tail&m[:,-1]),
            'forward_last_EPE_same_reliable_tracks':nullable_avg(forward_error[:,-1],tail&m[:,-1]),
            'raw_history_EPE_same_mask':nullable_avg(raw_error,selected),
            'local_history_EPE_same_mask':nullable_avg((local-gt).norm(dim=-1)*127,selected),
            'forward_history_EPE_same_mask':nullable_avg(forward_error,selected),
            'raw_history_RGB_MAE_same_mask':nullable_avg((a-g).abs(),selected[None]),
            'local_history_RGB_MAE_same_mask':nullable_avg((l-g).abs(),selected[None]),
            'forward_history_RGB_MAE_same_mask':nullable_avg((b-g).abs(),selected[None]),
            'D4RT_last_error_when_proxy_flagged':nullable_avg(raw_error[:,-1],available&flag),
            'D4RT_last_error_when_proxy_passed':nullable_avg(raw_error[:,-1],available&~flag)}
        counts={'tail_label_points':int(tail.sum()),'tail_label_drift':int((tail&truth).sum()),
            'proxy_evaluable':int(available.sum()),'proxy_true_positive':int((available&truth&flag).sum()),
            'proxy_false_positive':int((available&~truth&flag).sum()),'proxy_false_negative':int((available&truth&~flag).sum()),
            'proxy_true_negative':int((available&~truth&~flag).sum()),'selected_labeled_samples':int(selected.sum())}
        rows.append({'path':path,'index':i,'metrics':values,'counts':counts});dump(OUT/'history_per_video.json',rows)
        dump(OUT/'progress.json',{'stage':'history','done':i+1,'total':40,'seconds':time.time()-start});print('FIRST_QUERY_HISTORY',i+1,40,flush=True)
    count={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
    dump(OUT/'history_summary.json',{'videos':40,'means':{k:mean_valid([r['metrics'][k] for r in rows]) for k in rows[0]['metrics']},
        'nonempty_videos':{k:sum(r['metrics'][k] is not None for r in rows) for k in rows[0]['metrics']},'pooled_counts':count,
        'proxy_precision_on_evaluable':count['proxy_true_positive']/max(1,count['proxy_true_positive']+count['proxy_false_positive']),
        'proxy_recall_on_evaluable':count['proxy_true_positive']/max(1,count['proxy_true_positive']+count['proxy_false_negative'])})
    dump(OUT/'HISTORY_COMPLETE.json',{'videos':40,'seconds':time.time()-start})


def configurations(old,r):
    p=old['p'];uv=old['uv'];valid=old['valid'];mask=r['memory_mask']
    anchor_ok=mask[:,-1]&valid[:,-1]
    shift=r['forward'][:,-1]-uv[:,-1]
    return {'raw':(uv,valid,p,None),'last_anchor':(old['refined'],old['mask'],p,None),
        'identity_filter':(uv,r['identity_mask'],p,None),'first_mask_raw':(uv,mask,p,None),
        'first_local':(r['local'],mask,p,None),'first_forward':(r['forward'],mask,p,None),
        'first_reanchor':(r['forward'],mask,p+shift[None,:,None],anchor_ok)}


def run_case(index):
    start=time.time();path=manifest()['evaluation'][index]
    old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False);r=torch.load(OUT/(key(path)+'_identity.pt'),weights_only=False)
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;gc.collect()
    frozen={k:v.clone() for k,v in net.state_dict().items()}
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
    x=VideoForecastDataset('','all',128,paths=[path],explicit_trajectory_interface=True)[0];d=load(path);hh=d['hidden'];ss=d['skip']
    with torch.no_grad():
        enc,_=encode(net,x['context_rgb'][None]);scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        base=core.decode_frozen(net,hh,ss);assert (base[0]-d['base']).abs().max()<1e-3
        assert torch.allclose(x['future_rgb'],d['target'],atol=2e-6,rtol=0)
    outputs={};support={};memcfg={}
    for arm,(coords,mask,endpoints,anchor_ok) in configurations(old,r).items():
        cf=old['cf'];timew=torch.exp((torch.arange(16)-15)/4);w=cf*mask*timew;w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        with torch.no_grad():memory=(sample(enc,coords[None])*w[None,...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
        fw=(cf*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        ww=fw[:,None]*(mask.sum(-1)>=3)[:,None]*inside(endpoints[0])
        if anchor_ok is not None:ww=ww*anchor_ok[:,None]
        support[arm]={'history_samples':int(mask.sum()),'accepted_writes':int((ww>0).sum()),'possible_writes':ww.numel()}
        if not (ww>0).any():p=base[0].clone()
        else:
            gh,gs,_=make_gradient(net,hh,ss,memory,endpoints,ww[None],scale)
            with torch.no_grad():p=pixel_bound(core.decode_frozen(net,hh+bounded_feature(gh,.002,normalize=True),ss+bounded_feature(gs,.002,normalize=True)),base)[0]
            del gh,gs
        outputs[arm]=p.detach();memcfg[arm]=(coords,mask);gc.collect();print('FIRST_QUERY_CASE',index,arm,support[arm],flush=True)
    # Terminal label-only metrics, including encoder-space appearance controls.
    with torch.no_grad():
        raw={'tau':metrics(base[0],d['target'],base[0],lp)};raw.update({a:metrics(p,d['target'],base[0],lp) for a,p in outputs.items()})
        positive=[v['budget'] for a,v in raw.items() if a!='tau' and v['budget']>1e-10];amount=min(positive) if positive else 0
        matched={'tau':raw['tau']};matched_predictions={};appearance={}
        ids=old['ids'];gt=x['history_uv_gt'][ids];gv=x['history_valid_gt'][ids];truth=sample(enc,gt[None])
        for arm,p in outputs.items():
            q=pixel_bound(p[None],base,budget=amount)[0];matched_predictions[arm]=q;matched[arm]=metrics(q,d['target'],base[0],lp)
            coords,mask=memcfg[arm];cm=mask&gv;feature=sample(enc,coords[None])
            appearance[arm]=nullable_avg(((feature-truth)/scale).square(),cm[None])
        assert all(torch.equal(v,frozen[k]) for k,v in net.state_dict().items())
        assert not any(p.grad is not None for p in net.parameters())
        assert all(v['budget']<=.002001 for v in raw.values())
        assert all(float((p-base[0]).abs().max())<=.030001 for p in outputs.values())
        dump(OUT/f'case_{index}.json',{'path':path,'index':index,'raw_metrics':raw,'matched_metrics':matched,'support':support,
            'common_positive_budget':amount,'same_support_encoder_appearance_MSE':appearance,
            'TAU_bitwise_verified':True,'weights_updated':False,'seconds':time.time()-start})
        torch.save({'base':base[0],'raw':outputs,'matched':matched_predictions},OUT/f'predictions_{index}.pt')


def finalize():
    rows=[json.loads((OUT/f'case_{i}.json').read_text()) for i in range(4)]
    summary={kind:{arm:{k:float(np.mean([r[kind][arm][k] for r in rows])) for k in ['mse','lpips','temporal','budget']} for arm in ['tau']+ARMS} for kind in ['raw_metrics','matched_metrics']}
    summary['per_case']=rows;summary['encoder_appearance_MSE']={a:mean_valid([r['same_support_encoder_appearance_MSE'][a] for r in rows]) for a in ARMS}
    dump(OUT/'final_summary.json',summary)
    dump(OUT/'COMPLETE.json',{'CPU_only':True,'weights_updated':False,'TAU_bitwise_verified':True,'training':False,'history_videos':40,'final_cases':4})
    print('FIRST_QUERY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--case',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(20261019)
    try:
        if a.audit:audit()
        elif a.finalize:finalize()
        else:run_case(a.case)
    except Exception as e:
        dump(OUT/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
