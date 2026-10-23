"""Predeclared held-out, frozen counterfactual diagnostic. Never trains weights.

Previously evaluated test videos are NOT a fresh unseen confirmation set.
Oracle masks/targets are diagnostic only; all arms use identical support/budget.
"""
import argparse, gc, hashlib, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
import cv2
from common import ROOT, PROJECT, dump, sha, group, metrics, pixel_bound
from trajectory_attention import sample
from trajectory_model import VelocityResidualImplicitForecaster
from simvp_geometry_model import build_video_predictor_from_checkpoint
from uv_diagnostics import uv_baselines
from cross_objective_cpu import mse_decomposition, direction_audit
import train_simvp_feature_warp as core

OUT=Path(os.environ.get('THREE_STAGE_OUT','/root/autodl-tmp/three_stage_audit_20261020'))
CP=ROOT/'tau128_full_fair_fast_20261009/best.pt'
TP=ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt'
SPLIT=ROOT/'propagation_fusion_20261014/split_paths.json'
ARMS=['D_P','H_P','D_G','H_G','T_G','RGB_GT']

def encode(net,rgb):
    b,t,c,h,w=rgb.shape;e,s=net.backbone.enc(rgb.reshape(b*t,c,h,w))
    return e.reshape(b,t,*e.shape[1:]),s.reshape(b,t,*s.shape[1:])

def bounded_feature(delta,budget,normalize=False):
    norm=delta.abs().mean((2,3,4),keepdim=True);scale=budget/norm.clamp_min(1e-20)
    if not normalize:scale=scale.clamp(max=1)
    return (delta*scale).clamp(-.05,.05)

def avg(x,w):
    while w.ndim<x.ndim:w=w[...,None]
    return float((x*w).sum()/w.expand_as(x).sum().clamp_min(1e-12))

def memories(features,h):
    vd=sample(features,h['hu']);vg=sample(features,h['gu'])
    md=(vd*h['ns'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    mg=(vg*h['ns'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    own=(vd*h['nd'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    return vd,vg,md,mg,own

def encode_stream(net,rgb):
    parts=[encode(net,rgb[:,i:i+1]) for i in range(rgb.shape[1])]
    return tuple(torch.cat([p[j] for p in parts],1) for j in [0,1])

def decode_stream(net,h,s):
    return torch.cat([core.decode_frozen(net,h[:,i:i+1],s[:,i:i+1]) for i in range(h.shape[1])],1)

def feature_gradient_stream(net,h,s,appearance,coords,ww,scale):
    gh=torch.zeros_like(h);gs=torch.zeros_like(s);value=0.
    denominator=ww.sum().clamp_min(1e-9)
    for t in range(h.shape[1]):
        hi=h[:,t:t+1].detach().requires_grad_(True);si=s[:,t:t+1].detach().requires_grad_(True)
        rgb=core.decode_frozen(net,hi,si);fe,_=encode(net,rgb)
        diff=(sample(fe,coords[:,:,t:t+1])-appearance[:,:,t:t+1].detach())/scale
        loss=(diff.square().mean(-1)*ww[:,:,t:t+1]).sum()/denominator
        a,b=torch.autograd.grad(loss,(hi,si));gh[:,t:t+1]=-a.detach();gs[:,t:t+1]=-b.detach()
        value+=float(loss.detach())
        del hi,si,rgb,fe,diff,loss,a,b
    return gh,gs,value

def ranked(value):
    return hashlib.sha256(('three-stage-20261020:'+str(value)).encode()).hexdigest()

def decompose(m,k):
    v={a:m[a][k] for a in ARMS}
    history=.5*((v['D_P']-v['H_P'])+(v['D_G']-v['H_G']))
    endpoint=.5*((v['D_P']-v['D_G'])+(v['H_P']-v['H_G']))
    target=v['H_G']-v['T_G'];fusion=v['T_G']-v['RGB_GT']
    total=v['D_P']-v['RGB_GT']
    return dict(history_shapley=history,endpoint_shapley=endpoint,
        appearance_target_gap=target,fusion_objective_gap=fusion,
        third_stage_total=target+fusion,total_recoverable_vs_RGB_GT=total,
        identity_residual=total-history-endpoint-target-fusion,
        interaction=v['H_G']-v['H_P']-v['D_G']+v['D_P'])

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())['source_sha256']==sha(__file__)
        return
    splits=json.loads(SPLIT.read_text())['paths']
    tau=json.loads((CP.parent/'split_paths.json').read_text())['selected']
    trspec=json.loads((TP.parent/'split_manifest.json').read_text())
    # Verify trajectory trainer calls the same combined_splits implementation.
    seen={group(p) for s in ['train','val'] for p in splits[s]+tau[s]}
    by={}
    for p in sorted(splits['test'],key=ranked):
        g=group(p)
        assert g not in seen,('train/validation overlap',g)
        if g[0]!='PointOdyssey':
            # Legacy domain names differ from original-video source labels.
            legacy_name=g[1].split('-')[0] if g[0]=='ds_mini' else g[1]
            matches=[v['groups'] for v in trspec['legacy'].values() if legacy_name in v['groups']['test']]
            assert matches,('trajectory manifest does not certify held-out legacy group',g)
            assert all(legacy_name not in v['groups'][s] for v in trspec['legacy'].values() for s in ['train','val'])
        by.setdefault(g,p)
    sources={s:sorted([g for g in by if g[0]==s],key=ranked) for s in sorted({g[0] for g in by})}
    chosen=[]
    while len(chosen)<8 and any(sources.values()):
        for s in sources:
            if sources[s] and len(chosen)<8:chosen.append(sources[s].pop(0))
    cases=[dict(index=i,path=by[g],group=list(g)) for i,g in enumerate(chosen)]
    assert len(cases)==8 and len({tuple(c['group']) for c in cases})==8
    spec=dict(cases=cases,source_sha256=sha(__file__),checkpoint_sha256={'tau':sha(CP),'trajectory':sha(TP)},
        split_sha256=sha(SPLIT),selection='SHA256 fixed seed; round-robin domains; one clip per original video; no metric reads',
        independence='disjoint from canonical and TAU train/val; same trajectory combined split; previously full-test evaluated, NOT fresh unseen data',
        no_training=True,no_threshold_tuning=True,CPU_only=True,arms=ARMS,
        labels='GT history/future UV, visibility, future features and RGB are explicit counterfactuals; NOT deployable performance',
        support='same D4RT historical validity intersect GT validity and finite in-bounds D/H UV; same GT future visibility intersect P/G bounds; >=3 observations',
        budgets='fixed latent .002, latent element cap .05; RGB cap .03, mean L1 ceiling .002; scale down to minimum positive actual L1 across six arms',
        attribution='two-factor Shapley history/endpoint, signed target and fusion gaps; algebraic effects, NOT independent percentages of total TAU error',
        prefix='full16 observed context recomputed fresh, no short-prefix replay',
        resource={'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip()})
    dump(OUT/'protocol.json',spec);print('LOCKED',json.dumps(cases),flush=True)

def trajectory():
    ck=torch.load(TP,map_location='cpu',weights_only=False)
    assert ck.get('model_variant')=='velocity_residual'
    model=VelocityResidualImplicitForecaster(ck['future_frames'],ck['hidden_dim']).eval().requires_grad_(False)
    model.load_state_dict(ck['model']);return model

def load_case(c):
    from video_forecast_dataset import VideoForecastDataset
    x=VideoForecastDataset('','all',128,paths=[c['path']],explicit_trajectory_interface=True)[0]
    x['_path']=c['path'];return x

def cache_rgb(index):
    c=json.loads((OUT/'protocol.json').read_text())['cases'][index]
    dest=OUT/f'rgb_{index}.pt'
    if dest.exists():return
    x=load_case(c)
    torch.save({k:x[k] for k in ['context_rgb','future_rgb']},dest)
    dump(OUT/f'rgb_{index}_receipt.json',{'path':c['path'],'sha256':sha(dest),
        'purpose':'isolated source decoding to avoid allocator retention during TAU evaluation'})

def geometry_audit():
    from run_cpu_history_endpoint import history_data, appearance_row
    spec=json.loads((OUT/'protocol.json').read_text());traj=trajectory();rows=[]
    for c in spec['cases']:
        x=load_case(c)
        with torch.no_grad():
            h=history_data(x,traj)
            # Add explicit GT bounds shared across feature arms.
            inside=lambda u:torch.isfinite(u).all(-1)&(u>=0).all(-1)&(u<=1).all(-1)
            h['ms'] &= inside(h['hu']) & inside(h['gu'])
            tw=torch.exp((torch.arange(16)-15)/4)[None,None]
            w=h['cf']*h['ms']*tw;h['ns']=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
            rel=(h['cf']*h['ms']).sum(-1)/h['ms'].sum(-1).clamp_min(1)
            h['fv'] &= (h['ms'].sum(-1)>=3)[...,None]
            h['fw']=rel[...,None]*h['fv']
            assert bool((h['fw']>0).any()),'No shared support: record failure, never replace the sample'
            row=appearance_row(x,h,x['context_rgb'][None])
            bases,_=uv_baselines(x['past_uv'][None].float(),x['past_valid'][None].bool(),8)
            coords={n:u[:,h['ids']] for n,u in bases.items()};coords['predicted']=h['p']
            row['UV_ADE_px128']={n:avg((u-h['fu']).norm(dim=-1)*127,h['fv']) for n,u in coords.items()}
            row['horizon_EPE_px128']={n:[avg((u[:,:,t]-h['fu'][:,:,t]).norm(dim=-1)*127,h['fv'][:,:,t]) for t in range(8)] for n,u in coords.items()}
            row['historical_gt_valid_fraction']=float(h['ms'].sum()/h['md'].sum().clamp_min(1))
        result={**c,'metrics':row,'cache_indices':h['ids'].tolist(),'source_query_ids':h['query_ids']}
        dump(OUT/f'geometry_{c["index"]}.json',result)
        torch.save(h,OUT/f'support_{c["index"]}.pt')
        rows.append(result);dump(OUT/'geometry_progress.json',{'done':len(rows),'total':8})
        print('GEOMETRY',c['index'],row['UV_ADE_px128'],flush=True)
        del x,h;gc.collect()
    keys=[k for k,v in rows[0]['metrics'].items() if isinstance(v,(int,float))]
    summary={k:float(np.mean([r['metrics'][k] for r in rows])) for k in keys}
    summary['UV_ADE_px128']={k:float(np.mean([r['metrics']['UV_ADE_px128'][k] for r in rows])) for k in rows[0]['metrics']['UV_ADE_px128']}
    dump(OUT/'geometry_summary.json',{'videos':8,'video_equal':summary})

def frozen_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def latent_rgb_gradient(net,h,s,target):
    gh=torch.zeros_like(h);gs=torch.zeros_like(s)
    for t in range(h.shape[1]):
        hi=h[:,t:t+1].detach().requires_grad_(True);si=s[:,t:t+1].detach().requires_grad_(True)
        loss=(core.decode_frozen(net,hi,si)[0]-target[t:t+1]).square().mean()/h.shape[1]
        a,b=torch.autograd.grad(loss,(hi,si));gh[:,t:t+1]=a.detach();gs[:,t:t+1]=b.detach()
    return gh,gs

def case_audit(index):
    start=time.time();spec=json.loads((OUT/'protocol.json').read_text());c=spec['cases'][index]
    if (OUT/f'case_{index}.json').exists():return
    if sha(__file__)!=spec['source_sha256']:
        dump(OUT/'runtime_revision.json',dict(locked_source=spec['source_sha256'],runtime_source=sha(__file__),
            reason='isolated RGB decoding, sequential-frame encoder/decoder/gradients and separate TAU/LPIPS lifetimes; same separable objective, masks, budgets and samples',
            preserved_completed_cases=[p.name for p in sorted(OUT.glob('case_*.json'))]))
    assert sha(CP)==spec['checkpoint_sha256']['tau'] and sha(TP)==spec['checkpoint_sha256']['trajectory']
    # Source decoding has a transient full-resolution allocation. Complete it
    # before loading TAU/LPIPS under the 2 GiB instance limit.
    dump(OUT/'progress.json',dict(case=index,arm='loading_source_before_models',seconds=0))
    x=torch.load(OUT/f'rgb_{index}.pt',weights_only=False);gc.collect()
    support=torch.load(OUT/f'support_{index}.pt',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x={k:v.to(device) for k,v in x.items()}
    support={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in support.items()}
    ck=torch.load(CP,map_location='cpu',weights_only=False,mmap=True)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model'],assign=True);del ck;gc.collect()
    net=net.to(device)
    import ctypes
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    from cpu_exact_depthwise import install,channelwise
    patched=install(net) if device.type=='cpu' else 0
    # TAU's temporal large-kernel depthwise blocks create the peak workspace.
    # Four-channel execution is the same grouped convolution with smaller tiles.
    import types
    for module in (net.modules() if device.type=='cpu' else []):
        if isinstance(module,torch.nn.Conv2d) and module.groups==module.in_channels==module.out_channels and module.in_channels>=64 and module.kernel_size!=(1,1) and module.padding_mode=='zeros':
            module.forward=types.MethodType(lambda self,x:channelwise(self,x,chunk=4),module)
    before=frozen_hash(net)
    dump(OUT/'progress.json',dict(case=index,arm='encoding_observed',seconds=time.time()-start))
    with torch.no_grad():
        enc,skip=encode_stream(net,x['context_rgb'][None])
        dump(OUT/'progress.json',dict(case=index,arm='observed_encoded_before_temporal',shape=list(enc.shape),seconds=time.time()-start))
        h=net.latent_time(net.backbone.hid(enc));s=net.skip_time(skip)
        dump(OUT/'progress.json',dict(case=index,arm='temporal_done_before_decode',shape=list(h.shape),seconds=time.time()-start))
        base=decode_stream(net,h,s)
        dump(OUT/'progress.json',dict(case=index,arm='baseline_decoded_before_memory',seconds=time.time()-start))
        _,_,md,mh,_=memories(enc,support)
        future,_=encode_stream(net,x['future_rgb'][None]);truth=sample(future,support['fu'])
        scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        cfg={'D_P':(md,support['p']),'H_P':(mh,support['p']),'D_G':(md,support['fu']),
            'H_G':(mh,support['fu']),'T_G':(truth,support['fu'])}
        memory_error={'D':avg(((md-truth)/scale).square(),support['fw']),'H':avg(((mh-truth)/scale).square(),support['fw'])}
    del enc,skip,future;gc.collect()
    rg=latent_rgb_gradient(net,h,s,x['future_rgb']);outputs={};diagnostics={}
    for arm in ARMS:
        dump(OUT/'progress.json',dict(case=index,arm=arm,seconds=time.time()-start))
        if arm=='RGB_GT':gh,gs=(-rg[0],-rg[1]);loss=float((base[0]-x['future_rgb']).square().mean())
        else:gh,gs,loss=feature_gradient_stream(net,h,s,*cfg[arm],support['fw'],scale)
        with torch.no_grad():
            dh=bounded_feature(gh,.002,True);ds=bounded_feature(gs,.002,True)
            outputs[arm]=pixel_bound(decode_stream(net,h+dh,s+ds),base)[0]
            diagnostics[arm]={'loss_before':loss,**direction_audit((dh,ds),rg)}
        del gh,gs,dh,ds;gc.collect();print('ARM',index,arm,flush=True)
    # LPIPS initialization transiently holds both the pretrained state and model.
    # Never overlap that allocation with TAU on this 2 GiB CPU instance.
    assert before==frozen_hash(net) and not any(p.grad is not None for p in net.parameters())
    del net,h,s,rg,cfg,md,mh,truth;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    import ctypes
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    dump(OUT/'progress.json',dict(case=index,arm='LPIPS_after_releasing_TAU',seconds=time.time()-start))
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        raw={'tau':metrics(base[0],x['future_rgb'],base[0],lp)}
        raw.update({a:metrics(p,x['future_rgb'],base[0],lp) for a,p in outputs.items()})
        amount=min(raw[a]['budget'] for a in ARMS)
        assert amount>1e-10,'zero step: cannot claim nonzero budget matching'
        matched={'tau':raw['tau']};predictions={}
        for a,p in outputs.items():
            q=pixel_bound(p[None],base,budget=amount)[0];predictions[a]=q
            matched[a]=metrics(q,x['future_rgb'],base[0],lp)
            assert abs(matched[a]['budget']-amount)<2e-7
            assert float((q-base[0]).abs().max())<=.030001
            diagnostics[a]['MSE_identity']=mse_decomposition(q,base[0],x['future_rgb'])
        result={**c,'raw_metrics':raw,'matched_metrics':matched,'common_budget':amount,
            'matched_attribution':{k:decompose(matched,k) for k in ['mse','lpips','temporal']},
            'raw_attribution':{k:decompose(raw,k) for k in ['mse','lpips','temporal']},
            'diagnostics':diagnostics,'historical_feature_target_error':memory_error,
            'weights_bitwise_verified':True,'runtime_source_sha256':sha(__file__),'device':str(device),'channelwise_conv_modules':patched,'channel_chunk':4 if patched else None,'seconds':time.time()-start}
        dump(OUT/f'case_{index}.json',result)
        # Scientific outputs kept, no full hidden tensors or new shared-disk inodes.
        torch.save({'base':base[0].cpu(),'target':x['future_rgb'].cpu(),'matched':{a:p.cpu() for a,p in predictions.items()}},OUT/f'predictions_{index}.pt')
    print('CASE_COMPLETE',index,time.time()-start,flush=True)

def finalize():
    rows=[json.loads((OUT/f'case_{i}.json').read_text()) for i in range(8)]
    result={'videos':8,'scope':'fixed previously evaluated held-out test subset; descriptive diagnostic, not fresh confirmation'}
    for kind in ['raw_metrics','matched_metrics']:
        result[kind]={a:{k:float(np.mean([r[kind][a][k] for r in rows])) for k in ['mse','lpips','temporal','budget']} for a in ['tau']+ARMS}
    for kind in ['raw_attribution','matched_attribution']:
        result[kind]={k:{n:float(np.mean([r[kind][k][n] for r in rows])) for n in rows[0][kind][k]} for k in ['mse','lpips','temporal']}
    dump(OUT/'final_summary.json',result)
    dump(OUT/'COMPLETE.json',dict(training=False,weights_updated=False,cases=8,all_frozen_verified=all(r['weights_bitwise_verified'] for r in rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','geometry','rgb','case','finalize']);p.add_argument('--index',type=int);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(20261020);np.random.seed(20261020)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    # oneDNN backward reserves a large primitive workspace in this CPU image.
    # Native convolutions plus small depthwise channel tiles bound that workspace.
    if a.mode=='case':torch.backends.mkldnn.enabled=False
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='geometry':geometry_audit()
        elif a.mode=='rgb':cache_rgb(a.index)
        elif a.mode=='case':case_audit(a.index)
        else:finalize()
    except Exception as e:
        OUT.mkdir(exist_ok=True);dump(OUT/'FAILED.json',dict(mode=a.mode,index=a.index,error=str(e),traceback=traceback.format_exc()));raise
