"""Two-phase future-blind replay calibration; final labels opened only in evaluate.

No geometry GT, full-context D4RT cache, or old TAU latent is read. RGB-prefix
8->4 calibrates,12->4 rejects/accepts,16->4 runs; horizons5..8 always keep.
"""
import argparse,ctypes,gc,hashlib,json,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT,manifest,dump,sha,metrics,pixel_bound
from run_alignment_v2 import encode,make_gradient,bounded_feature
from trajectory_attention import sample
from simvp_geometry_model import build_video_predictor_from_checkpoint
import train_simvp_feature_warp as core
from eval_track3d_in_worldtrack import _decode_jpeg_rgb
from replay_policy_cpu import ALPHAS,MOTIONS,digest,padded_prefix,key,improves,select,bounded_output,track_prefix
from cpu_exact_depthwise import install as install_channelwise

OUT=ROOT/'cpu_causal_replay_policy_20261019'
CASES=list(range(6))


def protocol():
    return {'CPU_only':True,'training':False,'weight_updates':False,'cases':CASES,
        'selection':'manifest first6, order-only; previous4 plus2 additional; reused validation exploratory',
        'information_barrier':'ALL6 generation decision files must exist before evaluate may decode frames16:24',
        'calibration':'prefix0:8 -> observed8:12; choose smallest passing alpha, then LPIPS and MSE',
        'validation':'prefix0:12 -> observed12:16; only accept/reject fixed calibration choice, never reselect',
        'final':'prefix0:16 ->16:20 corrected; horizons5..8 remain TAU unchanged',
        'baseline':'TAU recomputed separately from each prefix, left-pad frame0 to16 if necessary',
        'low_memory_runtime':'independent process per8/12/16 stage; LPIPS loaded only after TAU freed; framewise encoder; exact channelwise DWconv32; byte-hash not clone; reclaim unused CPU arenas',
        'distribution_shift':'short prefixes use padding, final16 does not. Replay safety does NOT guarantee final safety.',
        'geometry':'RGB frame0 corners64 + independent adjacent LK per prefix; hold/robust-CV endpoints',
        'D4RT_status':'strict prefix XYZ cache absent; original learned D4RT trajectory route is NOT calibrated or enabled by this fallback',
        'reliability':'observed-prefix inner4-step motion error<=2px; loop<=2px, adjacent cycle<=1px; transfer only point/horizon support eligible in earlier stages',
        'alphas':list(ALPHAS),'motions':list(MOTIONS),'accept_margin':{'MSE':'improve>max(1e-8,1e-4*baseline)','LPIPS':'improve>1e-5'},
        'step':.002,'feature_cap':.05,'image_budget_ceiling_first4':.002,'pixel_cap':.03,
        'controls':['keep','fixed gated CV alpha=.25','ungated CV alpha=.25','calibration-only without validation','full calibrated policy'],
        'labels':'only historical RGB holdouts in generation; no UV/visibility GT, no future RGB, no old full-context latent/track cache',
        'hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('replay_policy_cpu.py'),Path(__file__).with_name('first_query_cpu.py'),Path(__file__).with_name('cpu_exact_depthwise.py')]}}


def cleanup():
    gc.collect()
    try:ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError,OSError):pass


def model_digest(net):
    h=hashlib.sha256()
    for name,tensor in net.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().contiguous().numpy().tobytes())
    cleanup();return h.hexdigest()


@torch.no_grad()
def prefix_features(net,rgb):
    embeddings=[];skips=[]
    for frame in rgb:
        e,s=net.backbone.enc(frame[None]);embeddings.append(e);skips.append(s)
    enc=torch.cat(embeddings);sk=torch.cat(skips)
    # Check real model framewise vs batch2 execution before using the prefix.
    check_e,check_s=net.backbone.enc(rgb[:2]);parity=max(float((check_e-enc[:2]).abs().max()),float((check_s-sk[:2]).abs().max()))
    assert parity<2e-5,('framewise encoder parity',parity)
    pe=torch.cat([enc[:1].expand(16-len(rgb),-1,-1,-1),enc])[None]
    ps=torch.cat([sk[:1].expand(16-len(rgb),-1,-1,-1),sk])[None]
    hidden=net.latent_time(net.backbone.hid(pe));skip=net.skip_time(ps)
    return enc[None],hidden,skip,parity


def read_rgb(path,start,stop,events,kind):
    if stop>16:
        assert kind=='read_real_future_AFTER_ALL_DECISIONS','normal stage cannot decode real future'
        assert all((OUT/f'case_{i}/decision.json').exists() for i in CASES)
    # Read only metadata source locator; no cached geometry or teacher fields.
    with np.load(path) as pack:source=Path(json.loads(str(pack['metadata_json']))['source_path'])
    with np.load(source,allow_pickle=True) as pack:
        encoded=pack['images_jpeg_bytes']
        assert len(encoded)>=stop
        frames=np.stack([cv2.resize(_decode_jpeg_rgb(encoded[i]),(128,128),interpolation=cv2.INTER_AREA) for i in range(start,stop)])
    events.append({'event':kind,'decoded_frame_range':[start,stop],'time':time.time()})
    return torch.from_numpy(frames).permute(0,3,1,2).float()/255


def generate(net,rgb,allowed=None,ungated_control=False):
    """No target argument, no disk/cache reads, and no caller's future RGB."""
    rgb=rgb.clone();tracking=track_prefix(rgb,allowed)
    cleanup();print('PREFIX_BASELINE_START',len(rgb),flush=True)
    with torch.no_grad():
        padded=padded_prefix(rgb);enc,h,s,parity=prefix_features(net,rgb)
        base=core.decode_frozen(net,h,s)[0]
        scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        memory=(sample(enc,tracking['uv'][None])*tracking['history_weights'][None,...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    cleanup();print('PREFIX_BASELINE_READY',len(rgb),flush=True)
    outputs={};support={};work=list(MOTIONS)+(['ungated_cv'] if ungated_control else [])
    for label in work:
        motion='cv' if label=='ungated_cv' else label
        first=tracking['raw_weights'][motion] if label=='ungated_cv' else tracking['weights'][motion]
        support[label]=int((first>0).sum())
        if first.any():
            coords=torch.cat([tracking['endpoints'][motion],tracking['endpoints'][motion][:,-1:].expand(-1,4,-1)],1)[None]
            ww=torch.cat([first,torch.zeros_like(first)],1)[None]
            gh,gs,_=make_gradient(net,h,s,memory,coords,ww,scale)
            with torch.no_grad():
                y=core.decode_frozen(net,h+bounded_feature(gh,.002,normalize=True),s+bounded_feature(gs,.002,normalize=True))[0]
                raw=base.clone();raw[:4]=pixel_bound(y[None,:4],base[None,:4])[0]
            del gh,gs
        else:raw=base.clone()
        for alpha in ([.25] if label=='ungated_cv' else ALPHAS):outputs[key(label,alpha)]=bounded_output(base,raw,alpha)
        cleanup()
        print('PREFIX_DIRECTION_READY',len(rgb),label,support[label],flush=True)
    return {'base':base,'outputs':outputs,'support':support,'allowed':{m:tracking['weights'][m]>0 for m in MOTIONS},
        'prefix_sha256':tracking['prefix_sha256'],'padded_sha256':digest(padded),'seed_sha256':tracking['seed_sha256'],
        'signals':tracking['signals'],'prefix_frames':len(rgb),'spatial_framewise_parity_max_abs':parity}


def locked_stage(folder,stage,pack,events):
    record={'prefix_frames':pack['prefix_frames'],'prefix_sha256':pack['prefix_sha256'],
        'padded_sha256':pack['padded_sha256'],'seed_sha256':pack['seed_sha256'],
        'spatial_framewise_parity_max_abs':pack['spatial_framewise_parity_max_abs'],
        'max_input_frame':pack['prefix_frames']-1,'old_prefix_cache_reused':False,
        'output_hashes':{k:digest(v) for k,v in pack['outputs'].items()},'support':pack['support'],'signals':pack['signals'],'time':time.time()}
    dump(folder/(stage+'_locked.json'),record);events.append({'event':stage+'_locked','time':record['time']})
    dump(OUT/'progress.json',{'case':folder.name,'stage':stage,'time':record['time']})


def replay_metrics(pack,target,lp):
    with torch.no_grad():
        baseline=metrics(pack['base'][:4],target,pack['base'][:4],lp)
        values={k:metrics(v[:4],target,pack['base'][:4],lp) for k,v in pack['outputs'].items()}
    return baseline,values


def run_case(index):
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol.json',protocol());folder=OUT/f'case_{index}';folder.mkdir(exist_ok=True)
    path=manifest()['evaluation'][index];events=[];start=time.time()
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;cleanup()
    before=model_digest(net)
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
    observed8=read_rgb(path,0,8,events,'read_prefix8');cleanup()
    cal=generate(net,observed8);locked_stage(folder,'prefix8',cal,events)
    target8=read_rgb(path,8,12,events,'read_observed_calibration_holdout')
    cb,cm=replay_metrics(cal,target8,lp);choice=select(cm,cb,cal['support'])
    allowed=cal['allowed'];seed=cal['seed_sha256'];cal_signals=cal['signals'];del cal;gc.collect()
    print('CAUSAL_REPLAY',index,'calibration_choice',choice,flush=True)
    observed12=torch.cat([observed8,target8]);validation=generate(net,observed12,allowed)
    assert validation['seed_sha256']==seed
    locked_stage(folder,'prefix12',validation,events)
    target12=read_rgb(path,12,16,events,'read_observed_validation_holdout')
    vb,vm=replay_metrics(validation,target12,lp)
    accepted=choice!='keep' and improves(vm[choice],vb);validation_pass=bool(accepted)
    allowed=validation['allowed'];val_signals=validation['signals'];del validation;gc.collect()
    observed16=torch.cat([observed12,target12]);final=generate(net,observed16,allowed,ungated_control=True)
    assert final['seed_sha256']==seed;locked_stage(folder,'prefix16',final,events)
    if accepted:
        motion=choice.split('_a')[0]
        if final['support'][motion]==0:accepted=False
    action=choice if accepted else 'keep'
    decisions={'index':index,'path':path,'calibration_choice':choice,'final_action':action,
        'validation_accepts_calibration_choice':validation_pass,'final_support_rejection':validation_pass and not accepted,
        'calibration_baseline':cb,'calibration_metrics':cm,
        'validation_baseline':vb,'validation_metrics':vm,'calibration_signals':cal_signals,'validation_signals':val_signals,
        'final_signals':final['signals'],'final_support':final['support'],'events':events,'future_RGB_decoded':False,
        'TAU_bitwise_verified':model_digest(net)==before,'model_state_sha256':before,
        'weights_updated':False,'seconds':time.time()-start,'decision_locked_at':time.time()}
    assert decisions['TAU_bitwise_verified'] and not any(p.grad is not None for p in net.parameters())
    for output in final['outputs'].values():
        assert torch.equal(output[4:],final['base'][4:])
        assert float((output[:4]-final['base'][:4]).abs().mean())<=.001001
        assert float((output-final['base']).abs().max())<=.015001
    torch.save({'base':final['base'],'outputs':final['outputs'],'decision':action},folder/'future_predictions.pt')
    dump(folder/'decision.json',decisions);print('CAUSAL_DECISION_LOCKED',index,action,flush=True)


def run_stage(index,n):
    """Fresh process per prefix. Safe caches contain earlier-prefix masks only."""
    assert n in (8,12,16)
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'protocol.json',protocol());folder=OUT/f'case_{index}';folder.mkdir(exist_ok=True)
    path=manifest()['evaluation'][index];start=time.time();previous=None;allowed=None
    if n>8:
        previous=json.loads((folder/f'stage{n-4}_result.json').read_text())
        allowed=torch.load(folder/f'allowed{n-4}.pt',map_location='cpu',weights_only=False)
        events=list(previous['events'])
    else:events=[]
    rgb=read_rgb(path,0,n,events,'read_prefix'+str(n));cleanup()
    if previous is not None:assert digest(rgb[:n-4])==previous['prefix_sha256']
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;cleanup()
    before=model_digest(net)
    if previous is not None:assert previous['model_state_sha256']==before
    if index==0 and n==8:
        with torch.no_grad():
            synthetic=torch.rand(1,16,3,32,32);native=net(synthetic)
        cleanup();changed=install_channelwise(net)
        with torch.no_grad():
            streamed=net(synthetic);error=float((native-streamed).abs().max());assert error<2e-5,('full-network runtime equivalence',error)
        dump(OUT/'runtime_equivalence.json',{'synthetic_frames':16,'synthetic_size':32,'max_abs_error':error,'depthwise_modules':changed,'weights_unchanged':model_digest(net)==before})
        del synthetic,native,streamed;cleanup()
    else:changed=install_channelwise(net)
    print('EXACT_DW_RUNTIME',index,n,changed,flush=True)
    pack=generate(net,rgb,allowed,ungated_control=n==16)
    if previous is not None:assert pack['seed_sha256']==previous['seed_sha256']
    locked_stage(folder,'prefix'+str(n),pack,events)
    assert model_digest(net)==before and not any(p.grad is not None for p in net.parameters())
    del net;cleanup()
    result={'index':index,'path':path,'prefix_frames':n,'prefix_sha256':pack['prefix_sha256'],
        'seed_sha256':pack['seed_sha256'],'model_state_sha256':before,'events':events,'TAU_bitwise_verified':True,
        'signals':pack['signals'],'support':pack['support'],'seconds':time.time()-start}
    if n<16:
        import lpips
        lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
        target=read_rgb(path,n,n+4,events,'read_observed_'+('calibration' if n==8 else 'validation')+'_holdout')
        baseline,values=replay_metrics(pack,target,lp);result.update(baseline=baseline,metrics=values)
        if n==8:result['calibration_choice']=select(values,baseline,pack['support'])
        else:
            choice=previous['calibration_choice'];result['calibration_choice']=choice
            result['validation_pass']=choice!='keep' and improves(values[choice],baseline)
        result['seconds']=time.time()-start
        torch.save(pack['allowed'],folder/f'allowed{n}.pt');dump(folder/f'stage{n}_result.json',result)
        print('CAUSAL_REPLAY_STAGE_LOCKED',index,n,result['calibration_choice'],flush=True)
    else:
        calibration=json.loads((folder/'stage8_result.json').read_text());validation=previous
        choice=calibration['calibration_choice'];valid=bool(validation['validation_pass'])
        accepted=valid and pack['support'][choice.split('_a')[0]]>0
        action=choice if accepted else 'keep'
        decisions={'index':index,'path':path,'calibration_choice':choice,'final_action':action,
            'validation_accepts_calibration_choice':valid,'final_support_rejection':valid and not accepted,
            'calibration_baseline':calibration['baseline'],'calibration_metrics':calibration['metrics'],
            'validation_baseline':validation['baseline'],'validation_metrics':validation['metrics'],
            'calibration_signals':calibration['signals'],'validation_signals':validation['signals'],
            'final_signals':pack['signals'],'final_support':pack['support'],'events':events,
            'future_RGB_decoded':False,'TAU_bitwise_verified':True,'model_state_sha256':before,
            'weights_updated':False,'seconds_total_stages':calibration['seconds']+validation['seconds']+time.time()-start,
            'decision_locked_at':time.time()}
        for output in pack['outputs'].values():
            assert torch.equal(output[4:],pack['base'][4:])
            assert float((output[:4]-pack['base'][:4]).abs().mean())<=.001001
            assert float((output-pack['base']).abs().max())<=.015001
        torch.save({'base':pack['base'],'outputs':pack['outputs'],'decision':action},folder/'future_predictions.pt')
        dump(folder/'stage16_result.json',result);dump(folder/'decision.json',decisions)
        print('CAUSAL_DECISION_LOCKED',index,action,flush=True)


def evaluate():
    # Hard global barrier: no case's real future can inform a later case's policy.
    decisions=[json.loads((OUT/f'case_{i}/decision.json').read_text()) for i in CASES]
    barrier=time.time();assert all(d['decision_locked_at']<barrier and not d['future_RGB_decoded'] for d in decisions)
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False);rows=[];reads=[]
    for d in decisions:
        i=d['index'];data=torch.load(OUT/f'case_{i}/future_predictions.pt',weights_only=False)
        target=read_rgb(d['path'],16,24,reads,'read_real_future_AFTER_ALL_DECISIONS')
        assert data['decision']==d['final_action'];base=data['base'];outputs=data['outputs']
        policy=base if d['final_action']=='keep' else outputs[d['final_action']]
        cal_only=base if d['calibration_choice']=='keep' else outputs[d['calibration_choice']]
        named={'tau':base,'policy':policy,'calibration_only':cal_only,
            'fixed_gated_cv_a0.25':outputs[key('cv',.25)],'fixed_ungated_cv_a0.25':outputs[key('ungated_cv',.25)]}
        with torch.no_grad():
            full={k:metrics(v,target,base,lp) for k,v in named.items()}
            first4={k:metrics(v[:4],target[:4],base[:4],lp) for k,v in named.items()}
            all_candidate={k:metrics(v[:4],target[:4],base[:4],lp) for k,v in outputs.items()}
        assert torch.equal(policy[4:],base[4:])
        rows.append({'index':i,'path':d['path'],'action':d['final_action'],'calibration_choice':d['calibration_choice'],
            'first4_metrics':first4,'full8_metrics':full,'all_candidates_first4':all_candidate,'decision':d})
    names=list(rows[0]['first4_metrics']);mean=lambda vals:float(np.mean(vals))
    result={kind:{a:{k:mean([r[kind][a][k] for r in rows]) for k in ['mse','lpips','temporal','budget']} for a in names} for kind in ['first4_metrics','full8_metrics']}
    accepts=[r for r in rows if r['action']!='keep']
    result['accepted_cases']=len(accepts);result['cases']=len(rows)
    result['policy_harm_cases']=[r['index'] for r in rows if any(r['first4_metrics']['policy'][k]>r['first4_metrics']['tau'][k] for k in ['mse','lpips'])]
    result['beneficial_candidate_available_cases']=[r['index'] for r in rows if any(v['mse']<r['first4_metrics']['tau']['mse'] and v['lpips']<r['first4_metrics']['tau']['lpips'] for v in r['all_candidates_first4'].values())]
    result['accepted_beneficial_cases']=[r['index'] for r in accepts if r['first4_metrics']['policy']['mse']<r['first4_metrics']['tau']['mse'] and r['first4_metrics']['policy']['lpips']<r['first4_metrics']['tau']['lpips']]
    result['per_case']=rows;result['future_read_events']=reads;result['global_decision_barrier_time']=barrier
    dump(OUT/'summary.json',result);dump(OUT/'COMPLETE.json',{'CPU_only':True,'training':False,'weights_updated':False,'cases':len(rows),
        'all_decisions_locked_before_future_RGB_decode':True,'D4RT_predictor_calibrated':False,'corrected_horizons':[1,2,3,4]})
    print('CAUSAL_REPLAY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=CASES);p.add_argument('--stage',type=int,choices=[8,12,16]);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(20261019)
    try:
        if a.evaluate:evaluate()
        else:run_stage(a.case,a.stage)
    except Exception as e:
        dump(OUT/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
