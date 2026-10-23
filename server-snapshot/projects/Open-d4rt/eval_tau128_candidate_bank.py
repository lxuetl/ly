"""Frozen candidate-bank architecture and budgeted region/horizon oracle probe.

Candidate construction takes observed inputs only. Oracle routing uses future
GT explicitly and is NOT deployable. Shared spatial permission, matched actual
correction budget, and post-routing cap isolate candidate content from magnitude.
Regions are fixed image grids, not instance segmentation.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import eval_tau128_ttpo_replay as previous

core,prior,old=previous.core,previous.prior,previous.old
NAMES=['tau','geometry','trajectory_half','trajectory_onehalf','flow_zero',
       'feature_transport','rgb_transport','rgb_local']
REPLAY_NAMES=['replay_clip','replay_horizon','replay_region16','replay_region16_horizon']

@torch.no_grad()
def build_bank(observed,backbone,joint,adapter,return_aux=False):
    assert set(observed)=={'context_rgb',*previous.PAST}
    source,hidden,skip=core.extract_frozen_features(backbone,observed['context_rgb'])
    tau=core.decode_frozen(backbone,hidden,skip)
    bundle=core.geometry_bundle('correct',joint,observed,tau.shape[1])
    adapter.mode='correct'
    corrected,_,_=adapter(source,hidden,bundle,bundle)
    raw_geo=prior.match_pixel_budget(core.decode_frozen(backbone,corrected,skip),tau,.002,'fixed',.08)
    b,k,_,h,w=tau.shape
    support=F.interpolate(adapter.last['support'].flatten(0,1),(h,w),mode='bilinear',align_corners=False).reshape(b,k,1,h,w)
    permission=((support-.01)/.05).clamp(0,1)
    flow=adapter.last['flow'].clone()
    feature_support=adapter.last['support'].clone()
    geometry=(tau+(raw_geo-tau)*permission).clamp(0,1)
    reference_budget=(geometry-tau).abs().mean((1,2,3,4),keepdim=True)
    def match(pred):
        delta=(pred-tau)*permission
        norm=delta.abs().mean((1,2,3,4),keepdim=True)
        delta=delta*(reference_budget/norm.clamp_min(1e-9))
        return (tau+delta.clamp(-.08,.08)).clamp(0,1)
    bank=[tau,geometry]
    # Scale the future XYZ displacement and UV displacement together. Anchors,
    # confidence and forecast times are retained; raw UV tokens follow UV.
    raw,uv,reliability,anchor=bundle
    for scale in [.5,1.5]:
        changed_uv=(anchor[:,:,None]+scale*(uv-anchor[:,:,None])).clamp(0,1)
        changed_raw=raw.clone(); changed_raw[...,:3]*=scale;changed_raw[...,3:5]=changed_uv
        changed=(changed_raw,changed_uv,reliability,anchor)
        altered,_,_=adapter(source,hidden,changed,changed)
        bank.append(match(core.decode_frozen(backbone,altered,skip)))
    # Zero-flow candidate keeps geometry tokens but removes feature transport.
    zero=(raw,anchor[:,:,None].expand_as(uv),reliability,anchor)
    # Use the existing counterfactual branch with exact zero raster displacement
    # and original correct geometry-derived support.
    adapter.mode='shuffled_flow'
    altered,_,_=adapter(source,hidden,bundle,zero)
    bank.append(match(core.decode_frozen(backbone,altered,skip)))
    # Explicit transport bypasses the learned residual proposal.
    transport=(core.backward_warp(source,flow)-hidden)*feature_support
    factor=.006/transport.abs().mean((1,2,3,4),keepdim=True).clamp_min(1e-7)
    altered=hidden+(transport*factor).clamp(-.12,.12)
    bank.append(match(core.decode_frozen(backbone,altered,skip)))
    rgbflow=F.interpolate(flow.flatten(0,1),(h,w),mode='bilinear',align_corners=False).reshape(b,k,2,h,w)
    bank.append(match(core.backward_warp(observed['context_rgb'][:,-1],rgbflow)))
    adapter.mode='local_rgb'
    altered,_,_=adapter(source,hidden,bundle,bundle)
    bank.append(match(core.decode_frozen(backbone,altered,skip)))
    adapter.mode='correct'
    result=torch.stack([x[0] for x in bank]) # candidate,T,C,H,W
    assert len(result)==len(NAMES)
    assert torch.isfinite(result).all()
    assert ((result-tau[0]).abs()*(permission[0]==0)).amax()<1e-6
    assert (result-tau[0]).abs().mean((1,2,3,4)).amax()<=reference_budget.item()+1e-6
    if return_aux:
        flow_rgb=F.interpolate(flow.flatten(0,1),(h,w),mode='bilinear',align_corners=False).reshape(b,k,2,h,w)
        return result,reference_budget.item(),torch.cat([flow_rgb[0],support[0],permission[0]],1)
    return result,reference_budget.item()

def cap_mix(pred,tau,budget):
    delta=pred-tau
    scale=min(1.,budget/max(delta.abs().mean().item(),1e-9))
    return (tau+delta*scale).clamp(0,1),scale

@torch.no_grad()
def history_choices(path,observed,d4rt,image_hw,backbone,joint,adapter,return_scores=False):
    cache=np.load(path);meta=json.loads(str(cache['metadata_json']))
    video=previous.load_worldtrack_sequence(Path(meta['source_path']),24)['video_rgb']
    loss=torch.zeros(len(NAMES),8,8,8,device=observed['context_rgb'].device)
    count=torch.zeros(1,8,1,1,device=loss.device)
    for cutoff in [8,12]:
        tracks=previous._infer_tracks(model=d4rt,video_model_rgb=previous._resize_video(video[:cutoff],image_hw=image_hw),query_uv_norm=np.asarray(cache['query_uv']),query_chunk_size=64)
        xyz=np.asarray(tracks['tracks_xyz_ref0'],np.float32)
        assert xyz.shape[1]==cutoff
        valid=np.asarray(tracks['tracks_visibility'],bool)&np.isfinite(xyz).all(-1)
        replay={'query_uv':observed['query_uv'],
                'past_xyz':torch.tensor(xyz,device=loss.device)[None],
                'past_uv':torch.tensor(np.asarray(tracks['tracks_uv_norm'],np.float32),device=loss.device)[None],
                'past_valid':torch.tensor(valid,device=loss.device)[None],
                'past_confidence':torch.tensor(np.asarray(tracks['tracks_confidence'],np.float32),device=loss.device)[None]}
        ctx=observed['context_rgb'][:,:cutoff]
        replay['context_rgb']=torch.cat([ctx[:,:1].expand(-1,16-cutoff,-1,-1,-1),ctx],1)
        bank,_=build_bank(replay,backbone,joint,adapter)
        k=16-cutoff
        err=(bank[:,:k]-observed['context_rgb'][0,cutoff:]).square().mean(2)
        loss[:,:k]+=F.avg_pool2d(err.reshape(-1,1,128,128),16).reshape(len(NAMES),k,8,8)
        count[:,:k]+=1
    loss=loss/count
    if return_scores:return loss
    return {'replay_clip':loss.mean((1,2,3)).argmin().expand(8,8,8),
            'replay_horizon':loss.mean((2,3)).argmin(0)[:,None,None].expand(8,8,8),
            'replay_region16':loss.mean(1).argmin(0)[None].expand(8,8,8),
            'replay_region16_horizon':loss.argmin(0)}

@torch.no_grad()
def apply_choice(bank,choice,budget):
    ids=choice.repeat_interleave(16,-2).repeat_interleave(16,-1)
    pred=bank.gather(0,ids[None,:,None].expand(1,8,3,128,128)).squeeze(0)
    pred,scale=cap_mix(pred,bank[0],budget)
    return pred,{'counts':torch.bincount(choice.flatten(),minlength=len(NAMES)).tolist(),'post_route_scale':scale,'switch_rate':(choice[1:]!=choice[:-1]).float().mean().item()}

@torch.no_grad()
def route_oracle(bank,target,patch,temporal,budget):
    n,t,c,h,w=bank.shape
    error=(bank-target).square().mean(2)
    cells=F.avg_pool2d(error.reshape(n*t,1,h,w),patch).reshape(n,t,h//patch,w//patch)
    if not temporal: cells=cells.mean(1,keepdim=True)
    choice=cells.argmin(0)
    if not temporal:choice=choice.expand(t,-1,-1)
    ids=choice.repeat_interleave(patch,-2).repeat_interleave(patch,-1)
    pred=bank.gather(0,ids[None,:,None].expand(1,t,c,h,w)).squeeze(0)
    pred,scale=cap_mix(pred,bank[0],budget)
    return pred,{'counts':torch.bincount(choice.flatten(),minlength=n).tolist(),'post_route_scale':scale,
                 'switch_rate':(choice[1:]!=choice[:-1]).float().mean().item() if t>1 else 0.}

@torch.no_grad()
def metrics(pred,target,tau,mask,lpips):
    err=pred-target;mse=err.square().mean().item()
    selected=mask.expand_as(err)
    return {'MSE':mse,'MAE':err.abs().mean().item(),'PSNR':-10*math.log10(max(mse,1e-12)),
            'LPIPS':lpips(pred*2-1,target*2-1).mean().item(),
            'temporal_MAE':((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean().item(),
            'dynamic_MSE':err.square()[selected].mean().item() if selected.any() else None,
            'pixel_delta':(pred-tau).abs().mean().item()}

def aggregate(rows,selection):
    data={}
    for name in rows[0]['metrics']:
        data[name]={key:float(np.mean([r['metrics'][name][key] for r in rows if r['metrics'][name][key] is not None]))
                    for key in rows[0]['metrics'][name] if any(r['metrics'][name][key] is not None for r in rows)}
    groups=defaultdict(list)
    for r in rows:groups[tuple(r['group'])].append(r)
    rng=np.random.default_rng(42);ci={}
    pairs=[('geometry',n) for n in data if n.startswith(('oracle_','replay_')) or n=='selected_val']
    pairs += [(n.replace('oracle_motion_','oracle_amplitude_'),n) for n in data if n.startswith('oracle_motion_')]
    for ref,name in pairs:
        sums=[];counts=[]
        for rs in groups.values():
            sums.append(sum(r['metrics'][ref]['MSE']-r['metrics'][name]['MSE'] for r in rs));counts.append(len(rs))
        idx=rng.integers(0,len(sums),(5000,len(sums)))
        boot=np.array(sums)[idx].sum(1)/np.array(counts)[idx].sum(1)
        ci[ref+' -> '+name]={'gain_MSE':sum(sums)/sum(counts),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    return {'clips':len(rows),'clusters':len(groups),'selected_fixed_candidate_on_validation':selection,'metrics':data,'paired_cluster_bootstrap':ci,
            'by_source':{d:{n:float(np.mean([r['metrics'][n]['MSE'] for r in rows if r['group'][0]==d])) for n in data} for d in sorted(set(r['group'][0] for r in rows))}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--history-replay',action='store_true')
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);prior.seed_all(42)
    base=Path('/root/autodl-fs/d4rt_training')
    cfg=SimpleNamespace(backbone_checkpoint=str(base/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(base/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(base/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    split=json.loads((base/'tau_v2_direction_rank_pb002_20261009/split_paths.json').read_text())['paths']
    split={s:split[s][:args.limit or None] for s in ['val','test']}
    previous.dump(out/'split_paths.json',{'paths':split})
    previous.dump(out/'protocol.json',{'architecture':'shared frozen TAU feature extractor -> geometry/transport candidate bank -> budget-constrained regional routing',
        'candidates':NAMES,'same_permission':True,'budget':'all candidates matched to original geometry actual per-clip L1; mixtures capped at same budget',
        'oracle_uses_future_truth':True,'deployable':'fixed candidates and optional prefix-only replay, chosen by validation',
        'regions':'fixed grids 8/16/32 pixels; not object instance masks','weights_updated':False,
        'scope':'candidate-space diagnostic, not an optimal constrained mathematical bound',
        'capacity_control':'rgb_local reuses geometry-trained weights: not a separately trained RGB capacity control',
        'candidate_count_control':'amplitude bank has same number of entries as motion bank',
        'future_labels':'never passed into build_bank','history_replay':args.history_replay,
        'replay_limitations':'left padded short contexts; fixed spatial cells do not follow moving objects'} )
    device=torch.device('cuda');backbone,joint,adapter,_=old.load_frozen(cfg,device)
    import lpips
    perceptual=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    d4rt=None;image_hw=None
    if args.history_replay:
        config=previous.load_yaml_config(str(previous.ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml'))
        d4rt=previous.build_model(config['model']).eval()
        state=previous._unwrap_state_dict(previous.load_checkpoint(str(previous.ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt'),map_location='cpu'))
        d4rt.load_state_dict(state,strict=True);del state
        d4rt.to(device).requires_grad_(False);image_hw=tuple(config.get_path('model.input.image_size',[256,256]))
    start=time.time();done=0;total=sum(map(len,split.values()));selected='geometry'
    print(json.dumps({'stage':'loaded','total':total,'candidates':NAMES}),flush=True)
    for phase in ['val','test']:
        dataset=previous.VideoForecastDataset('',phase,128,paths=split[phase]);rows=[]
        rowpath=out/f'{phase}_per_clip.jsonl'
        if rowpath.exists():rows=[json.loads(x) for x in rowpath.read_text().splitlines()]
        completed={r['path'] for r in rows};done+=len(rows)
        for ix,path in enumerate(split[phase]):
            if path in completed:continue
            tick=time.time();item=dataset[ix]
            observed={k:item[k][None].to(device) for k in previous.PAST};observed['context_rgb']=item['context_rgb'][None].to(device)
            choices=history_choices(path,observed,d4rt,image_hw,backbone,joint,adapter) if args.history_replay else {}
            bank,budget=build_bank(observed,backbone,joint,adapter)
            amplitude=torch.stack([bank[0]+s*(bank[1]-bank[0]) for s in np.linspace(0,1,len(NAMES))])
            # Prediction finishes before oracle target access.
            target=item['future_rgb'].to(device)
            batch=core.move(previous.video_pad_collate([item]),device)
            mask,_=prior.camera_compensated_dynamic_mask(batch,128,128,4,6);mask=mask[0,:,None]
            predictions=dict(zip(NAMES,bank));routing={}
            for name,choice in choices.items():predictions[name],routing[name]=apply_choice(bank,choice,budget)
            layouts={'clip':(128,False),'horizon':(128,True),'region32':(32,False),'region16':(16,False),'region8':(8,False),'region16_horizon':(16,True),'region8_horizon':(8,True)}
            for kind,candidates in [('amplitude',amplitude),('motion',bank)]:
                for label,(patch,temporal) in layouts.items():
                    name=f'oracle_{kind}_{label}'
                    predictions[name],routing[name]=route_oracle(candidates,target,patch,temporal,budget)
            predictions['selected_val']=predictions[selected]
            result={n:metrics(pred,target,bank[0],mask,perceptual) for n,pred in predictions.items()}
            # Selection can't improve pixelwise MSE without budget projection;
            # after projection report actual result, never claim guaranteed gain.
            assert all(m['pixel_delta']<=budget+1e-6 for m in result.values())
            record={'path':path,'group':previous.group(path),'metrics':result,'routing':routing,'budget':budget,'seconds':time.time()-tick}
            rows.append(record)
            with rowpath.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            done+=1;speed=float(np.mean([r['seconds'] for r in rows[-20:]]))
            progress={'phase':phase,'done':done,'total':total,'seconds_per_clip':speed,'eta_minutes':speed*(total-done)/60,'elapsed_seconds':time.time()-start,'gpu_peak_GB':torch.cuda.max_memory_allocated()/2**30}
            previous.dump(out/'progress.json',progress)
            if done%10==0 or args.limit:print(json.dumps(progress),flush=True)
        if phase=='val':
            selected=min(NAMES+(REPLAY_NAMES if args.history_replay else []),key=lambda n:np.mean([r['metrics'][n]['MSE'] for r in rows]))
            for r in rows:r['metrics']['selected_val']=r['metrics'][selected]
            previous.dump(out/'selection.json',{'selected_fixed_candidate':selected})
        previous.dump(out/f'{phase}_summary.json',aggregate(rows,selected))
    previous.dump(out/'COMPLETE.json',{'seconds':time.time()-start,'selected_fixed_candidate':selected})
    print(json.dumps({'stage':'complete'}),flush=True)

if __name__=='__main__':main()
