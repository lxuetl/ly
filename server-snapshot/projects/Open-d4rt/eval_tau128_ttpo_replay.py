"""Causal observed-prefix preference selection; frozen models, no test fitting.

D4RT is rerun on each RGB prefix. TAU gets first-frame left padding to its
fixed context length. Shorter-context replay is a documented distribution shift.
Future labels are exposed only after all deployable choices have been made.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'forecasting')]
import train_tau128_experience_router as old
import train_tau128_scaled_cap as prior
import train_simvp_feature_warp as core
from video_forecast_dataset import VideoForecastDataset, video_pad_collate
from eval_track3d_in_worldtrack import load_worldtrack_sequence
from infer_track_3d import _infer_tracks, _resize_video, _unwrap_state_dict
from src.core import load_checkpoint, load_yaml_config
from src.model import build_model

SCALES = [0., .25, .5, .75, 1.]
PAST = ('query_uv', 'past_xyz', 'past_uv', 'past_valid', 'past_confidence')

def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

@torch.no_grad()
def proposals(observed, backbone, joint, adapter):
    assert set(observed) == {'context_rgb', *PAST}
    source, hidden, skip = core.extract_frozen_features(backbone, observed['context_rgb'])
    tau = core.decode_frozen(backbone, hidden, skip)
    bundle = core.geometry_bundle('correct', joint, observed, tau.shape[1])
    corrected, _, _ = adapter(source, hidden, bundle, bundle)
    geometry = prior.match_pixel_budget(core.decode_frozen(backbone, corrected, skip), tau, .002, 'fixed', .08)
    b, k, _, h, w = tau.shape
    support = F.interpolate(adapter.last['support'].flatten(0, 1), (h, w), mode='bilinear', align_corners=False).reshape(b,k,1,h,w)
    delta = (geometry-tau) * ((support-.01)/.05).clamp(0,1)
    return [(tau+s*delta).clamp(0,1) for s in SCALES]

@torch.no_grad()
def scores(predictions, target, lpips_model, mask=None):
    values=[]
    for pred in predictions:
        err=pred-target
        mse=err.square().mean().item()
        perceptual=lpips_model(pred.flatten(0,1)*2-1,target.flatten(0,1)*2-1).mean().item()
        temporal=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean().item()
        row={'MSE':mse,'MAE':err.abs().mean().item(),'PSNR':-10*math.log10(max(mse,1e-12)), 'LPIPS':perceptual,'temporal_MAE':temporal}
        if mask is not None:
            selected=mask.expand_as(err)
            row['dynamic_MSE']=err.square()[selected].mean().item() if selected.any() else None
            row['dynamic_MAE']=err.abs()[selected].mean().item() if selected.any() else None
        values.append(row)
    return values

def group(path):
    p=Path(path)
    domain='PointOdyssey' if 'po50' in p.parts else p.parent.name
    original=re.sub(r'_clip_\d+$','',p.stem)
    if domain!='PointOdyssey': original=re.sub(r'_\d+$','',original)
    return domain,original

def summarize(records, selected_rule):
    variants=['tau','all_geometry','fixed_val','random','replay_mse','replay_dynamic','replay_perceptual','selected_val','oracle_not_deployable']
    out={}
    for variant in variants:
        rows=[r['metrics'][r['choices'][variant]] for r in records]
        out[variant]={key:float(np.mean([v[key] for v in rows if v.get(key) is not None])) for key in rows[0] if any(v.get(key) is not None for v in rows)}
    grouped=defaultdict(list)
    for r in records:
        grouped[tuple(r['group'])].append(r)
    ci={}
    rng=np.random.default_rng(42)
    for variant in ['replay_mse','replay_dynamic','replay_perceptual','selected_val','oracle_not_deployable']:
        sums=[]; counts=[]
        for rows in grouped.values():
            ds=[r['metrics'][4]['MSE']-r['metrics'][r['choices'][variant]]['MSE'] for r in rows]
            sums.append(sum(ds));counts.append(len(ds))
        idx=rng.integers(0,len(sums),(5000,len(sums)))
        boot=np.array(sums)[idx].sum(1)/np.array(counts)[idx].sum(1)
        ci[variant]={'gain_vs_all_geometry':sum(sums)/sum(counts),'ci95':np.quantile(boot,[.025,.975]).tolist(),'original_video_clusters':len(sums)}
    hist={v:np.bincount([r['choices'][v] for r in records],minlength=5).tolist() for v in variants}
    x=np.array([r['replay_scores']['replay_mse'][0]-r['replay_scores']['replay_mse'][4] for r in records])
    y=np.array([r['metrics'][0]['MSE']-r['metrics'][4]['MSE'] for r in records])
    corr=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
    return {'clips':len(records),'selected_on_validation':selected_rule,'metrics':out,'paired_cluster_bootstrap':ci,'choice_counts':hist,'replay_future_gain_pearson':corr,
            'by_source':{d:{v:float(np.mean([r['metrics'][r['choices'][v]]['MSE'] for r in records if r['group'][0]==d])) for v in variants} for d in sorted(set(r['group'][0] for r in records))}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',required=True)
    p.add_argument('--max-val',type=int,default=0)
    p.add_argument('--max-test',type=int,default=0)
    p.add_argument('--cutoffs',default='8,12')
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cutoffs=[int(x) for x in a.cutoffs.split(',')]
    torch.set_num_threads(4); prior.seed_all(42)
    base=Path('/root/autodl-fs/d4rt_training')
    cfg=SimpleNamespace(backbone_checkpoint=str(base/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(base/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(base/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    manifest=base/'tau_v2_direction_rank_pb002_20261009/split_paths.json'
    split=json.loads(manifest.read_text())['paths']
    # Same exact paths as prior experiments; limits are smoke-test only.
    split={s:ps[:getattr(a,'max_'+s) or None] for s,ps in split.items() if s in ['val','test']}
    dump(out/'split_paths.json',{'paths':split})
    dump(out/'protocol.json',{'config':vars(cfg),'cutoffs':cutoffs,'scales':SCALES,'prefix_geometry':'D4RT rerun on prefix-only raw RGB','TAU_padding':'left repeat first frame to 16','future_label_access':'metrics only, after replay choices','training':False,'selection_rule':'choose replay rule or fixed scale by validation MSE only','dynamic_replay':'observed RGB change proxy, not object GT','bootstrap':'original-video cluster resampling, clip-weighted mean','budget':'shared maximum 0.002, candidates vary actual correction amplitude'})
    device=torch.device('cuda')
    backbone,joint,adapter,_=old.load_frozen(cfg,device)
    import lpips
    perceptual=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    config=load_yaml_config(str(ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml'))
    d4rt=build_model(config['model']).eval()
    state=_unwrap_state_dict(load_checkpoint(str(ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt'),map_location='cpu'))
    loaded=d4rt.load_state_dict(state,strict=False);del state
    if loaded.missing_keys or loaded.unexpected_keys: raise RuntimeError(str(loaded))
    d4rt.to(device).requires_grad_(False)
    image_hw=tuple(config.get_path('model.input.image_size',[256,256]))
    print(json.dumps({'stage':'loaded','counts':{s:len(v) for s,v in split.items()},'cutoffs':cutoffs,'gpu':torch.cuda.get_device_name(0)}),flush=True)
    started=time.time(); total=sum(map(len,split.values())); done_count=0; fixed=4; selected='replay_mse'
    for phase in ['val','test']:
        paths=split[phase]; dataset=VideoForecastDataset('',phase,128,paths=paths)
        record_path=out/f'{phase}_per_clip.jsonl'
        records=[json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
        completed={r['path'] for r in records}; done_count+=len(records)
        for ix,path in enumerate(paths):
            if path in completed: continue
            t0=time.time()
            item=dataset[ix]
            assert item['context_rgb'].shape[0]==16 and item['future_rgb'].shape[0]==8
            observed={k:item[k].unsqueeze(0).to(device) for k in PAST}
            observed['context_rgb']=item['context_rgb'].unsqueeze(0).to(device)
            cache=np.load(path); meta=json.loads(str(cache['metadata_json']))
            raw=load_worldtrack_sequence(Path(meta['source_path']),24)['video_rgb']
            replay_scores={'replay_mse':np.zeros(5),'replay_dynamic':np.zeros(5),'replay_perceptual':np.zeros(5)}
            windows=[]
            for cutoff in cutoffs:
                prefix=_resize_video(raw[:cutoff],image_hw=image_hw)
                # No withheld RGB enters this call or any other predictor.
                geom=_infer_tracks(model=d4rt,video_model_rgb=prefix,query_uv_norm=np.asarray(cache['query_uv']),query_chunk_size=64)
                xyz=np.asarray(geom['tracks_xyz_ref0'],np.float32)
                valid=np.asarray(geom['tracks_visibility'],bool)&np.isfinite(xyz).all(-1)
                replay={'query_uv':observed['query_uv'],'past_xyz':torch.tensor(xyz,device=device)[None],
                        'past_uv':torch.tensor(np.asarray(geom['tracks_uv_norm'],np.float32),device=device)[None],
                        'past_valid':torch.tensor(valid,device=device)[None],
                        'past_confidence':torch.tensor(np.asarray(geom['tracks_confidence'],np.float32),device=device)[None]}
                ctx=observed['context_rgb'][:,:cutoff]
                replay['context_rgb']=torch.cat([ctx[:,:1].expand(-1,16-cutoff,-1,-1,-1),ctx],1)
                predictions=[x[:,:16-cutoff] for x in proposals(replay,backbone,joint,adapter)]
                target=observed['context_rgb'][:,cutoff:]
                mask=(target-ctx[:,-1:]).abs().mean(2,keepdim=True)>.05
                rs=scores(predictions,target,perceptual,mask)
                replay_scores['replay_mse']+=np.array([x['MSE'] for x in rs])/len(cutoffs)
                replay_scores['replay_dynamic']+=np.array([x['MSE']+.5*(x['dynamic_MSE'] or x['MSE']) for x in rs])/len(cutoffs)
                replay_scores['replay_perceptual']+=np.array([x['MSE']+.03*x['LPIPS']+.1*x['temporal_MAE'] for x in rs])/len(cutoffs)
                windows.append({'cutoff':cutoff,'scores':rs,'geometry_frames':xyz.shape[1]})
                assert xyz.shape[1]==cutoff
            choices={k:int(np.argmin(v)) for k,v in replay_scores.items()}
            choices.update(tau=0,all_geometry=4,fixed_val=fixed,random=int(hashlib.sha256(path.encode()).hexdigest()[:8],16)%5)
            choices['selected_val']=fixed if selected=='fixed_val' else choices[selected]
            # Only now may actual future GT be accessed for evaluation.
            full_predictions=proposals(observed,backbone,joint,adapter)
            target=item['future_rgb'][None].to(device)
            batch=core.move(video_pad_collate([item]),device)
            dynamic,_=prior.camera_compensated_dynamic_mask(batch,128,128,4,6)
            ms=scores(full_predictions,target,perceptual,dynamic[:,:,None])
            for mi,pred in zip(ms,full_predictions):
                mi['pixel_delta']=(pred-full_predictions[0]).abs().mean().item()
                assert mi['pixel_delta'] <= .00201, mi['pixel_delta']
                assert all(math.isfinite(v) for v in mi.values() if v is not None)
            choices['oracle_not_deployable']=int(np.argmin([m['MSE'] for m in ms]))
            record={'path':path,'name':item['name'],'group':group(path),'choices':choices,'metrics':ms,'replay_scores':{k:v.tolist() for k,v in replay_scores.items()},'windows':windows,'seconds':time.time()-t0}
            records.append(record)
            with record_path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            done_count+=1
            mean_seconds=float(np.mean([r['seconds'] for r in records[-20:]]))
            progress={'stage':phase,'done':done_count,'total':total,'seconds_per_clip':mean_seconds,'eta_hours':mean_seconds*(total-done_count)/3600,'elapsed_seconds':time.time()-started,'gpu_peak_GB':torch.cuda.max_memory_allocated()/2**30}
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if phase=='val':
            fixed=int(np.argmin(np.mean([[m['MSE'] for m in r['metrics']] for r in records],0)))
            for r in records:r['choices']['fixed_val']=fixed
            candidates=['fixed_val','replay_mse','replay_dynamic','replay_perceptual']
            selected=min(candidates,key=lambda v:np.mean([r['metrics'][r['choices'][v]]['MSE'] for r in records]))
            for r in records:r['choices']['selected_val']=r['choices'][selected]
            dump(out/'validation_selection.json',{'selected_rule':selected,'fixed_scale':SCALES[fixed],'fixed_index':fixed})
        dump(out/f'{phase}_summary.json',summarize(records,selected))
    dump(out/'COMPLETE.json',{'elapsed_seconds':time.time()-started,'selected_rule':selected})
    print(json.dumps({'stage':'complete','selected_rule':selected}),flush=True)

if __name__=='__main__':main()
