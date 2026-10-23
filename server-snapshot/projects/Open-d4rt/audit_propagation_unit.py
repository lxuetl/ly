"""Frozen propagation audit; future GT is diagnostic-only, never deployment input."""
import json, time, random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import audit_coordinate_gt as old
e, prev, core = old.e, old.prev, old.core
OUT = e.p.run.fit.ROOT / 'propagation_unit_20261014'

def sample_points(images, uv):
    # images K,C,H,W; uv N,K,2 -> N,K,C
    return F.grid_sample(images, uv.permute(1,0,2)[:,:,None]*2-1,
                         align_corners=True, padding_mode='border').squeeze(-1).permute(2,0,1)

def mean_mask(error, mask):
    return float(error[mask].mean()) if mask.any() else None

def region_mask(uv, weight, size=128):
    _, d = core.SparseDenseFlow._splat(torch.ones_like(uv[...,:1]), uv, weight.float(), size, size)
    return F.max_pool2d((d>0).float(),7,1,3)>0

def ci(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return {'n':0,'mean':None,'ci95':None}
    rng=np.random.default_rng(42)
    return {'n':len(a),'mean':float(a.mean()),'ci95':np.quantile(a[rng.integers(0,len(a),(5000,len(a)))].mean(1),[.025,.975]).tolist()}

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(4); prev.prior.seed_all(42); start=time.time()
    groups=defaultdict(list)
    for p in json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']['val']:
        groups[tuple(prev.group(p))].append(p)
    rng=random.Random(42); paths=[rng.choice(sorted(ps)) for _,ps in sorted(groups.items())]
    order=list(range(len(paths))); random.Random(2026).shuffle(order); tune=set(order[:len(order)//2])
    root=e.p.run.fit.ROOT
    cfg=SimpleNamespace(backbone_checkpoint=str(root/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(root/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(root/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    backbone,joint,adapter,_=prev.old.load_frozen(cfg,torch.device('cuda'))
    ds=prev.VideoForecastDataset('','val',128,paths=paths)
    import lpips
    lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def metrics(a,b):
        mse=float((a-b).square().mean()); per=float(lp(a*2-1,b*2-1).mean()); temporal=float(((a[1:]-a[:-1])-(b[1:]-b[:-1])).abs().mean())
        return {'mse':mse,'mae':float((a-b).abs().mean()),'lpips':per,'temporal':temporal,'cost':mse+.03*per+.1*temporal}
    rows=[]
    for i,path in enumerate(paths):
        x=ds[i]; obs={k:x[k][None].cuda() for k in prev.PAST}; obs['context_rgb']=x['context_rgb'][None].cuda()
        src,hid,skip=core.extract_frozen_features(backbone,obs['context_rgb']); tau=core.decode_frozen(backbone,hid,skip)[0]
        raw,puv,rel,anchor=core.geometry_bundle('correct',joint,obs,8)
        z=np.load(path); meta=json.loads(str(z['metadata_json'])); s=prev.load_worldtrack_sequence(Path(meta['source_path']),24); ids=np.asarray(z['query_ids'],dtype=np.int64)
        uv=np.asarray(s['tracks_uv'][:,ids],np.float32).transpose(1,0,2).copy(); hh,ww=s['video_rgb'].shape[1:3]; uv/=np.array([ww-1,hh-1],np.float32)
        inside=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
        vis=np.asarray(s['visibility'][:,ids]).T.astype(bool)&inside
        uv=torch.tensor(np.nan_to_num(uv),device='cuda')[None]; valid=torch.tensor(vis,device='cuda')[None]; ins=torch.tensor(inside,device='cuda')[None]
        ga=uv[:,:,15]; gf=uv[:,:,16:24]; pair=valid[:,:,15,None]&valid[:,:,16:24]
        last=obs['past_valid'][:,:,-1].bool(); aligned=pair & last[:,:,None]
        moving=(gf-ga[:,:,None]).norm(dim=-1)*127>2
        last_rgb=obs['context_rgb'][0,-1]; target=x['future_rgb'].cuda()
        target_pts=sample_points(target,gf[0]); repeated=last_rgb[None].expand(8,-1,-1,-1)
        exact=sample_points(repeated,ga[0,:,None].expand(-1,8,-1))
        zero=sample_points(repeated,gf[0]); reverse=sample_points(repeated,2*gf[0]-ga[0,:,None])
        point={}
        for name,pred in [('zero',zero),('gt_exact',exact),('gt_reverse',reverse)]:
            er=(pred-target_pts).square().mean(-1)
            point[name]={label:mean_mask(er,m[0]) for label,m in [('visible',pair),('moving_visible',pair&moving),('small_motion_visible',pair&~moving)]}
        regions={'visible':region_mask(gf,pair),'moving_visible':region_mask(gf,pair&moving),
                 'occluded_projection':region_mask(gf,valid[:,:,15,None]&~valid[:,:,16:24]&ins[:,:,16:24]),
                 'newly_visible':region_mask(gf,~valid[:,:,15,None]&valid[:,:,16:24])}
        # Exclude known visible samples from occluded/newly-visible proxy masks.
        for n in ['occluded_projection','newly_visible']: regions[n]&=~regions['visible']
        rgb={}; feature={}; backmap={}
        for kernel in [9,33]:
            raster=core.SparseDenseFlow(kernel).cuda()
            flow,support=raster(ga,gf,aligned.float(),*hid.shape[-2:])
            pred_flow,_=raster(anchor,puv,rel*last[:,:,None],*hid.shape[-2:])
            stale_flow,_=raster(anchor,puv,rel,*hid.shape[-2:])
            # Same diagnostic support for all modes; no learned permission suppression.
            gate=(support>1e-4).float()
            flows={'gt':flow,'zero':torch.zeros_like(flow),'reverse':-flow,'pred_aligned':pred_flow,'pred_stale':stale_flow}
            er=[]
            for t in range(8):
                sf=F.grid_sample(flow[:,t],gf[:,:,t,None]*2-1,align_corners=True).squeeze(-1).transpose(1,2)
                err=(gf[:,:,t]-sf-ga).norm(dim=-1)*127
                if aligned[:,:,t].any(): er.extend(err[aligned[:,:,t]].tolist())
            backmap[str(kernel)]=float(np.mean(er)) if er else None
            for mode,fl in flows.items():
                fl128=F.interpolate(fl.flatten(0,1),(128,128),mode='bilinear',align_corners=True).reshape(1,8,2,128,128)
                warped=core.backward_warp(last_rgb[None],fl128)[0]
                err=(warped-target).square().mean(1,keepdim=True)
                rgb[f'k{kernel}_{mode}']={n:mean_mask(err,m) for n,m in regions.items()}
                if mode=='zero': continue
                delta=(core.backward_warp(src,fl)-src[:,None])*gate
                for gain in [.25,1.,4.]:
                    corrected=core.decode_frozen(backbone,hid+(gain*delta).clamp(-.12,.12),skip)[0]
                    pix=(corrected-tau).clamp(-.08,.08)
                    pix*= (.002/pix.abs().mean().clamp_min(1e-9)).clamp(max=1)
                    pred=(tau+pix).clamp(0,1); mm=metrics(pred,target)
                    mm['actual_budget']=float((pred-tau).abs().mean()); mm['moving_mse']=mean_mask((pred-target).square().mean(1,keepdim=True),regions['moving_visible'])
                    feature[f'k{kernel}_{mode}_g{gain}']=mm
        baseline=metrics(tau,target); baseline['moving_mse']=mean_mask((tau-target).square().mean(1,keepdim=True),regions['moving_visible'])
        row={'path':path,'group':list(prev.group(path)),'partition':'tune' if i in tune else 'held','point':point,'rgb':rgb,'feature':feature,'baseline':baseline,'backmap_px':backmap,
             'counts':{'visible_pairs':int(pair.sum()),'aligned_pairs':int(aligned.sum())},'region_pixels':{n:int(m.sum()) for n,m in regions.items()}}
        rows.append(row)
        with (OUT/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'clips':i+1,'seconds':time.time()-start}),flush=True)
    summary={'clips':len(rows),'seconds':time.time()-start,'scope':'40 validation video groups, one clip each; 20 tune / 20 held diagnostic split, not untouched test. GT visibility/support diagnostic only. Regions are sparse-track projections, not dense segmentation; motion includes camera motion. Budget is maximum, not forced equal. No training.', 'point':{},'rgb':{},'feature_selected':{},'backmap_px':{}}
    for n in rows[0]['point']:
        summary['point'][n]={r:ci([x['point'][n][r]-x['point']['zero'][r] for x in rows if x['point'][n][r] is not None]) for r in rows[0]['point'][n]}
    for n in rows[0]['rgb']:
        zero=n.split('_')[0]+'_zero'
        summary['rgb'][n]={r:ci([x['rgb'][n][r]-x['rgb'][zero][r] for x in rows if x['rgb'][n][r] is not None]) for r in rows[0]['rgb'][n]}
    for mode in ['gt','reverse','pred_aligned','pred_stale']:
        names=[n for n in rows[0]['feature'] if '_'+mode+'_g' in n]
        best=min(names,key=lambda n:np.mean([x['feature'][n]['cost'] for x in rows if x['partition']=='tune']))
        held=[x for x in rows if x['partition']=='held']
        summary['feature_selected'][mode]={'selected':best,'held_delta_vs_tau':{m:ci([x['feature'][best][m]-x['baseline'][m] for x in held if x['baseline'].get(m) is not None]) for m in ['mse','mae','lpips','temporal','cost','moving_mse']},'held_mean_budget':float(np.mean([x['feature'][best]['actual_budget'] for x in held]))}
    for k in ['9','33']: summary['backmap_px'][k]=ci([x['backmap_px'][k] for x in rows if x['backmap_px'][k] is not None])
    prev.dump(OUT/'summary.json',summary); prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start}); print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
