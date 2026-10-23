"""Frozen validation diagnostics, original-source clustered bootstrap, no training."""
import json,time,hashlib,re
from pathlib import Path
from collections import defaultdict
import numpy as np,torch,lpips
from internal_geometry import Model,Data,RUN,BASE
from stage1_baseline import ROOT,save
from token_audit import metrics
OUT=Path('/root/autodl-tmp/predformer_internal_diagnostics_20261022')
def means(rows,keys):
    return {k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None for k in keys}
def main():
    torch.set_num_threads(2);torch.manual_seed(42);OUT.mkdir(exist_ok=False);start=time.time()
    splits=json.loads((RUN/'split_paths.json').read_text())['paths'];manifest=json.loads((RUN/'manifest.json').read_text());mapping={}
    for p in splits['train']+splits['val']:
        pp=Path(p);pp=pp if pp.is_absolute() else ROOT/pp
        with np.load(pp) as z:meta=json.loads(str(z['metadata_json']))
        sp=Path(meta['source_path']);sp=sp if sp.is_absolute() else ROOT/sp
        source=manifest[p]['source'];method='filename without terminal clip index'
        if meta['subset']=='po50':
            with np.load(sp) as z:original=json.loads(str(z['metadata_json']))
            source='po50/'+original['source_sequence'];method='source_sequence metadata'
        elif meta['subset']=='ds_mini':source='ds_mini/'+meta['video_name'].split('_obj_')[0];method='conservative shared DS source prefix'
        mapping[p]={'source':source,'domain':meta['subset'],'method':method}
    train_sources={mapping[p]['source'] for p in splits['train']};val_sources={mapping[p]['source'] for p in splits['val']}
    save(OUT/'grouping.json',{'paths':mapping,'val_sequences':len(val_sources),'train_val_source_overlap':sorted(train_sources&val_sources)})
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);models={};hashes={}
    for arm in ['plain','rgb','null','omega']:
        path=RUN/arm/'best.pt';hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
        m=Model(ck,arm).cuda();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models[arm]=m.eval().requires_grad_(False)
    net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    ds=Data(splits['val'],BASE/'rgb_cache','omega');cliprows=[];frames=[]
    save(OUT/'protocol.json',{'checkpoint_hashes':hashes,'training':False,'test_used':False,'val_clips':len(ds),'val_sequences':len(val_sources),'bootstrap':'5000 paired original-source cluster resamples, seed42; macro sequence and clip-weighted cluster CI; unadjusted descriptive validation CIs after checkpoint selection','mask':'mean RGB abs(GT_future-last_observed)>0.05; fixed before evaluation; change proxy NOT object-motion truth','regional_lpips':'mean spatial LPIPS map within mask; receptive fields cross boundaries; not standard cropped-image LPIPS','spatial_shuffle':'three fixed permutations, identical across timestamps, content shuffled with positional coordinates held fixed','camera_only':'remove spatial K/V after projection; one camera K/V per frame','spatial_only':'remove camera K/V after projection; sixteen spatial K/V per frame','paired_permutation':'permute content+coordinate K/V together; invariance sanity check','grouping_caveat':'PO original metadata; legacy filename grouping may not establish physical scene independence','code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with torch.inference_mode():
        for i in range(len(ds)):
            x,y,p,g=ds[i];x=x[None].cuda();y=y.cuda();g=g[None].cuda();info=mapping[p]
            candidates=[j for j,q in enumerate(ds.paths) if mapping[q]['source']!=info['source']]
            j=next((j for j in candidates if mapping[ds.paths[j]]['domain']==info['domain']),candidates[0]);dx,_,dp,dg=ds[j]
            preds={}
            with torch.autocast('cuda',dtype=torch.float16):
                for arm,m in models.items():preds[arm]=m(x,g).float()[0]
                m=models['omega'];preds['omega_mismatch']=m(x,g,external=dg[None].cuda().to(torch.float16)).float()[0]
                preds['omega_bypass']=m(x,g,mode='bypass').float()[0]
                for name,idx in [('omega_camera_only',[16]),('omega_spatial_only',list(range(16)))]:
                    def hook(module,args,idx=idx):return (args[0],args[1][:,idx],args[2][:,idx])
                    h=m.read.register_forward_pre_hook(hook);preds[name]=m(x,g).float()[0];h.remove()
                for seed in [42,123,3407]:
                    perm=torch.randperm(16,generator=torch.Generator().manual_seed(seed)).to(g.device)
                    shuffled=g.clone();shuffled[:,:,:16]=g[:,:,perm]
                    preds['omega_shuffle_'+str(seed)]=m(x,shuffled).float()[0]
                idx=list(range(15,-1,-1))+[16]
                h=m.read.register_forward_pre_hook(lambda module,args:(args[0],args[1][:,idx],args[2][:,idx]))
                preds['omega_joint_permutation']=m(x,g).float()[0];h.remove()
            mask=(y-x[0,-1]).abs().mean(1)>.05
            for arm,pred in preds.items():
                r=metrics(pred,y,net);r.update(path=p,arm=arm,**info,output_l1_vs_omega=float((pred-preds['omega']).abs().mean()));cliprows.append(r)
                net.spatial=True;lpmap=net(pred.clamp(0,1)*2-1,y*2-1)[:,0];net.spatial=False
                global_lp=net(pred.clamp(0,1)*2-1,y*2-1).flatten()
                sq=(pred-y).square().mean(1);ab=(pred-y).abs().mean(1)
                for t in range(8):
                    rr={'path':p,'arm':arm,**info,'horizon':t+1,'mse':float(sq[t].mean()),'mae':float(ab[t].mean()),'lpips':float(global_lp[t]),'dynamic_fraction':float(mask[t].float().mean())}
                    rr['psnr']=float(-10*np.log10(max(rr['mse'],1e-12)))
                    for region,mm in [('dynamic',mask[t]),('static',~mask[t])]:
                        count=int(mm.sum());rr[region+'_pixels']=count
                        for k,arr in [('mse',sq),('mae',ab),('lpips_proxy',lpmap)]:rr[region+'_'+k]=float(arr[t][mm].mean()) if count else None
                    frames.append(rr)
            if (i+1)%10==0:
                elapsed=time.time()-start;r={'clips':i+1,'total':len(ds),'seconds':elapsed,'eta_seconds':elapsed/(i+1)*(len(ds)-i-1)};save(OUT/'progress.json',r);print(json.dumps(r),flush=True)
    keys=['mse','mae','psnr','lpips','temporal','output_l1_vs_omega']
    summaries={arm:means([r for r in cliprows if r['arm']==arm],keys) for arm in preds}
    for arm in models:
        old=json.loads((RUN/arm/'COMPLETE.json').read_text())['metrics']
        assert abs(summaries[arm]['mse']-old['mse'])<1e-8
    save(OUT/'per_clip.json',cliprows);save(OUT/'per_frame_region.json',frames);save(OUT/'summary.json',summaries)
    fkeys=['mse','mae','psnr','lpips','dynamic_fraction','dynamic_mse','static_mse','dynamic_mae','static_mae','dynamic_lpips_proxy','static_lpips_proxy']
    for field in ['source','domain','horizon']:
        grouped=defaultdict(list)
        for r in frames:grouped[(r['arm'],r[field])].append(r)
        save(OUT/'by_{}.json'.format(field),[{'arm':a,field:v,'frame_records':len(rs),**means(rs,fkeys)} for (a,v),rs in grouped.items()])
    # Pair exact clip paths, then resample entire original-video groups.
    indexes={a:{r['path']:r for r in cliprows if r['arm']==a} for a in preds};cis=[]
    comparisons=[('omega',a) for a in ['plain','rgb','null','omega_mismatch','omega_camera_only','omega_spatial_only','omega_shuffle_42','omega_shuffle_123','omega_shuffle_3407']]
    for a,b in comparisons:
        assert indexes[a].keys()==indexes[b].keys()
        groups=defaultdict(list)
        for p in indexes[a]:groups[mapping[p]['source']].append(p)
        for k in ['mse','mae','psnr','lpips','temporal']:
            diffs=[np.array([indexes[a][p][k]-indexes[b][p][k] for p in ps]) for ps in groups.values()]
            avg=np.array([v.mean() for v in diffs]);sums=np.array([v.sum() for v in diffs]);counts=np.array([len(v) for v in diffs])
            rng=np.random.default_rng(42);ids=rng.integers(0,len(avg),(5000,len(avg)))
            for weighting,estimate,draws in [('video_equal',avg.mean(),avg[ids].mean(1)),('clip_weighted_cluster',sums.sum()/counts.sum(),sums[ids].sum(1)/counts[ids].sum(1))]:
                ci=np.quantile(draws,[.025,.975]);cis.append({'a':a,'b':b,'metric':k,'weighting':weighting,'difference_a_minus_b':float(estimate),'ci95':ci.tolist(),'videos':len(avg),'clips':int(counts.sum())})
    save(OUT/'bootstrap.json',cis)
    assert all(hashlib.sha256((RUN/a/'best.pt').read_bytes()).hexdigest()==h for a,h in hashes.items())
    save(OUT/'COMPLETE.json',{'seconds':time.time()-start,'checkpoint_hashes_unchanged':True,'clips':len(ds),'videos':len(val_sources),'train_val_source_overlap':sorted(train_sources&val_sources)})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
