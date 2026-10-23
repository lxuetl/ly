"""Frozen validation-only audit. Memory permutation is an invariance probe."""
import json,time,hashlib,re
from pathlib import Path
import numpy as np
import torch
import lpips
from stage2_queries import QueryModel,BASE
from stage1_baseline import Videos,ROOT,save

OUT=Path('/root/autodl-tmp/predformer_token_audit_v2_20261021')
STAGE=Path('/root/autodl-tmp/predformer_queries_20261021')

def encode(m,x):
    z=m.base.to_patch_embedding(x)+m.base.pos_embedding.to(x.device)
    for block in m.base.blocks:z=block(z)
    return z

def memory(z):
    b,t,p,d=z.shape
    return torch.nn.functional.adaptive_avg_pool2d(z.reshape(b*t,16,16,d).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2).reshape(b,t*16,d)

def image(m,f):
    b,t,p,d=f.shape
    return m.head(f).reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)

def metrics(p,y,net):
    p=p.float();y=y.float();mse=float((p-y).square().mean())
    pc=p.clamp(0,1)
    return {'mse':mse,'mae':float((p-y).abs().mean()),'psnr':float(-10*np.log10(max(mse,1e-12))),
            'temporal':float(((p[1:]-p[:-1])-(y[1:]-y[:-1])).abs().mean()),
            'lpips':float(net(pc*2-1,y*2-1).mean()),'clamped_mse':float((pc-y).square().mean()),
            'out_of_range_fraction':float(((p<0)|(p>1)).float().mean())}

def main():
    torch.set_num_threads(2);torch.manual_seed(42);OUT.mkdir(exist_ok=True)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    models={}
    for arm in ['query','null','rgb']:
        m=QueryModel(ck,arm).cuda().eval()
        state=torch.load(STAGE/arm/'best.pt',map_location='cpu',weights_only=False)
        msg=m.load_state_dict(state['model'],strict=False)
        assert not msg.unexpected_keys and all(k.startswith('base.') for k in msg.missing_keys)
        models[arm]=m
    net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    paths=json.loads((BASE/'split_paths.json').read_text())['paths']['val']
    ds=Videos(paths,BASE/'rgb_cache');groups=[];domains=[]
    for path in paths:
        pp=Path(path);pp=pp if pp.is_absolute() else ROOT/pp
        with np.load(pp) as z:meta=json.loads(str(z['metadata_json']))
        domains.append(meta['subset'])
        name=re.sub(r'_clip_\d+$','',meta['video_name'])
        if name==meta['video_name']:name=re.sub(r'_\d+$','',name)
        groups.append(meta['subset']+'/'+name)
    # Rotating donors from a different sequence in the same source domain.
    donors=[]
    for i in range(len(paths)):
        candidates=[(i+k)%len(paths) for k in range(1,len(paths)) if groups[(i+k)%len(paths)]!=groups[i]]
        donors.append(next((j for j in candidates if domains[j]==domains[i]),candidates[0]))
    save(OUT/'protocol.json',{'test_used':False,'checkpoints_frozen':True,'count':len(paths),'lpips':'alex v0.1; clamp RGB [0,1], normalize [-1,1]; mean frames then clips','other_metrics':'raw predictions; also clamped MSE','time_reverse_memory':'permutes pooled encoded memory tokens, an attention-set invariance probe, NOT a causal temporal test','time_reverse_input':'re-encodes reversed observed RGB frames for adapter memory only; future query input unchanged','donor_group':'subset + video_name with terminal clip index removed; same-domain different sequence','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows=[];start=time.time();donor_cache={}
    with torch.inference_mode():
        for i in range(len(ds)):
            x,y,path=ds[i];x=x[None].cuda();y=y.cuda()
            with torch.autocast('cuda',dtype=torch.float16):
                z=encode(models['rgb'],x);mem=memory(z)
                native=models['rgb'].base(x)[:,:8]
                preds={'native':native}
                for arm,m in models.items():
                    f=m.query(z)
                    if m.adapter is not None:
                        mm=mem if arm=='rgb' else torch.zeros_like(mem)
                        mm=mm[:,None].expand(-1,8,-1,-1)
                        f=m.adapter(f,mm,torch.ones(mm.shape[:3],device='cuda',dtype=torch.bool))
                    preds[arm]=image(m,f)
                m=models['rgb'];f=m.query(z);di=donors[i]
                if di not in donor_cache:
                    dx,_,_=ds[di];donor_cache[di]=memory(encode(m,dx[None].cuda()))
                variants={'rgb_zero':torch.zeros_like(mem),'rgb_cross_video':donor_cache[di],
                          'rgb_time_reverse_input':memory(encode(m,x.flip(1))),
                          'rgb_time_reverse_memory':mem.reshape(1,16,16,256).flip(1).reshape_as(mem)}
                for name,mm in variants.items():
                    mm=mm[:,None].expand(-1,8,-1,-1)
                    preds[name]=image(m,m.adapter(f,mm,torch.ones(mm.shape[:3],device='cuda',dtype=torch.bool)))
                preds['rgb_adapter_bypass']=image(m,f)
            for name,p in preds.items():
                r=metrics(p[0],y,net);r.update(path=path,source=groups[i],arm=name,donor_source=groups[di],donor_same_domain=domains[i]==domains[di],output_l1_vs_rgb=float((p.float()-preds['rgb'].float()).abs().mean()))
                rows.append(r)
            if (i+1)%20==0:
                save(OUT/'progress.json',{'clips':i+1,'total':len(ds),'seconds':time.time()-start});print(i+1,len(ds),time.time()-start,flush=True)
    save(OUT/'per_clip.json',rows)
    keys=['mse','mae','psnr','temporal','lpips','clamped_mse','out_of_range_fraction','output_l1_vs_rgb']
    result={name:{k:float(np.mean([r[k] for r in rows if r['arm']==name])) for k in keys} for name in preds}
    save(OUT/'summary.json',result);save(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
