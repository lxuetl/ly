"""Read-only checkpoint audit: interventions, weight changes, backward-only probes."""
import json,time
from pathlib import Path
import torch,numpy as np,lpips
from residual_queries import ResidualModel
from stage2_queries import BASE
from stage1_baseline import Videos,save
from token_audit import metrics

OUT=Path('/root/autodl-tmp/predformer_residual_adapter_audit_20261021')
RUN=Path('/root/autodl-tmp/predformer_residual_v2_20261021/rgb')
def main():
    torch.set_num_threads(2);torch.manual_seed(42);OUT.mkdir(exist_ok=False)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    m=ResidualModel(ck,'rgb').cuda().eval()
    initial={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('base.')}
    trained=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
    missing=m.load_state_dict(trained['model'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('base.') for k in missing.missing_keys)
    changes={}
    for prefix in ['query.','head.','adapter.']:
        vals=[(v.detach()-initial[k]).float().flatten() for k,v in m.state_dict().items() if k.startswith(prefix)]
        changes[prefix]={'change_l2':float(torch.cat(vals).norm()),'initial_l2':float(torch.cat([v.float().flatten() for k,v in initial.items() if k.startswith(prefix)]).norm())}
    changes['scale']={'initial':float(initial['adapter.scale']),'trained':float(m.adapter.scale)}
    net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    paths=json.loads((BASE/'split_paths.json').read_text())['paths']['val'];ds=Videos(paths,BASE/'rgb_cache')
    prior=json.loads(Path('/root/autodl-tmp/predformer_token_audit_v2_20261021/per_clip.json').read_text())
    groups={r['path']:r['source'] for r in prior};donors={r['path']:r['donor_source'] for r in prior}
    indices={}
    for i,p in enumerate(paths):indices.setdefault(groups[p],i)
    captured={};rows=[];start=time.time();cache={};gradrows=[]
    def capture(mod,args,output):
        captured['mem']=args[1].detach();captured['delta']=float((output-args[0]).float().abs().mean());captured['qabs']=float(args[0].float().abs().mean())
    for i in range(len(ds)):
        x,y,path=ds[i];x=x[None].cuda();y=y.cuda()
        with torch.no_grad():
            h=m.adapter.register_forward_hook(capture)
            with torch.autocast('cuda',dtype=torch.float16):correct=m(x).float()
            h.remove();delta=captured['delta'];qabs=captured['qabs'];normal=captured['mem']
            di=indices[donors[path]]
            if di not in cache:
                dx,_,_=ds[di];h=m.adapter.register_forward_hook(capture)
                with torch.autocast('cuda',dtype=torch.float16):m(dx[None].cuda())
                h.remove();cache[di]=captured['mem']
            preds={'correct':correct}
            for name in ['zero','cross_video','bypass']:
                if name=='bypass':h=m.adapter.register_forward_hook(lambda mod,args,out:args[0])
                else:
                    replacement=torch.zeros_like(normal) if name=='zero' else cache[di]
                    h=m.adapter.register_forward_pre_hook(lambda mod,args:(args[0],replacement,args[2]))
                with torch.autocast('cuda',dtype=torch.float16):preds[name]=m(x).float()
                h.remove()
            for name,p in preds.items():
                r=metrics(p[0],y,net);r.update(path=path,source=groups[path],arm=name,output_l1_vs_correct=float((p-correct).abs().mean()),adapter_delta_l1=delta,query_abs=qabs);rows.append(r)
        if i<16:
            m.zero_grad(set_to_none=True)
            # Full fp32 backward probe avoids conflating AMP scaling with connectivity.
            p=m(x);loss=(p[0]-y).square().mean();loss.backward()
            norms={prefix:float(torch.sqrt(sum((p.grad.float().square().sum() for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None),torch.tensor(0.,device='cuda')))) for prefix in ['query.','head.','adapter.']}
            assert all(p.grad is None for p in m.base.parameters());gradrows.append({'path':path,**norms})
        if (i+1)%40==0:print(i+1,len(ds),time.time()-start,flush=True)
    keys=['mse','mae','lpips','psnr','temporal','output_l1_vs_correct','adapter_delta_l1','query_abs']
    summary={a:{k:float(np.mean([r[k] for r in rows if r['arm']==a])) for k in keys} for a in preds}
    # Ensure diagnostic normal forward reproduces saved checkpoint metrics.
    assert abs(summary['correct']['mse']-trained['metrics']['mse'])<1e-8
    save(OUT/'per_clip.json',rows);save(OUT/'summary.json',summary);save(OUT/'weights.json',changes);save(OUT/'gradients.json',gradrows)
    save(OUT/'COMPLETE.json',{'seconds':time.time()-start,'checkpoint_epoch':trained['epoch'],'test_used':False,'optimizer_steps':0,'gradient_probe':'16 val clips, full fp32, eval mode, backward only; no checkpoint mutation'})
    print(json.dumps({'summary':summary,'weights':changes,'gradient_mean':{k:float(np.mean([r[k] for r in gradrows])) for k in ['query.','head.','adapter.']}}),flush=True)
if __name__=='__main__':main()
