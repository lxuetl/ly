"""Frozen feature-transport flow intervention, paired identical regions."""
import json,time,hashlib,random,argparse
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import train_single_region_risk_expanded as e
tr=e.tr;prev=tr.prev;core=prev.core
OUT=e.p.run.fit.ROOT/'transport_direction_audit_20261013'
NAMES=['correct','zero','reverse','shuffle']

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);prev.prior.seed_all(42);start=time.time()
    paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']['val'][:args.limit or None]
    base=e.p.run.fit.ROOT
    cfg=SimpleNamespace(backbone_checkpoint=str(base/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(base/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(base/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    backbone,joint,adapter,_=prev.old.load_frozen(cfg,torch.device('cuda'))
    ds=prev.VideoForecastDataset('','val',128,paths=paths)
    import lpips
    lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def cost(pred,target):return (pred-target).square().mean()+.03*lp(pred*2-1,target*2-1).mean()+.1*((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    protocol={'clips':len(paths),'test_used':False,'frozen':True,'variants':NAMES,
      'shuffle':'same spatial permutation applied to all times, per clip seeded; preserves global vector values, changes assignment',
      'matched':'same source/hidden/support/permission, feature budget .006 cap .12, pixel budget reference cap .08, same sampled two-frame regions; normalization policy equal, realized magnitudes logged',
      'zero_note':'zero flow still propagates identity-warp source features; not no-op',
      'reference':'geometry output; unchanged elsewhere; Oracle selects one sampled region or abstains; no future input to candidates'}
    prev.dump(OUT/'protocol.json',protocol)
    file=OUT/'rows.jsonl';rows=[json.loads(x) for x in file.read_text().splitlines()] if file.exists() else [];done={r['path'] for r in rows}
    for i,path in enumerate(paths):
        if path in done:continue
        x=ds[i];observed={k:x[k][None].cuda() for k in prev.PAST};observed['context_rgb']=x['context_rgb'][None].cuda()
        source,hidden,skip=core.extract_frozen_features(backbone,observed['context_rgb']);tau=core.decode_frozen(backbone,hidden,skip)
        bundle=core.geometry_bundle('correct',joint,observed,8);adapter.mode='correct';corrected,_,_=adapter(source,hidden,bundle,bundle)
        flow=adapter.last['flow'].clone();support=adapter.last['support'].clone()
        perm=((F.interpolate(support.flatten(0,1),(128,128),mode='bilinear',align_corners=False).reshape(1,8,1,128,128)-.01)/.05).clamp(0,1)
        geo=(tau+(prev.prior.match_pixel_budget(core.decode_frozen(backbone,corrected,skip),tau,.002,'fixed',.08)-tau)*perm).clamp(0,1)
        budget=(geo-tau).abs().mean();key=hashlib.sha256(path.encode()).hexdigest()[:24]
        gen=torch.Generator(device='cuda');gen.manual_seed(int(key[:12],16))
        order=torch.randperm(flow.shape[-2]*flow.shape[-1],generator=gen,device='cuda')
        flows={'correct':flow,'zero':torch.zeros_like(flow),'reverse':-flow,'shuffle':flow.flatten(-2)[...,order].reshape_as(flow)}
        candidates={};magnitudes={}
        for name,f in flows.items():
            transport=(core.backward_warp(source,f)-hidden)*support
            transport=(transport*(.006/transport.abs().mean().clamp_min(1e-7))).clamp(-.12,.12)
            delta=(core.decode_frozen(backbone,hidden+transport,skip)-tau)*perm
            pred=(tau+(delta*(budget/delta.abs().mean().clamp_min(1e-9))).clamp(-.08,.08)).clamp(0,1)
            candidates[name]=pred[0];magnitudes[name]={'pixel':float((pred-tau).abs().mean()),'feature':float(transport.abs().mean())}
        # Candidates have been formed before loading future target.
        cached=e.p.run.load(path)
        error=float((candidates['correct'].cpu()-cached['bank'][5]).abs().max())
        mean_error=float((candidates['correct'].cpu()-cached['bank'][5]).abs().mean())
        # Fresh four-way paired audit, not subtraction against historical outputs.
        # Preserve cache differences for audit; only guard gross model/data mismatch.
        assert error<.05 and mean_error<.001,('gross candidate reproduction mismatch',error,mean_error,path)
        lab=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False)
        cells=torch.nonzero(lab['mask'][:,0]).tolist();target=x['future_rgb'].cuda();refcost=cost(geo[0],target)
        trials=[]
        for t,y,z in cells:
            result={}
            for name,pred in candidates.items():
                bank=torch.stack([tau[0],geo[0],pred])[None];w=torch.zeros(1,8,3,8,8,device='cuda');w[:,:,1]=1;w[:,2*t:2*t+2,:,y,z]=0;w[:,2*t:2*t+2,2,y,z]=1
                output=tr.synthesize(bank,w,budget[None])[0];result[name]=float(refcost-cost(output,target))
            trials.append({'tube':t,'y':y,'x':z,'gains':result})
        row={'path':path,'group':list(prev.group(path)),'reproduction_max':error,'reproduction_mean':mean_error,'magnitudes':magnitudes,'trials':trials}
        rows.append(row)
        with file.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'clips':len(rows),'total':len(paths),'seconds':time.time()-start}),flush=True)
    def aggregate(rs):
        groups=defaultdict(list)
        for r in rs:groups[tuple(r['group'])].append(r)
        out={}
        for horizon in ['all','0','1','2','3']:
            vm={name:[] for name in NAMES};rm={name:[] for name in NAMES}
            for gr in groups.values():
                for name in NAMES:
                    values=[[q['gains'][name] for q in r['trials'] if horizon=='all' or q['tube']==int(horizon)] for r in gr]
                    vm[name].append(float(np.mean([max([0.]+v) for v in values])))
                    rm[name].append(float(np.mean([np.mean(v) if v else 0 for v in values])))
            ci={};rng=np.random.default_rng(42)
            for name in NAMES[1:]:
                delta=np.array(vm['correct'])-vm[name];boots=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
                ci[name]={'correct_minus_control':float(delta.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist()}
            out[horizon]={'oracle_video_mean':{k:float(np.mean(v)) for k,v in vm.items()},'uniform_region_video_mean':{k:float(np.mean(v)) for k,v in rm.items()},'oracle_paired_ci':ci}
        return out
    summary={'clips':len(rows),'videos':len({tuple(r['group']) for r in rows}),'aggregate':aggregate(rows),
       'by_source':{d:aggregate([r for r in rows if r['group'][0]==d]) for d in sorted({r['group'][0] for r in rows})},
       'mean_actual_pixel_budget':{n:float(np.mean([r['magnitudes'][n]['pixel'] for r in rows])) for n in NAMES},
       'max_reproduction_error':max(r['reproduction_max'] for r in rows)}
    prev.dump(OUT/'summary.json',summary);prev.dump(OUT/'COMPLETE.json',{'clips':len(rows),'seconds':time.time()-start})
    print(json.dumps({'stage':'complete','summary':summary['aggregate']}),flush=True)

if __name__=='__main__':main()
