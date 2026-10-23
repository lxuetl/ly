"""Train modified-state scorer using original training videos only; no test/TUM.

Resume compact label caches; distinct original-video fit/dev/calibration subsets.
Final validation is a forced-state decision diagnostic, NOT rollout improvement.
"""
import argparse,json,time,random,hashlib,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import iterative_video_repair as ir
from build_iterative_repair_diagnostic import metrics,label
import train_propagation_fusion as t
import evaluate_fixed_history_expanded as e
import history_texture_candidates as h


def dump(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(path)


def warp(last,flow):
    ht,wd=last.shape[-2:];yy,xx=torch.meshgrid(torch.arange(ht,device=last.device),torch.arange(wd,device=last.device),indexing='ij')
    fl=torch.as_tensor(flow,device=last.device);grid=torch.stack([2*(xx+fl[...,0])/(wd-1)-1,2*(yy+fl[...,1])/(ht-1)-1],-1)[None]
    return F.grid_sample(last,grid,mode='bicubic',padding_mode='border',align_corners=True).clamp(0,1)


@torch.no_grad()
def collect(path,index,split,lp,budget):
    if split=='val':
        d=torch.load(t.ROOT/'cpu_rejection_smoothing_20261016/cache'/f'{index:04d}.pt',map_location='cpu',weights_only=False)
        tau=d['tau'].float().cuda();sources={k:d[k].float().cuda() for k in ['align','velocity']}
    else:
        d=torch.load(t.OUT/'cache'/(t.key(path)+'.pt'),map_location='cpu',weights_only=False)
        assert d['path']==path
        tau=d['tau'].float().cuda();rgb,_=e.rgb24(path);obs=rgb[:16].half().float();last=obs[-1:].cuda();base=h.flow(obs[-1],obs[-2])
        sources={'align':torch.cat([warp(last,h.flow(tau[j].cpu(),obs[-1])) for j in range(8)]),
                 'velocity':torch.cat([warp(last,base*(j+1)) for j in range(8)])}
        # Detect incompatible source/target indexing rather than silently mixing clips.
        assert torch.equal(rgb[16:24].half(),d['target'].half())
    state=ir.RepairState.start(tau,budget=budget)
    rng=random.Random(int(hashlib.sha256(path.encode()).hexdigest()[:8],16));pending=[]
    for step in range(3):
        candidates=ir.propose(state,sources)
        pending.append((step,{n:ir.features(state,x,sources).cpu().tolist() for n,x in candidates.items()},
                        {n:x.clone() for n,x in candidates.items()},state.current.clone()))
        choices=[n for n in candidates if n not in ['keep','undo']]
        if not ir.execute(state,rng.choice(choices),candidates):break
    # Only now read future truth. It never controls random rollouts or features.
    gt=d['target'].float().cuda();base_metrics=metrics(tau,gt,lp);rows=[]
    for step,feat,cand,current in pending:
        current_metrics=metrics(current,gt,lp)
        rows.append({'path':path,'group':list(t.prev.group(path)),'step':step,'features':feat,
                     'labels':{n:label(metrics(x,gt,lp),current_metrics,base_metrics) for n,x in cand.items()}})
    return rows


def tensors(rows):
    xs=[];ys=[];safe=[];pairs=[];meta=[]
    for r in rows:
        start=len(xs)
        for n,x in r['features'].items():
            z=r['labels'][n];xs.append(x)
            ys.append([z[k] for k in ['gain','mse_delta','temporal_delta','baseline_mse_delta','baseline_temporal_delta']])
            safe.append(z['safe_improvement']);meta.append((r,n))
        pairs.extend((i,j) for i in range(start,len(xs)) for j in range(start,i))
    return (torch.tensor(xs,device='cuda'),torch.tensor(ys,device='cuda'),
            torch.tensor(safe,device='cuda'),torch.tensor(pairs,device='cuda',dtype=torch.long).reshape(-1,2),meta)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--budget',type=float,default=.025);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'records').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    ps=json.loads((t.OUT/'split_paths.json').read_text())['paths']
    val_manifest=json.loads((t.ROOT/'history_ablation_val_locked_20261015/manifest.json').read_text())['val']
    assert ps['val']==val_manifest
    groups={k:{tuple(t.prev.group(p)) for p in ps[k]} for k in ['train','val','test']}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    ordered=sorted(groups['train']);random.Random(42).shuffle(ordered);n=max(1,len(ordered)//10)
    cal=set(ordered[:n]);dev=set(ordered[n:2*n]);fit=set(ordered[2*n:]);assert fit
    protocol={'epochs':args.epochs,'budget':args.budget,'limit':args.limit,'seed':42,'tau_frozen':True,
              'feature_version':ir.FEATURE_VERSION,'groups':{'fit':sorted(fit),'dev':sorted(dev),'calibration':sorted(cal)},
              'counts':{k:len(v) for k,v in ps.items()},'test_used':False,'TUM_used':False,
              'selection':'best dev normalized supervised loss; original training videos only',
              'calibration':'95th percentile of per-original-video maximum one-sided residual, conservative empirical margins',
              'validation':'modified-state decisions on random forced trajectories; NOT autonomous rollout quality',
              'train_sources':'cached frozen TAU; RGB-only Farneback from historical frames',
              'hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'iterative_video_repair.py']}}
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:dump(out/'protocol.json',protocol)
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval();start=time.time();sets={'fit':[],'dev':[],'calibration':[],'val':[]}
    total=sum(min(len(ps[k]),args.limit) if args.limit else len(ps[k]) for k in ['train','val']);done=0
    for split in ['train','val']:
        paths=ps[split][:args.limit] if args.limit else ps[split]
        for i,path in enumerate(paths):
            dest=out/'records'/(split+'-'+t.key(path)+'.json')
            if dest.exists():rows=json.loads(dest.read_text())
            else:rows=collect(path,i,split,lp,args.budget);dump(dest,rows)
            group=tuple(t.prev.group(path));bucket='val' if split=='val' else ('calibration' if group in cal else 'dev' if group in dev else 'fit')
            sets[bucket].extend(rows);done+=1
            if done%10==0:
                elapsed=time.time()-start;progress={'stage':'candidate_labels','done':done,'total':total,'elapsed':elapsed,'remaining_seconds':elapsed/done*(total-done)}
                dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    del lp;torch.cuda.empty_cache();dump(out/'DATA_COMPLETE.json',{'states':{k:len(v) for k,v in sets.items()},'seconds':time.time()-start})
    assert all(sets.values()),'Need nonempty original-video fit/dev/cal/val subsets'
    data={k:tensors(v) for k,v in sets.items()};x,y,safe,pairs,_=data['fit'];model=ir.PreferenceEvaluator().cuda()
    model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0).clamp_min(1e-5));target_scale=y.std(0).clamp_min(1e-5)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);best=float('inf');train_start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train()
        # Full compact feature batches; 20 optimizer updates per epoch.
        for _ in range(20):
            pred=model(x);loss=ir.preference_loss(pred,y,safe,pairs,target_scale)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            dx,dy,ds,dp,_=data['dev'];dev_loss=float(ir.preference_loss(model(dx),dy,ds,dp,target_scale))
        record={'epoch':epoch,'train_loss':float(loss),'dev_loss':dev_loss,'elapsed':time.time()-train_start}
        dump(out/'progress.json',record);print(json.dumps(record),flush=True)
        torch.save({'model':model.state_dict(),'epoch':epoch,'target_scale':target_scale,'protocol':protocol},out/'last.pt')
        if dev_loss<best:
            best=dev_loss;torch.save({'model':model.state_dict(),'epoch':epoch,'target_scale':target_scale,'protocol':protocol},out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=False)['model']);model.eval()
    with torch.no_grad():
        cx,cy,_,_,cm=data['calibration'];cp=model(cx)[:,:5]
        residual=cy-cp;residual[:,0]=-residual[:,0];vg={}
        for j,(r,_) in enumerate(cm):vg.setdefault(tuple(r['group']),[]).append(residual[j])
        maxima=torch.stack([torch.stack(v).max(0).values for v in vg.values()])
        margins=torch.quantile(maxima,.95,dim=0,interpolation='higher').clamp_min(0)
        dump(out/'calibration.json',{'margins':margins.tolist(),'videos':len(vg),'guarantee':False})
        vx,vy,vs,_,vm=data['val'];vp=model(vx);decisions=[];offset=0
        for r in sets['val']:
            names=list(r['features']);pr={}
            for j,name in enumerate(names):
                p=vp[offset+j];pr[name]={'gain_lower':float(p[0]-margins[0]),'mse_harm_upper':float(p[1]+margins[1]),'temporal_harm_upper':float(p[2]+margins[2]),'baseline_harm_upper':float(torch.maximum(p[3]+margins[3],p[4]+margins[4]))}
            choice=ir.choose(pr,calibrated=True);lab=r['labels'][choice]
            decisions.append({'path':r['path'],'group':r['group'],'step':r['step'],'choice':choice,'label':lab,'safe_available':any(z['safe_improvement'] for z in r['labels'].values()),'undo_safe':r['labels'].get('undo',{}).get('safe_improvement',False)})
            offset+=len(names)
        dump(out/'validation_decisions.json',decisions)
        summary={}
        for step in range(3):
            rows=[r for r in decisions if r['step']==step];accepted=[r for r in rows if r['choice']!='keep']
            summary[str(step)]={'states':len(rows),'accepted':len(accepted),'safe_accepted':sum(r['label']['safe_improvement'] for r in accepted),'unsafe_accepted':sum(not r['label']['safe_improvement'] for r in accepted),'safe_available':sum(r['safe_available'] for r in rows),'undo_available_safe':sum(r['undo_safe'] for r in rows),'undo_selected':sum(r['choice']=='undo' for r in rows)}
        dump(out/'summary.json',{'per_step':summary,'warning':'forced-state diagnostic; unsafe means fails composite safe-improvement criteria, not necessarily worsens every metric'})
    dump(out/'COMPLETE.json',{'seconds':time.time()-start,'training':True,'test_used':False});print('COMPLETE',flush=True)

if __name__=='__main__':main()
