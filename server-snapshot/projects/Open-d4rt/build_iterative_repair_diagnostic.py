"""Build labeled modified-state diagnostics from original validation caches.

No model fitting, no TUM, no test. Random rollout selection occurs before labels
are accessed. These are exploratory validation records, NOT training data for
subsequent claims on the same validation videos.
"""
import argparse,json,time,hashlib
from pathlib import Path
import torch
import iterative_video_repair as ir


def metrics(x,gt,lp):
    return {'mse':float((x-gt).square().mean()),
            'lpips':float(lp(x*2-1,gt*2-1).mean()),
            'temporal':float(((x[1:]-x[:-1])-(gt[1:]-gt[:-1])).abs().mean())}


def label(c,cur,base,tolerance=.01):
    def safe(ref):return all(c[k] <= ref[k]+tolerance*max(ref[k],1e-6) for k in ['mse','temporal'])
    return {'gain':cur['lpips']-c['lpips'],
            'mse_delta':c['mse']-cur['mse'],'temporal_delta':c['temporal']-cur['temporal'],
            'baseline_mse_delta':c['mse']-base['mse'],
            'baseline_temporal_delta':c['temporal']-base['temporal'],
            'safe_improvement':safe(cur) and safe(base) and c['lpips']<cur['lpips']-1e-4,
            'metrics':c}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=32)
    p.add_argument('--device',default='cuda');p.add_argument('--budget',type=float,default=.025)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    import lpips
    paths=json.loads(Path(args.manifest).read_text())['val']
    assert args.limit>0
    # Original grouping supplied by established pipeline, never random clip split.
    import train_propagation_fusion as t
    records=[];targets={};start=time.time();torch.set_num_threads(2)
    rng=torch.Generator().manual_seed(42)
    protocol={'purpose':'modified-state evaluator diagnostic corpus, not evaluator training',
              'split':'original validation only','feature_version':ir.FEATURE_VERSION,
              'rollout':'seed42 random eligible edit, up to3 rounds; no quality labels in rollout',
              'sources':'cached align and velocity; residuals recomputed, flow NOT recomputed',
              'budget':args.budget,'selection':'first N manifest clips; not representative full validation',
              'limitation':'evaluation of evaluator still requires a frozen scorer trained on disjoint original videos',
              'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    # Phase 1: all trajectories and input features fixed before accessing target key.
    with torch.no_grad():
        for i,path in enumerate(paths[:args.limit]):
            d=torch.load(Path(args.cache)/f'{i:04d}.pt',map_location='cpu',weights_only=False)
            state=ir.RepairState.start(d['tau'].float().to(args.device),budget=args.budget)
            sources={k:d[k].float().to(args.device) for k in ['align','velocity']}
            for step in range(3):
                cand=ir.propose(state,sources)
                prefix=f'{i}:{step}'
                records.append({'id':prefix,'path':path,'group':list(t.prev.group(path)),
                                'step':step,'features':{n:ir.features(state,v,sources).cpu().tolist() for n,v in cand.items()}})
                # Per-clip small intermediates only: bounded host memory, no global GPU cache.
                torch.save({'baseline':state.baseline.cpu(),'current':state.current.cpu(),
                            'candidates':{k:v.cpu() for k,v in cand.items()}},out/f'state-{i}-{step}.pt')
                edits=[n for n in cand if n not in ['keep','undo']]
                action=edits[int(torch.randint(len(edits),(1,),generator=rng))]
                records[-1]['rollout_action']=action
                if not ir.execute(state,action,cand):break
    (out/'features-no-labels.json').write_text(json.dumps(records))
    # Phase 2: ground truth only produces offline labels, never proposals/features/actions.
    lp=lpips.LPIPS(net='alex',verbose=False).to(args.device).eval();labeled=[]
    with torch.no_grad():
        for rec in records:
            i,step=map(int,rec['id'].split(':'))
            gt=torch.load(Path(args.cache)/f'{i:04d}.pt',map_location='cpu',weights_only=False)['target'].float().to(args.device)
            pack=torch.load(out/f'state-{i}-{step}.pt',map_location=args.device,weights_only=False)
            base=metrics(pack['baseline'],gt,lp);cur=metrics(pack['current'],gt,lp)
            labels={n:label(metrics(x,gt,lp),cur,base) for n,x in pack['candidates'].items()}
            labeled.append({'id':rec['id'],'group':rec['group'],'labels':labels})
    (out/'labels.json').write_text(json.dumps(labeled))
    (out/'COMPLETE.json').write_text(json.dumps({'states':len(records),'seconds':time.time()-start,'scorer_evaluated':False}))

if __name__=='__main__':main()
