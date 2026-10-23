"""TTPO-inspired pixel-residual mechanism pilot, NOT diffusion TTPO reproduction.

Frozen scorer supplies preferred/dispreferred examples; optimize current video
with pairwise gradients, history anchoring and baseline temporal increments.
All labels are read only AFTER all branches finish. No training or test tuning.
"""
import argparse,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import iterative_video_repair as ir

BUDGET=.002
PIXEL_CAP=.03
MODES=['select','guided','no_preference','swapped']


def pref_loss(x,win,lose,mask):
    high=lambda a:a-ir.low(a)
    z,p,n=high(x),high(win),high(lose)
    den=(mask.sum()*x.shape[1]).clamp_min(1)
    distance=lambda a,b:((a-b).square()*mask).sum()/den
    scale=distance(p,n).detach().clamp_min(1e-5)
    return F.softplus((distance(z,p)-distance(z,n))/scale)


def constraints(x,base,history,confidence):
    # Preserve low-frequency layout, high-frequency historical evidence only
    # where cached align/velocity agree; confidence is NOT true visibility.
    low=(ir.low(x)-ir.low(base)).square().mean()/1e-4
    texture=lambda a:a-ir.low(a)
    content=((texture(x)-texture(history)).square()*confidence).mean()/1e-4
    delta=x-base
    temporal=(delta[1:]-delta[:-1]).square().mean()/1e-4
    return low+.25*content+temporal


def project(x,base):
    d=(x.clamp(0,1)-base).clamp(-PIXEL_CAP,PIXEL_CAP)
    d=d*min(1.,BUDGET/max(float(d.abs().mean()),1e-12))
    return (base+d).detach()


def optimize(current,base,win,lose,sources,mode,steps=5):
    if mode=='swapped':win,lose=lose,win
    mask=((win-current).abs().mean(1,keepdim=True)+(lose-current).abs().mean(1,keepdim=True)>1e-8).float()
    if not mask.any():return current.clone(),{'movement':0.,'pair_change':0.}
    history=(sources['align']+sources['velocity'])/2
    confidence=torch.exp(-(sources['align']-sources['velocity']).abs().mean(1,keepdim=True)/.06).detach()
    start_pref=float(pref_loss(current,win,lose,mask));x=current.detach()
    for _ in range(steps):
        x=x.detach().requires_grad_(True)
        loss=constraints(x,base,history,confidence)
        if mode!='no_preference':loss=loss+pref_loss(x,win,lose,mask)
        grad=torch.autograd.grad(loss,x)[0]*mask
        assert torch.isfinite(grad).all()
        denom=(grad.abs().sum()/(mask.sum()*x.shape[1]).clamp_min(1)).clamp_min(1e-8)
        # Fixed masked normalized-gradient step; no validation quality line search.
        x=project(x-.001*(grad/denom).clamp(-2,2),base)
    return x,{'movement':float((x-current).abs().mean()),'pair_change':float(pref_loss(x,win,lose,mask))-start_pref}


@torch.no_grad()
def pair(model,state,sources):
    candidates=ir.propose(state,sources)
    names=list(candidates)
    scores=model(torch.stack([ir.features(state,candidates[n],sources) for n in names]))[:,0]
    top=int(scores.argmax());bottom=int(scores.argmin())
    return candidates,names[top],names[bottom],{n:float(v) for n,v in zip(names,scores)}


def generate(tau,sources,model):
    outputs={'tau':tau.clone()};logs=[]
    for mode in MODES:
        state=ir.RepairState.start(tau,budget=BUDGET)
        for step in range(3):
            cand,positive,negative,scores=pair(model,state,sources)
            before=state.current.clone()
            if mode=='select':
                after=project(cand[positive],tau);diag={}
            else:
                after,diag=optimize(before,tau,cand[positive],cand[negative],sources,mode)
            assert float((after-tau).abs().mean())<=BUDGET+1e-7
            assert float((after-tau).abs().max())<=PIXEL_CAP+1e-6
            # Keep is offered at each round, but calibrated risk rejection is NOT
            # invoked: this is an explicit forced-guidance diagnostic, not safe use.
            state.stack.append(before);state.current=after;state.steps+=1
            outputs[f'{mode}_{step+1}']=after.clone()
            logs.append({'mode':mode,'round':step+1,'positive':positive,'negative':negative,'scores':scores,
                         'diagnostics':diag,'positive_image':cand[positive],'negative_image':cand[negative]})
    return outputs,logs


def ci(x):
    x=np.array(x);rng=np.random.default_rng(42)
    return {'mean':float(x.mean()),'ci95':np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    import train_propagation_fusion as t
    from build_iterative_repair_diagnostic import metrics
    import lpips
    torch.set_num_threads(2);torch.manual_seed(42);start=time.time()
    paths=json.loads((t.ROOT/'history_ablation_val_locked_20261015/manifest.json').read_text())['val']
    # First manifest clip per original video, independent of outcomes: 40 videos.
    chosen={}
    for i,path in enumerate(paths):chosen.setdefault(tuple(t.prev.group(path)),(i,path))
    checkpoint=t.ROOT/'iterative_preference_v1_20261017/best.pt'
    model=ir.PreferenceEvaluator().cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
    lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)
    protocol={'method':'pixel-space TTPO-inspired pair-guidance; no diffusion prior, not full TTPO',
      'selection':'first manifest clip per original validation video, no outcome filtering','clips':len(chosen),
      'modes':MODES,'rounds':3,'steps_per_round':5,'budget':BUDGET,'pixel_cap':PIXEL_CAP,
      'constraints':'baseline lowpass, agreement-weighted historical highpass, correction temporal increments',
      'preference':'frozen v1 predicted gain top/bottom INCLUDING keep and undo; no ground truth',
      'limitations':['known imperfect scorer; forced guidance bypasses failed calibration explicitly',
                    'cached history propagation; no flow regeneration; temporal constraint is not true motion consistency',
                    'same budget ceiling, not matched realized budget; report realized changes',
                    'closed-loop arms diverge after first round; first round shares exact candidates/pair',
                    'three-round candidate selection has same outer candidate count, but no inner-gradient compute',
                    'reused validation, not fresh benchmark; no parameter sweep'],
      'training':False,'test_used':False,'TUM_used':False,
      'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for j,(group,(i,path)) in enumerate(chosen.items()):
        d=torch.load(t.ROOT/'cpu_rejection_smoothing_20261016/cache'/f'{i:04d}.pt',map_location='cpu',weights_only=False)
        tau=d['tau'].float().cuda();sources={k:d[k].float().cuda() for k in ['align','velocity']}
        outputs,logs=generate(tau,sources,model)
        # Future truth is never an input to generation, ranking or optimization.
        gt=d['target'].float().cuda()
        with torch.no_grad():
            mets={n:{**metrics(v,gt,lp),'budget':float((v-tau).abs().mean())} for n,v in outputs.items()}
            for z in logs:
                positive=metrics(z.pop('positive_image'),gt,lp);negative=metrics(z.pop('negative_image'),gt,lp)
                z['offline_positive_minus_negative']={k:positive[k]-negative[k] for k in positive}
        row={'group':list(group),'path':path,'metrics':mets,'trace':logs};rows.append(row)
        with (out/'per_video.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        elapsed=time.time()-start
        print(json.dumps({'done':j+1,'total':len(chosen),'seconds':elapsed,'remaining_seconds':elapsed/(j+1)*(len(chosen)-j-1)}),flush=True)
    means={n:{k:float(np.mean([r['metrics'][n][k] for r in rows])) for k in rows[0]['metrics'][n]} for n in rows[0]['metrics']}
    comparisons={}
    for step in range(1,4):
        name=f'guided_{step}'
        for ref in ['tau',f'select_{step}',f'no_preference_{step}',f'swapped_{step}']:
            comparisons[name+'-vs-'+ref]={k:ci([r['metrics'][name][k]-r['metrics'][ref][k] for r in rows]) for k in ['mse','lpips','temporal']}
    summary={'video_equal':means,'paired_ci':comparisons,'videos':len(rows),'per_source':{},'pair_ordering':{}}
    for source in sorted({r['group'][0] for r in rows}):
        rr=[r for r in rows if r['group'][0]==source]
        summary['per_source'][source]={'videos':len(rr),'metrics':{n:{k:float(np.mean([r['metrics'][n][k] for r in rr])) for k in ['mse','lpips','temporal']} for n in means}}
    for step in range(1,4):
        trace=[z for r in rows for z in r['trace'] if z['mode']=='guided' and z['round']==step]
        summary['pair_ordering'][step]={'positive_better_lpips':sum(z['offline_positive_minus_negative']['lpips']<0 for z in trace),'total':len(trace)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'COMPLETE.json').write_text(json.dumps({'seconds':time.time()-start}));print('COMPLETE',flush=True)

if __name__=='__main__':main()
