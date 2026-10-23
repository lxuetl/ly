"""Sequential G0/G1/G2 then matched global-vs-local one-shot scorer pilot."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,OUT,PROJECT,SEED,manifest,dump,sha,key,group,seed,lpips_model,metrics,summarize,ci,pixel_bound
from local_risk import RiskScorer,local_descriptor,choose,supervised_loss
import iterative_video_repair as ir
import train_propagation_fusion as t
import evaluate_fixed_history_expanded as e
import history_texture_candidates as h

NAMES=['align_rgb','align_texture','velocity_rgb','velocity_texture']
BUDGET=.002


def warp(last,fl):
    ht,wd=last.shape[-2:];yy,xx=torch.meshgrid(torch.arange(ht,device=last.device),torch.arange(wd,device=last.device),indexing='ij')
    fl=torch.as_tensor(fl,device=last.device)
    grid=torch.stack([2*(xx+fl[...,0])/(wd-1)-1,2*(yy+fl[...,1])/(ht-1)-1],-1)[None]
    return F.grid_sample(last,grid,mode='bicubic',padding_mode='border',align_corners=True).clamp(0,1)


def get_sources(path,split):
    # Regenerate using the exact first-round source definition from prior scorer.
    d=torch.load(t.OUT/'cache'/(key(path)+'.pt'),map_location='cpu',weights_only=False)
    assert d['path']==path
    base=d['tau'].float().cuda();rgb,_=e.rgb24(path);obs=rgb[:16].half().float()
    assert torch.equal(rgb[16:24].half(),d['target'].half())
    last=obs[-1:].cuda();fl=h.flow(obs[-1],obs[-2])
    sources={'align':torch.cat([warp(last,h.flow(base[k].cpu(),obs[-1])) for k in range(8)]),
             'velocity':torch.cat([warp(last,fl*(k+1)) for k in range(8)])}
    # Read label tensor only after no-label source construction.
    return base,sources,d['target'].float().cuda()


def candidates(base,sources):
    state=ir.RepairState.start(base,budget=BUDGET,max_steps=1)
    bank=ir.propose(state,sources)
    # Reuse region/alpha exactly, add shared pixel cap identically to every arm.
    bank={n:pixel_bound(bank[n][None],base[None],BUDGET)[0] for n in NAMES}
    return state,bank


@torch.no_grad()
def prepare(m,lp):
    folder=OUT/'local_risk/cache';folder.mkdir(parents=True,exist_ok=True)
    started=time.time();total=sum(len(m[k]) for k in ['fit','dev','calibration','evaluation']);done=0
    for split in ['fit','dev','calibration','evaluation']:
        for path in m[split]:
            dest=folder/(key(path)+'.pt')
            if dest.exists():done+=1;continue
            base,sources,target=get_sources(path,split)
            state,bank=candidates(base,sources)
            # All features/proposals fixed before metrics see future target.
            gf=torch.stack([ir.features(state,bank[n],sources) for n in NAMES])
            lf=torch.stack([local_descriptor(base,bank[n],sources) for n in NAMES])
            bm=metrics(base,target,base,lp);mets=[metrics(bank[n],target,base,lp) for n in NAMES]
            labels=torch.tensor([[bm['lpips']-z['lpips'],(z['mse']-bm['mse'])/max(bm['mse'],1e-6),(z['temporal']-bm['temporal'])/max(bm['temporal'],1e-6)] for z in mets])
            # Extra global fixed blends selected solely on dev, not evaluation.
            fixed={}
            for alpha in [.03,.1,.25]:
                pred=pixel_bound((base+alpha*(sources['velocity']-base))[None],base[None])[0]
                fixed[str(alpha)]=metrics(pred,target,base,lp)
            pack={'path':path,'group':list(group(path)),'global':gf.cpu(),'local':lf.cpu().half(),'labels':labels,
                  'base_metrics':bm,'candidate_metrics':mets,'fixed_metrics':fixed}
            temp=dest.with_suffix('.tmp');torch.save(pack,temp);temp.replace(dest);done+=1
            if done%8==0 or done==total:
                z={'stage':'local_risk_cache','done':done,'total':total,'seconds':time.time()-started}
                dump(OUT/'progress.json',z);print(json.dumps(z),flush=True)


def assemble(rows,device='cuda'):
    return (torch.cat([r['global'] for r in rows]).to(device),torch.cat([r['local'] for r in rows]).to(device).float(),torch.cat([r['labels'] for r in rows]).to(device))


@torch.no_grad()
def predict(model,rows):
    result=[]
    model.eval()
    for r in rows:
        result.append(model(r['global'].cuda(),r['local'].cuda().float()).cpu())
    return result


def decision_audit(rows,preds,margin):
    accepted=0;safe=0;available=0;deltas=[];pair_acc=[]
    for r,p in zip(rows,preds):
        lab=r['labels'];truth=(lab[:,0]>1e-4)&(lab[:,1:]<=.01).all(-1);available+=int(truth.any())
        pick=choose(p,margin)
        if pick>=0:accepted+=1;safe+=int(truth[pick]);deltas.append(lab[pick].tolist())
        signs=(lab[:,0,None]-lab[None,:,0]).sign();est=(p[:,0,None]-p[None,:,0]).sign()
        mask=torch.triu(torch.ones(4,4,dtype=torch.bool),1)&(signs!=0)
        if mask.any():pair_acc.append(float((est[mask]==signs[mask]).float().mean()))
    return {'videos':len(rows),'accepted':accepted,'safe_accepted':safe,'harmful_accepted':accepted-safe,
            'safe_available':available,'coverage':accepted/len(rows),
            'acceptance_precision':safe/accepted if accepted else None,
            'beneficial_coverage':safe/available if available else None,
            'pair_accuracy':float(np.mean(pair_acc)) if pair_acc else None,
            'selected_label_mean':np.mean(deltas,axis=0).tolist() if deltas else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--normalized-outputs',action='store_true');ap.add_argument('--out-name',default='local_risk');args=ap.parse_args()
    seed();m=manifest();dest=OUT/args.out_name;dest.mkdir(parents=True,exist_ok=True);lp=lpips_model()
    protocol={'manifest':m,'epochs':args.epochs,'seed':SEED,'budget':BUDGET,'pixel_cap':.03,'alpha':.25,
      'stage_order':['G0 invariants/label contract','G1/G2 aligned global score and calibration','P2 local representation, same candidates and tile','one-shot held-video evaluation'],
      'tau_frozen':True,'trajectory_not_used':True,'single_round':True,'names':NAMES,
      'candidate_change':'same prior max-disagreement tile/RGB/highpass actions; shared .002 budget and .03 cap, prior scorer retrained fairly',
      'labels':'absolute LPIPS gain; MSE and temporal changes relative to baseline GT errors, supervised offline only',
      'safety_tolerance':.01,'gain_floor':.0001,'rank_head':'same gain head used for inference',
      'selection':'dev standardized regression loss, no calibration/evaluation model choice',
      'calibration':'per-metric q90 of per-video maximum one-sided residual; conservative empirical heuristic, not a selected-risk guarantee',
      'coverage_curves':'different calibration quantiles descriptive only; main q90 locked before evaluation',
      'test_used':False,'TUM_used':False,'hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('local_risk.py'),PROJECT/'iterative_video_repair.py']}}
    protocol['normalized_output_head']=args.normalized_outputs
    protocol['numerical_conditioning_control']='fit-only mean/std undo-transform on outputs; same loss/candidates/splits, no evaluation-based threshold tuning'
    dump(dest/'protocol.json',protocol);prepare(m,lp)
    data={sp:[torch.load(OUT/'local_risk/cache'/(key(p)+'.pt'),map_location='cpu',weights_only=False) for p in m[sp]] for sp in ['fit','dev','calibration','evaluation']}
    models={kind:RiskScorer(kind=='local',args.normalized_outputs).cuda() for kind in ['global','local']}
    fit_global=torch.cat([r['global'] for r in data['fit']]);target_scale=torch.cat([r['labels'] for r in data['fit']]).std(0).clamp_min(1e-5).cuda()
    for model in models.values():
        model.mean.copy_(fit_global.mean(0));model.scale.copy_(fit_global.std(0).clamp_min(1e-5))
        if args.normalized_outputs:
            model.target_mean.copy_(torch.cat([r['labels'] for r in data['fit']]).mean(0));model.target_scale.copy_(target_scale)
    params={k:sum(p.numel() for p in v.parameters()) for k,v in models.items()};dump(dest/'parameter_counts.json',params)
    optim={k:torch.optim.AdamW(v.parameters(),lr=1e-3,weight_decay=1e-4) for k,v in models.items()};best={k:float('inf') for k in models};history=[];start=time.time()
    for epoch in range(1,args.epochs+1):
        order=list(range(len(data['fit'])));random.Random(SEED+epoch).shuffle(order);sums={k:0. for k in models};steps=0
        for off in range(0,len(order),8):
            g,l,y=assemble([data['fit'][j] for j in order[off:off+8]])
            for kind,model in models.items():
                model.train();optim[kind].zero_grad(set_to_none=True);pred=model(g,l)
                loss=supervised_loss(pred,y,target_scale);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim[kind].step();sums[kind]+=float(loss.detach())
            steps+=1
        dv={}
        for kind,model in models.items():
            pp=predict(model,data['dev']);y=torch.cat([r['labels'] for r in data['dev']]);p=torch.cat(pp)
            score=float(F.smooth_l1_loss((p-y)/target_scale.cpu(),torch.zeros_like(p)));dv[kind]=score
            if score<best[kind]:best[kind]=score;torch.save({'model':model.state_dict(),'epoch':epoch},dest/(kind+'_best.pt'))
        row={'stage':'local_risk_training','epoch':epoch,'train':{k:v/steps for k,v in sums.items()},'dev':dv,'seconds':time.time()-start};history.append(row)
        if epoch%5==0 or epoch==1:dump(dest/'history.json',history);dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    # Pick fixed velocity coefficient solely on dev, including keep.
    choices=['keep','.03','.1','.25'];costs={}
    for alpha in choices:
        mm=[r['base_metrics'] if alpha=='keep' else r['fixed_metrics'][str(float(alpha))] for r in data['dev']]
        costs[alpha]=float(np.mean([z['mse']+.03*z['lpips']+.1*z['temporal'] for z in mm]))
    alpha=min(costs,key=costs.get);dump(dest/'fixed_selection.json',{'alpha':alpha,'dev_costs':costs})
    summaries={};predictions={};margins={};chosen_epochs={};calibration={}
    for kind,model in models.items():
        ck=torch.load(dest/(kind+'_best.pt'),map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);chosen_epochs[kind]=ck['epoch']
        cp=predict(model,data['calibration']);residuals=[]
        for r,p in zip(data['calibration'],cp):
            err=r['labels']-p;err[:,0]=-err[:,0];residuals.append(err.max(0).values)
        rv=torch.stack(residuals)
        margin=torch.quantile(rv,.9,dim=0,interpolation='higher').clamp_min(0);margins[kind]=margin
        predictions[kind]=predict(model,data['evaluation'])
        curves={str(q):decision_audit(data['evaluation'],predictions[kind],torch.quantile(rv,q,dim=0,interpolation='higher').clamp_min(0)) for q in [0,.5,.8,.9,1.]}
        summaries[kind]={'calibrated':decision_audit(data['evaluation'],predictions[kind],margin),
                         'uncalibrated_diagnostic':decision_audit(data['evaluation'],predictions[kind],torch.zeros(3)),
                         'descriptive_coverage_curve_not_for_selection':curves}
        calibration[kind]={'margins':margin.tolist(),'videos':len(residuals),'selected_risk_guarantee':False}
    dump(dest/'calibration.json',calibration);rows=[]
    for idx,r in enumerate(data['evaluation']):
        base,sources,target=get_sources(r['path'],'evaluation');state,bank=candidates(base,sources)
        outputs={'tau':base,'fixed_velocity':base if alpha=='keep' else pixel_bound((base+float(alpha)*(sources['velocity']-base))[None],base[None])[0]}
        decisions={}
        for kind in models:
            pick=choose(predictions[kind][idx],margins[kind]);decisions[kind]=NAMES[pick] if pick>=0 else 'keep'
            outputs[kind]=bank[NAMES[pick]] if pick>=0 else base
            diagnostic=choose(predictions[kind][idx],torch.zeros(3))
            outputs[kind+'_uncalibrated']=bank[NAMES[diagnostic]] if diagnostic>=0 else base
        # Match local accepted amount using the SAME source and edit family,
        # but spread it uniformly in space; no future label sets this amount.
        target_budget=float((outputs['local']-base).abs().mean())
        if decisions['local']=='keep':uniform=base
        else:
            source,kind=decisions['local'].split('_');res=sources[source]-base
            if kind=='texture':res=res-ir.low(res)
            uniform=pixel_bound((base+.25*res)[None],base[None],target_budget)[0]
        outputs['local_budget_uniform_same_action']=uniform
        budget_error=abs(float((uniform-base).abs().mean())-target_budget)
        assert budget_error<2e-6,(budget_error,decisions['local'])
        with torch.no_grad():mets={n:metrics(x,target,base,lp) for n,x in outputs.items()}
        rows.append({'path':r['path'],'group':r['group'],'metrics':mets,'decisions':decisions,'budget_match_error':budget_error})
    dump(dest/'per_video.json',rows);result=summarize(rows);result.update(scoring=summaries,best_epochs=chosen_epochs,parameter_counts=params)
    result['local_vs_global']={metric:ci([r['metrics']['local'][metric]-r['metrics']['global'][metric] for r in rows]) for metric in ['mse','lpips','temporal']}
    dump(dest/'summary.json',result);dump(dest/'COMPLETE.json',{'seconds':time.time()-start,'formal_training':False});print('LOCAL_RISK_COMPLETE',flush=True)


if __name__=='__main__':main()
