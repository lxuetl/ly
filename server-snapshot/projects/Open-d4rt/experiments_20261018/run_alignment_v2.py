"""Visibility-corrected, no-fit representation-alignment ablation.

No optimizer or weight updates. A single gradient step changes *input latents*
to match historical appearance in a shared frozen encoder representation.
Oracle UV/visibility/appearance branches are label-only diagnostics.
"""
import json,time,traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,manifest,seed,key,dump,sha,lpips_model,metrics,summarize,pixel_bound,ci
from run_trajectory import load_models,load,device
from trajectory_attention import TrackCrossAttention,sample,splat
from visibility_interface import future_gate
from video_forecast_dataset import VideoForecastDataset
import train_simvp_feature_warp as core

DEST=ROOT/'visibility_alignment_v2_20261018'
STEPS=[.002,.02]
ARMS=['raw_replace_oracle','unaligned_residual_oracle','aligned_hidden_oracle','aligned_oracle','aligned_oracle_uniform',
      'aligned_fixed_memory','aligned_shuffle_memory','aligned_pred_gtvis','aligned_pred_unknown',
      'aligned_true_appearance','rgb_oracle','rgb_pred_gtvis']

def encode(net,rgb):
    b,t,c,h,w=rgb.shape;e,s=net.backbone.enc(rgb.reshape(b*t,c,h,w))
    return e.reshape(b,t,*e.shape[1:]),s.reshape(b,t,*s.shape[1:])

def bounded_feature(delta,budget,normalize=False):
    norm=delta.abs().mean((2,3,4),keepdim=True)
    scale=budget/norm.clamp_min(1e-20)
    if not normalize:scale=scale.clamp(max=1)
    return (delta*scale).clamp(-.05,.05)

def inputs(path,x,d):
    uv=x['past_uv'][None].cuda().float();valid=x['past_valid'][None].cuda().bool()
    conf=torch.nan_to_num(x['past_confidence'][None].cuda().float(),nan=0.,posinf=0.,neginf=0.).clamp_min(0);conf=conf/(1+conf)
    good=valid&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
    ids=(good.float()*conf).sum(-1).topk(min(128,uv.shape[1]),dim=1).indices[0]
    assert torch.equal(torch.nan_to_num(uv[:,ids]),d['uv'])
    fu=x['future_uv_raw'].cuda()[ids][None].float();fv=x['future_visibility_gt'].cuda()[ids][None].bool()
    assert torch.equal(x['future_uv_teacher'],x['future_uv_raw']) or torch.allclose(x['future_uv_teacher'],x['future_uv_raw'],equal_nan=True)
    return fu,fv

def historical_weights(d,uniform=False):
    w=d['valid'].float()*d['confidence']
    # Most recent valid observations are a fixed, label-independent appearance prior.
    if not uniform:w=w*torch.exp((torch.arange(16,device='cuda')-15)/4)[None,None]
    return w/w.sum(-1,keepdim=True).clamp_min(1e-12)

def history_target(enc,d,coords,kind):
    if kind=='fixed':
        mem=torch.stack([sample(enc,coords[:,:,k:k+1].expand(-1,-1,16,-1)) for k in range(8)],2)
    else:
        mem=sample(enc,d['uv'])[:,:,None].expand(-1,-1,8,-1,-1)
        if kind=='shuffle':mem=mem.roll(1,1)
    w=historical_weights(d,kind=='uniform')
    return (mem*w[:,:,None,:,None]).sum(3)

def weights(d,coords,vis,source):
    rel=(d['confidence']*d['valid']).sum(-1)/d['valid'].sum(-1).clamp_min(1)
    return rel[:,:,None]*future_gate(coords,vis,source=source,oracle_diagnostic=source=='oracle_gt')

def weighted_loss(features,appearance,coords,ww,channel_scale):
    diff=(sample(features,coords)-appearance)/channel_scale
    return (diff.square().mean(-1)*ww).sum()/ww.sum().clamp_min(1e-9)

def make_gradient(net,h,s,appearance,coords,ww,scale):
    # Feature objective contains no future RGB except explicit true_appearance arm.
    hi=h.detach().requires_grad_(True);si=s.detach().requires_grad_(True)
    rgb=core.decode_frozen(net,hi,si);fe,_=encode(net,rgb)
    loss=weighted_loss(fe,appearance.detach(),coords,ww,scale)
    gh,gs=torch.autograd.grad(loss,(hi,si))
    return -gh.detach(),-gs.detach(),float(loss.detach())

def selection_cost(mm):return mm['mse']+.03*mm['lpips']+.1*mm['temporal']

def run_split(net,model,paths,split,lp,selected=None):
    ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True)
    rows=[];grid_costs={a:{str(v):[] for v in [0]+STEPS} for a in ARMS}
    for index,path in enumerate(paths):
      x=ds[index];d=device(load(path));h=d['hidden'];s=d['skip'];base=d['base'][None];target=d['target']
      with torch.no_grad():
        fu,fv=inputs(path,x,d);enc,_=encode(net,x['context_rgb'][None].cuda());baseenc,_=encode(net,base)
        scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        # Fixed historical-only track mask, confidence and selection in every arm.
        wo=weights(d,fu,fv,'oracle_gt');wp=weights(d,d['pred'],fv,'oracle_gt');wu=weights(d,d['pred'],None,'unknown')
        source=history_target(enc,d,fu,'recent');uniform=history_target(enc,d,fu,'uniform')
        fixed=history_target(enc,d,fu,'fixed');shuffle=history_target(enc,d,fu,'shuffle')
        predsrc=history_target(enc,d,d['pred'],'recent')
        bm=metrics(d['base'],target,d['base'],lp);mets={'tau':bm};trace={}
        legacy,_=model(h,d['appearance'],d['uv'],d['valid'],d['confidence'],d['oracle'])
        corrected,sup=model(h,d['appearance'],d['uv'],d['valid'],d['confidence'],fu,
            future_visibility=fv,visibility_source='oracle_gt',oracle_diagnostic=True)
        for name,hh in [('legacy_oracle',legacy),('corrected_attention',corrected)]:
            pred=pixel_bound(core.decode_frozen(net,hh,s),base)[0];mets[name]=metrics(pred,target,d['base'],lp)
        basetokens=sample(baseenc,fu);raw,rsup=splat(source,fu,wo,*h.shape[-2:])
        rawreplace=raw-h*rsup
        unaligned,_=splat(source-basetokens,fu,wo,*h.shape[-2:])
        futureenc,_=encode(net,target[None]);trueappearance=sample(futureenc,fu)
        rgbhist=x['context_rgb'][None].cuda();rgbsource=history_target(rgbhist,d,fu,'recent');rgbpredsrc=history_target(rgbhist,d,d['pred'],'recent')
        rgbdelta,_=splat(rgbsource-sample(base,fu),fu,wo,128,128)
        rgbpreddelta,_=splat(rgbpredsrc-sample(base,d['pred']),d['pred'],wp,128,128)
        gradients={}
      configs={'aligned_oracle':(source,fu,wo),'aligned_oracle_uniform':(uniform,fu,wo),'aligned_fixed_memory':(fixed,fu,wo),
        'aligned_shuffle_memory':(shuffle,fu,wo),'aligned_pred_gtvis':(predsrc,d['pred'],wp),
        'aligned_pred_unknown':(predsrc,d['pred'],wu),'aligned_true_appearance':(trueappearance,fu,wo)}
      for name,(appearance,coords,ww) in configs.items():
        gradients[name]=make_gradient(net,h,s,appearance,coords,ww,scale)
      gradients['aligned_hidden_oracle']=gradients['aligned_oracle']
      configs['aligned_hidden_oracle']=configs['aligned_oracle']
      with torch.no_grad():
        for name in ARMS:
          values=STEPS if selected is None else sorted(set(STEPS+[selected[name]]))
          if selected is None:grid_costs[name]['0'].append(selection_cost(bm))
          for step in values:
            if step==0:pred=d['base'];extra={}
            elif name.startswith('rgb_'):
                delta=rgbdelta if name=='rgb_oracle' else rgbpreddelta
                # step is a fixed blend coefficient (.002/.02); same shared image cap.
                pred=pixel_bound(base+step*delta,base)[0];extra={}
            else:
                if name in ['raw_replace_oracle','unaligned_residual_oracle']:
                    dh=bounded_feature(rawreplace if name.startswith('raw_replace') else unaligned,step);ss=s;oldloss=None
                else:
                    gh,gs,oldloss=gradients[name]
                    dh=bounded_feature(gh,step,normalize=True)
                    ss=s if name=='aligned_hidden_oracle' else s+bounded_feature(gs,step,normalize=True)
                decoded=core.decode_frozen(net,h+dh,ss);pred=pixel_bound(decoded,base)[0]
                extra={'hidden_abs':float(dh.abs().mean()),'skip_abs':float((ss-s).abs().mean())}
                if oldloss is not None:
                    # Measure feature objective AFTER final pixel clipping/budget too.
                    fe,_=encode(net,pred[None]);appearance,coords,ww=configs[name]
                    extra.update(feature_loss_before=oldloss,feature_loss_after=float(weighted_loss(fe,appearance,coords,ww,scale)))
            mm=metrics(pred,target,d['base'],lp)
            assert mm['budget']<=.002001 and float((pred-d['base']).abs().max())<=.030001
            if selected is None:grid_costs[name][str(step)].append(selection_cost(mm))
            # Save all predeclared positive-step diagnostics; main selected policy separate.
            if step>0:
                mets[name+'@'+str(step)]=mm;trace[name+'@'+str(step)]=extra
            if selected is not None and step==selected[name]:
                mets[name]=mm;trace[name]=extra
        invalid=(wo==0)
        rows.append({'path':path,'group':d['group'],'metrics':mets,'feature_objectives':trace,
          'visibility':{'oracle_invalid_fraction':float(invalid.float().mean()),'support_fraction':float((sup>0).float().mean()),'raw_future_outside_fraction':float((future_gate(fu)==0).float().mean())}})
        dump(DEST/(split+'_per_video.json'),rows)
        dump(DEST/'progress.json',{'stage':split,'done':index+1,'total':len(paths)})
        print('ALIGNMENT',split,index+1,len(paths),flush=True)
      assert not any(p.grad is not None for p in net.parameters())
    if selected is None:
        costs={a:{k:float(np.mean(v)) for k,v in grid.items()} for a,grid in grid_costs.items()}
        chosen={a:float(min(grid,key=grid.get)) for a,grid in costs.items()}
        dump(DEST/'dev_selection.json',{'coefficients':chosen,'costs':costs,'criterion':'MSE+.03LPIPS+.1temporal, keep included; no eval selection'})
    else:chosen=selected
    dump(DEST/(split+'_summary.json'),summarize(rows));return chosen,rows

def main():
    seed();DEST.mkdir(parents=True,exist_ok=True);m=manifest();start=time.time()
    dump(DEST/'protocol.json',{'weights_updated':False,'formal_training':False,'parameter_fitting':False,'optimizer':False,
      'gradient':'one input-latent step, frozen encoder/decoder; historical-feature objective, not target RGB except explicit true_appearance arm',
      'steps':STEPS,'latent_pixel_cap':.05,'image_budget':.002,'image_pixel_cap':.03,'temporal_memory':'recent exp((t-15)/4) vs uniform',
      'visibility':'raw UV, in-bounds separate; oracle_gt explicit diagnostic; unknown NOT a future probability',
      'selection':'existing dev12, include keep, then reused validation40 exploratory; no test/TUM','arms':ARMS,
      'dev_paths':m['dev'],'evaluation_paths':m['evaluation'],'hashes':{n:sha(Path(__file__).with_name(n)) for n in ['run_alignment_v2.py','trajectory_attention.py','visibility_interface.py']}})
    net,traj=load_models();del traj;before={k:v.cpu().clone() for k,v in net.state_dict().items()}
    ch=load(m['dev'][0])['hidden'].shape[2];model=TrackCrossAttention(ch).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(OUT/'trajectory/oracle_attention_best.pt',weights_only=False)['model']);lp=lpips_model()
    selected,dev=run_split(net,model,m['dev'],'dev',lp)
    selected,ev=run_split(net,model,m['evaluation'],'evaluation',lp,selected)
    # UV and correspondence comparisons share visibility; predicted_unknown remains deployable input provenance.
    comparisons={b:{metric:ci([r['metrics']['aligned_oracle'][metric]-r['metrics'][b][metric] for r in ev]) for metric in ['mse','lpips','temporal']} for b in ['aligned_pred_gtvis','aligned_fixed_memory','aligned_shuffle_memory','unaligned_residual_oracle']}
    dump(DEST/'comparisons.json',comparisons)
    fixed_comparisons={str(step):{b:{metric:ci([r['metrics']['aligned_oracle@'+str(step)][metric]-r['metrics'][b+'@'+str(step)][metric] for r in ev]) for metric in ['mse','lpips','temporal']} for b in ['aligned_pred_gtvis','aligned_fixed_memory','aligned_shuffle_memory','aligned_oracle_uniform']} for step in STEPS}
    fixed_comparisons['hidden_alignment_only']={str(step):{metric:ci([r['metrics']['aligned_hidden_oracle@'+str(step)][metric]-r['metrics']['unaligned_residual_oracle@'+str(step)][metric] for r in ev]) for metric in ['mse','lpips','temporal']} for step in STEPS}
    dump(DEST/'fixed_coefficient_comparisons.json',fixed_comparisons)
    assert all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items())
    dump(DEST/'COMPLETE.json',{'seconds':time.time()-start,'weights_updated':False,'frozen_backbone_bitwise_verified':True,'selected':selected})
    print('ALIGNMENT_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as ex:
        dump(DEST/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc()});raise
