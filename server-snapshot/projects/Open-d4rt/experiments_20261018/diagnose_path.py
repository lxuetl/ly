"""Frozen decoder sensitivity and attention-content diagnostics, no optimization."""
import math
import torch
import numpy as np
from common import manifest,seed,dump,pixel_bound,OUT,sha
from diagnose_interfaces import DEST,mse,stats
from run_trajectory import load_models,load,device
from trajectory_attention import TrackCrossAttention,sample
import train_simvp_feature_warp as core

def attn_stats(model,d):
    vals=d['appearance'];hu=d['uv'];cf=d['confidence'];valid=d['valid'];fu=d['oracle'];h=d['hidden']
    b,n,t,c=vals.shape;k=fu.shape[2];dh=model.dim//model.heads
    horizon=torch.linspace(0,1,k,device='cuda').view(1,1,k,1).expand(b,n,-1,-1)
    ht=torch.linspace(0,1,t,device='cuda').view(1,1,t,1).expand(b,n,-1,-1)
    q=model.q(torch.cat([sample(h,fu),fu,horizon],-1)).view(b,n,k,model.heads,dh).permute(0,1,3,2,4)
    kk=model.k(torch.cat([vals,hu-hu[:,:,-1:],ht,cf[...,None]],-1)).view(b,n,t,model.heads,dh).permute(0,1,3,2,4)
    vv=model.v(vals).view(b,n,t,model.heads,dh).permute(0,1,3,2,4)
    good=valid&torch.isfinite(cf)&(cf>0)
    content=q@kk.transpose(-1,-2)/math.sqrt(dh)
    logprior=cf.clamp_min(1e-6).log()[:,:,None,None,:]
    logits=(content+logprior).masked_fill(~good[:,:,None,None,:],-1e4)
    a=logits.softmax(-1)*good[:,:,None,None,:];a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
    prior=cf*good;prior=prior/prior.sum(-1,keepdim=True).clamp_min(1e-9)
    before=(a@vv).permute(0,1,3,2,4).reshape(b,n,k,model.dim)
    projected=torch.nn.functional.linear(before,model.out.weight,None)
    output=model.out(before)
    return {'effective_history_frames':float((-(a*a.clamp_min(1e-9).log()).sum(-1)).exp().mean()),'attention_abs_difference_from_confidence_prior':float((a-prior[:,:,None,None,:]).abs().mean()),'content_logits_std':float(content.std()),'projected_content_abs':float(projected.abs().mean()),'output_bias_abs':float(model.out.bias.abs().mean()),'output_abs':float(output.abs().mean()),'output_std_across_tracks':float(output.std(dim=1).mean())}

def main():
    seed();net,traj=load_models();del traj
    frozen={k:v.cpu().clone() for k,v in net.state_dict().items()};rows=[]
    for i,p in enumerate(manifest()['dev']):
        d=device(load(p));h=d['hidden'];s=d['skip'];target=d['target'][None];base=d['base'][None]
        model=TrackCrossAttention(h.shape[2]).cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(OUT/'trajectory/oracle_attention_best.pt',weights_only=False)['model'])
        with torch.no_grad():
            ast=attn_stats(model,d);original,_=model(h,d['appearance'],d['uv'],d['valid'],d['confidence'],d['oracle'])
            residual=original-h;probes={};sensitivity={}
            for name,values in [('original',d['appearance']),('zero_values',torch.zeros_like(d['appearance'])),('shuffle_track_values',d['appearance'].roll(1,1))]:
                hh,_=model(h,values,d['uv'],d['valid'],d['confidence'],d['oracle'])
                pred=pixel_bound(core.decode_frozen(net,hh,s),base)
                probes[name]={'mse':mse(pred,target),'rgb_abs':float((pred-base).abs().mean()),'hidden_abs':float((hh-h).abs().mean()),'hidden_abs_diff_vs_original':float((hh-original).abs().mean())}
        # Label-dependent gradient is an input-feature diagnostic, never a weight update.
        hi=h.detach().requires_grad_(True);si=s.detach().requires_grad_(True)
        loss=(core.decode_frozen(net,hi,si)-target).square().mean()
        gh,gs=torch.autograd.grad(loss,(hi,si));gh=gh.detach();gs=gs.detach()
        with torch.no_grad():
            dot=float((gh*residual).sum());cos=float((gh*residual).sum()/(gh.norm()*residual.norm()).clamp_min(1e-20))
            sensitivity={'mse_gradient_abs_hidden':float(gh.abs().mean()),'mse_gradient_abs_skip':float(gs.abs().mean()),'original_residual_mse_first_order':dot,'original_residual_gradient_cosine':cos,'hidden_abs':float(h.abs().mean()),'skip_abs':float(s.abs().mean())}
            for budget in [.002,.02]:
                dh=-gh/gh.abs().mean().clamp_min(1e-20)*budget
                ds=-gs/gs.abs().mean().clamp_min(1e-20)*budget
                for name,hh,ss in [('oracle_gradient_hidden',h+dh,s),('oracle_gradient_skip',h,s+ds),('oracle_gradient_both',h+dh,s+ds)]:
                    pred=pixel_bound(core.decode_frozen(net,hh,ss),base)
                    probes[name+'_'+str(budget)]={'mse':mse(pred,target),'rgb_abs':float((pred-base).abs().mean())}
        assert not any(p.grad is not None for p in net.parameters())
        rows.append({'path':p,'group':d['group'],'tau_mse':mse(base,target),'attention':ast,'sensitivity':sensitivity,'probes':probes});print('PATH_AUDIT',i+1,flush=True)
    dump(DEST/'path_per_video.json',rows)
    summary={'videos':len(rows),'weights_updated':False,'warning':'GT-gradient input-feature steps are diagnostic upper-direction probes, not deployable or training; no claim of global ceiling',
      'tau_mse':float(np.mean([r['tau_mse'] for r in rows])),
      'attention':{k:float(np.mean([r['attention'][k] for r in rows])) for k in rows[0]['attention']},
      'sensitivity':{k:float(np.mean([r['sensitivity'][k] for r in rows])) for k in rows[0]['sensitivity']},
      'probes':{n:{k:float(np.mean([r['probes'][n][k] for r in rows])) for k in rows[0]['probes'][n]} for n in rows[0]['probes']},
      'frozen_backbone_bitwise_verified':all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()),'source_sha256':sha(__file__)}
    dump(DEST/'path_summary.json',summary);print('PATH_COMPLETE',flush=True)

if __name__=='__main__':main()
