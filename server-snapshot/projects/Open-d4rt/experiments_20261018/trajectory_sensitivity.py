"""Post-hoc NO-TRAINING amplitude diagnostic. Dev chooses scale, never eval.

Original pilot results remain unchanged. This is a separate exploratory check,
not a fresh benchmark or a proof that oracle trajectories have no information.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from common import OUT,manifest,dump,metrics,summarize,lpips_model,pixel_bound,seed,sha
from run_trajectory import load_models,load,device,ARMS,loss
from trajectory_attention import TrackCrossAttention

SCALES=[0.,1.,4.,16.]


def main():
    seed();dest=OUT/'trajectory_sensitivity';dest.mkdir(parents=True,exist_ok=True);m=manifest();net,traj=load_models();del traj
    lp=lpips_model();models={};start=time.time()
    for arm in ARMS:
        ck=torch.load(OUT/'trajectory'/(arm+'_best.pt'),map_location='cuda',weights_only=False)
        channels=net.backbone_config['hid_s'];model=TrackCrossAttention(channels).cuda().eval().requires_grad_(False)
        model.load_state_dict(ck['model']);models[arm]=model
    dump(dest/'protocol.json',{'scales':SCALES,'training':False,'post_hoc':True,'selection':'dev composite only; original evaluation already reused',
          'feature_cap':'amplifies the original capped residual; intentionally relaxes original latent cap to diagnose its limitation',
          'pixel_mean_budget_max':.002,'pixel_cap':.03,'frozen_checkpoints':True,
          'dynamic_slice':'target adjacent-frame RGB change >.05, proxy only; no object/3D semantics',
          'script_sha256':sha(__file__)})
    @torch.no_grad()
    def proposals(d,arm):
        model=models[arm]
        if arm=='fixed_attention':z,_=model.forward_fixed(d['hidden'],d['fixed'],d['pred'],d['valid'],d['confidence'])
        else:z,_=model(d['hidden'],d['appearance'],d['uv'],d['valid'],d['confidence'],d['oracle'] if arm=='oracle_attention' else d['pred'],uniform=arm=='predicted_uniform')
        return z-d['hidden']
    @torch.no_grad()
    def render(d,delta,scale):
        import train_simvp_feature_warp as core
        if scale==0:return d['base']
        return pixel_bound(core.decode_frozen(net,d['hidden']+scale*delta,d['skip']),d['base'][None])[0]
    costs={a:{str(s):[] for s in SCALES} for a in ARMS}
    with torch.no_grad():
        for path in m['dev']:
            d=device(load(path))
            for arm in ARMS:
                delta=proposals(d,arm)
                for s in SCALES:costs[arm][str(s)].append(float(loss(render(d,delta,s),d['target'],lp)))
        costs={a:{s:float(np.mean(v)) for s,v in grid.items()} for a,grid in costs.items()}
        chosen={a:float(min(grid,key=grid.get)) for a,grid in costs.items()};dump(dest/'dev_selection.json',{'costs':costs,'chosen':chosen})
        rows=[]
        for path in m['evaluation']:
            d=device(load(path));outputs={'tau':d['base']}
            for arm in ARMS:outputs[arm]=render(d,proposals(d,arm),chosen[arm])
            target=d['target'];change=(target[1:]-target[:-1]).abs().mean(1,keepdim=True)>.05
            mask=torch.cat([change[:1],change],0).expand_as(target)
            mm={a:metrics(x,target,d['base'],lp) for a,x in outputs.items()}
            for a,x in outputs.items():mm[a]['rgb_change_mse']=float((x-target).square()[mask].mean()) if mask.any() else None
            rows.append({'path':path,'group':d['group'],'metrics':mm})
    result=summarize(rows);result['chosen_scales']=chosen
    result['rgb_change_mse']={a:float(np.mean([r['metrics'][a]['rgb_change_mse'] for r in rows if r['metrics'][a]['rgb_change_mse'] is not None])) for a in rows[0]['metrics']}
    dump(dest/'per_video.json',rows);dump(dest/'summary.json',result);dump(dest/'COMPLETE.json',{'seconds':time.time()-start,'training':False})
    print('SENSITIVITY_COMPLETE',flush=True)


if __name__=='__main__':main()
