"""Bounded single-case CPU workers; final3 purposive cases under .5 CPU/2 GiB."""
import argparse,gc,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from run_cpu_history_endpoint import (DEST,ROOT,ARMS,history_data,memories,contact_sheet,factor_effects,
    VideoForecastDataset,load,encode,sample,avg,make_gradient,bounded_feature,weighted_loss,
    core,pixel_bound,metrics,summarize,dump,sha,key,build_video_predictor_from_checkpoint,VelocityResidualImplicitForecaster)

OUT=DEST/'isolated'
CASE_INDICES=[0,3,4]

def run_case(index):
    started=time.time();torch.manual_seed(20261019);np.random.seed(20261019);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(__file__).with_name('cpu_history_endpoint_selection.json').read_text());case=selection['cases'][index];path=case['path']
    dump(OUT/'protocol.json',{'CPU_only':True,'threads':1,'observed_cpu_quota':.5,'observed_memory_limit_bytes':2147483648,
      'formal_training':False,'parameter_fitting':False,'optimizer':False,'case_indices':CASE_INDICES,
      'scope':'3 purposive cases from previously declared6; reduced for resource budget, not based on new factorial effects',
      'roles':['largest cached LPIPS harm','largest remaining cached Oracle gap','best remaining cached composite control'],
      'factorial':'identical masks/weights; D/G history sample coordinates x P/G future endpoints; no position/probability conflation',
      'step':.002,'feature_cap':.05,'image_budget':.002,'image_cap':.03,'source_sha256':sha(__file__),
      'helper_source_sha256':sha(Path(__file__).with_name('run_cpu_history_endpoint.py')),'case_selection':selection})
    dump(OUT/'progress.json',{'case_index':index,'stage':'loading_CPU_models','seconds':0})
    ck=torch.load(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt',map_location='cpu',weights_only=False)
    traj=VelocityResidualImplicitForecaster(ck['future_frames'],ck['hidden_dim']).eval().requires_grad_(False);traj.load_state_dict(ck['model']);del ck;gc.collect()
    traj_before={k:v.clone() for k,v in traj.state_dict().items()}
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);del ck;gc.collect()
    frozen={k:v.clone() for k,v in net.state_dict().items()}
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
    x=VideoForecastDataset('','all',128,paths=[path],explicit_trajectory_interface=True)[0];x['_path']=path;d=load(path);hh=d['hidden'];ss=d['skip']
    with torch.no_grad():
        h=history_data(x,traj);enc,_=encode(net,x['context_rgb'][None]);fe,_=encode(net,x['future_rgb'][None])
        _,_,md,mg,own=memories(enc,h);true=sample(fe,h['fu']);scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
        base=core.decode_frozen(net,hh,ss);parity=float((base[0]-d['base']).abs().max());assert parity<1e-3
        assert torch.allclose(x['future_rgb'],d['target'],atol=2e-6,rtol=0)
        cfg={'D_P':(md,h['p']),'G_P':(mg,h['p']),'D_G':(md,h['fu']),'G_G':(mg,h['fu']),
             'D_P_own_history_mask':(own,h['p']),'true_future_G':(true,h['fu'])}
        raw={'tau':metrics(base[0],d['target'],base[0],lp)};outputs={};objectives={}
        memory_metrics={'D_to_true_future_MSE':avg(((md-true)/scale).square(),h['fw']),
                        'G_to_true_future_MSE':avg(((mg-true)/scale).square(),h['fw'])}
    for name,(memory,coords) in cfg.items():
        gh,gs,oldloss=make_gradient(net,hh,ss,memory,coords,h['fw'],scale)
        with torch.no_grad():
            p=pixel_bound(core.decode_frozen(net,hh+bounded_feature(gh,.002,normalize=True),ss+bounded_feature(gs,.002,normalize=True)),base)[0]
            outputs[name]=p;raw[name]=metrics(p,d['target'],base[0],lp)
            assert raw[name]['budget']<=.002001 and float((p-base[0]).abs().max())<=.030001
            ee,_=encode(net,p[None]);objectives[name]={'before':oldloss,'after':float(weighted_loss(ee,memory,coords,h['fw'],scale))}
        del gh,gs;gc.collect()
        dump(OUT/'progress.json',{'case_index':index,'stage':name,'seconds':time.time()-started});print('ISOLATED_CPU_ARM',index,name,flush=True)
    with torch.no_grad():
        amount=min(float((p-base[0]).abs().mean()) for p in outputs.values());matched={'tau':raw['tau']};matched_outputs={}
        for name,p in outputs.items():
            q=pixel_bound(p[None],base,budget=amount)[0];matched_outputs[name]=q;matched[name]=metrics(q,d['target'],base[0],lp)
            assert abs(matched[name]['budget']-amount)<2e-7
        result={'path':path,'group':case['group'],'case_index':index,'reason':case['reason'],'metrics':matched,'raw_metrics':raw,
          'effects_matched':factor_effects(matched),'effects_raw':factor_effects(raw),'common_budget':amount,
          'CPU_decoder_vs_GPU_cache_max_abs':parity,'feature_memory':memory_metrics,'objectives':objectives,
          'cache_indices':h['ids'].tolist(),'source_query_ids':h['query_ids'],'support_samples':int((h['fw']>0).sum()),
          'seconds':time.time()-started,'weights_updated':False}
        assert all(torch.equal(v,frozen[k]) for k,v in net.state_dict().items())
        assert all(torch.equal(v,traj_before[k]) for k,v in traj.state_dict().items())
        assert not any(p.grad is not None for model in [net,traj] for p in model.parameters())
        result['tau_and_trajectory_bitwise_verified']=True
        dump(OUT/('case_'+str(index)+'.json'),result)
        torch.save({'path':path,'base':base[0],'target':d['target'],'matched_predictions':matched_outputs,'raw_predictions':outputs},OUT/('predictions_'+str(index)+'.pt'))
        contact_sheet(path,x,h,OUT/('case_'+str(index)+'_coordinates.png'))
    print('ISOLATED_CPU_CASE_COMPLETE',index,flush=True)

def finalize():
    cases=[json.loads((OUT/('case_'+str(index)+'.json')).read_text()) for index in CASE_INDICES]
    assert all(r['tau_and_trajectory_bitwise_verified'] for r in cases)
    result=summarize(cases);result['scope']='3 purposive failure/control cases, not independent performance estimate'
    for kind in ['matched','raw']:
        result['effects_'+kind+'_case_mean']={e:{m:float(np.mean([r['effects_'+kind][e][m] for r in cases])) for m in ['mse','lpips','temporal']} for e in cases[0]['effects_matched']}
    dump(OUT/'factorial_per_case.json',cases);dump(OUT/'factorial_summary.json',result)
    dump(OUT/'COMPLETE.json',{'CPU_only':True,'weights_updated':False,'tau_and_trajectory_bitwise_verified':True,'frozen_cases':len(cases),'case_indices':CASE_INDICES})
    print('ISOLATED_CPU_FACTORIAL_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=CASE_INDICES);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    try:
        if a.finalize:finalize()
        else:run_case(a.case)
    except Exception as ex:
        OUT.mkdir(parents=True,exist_ok=True);dump(OUT/('FAILED_'+str(a.case)+'.json'),{'error':str(ex),'traceback':traceback.format_exc()});raise
