"""Post-v2 attribution: match actual RGB change, never tune on validation.

All models frozen. Fixed .002 input feature step was chosen on dev in v2.
Oracle coordinates and visibility remain label-only diagnostic branches.
"""
import time,json
import torch
from common import manifest,seed,dump,lpips_model,metrics,pixel_bound,summarize,ci,sha
from run_alignment_v2 import DEST,load_models,VideoForecastDataset,load,device,inputs,encode,weights,history_target,make_gradient,bounded_feature,core

ARMS=['oracle_gtvis','pred_gtvis','fixed_gtvis','shuffle_gtvis','oracle_unknown','pred_unknown']

def main():
    seed();start=time.time();m=manifest();net,traj=load_models();del traj
    before={k:v.cpu().clone() for k,v in net.state_dict().items()};lp=lpips_model();folder=DEST/'budget_matched';folder.mkdir(exist_ok=True)
    dump(folder/'protocol.json',{'formal_training':False,'weights_updated':False,'step':.002,'source_sha256':sha(__file__),
        'scope':'post-hoc descriptive actual-budget attribution on reused validation40; no threshold/model selection',
        'actual_budget':'per-video minimum actual RGB L1 among all 6 arms, scale down only; oracle-dependent, diagnostic not deployment',
        'comparisons':'same future visibility for UV-only comparison; unknown-vs-oracle visibility explicitly separated','arms':ARMS})
    rows=[];ds=VideoForecastDataset('','all',128,paths=m['evaluation'],explicit_trajectory_interface=True)
    for i,path in enumerate(m['evaluation']):
        x=ds[i];d=device(load(path));h=d['hidden'];s=d['skip'];base=d['base'][None]
        with torch.no_grad():
            fu,fv=inputs(path,x,d);enc,_=encode(net,x['context_rgb'][None].cuda());scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
            wo=weights(d,fu,fv,'oracle_gt');wp=weights(d,d['pred'],fv,'oracle_gt')
            wu=weights(d,fu,None,'unknown');wpu=weights(d,d['pred'],None,'unknown')
            source=history_target(enc,d,fu,'recent');predsrc=history_target(enc,d,d['pred'],'recent')
            cfg={'oracle_gtvis':(source,fu,wo),'pred_gtvis':(predsrc,d['pred'],wp),
                 'fixed_gtvis':(history_target(enc,d,fu,'fixed'),fu,wo),
                 'shuffle_gtvis':(history_target(enc,d,fu,'shuffle'),fu,wo),
                 'oracle_unknown':(source,fu,wu),'pred_unknown':(predsrc,d['pred'],wpu)}
        predictions={}
        for name,(appearance,coords,ww) in cfg.items():
            gh,gs,_=make_gradient(net,h,s,appearance,coords,ww,scale)
            with torch.no_grad():
                pred=core.decode_frozen(net,h+bounded_feature(gh,.002,normalize=True),s+bounded_feature(gs,.002,normalize=True))
                predictions[name]=pixel_bound(pred,base)[0]
        with torch.no_grad():
            amounts={n:float((p-d['base']).abs().mean()) for n,p in predictions.items()};amount=min(amounts.values())
            mets={'tau':metrics(d['base'],d['target'],d['base'],lp)};raw={}
            for name,pred in predictions.items():
                raw[name]=metrics(pred,d['target'],d['base'],lp)
                # Pixel_bound may further clip only at numeric epsilon; no amplification.
                matched=pixel_bound(pred[None],base,budget=amount)[0]
                mets[name]=metrics(matched,d['target'],d['base'],lp)
                assert abs(mets[name]['budget']-amount)<2e-7
                assert float((matched-d['base']).abs().max())<=.030001
            rows.append({'path':path,'group':d['group'],'metrics':mets,'raw_metrics':raw,'common_budget':amount,'original_budgets':amounts})
            dump(folder/'per_video.json',rows);print('BUDGET_MATCHED',i+1,40,flush=True)
    dump(folder/'summary.json',summarize(rows))
    comparisons={b:{metric:ci([r['metrics']['oracle_gtvis'][metric]-r['metrics'][b][metric] for r in rows]) for metric in ['mse','lpips','temporal']} for b in ['pred_gtvis','fixed_gtvis','shuffle_gtvis','oracle_unknown']}
    comparisons['UV_without_future_visibility']={metric:ci([r['metrics']['oracle_unknown'][metric]-r['metrics']['pred_unknown'][metric] for r in rows]) for metric in ['mse','lpips','temporal']}
    dump(folder/'comparisons.json',comparisons)
    assert all(torch.equal(v.cpu(),before[k]) for k,v in net.state_dict().items())
    assert not any(p.grad is not None for p in net.parameters())
    dump(folder/'COMPLETE.json',{'seconds':time.time()-start,'weights_updated':False,'frozen_backbone_bitwise_verified':True});print('BUDGET_MATCHED_COMPLETE',flush=True)

if __name__=='__main__':main()
