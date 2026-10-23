"""CPU-only history-sampling vs future-endpoint audit; no parameter fitting."""
import json,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from common import ROOT,PROJECT,manifest,seed,dump,sha,metrics,pixel_bound,summarize,key
from trajectory_attention import sample
from visibility_interface import future_gate
from video_forecast_dataset import VideoForecastDataset
from simvp_geometry_model import build_video_predictor_from_checkpoint
from trajectory_model import VelocityResidualImplicitForecaster
from run_trajectory import load
from run_alignment_v2 import encode,bounded_feature,make_gradient,weighted_loss
import train_simvp_feature_warp as core

DEST=ROOT/'cpu_history_endpoint_audit_20261019'
ARMS=['D_P','G_P','D_G','G_G','D_P_own_history_mask','true_future_G']

def avg(x,w):
    while w.ndim<x.ndim:w=w[...,None]
    return float((x*w).sum()/w.expand_as(x).sum().clamp_min(1e-12))

def history_data(x,traj):
    uv=x['past_uv'][None].float();valid=x['past_valid'][None].bool()
    rawcf=torch.nan_to_num(x['past_confidence'][None].float(),nan=0.,posinf=0.,neginf=0.).clamp_min(0);cf=rawcf/(1+rawcf)
    good=valid&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
    # Stable CPU tie rule and explicit source IDs: no reverse-matching old GPU cache.
    ids=torch.argsort((good*cf).sum(-1)[0],descending=True,stable=True)[:128]
    _,pred=traj(x['past_xyz'][None].float(),uv,valid,x['past_confidence'][None].float())
    hu=uv[:,ids];gu=x['history_uv_gt'][ids][None].float();gd=x['history_valid_gt'][ids][None].bool()
    md=good[:,ids];ms=md&gd;cf=cf[:,ids]
    time_weight=torch.exp((torch.arange(16,dtype=torch.float32)-15)/4)[None,None]
    wd=cf*md*time_weight;ws=cf*ms*time_weight
    nd=wd/wd.sum(-1,keepdim=True).clamp_min(1e-12);ns=ws/ws.sum(-1,keepdim=True).clamp_min(1e-12)
    fu=x['future_uv_raw'][ids][None].float();p=pred[:,ids]
    fv=x['future_visibility_gt'][ids][None].bool()&future_gate(fu).bool()&future_gate(p).bool()
    rel=(cf*ms).sum(-1)/ms.sum(-1).clamp_min(1)
    fw=rel[:,:,None]*fv
    qids=np.load(x['_path'])['query_ids'][ids.numpy()].tolist()
    return dict(ids=ids,query_ids=qids,hu=hu,gu=gu,md=md,ms=ms,cf=cf,nd=nd,ns=ns,fu=fu,p=p,fv=fv,fw=fw)

def memories(features,h):
    vd=sample(features,h['hu']);vg=sample(features,h['gu'])
    md=(vd*h['ns'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    mg=(vg*h['ns'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    own=(vd*h['nd'][...,None]).sum(2)[:,:,None].expand(-1,-1,8,-1)
    return vd,vg,md,mg,own

@torch.no_grad()
def appearance_row(x,h,features):
    vd,vg,md,mg,own=memories(features,h)
    future=x['future_rgb'][None]
    target=sample(future,h['fu']);atpred=sample(future,h['p'])
    return {'history_coordinate_EPE_px128':avg((h['hu']-h['gu']).norm(dim=-1)*127,h['ms']),
      'last_history_coordinate_EPE_px128':avg((h['hu'][:,:,15]-h['gu'][:,:,15]).norm(dim=-1)*127,h['ms'][:,:,15]),
      'same_frame_history_RGB_MAE':avg((vd-vg).abs(),h['ms']),
      'history_D_to_true_future_RGB_MAE':avg((md-target).abs(),h['fw']),
      'history_G_to_true_future_RGB_MAE':avg((mg-target).abs(),h['fw']),
      'history_mask_only_to_true_future_RGB_MAE':avg((own-target).abs(),h['fw']),
      'endpoint_same_future_RGB_MAE':avg((atpred-target).abs(),h['fw']),
      'endpoint_EPE_px128':avg((h['p']-h['fu']).norm(dim=-1)*127,h['fv']),
      'history_mask_removed_fraction':float((h['md']&~h['ms']).sum()/h['md'].sum().clamp_min(1)),
      'common_history_samples':int(h['ms'].sum()),'common_future_samples':int((h['fw']>0).sum())}

def contact_sheet(path,x,h,out):
    # Scientific annotations of sampled coordinates, not a generative edit.
    canvas=Image.new('RGB',(768,290),'white');draw=ImageDraw.Draw(canvas)
    positions=[(x['context_rgb'][15],h['hu'][0,:,15],h['gu'][0,:,15],'t15 history: red D4RT / green GT'),
               (x['future_rgb'][0],h['p'][0,:,0],h['fu'][0,:,0],'t16 endpoint: red predicted / green GT'),
               (x['future_rgb'][7],h['p'][0,:,7],h['fu'][0,:,7],'t23 endpoint: red predicted / green GT')]
    error=(h['hu'][0,:,15]-h['gu'][0,:,15]).norm(dim=-1)
    allowed=h['ms'][0,:,15]&h['fv'][0].any(-1)
    chosen=error.masked_fill(~allowed,-1).argsort(descending=True)[:8];chosen=chosen[allowed[chosen]]
    for j,(rgb,a,b,title) in enumerate(positions):
        img=Image.fromarray((rgb.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)).resize((256,256))
        canvas.paste(img,(j*256,24));draw.text((j*256+3,4),title,fill='black')
        for index in chosen.tolist():
            points=[]
            for uv,color in [(a[index],'red'),(b[index],'lime')]:
                if not torch.isfinite(uv).all():continue
                xx=j*256+float(uv[0])*255;yy=24+float(uv[1])*255
                if j*256<=xx<(j+1)*256 and 24<=yy<280:
                    draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline=color,width=2);points.append((xx,yy))
            if len(points)==2:draw.line(points,fill='yellow',width=1)
    canvas.save(out)

def factor_effects(mets):
    pairs={'history_at_pred':('G_P','D_P'),'history_at_oracle':('G_G','D_G'),
      'endpoint_with_D_history':('D_G','D_P'),'endpoint_with_G_history':('G_G','G_P'),
      'history_mask_only':('D_P','D_P_own_history_mask'),'future_appearance_reference':('true_future_G','G_G')}
    result={n:{metric:mets[a][metric]-mets[b][metric] for metric in ['mse','lpips','temporal']} for n,(a,b) in pairs.items()}
    result['interaction']={metric:mets['G_G'][metric]-mets['G_P'][metric]-mets['D_G'][metric]+mets['D_P'][metric] for metric in ['mse','lpips','temporal']}
    return result

def main():
    seed();torch.set_num_threads(4);DEST.mkdir(parents=True,exist_ok=True);start=time.time()
    selection=json.loads((Path(__file__).with_name('cpu_history_endpoint_selection.json')).read_text())
    selected={r['path']:r for r in selection['cases']};paths=manifest()['evaluation']
    dump(DEST/'protocol.json',{'CPU_only':True,'formal_training':False,'parameter_fitting':False,'optimizer':False,
      'cache_audit_videos':40,'frozen_model_cases':len(selected),'case_selection':selection,
      'point_selection':'stable CPU history-only score top128, actual source query IDs saved, not old GPU top-k replay',
      'factorial':'D/G historical UV x P/G future UV; identical intersection historical visibility, confidence, time weights, future writes',
      'labels':'all GT coordinate/visibility/appearance replacements label-only diagnostics, not deployment',
      'step':.002,'feature_cap':.05,'image_budget':.002,'image_cap':.03,
      'budget_matching':'minimum actual RGB L1 across six methods per clip, scale down only; also raw-budget effects saved',
      'hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('cpu_history_endpoint_selection.json')]}})
    ck=torch.load(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt',map_location='cpu',weights_only=False)
    traj=VelocityResidualImplicitForecaster(ck['future_frames'],ck['hidden_dim']).eval().requires_grad_(False);traj.load_state_dict(ck['model'])
    traj_before={k:v.clone() for k,v in traj.state_dict().items()}
    ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True);audit=[]
    for i,path in enumerate(paths):
        x=ds[i];x['_path']=path
        with torch.no_grad():h=history_data(x,traj);rr=appearance_row(x,h,x['context_rgb'][None])
        audit.append({'path':path,'group':list(__import__('common').group(path)),'metrics':rr,'cache_indices':h['ids'].tolist(),'source_query_ids':h['query_ids']})
        dump(DEST/'appearance_per_video.json',audit);print('CPU_PIXEL_AUDIT',i+1,40,flush=True)
    summary={k:float(np.mean([r['metrics'][k] for r in audit])) for k in audit[0]['metrics']}
    summary.update(videos=40,GT_history_reduces_future_RGB_MAE_videos=sum(r['metrics']['history_G_to_true_future_RGB_MAE']<r['metrics']['history_D_to_true_future_RGB_MAE'] for r in audit))
    dump(DEST/'appearance_summary.json',summary)
    ck=torch.load(ROOT/'tau128_full_fair_fast_20261009/best.pt',map_location='cpu',weights_only=False)
    net=build_video_predictor_from_checkpoint(ck).eval().requires_grad_(False);net.load_state_dict(ck['model']);frozen={k:v.clone() for k,v in net.state_dict().items()}
    import lpips
    lp=lpips.LPIPS(net='alex',verbose=False).cpu().eval().requires_grad_(False)
    cases=[];folder=DEST/'predictions';folder.mkdir(exist_ok=True)
    casepaths=list(selected);ds=VideoForecastDataset('','all',128,paths=casepaths,explicit_trajectory_interface=True)
    for i,path in enumerate(casepaths):
        x=ds[i];x['_path']=path;d=load(path);hh=d['hidden'];ss=d['skip']
        with torch.no_grad():
            h=history_data(x,traj);enc,_=encode(net,x['context_rgb'][None]);fe,_=encode(net,x['future_rgb'][None])
            _,_,md,mg,own=memories(enc,h);true=sample(fe,h['fu'])
            scale=enc.std((0,1,3,4),unbiased=False).clamp_min(.05)[None,None,None]
            base=core.decode_frozen(net,hh,ss);parity=float((base[0]-d['base']).abs().max());assert parity<1e-3
            assert torch.allclose(x['future_rgb'],d['target'],atol=2e-6,rtol=0)
            cfg={'D_P':(md,h['p']),'G_P':(mg,h['p']),'D_G':(md,h['fu']),'G_G':(mg,h['fu']),
                 'D_P_own_history_mask':(own,h['p']),'true_future_G':(true,h['fu'])}
            raw={'tau':metrics(base[0],d['target'],base[0],lp)};outputs={};objectives={}
            feature_memory={'D_to_true_future_MSE':avg(((md-true)/scale).square(),h['fw']),
              'G_to_true_future_MSE':avg(((mg-true)/scale).square(),h['fw'])}
        for name,(memory,coords) in cfg.items():
            gh,gs,oldloss=make_gradient(net,hh,ss,memory,coords,h['fw'],scale)
            with torch.no_grad():
                p=pixel_bound(core.decode_frozen(net,hh+bounded_feature(gh,.002,normalize=True),ss+bounded_feature(gs,.002,normalize=True)),base)[0]
                outputs[name]=p;raw[name]=metrics(p,d['target'],base[0],lp)
                assert raw[name]['budget']<=.002001 and float((p-base[0]).abs().max())<=.030001
                ee,_=encode(net,p[None]);objectives[name]={'before':oldloss,'after':float(weighted_loss(ee,memory,coords,h['fw'],scale))}
            print('CPU_FROZEN_ARM',i+1,len(casepaths),name,flush=True)
        with torch.no_grad():
            amount=min(float((p-base[0]).abs().mean()) for p in outputs.values());matched={'tau':raw['tau']};matched_outputs={}
            for name,p in outputs.items():
                q=pixel_bound(p[None],base,budget=amount)[0];matched_outputs[name]=q
                matched[name]=metrics(q,d['target'],base[0],lp);assert abs(matched[name]['budget']-amount)<2e-7
            z={'path':path,'group':selected[path]['group'],'reason':selected[path]['reason'],'metrics':matched,'raw_metrics':raw,
              'effects_matched':factor_effects(matched),'effects_raw':factor_effects(raw),'common_budget':amount,
              'CPU_decoder_vs_GPU_cache_max_abs':parity,'feature_memory':feature_memory,'objectives':objectives,
              'cache_indices':h['ids'].tolist(),'source_query_ids':h['query_ids'],'support_samples':int((h['fw']>0).sum())}
            cases.append(z);dump(DEST/'factorial_per_case.json',cases)
            torch.save({'path':path,'base':base[0],'target':d['target'],'matched_predictions':matched_outputs,'raw_predictions':outputs},folder/(key(path)+'.pt'))
            contact_sheet(path,x,h,DEST/('case_'+str(i+1)+'_coordinates.png'))
            dump(DEST/'progress.json',{'stage':'CPU_frozen_cases','done':i+1,'total':len(casepaths),'seconds':time.time()-start})
    final=summarize(cases);final['scope']='six purposively selected cases, not unbiased performance estimate'
    final['effects_matched_case_mean']={effect:{metric:float(np.mean([r['effects_matched'][effect][metric] for r in cases])) for metric in ['mse','lpips','temporal']} for effect in cases[0]['effects_matched']}
    final['effects_raw_case_mean']={effect:{metric:float(np.mean([r['effects_raw'][effect][metric] for r in cases])) for metric in ['mse','lpips','temporal']} for effect in cases[0]['effects_raw']}
    dump(DEST/'factorial_summary.json',final)
    assert all(torch.equal(v,frozen[k]) for k,v in net.state_dict().items())
    assert all(torch.equal(v,traj_before[k]) for k,v in traj.state_dict().items())
    assert not any(p.grad is not None for model in [net,traj] for p in model.parameters())
    dump(DEST/'COMPLETE.json',{'seconds':time.time()-start,'CPU_only':True,'weights_updated':False,'tau_and_trajectory_bitwise_verified':True,'frozen_cases':len(cases),'pixel_audit_videos':40})
    print('CPU_HISTORY_ENDPOINT_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as ex:
        dump(DEST/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc()});raise
