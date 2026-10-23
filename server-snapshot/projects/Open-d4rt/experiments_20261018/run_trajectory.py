"""Small equal-budget training pilot; TAU and trajectory predictor stay frozen."""
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,OUT,PROJECT,SEED,dump,sha,key,group,manifest,seed,lpips_model,metrics,summarize,pixel_bound
from trajectory_attention import TrackCrossAttention,sample
import train_simvp_feature_warp as core
from video_forecast_dataset import VideoForecastDataset
from simvp_geometry_model import build_video_predictor_from_checkpoint
from trajectory_model import VelocityResidualImplicitForecaster,ImplicitConditionedJointForecaster

ARMS=['oracle_attention','predicted_attention','predicted_uniform','fixed_attention']


def load_models():
    cp=ROOT/'tau128_full_fair_fast_20261009/best.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False);net=build_video_predictor_from_checkpoint(ck).cuda().eval().requires_grad_(False)
    net.load_state_dict(ck['model']);assert net.context_frames==16 and net.future_frames==8
    ck=torch.load(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt',map_location='cpu',weights_only=False)
    cls=VelocityResidualImplicitForecaster if ck.get('model_variant')=='velocity_residual' else ImplicitConditionedJointForecaster
    traj=cls(ck['future_frames'],ck['hidden_dim']).cuda().eval().requires_grad_(False);traj.load_state_dict(ck['model'])
    return net,traj


@torch.no_grad()
def prepare(net,traj,m):
    folder=OUT/'trajectory/cache';folder.mkdir(parents=True,exist_ok=True)
    paths=m['fit']+m['dev']+m['evaluation'];ds=VideoForecastDataset('', 'all',128,paths=paths)
    started=time.time()
    for i,p in enumerate(paths):
        dest=folder/(key(p)+'.pt')
        if dest.exists():continue
        x=ds[i];context=x['context_rgb'][None].cuda()
        b,t,c,h,w=context.shape
        enc,sk=net.backbone.enc(context.reshape(b*t,c,h,w));ch,hh,ww=enc.shape[1:]
        enc=enc.reshape(b,t,ch,hh,ww)
        hidden=net.latent_time(net.backbone.hid(enc));skip=net.skip_time(sk.reshape(b,t,*sk.shape[1:]))
        # Assert baseline forward parity; no geometry path is activated.
        baseline=core.decode_frozen(net,hidden,skip)
        if i==0:
            direct=net(context)
            assert torch.allclose(baseline,direct,atol=2e-6,rtol=0)
        valid=x['past_valid'][None].cuda().bool();uv=x['past_uv'][None].cuda().float()
        conf=torch.nan_to_num(x['past_confidence'][None].cuda().float()).clamp_min(0)
        conf=conf/(1+conf) # historical reliability heuristic, not a future probability
        good=valid&torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
        # Track selection depends ONLY on observed history, not teacher visibility.
        score=(good.float()*conf).sum(-1);ids=score.topk(min(128,score.shape[1]),dim=1).indices
        def select(z):return z.gather(1,ids.reshape(1,-1,*([1]*(z.ndim-2))).expand(-1,-1,*z.shape[2:]))
        past_uv=select(torch.nan_to_num(uv));good=select(good);conf=select(conf)*good
        _,pred=traj(x['past_xyz'][None].cuda().float(),uv,valid,x['past_confidence'][None].cuda().float())
        pred=select(pred)
        appearance=sample(enc,past_uv)
        memories=[]
        for k in range(8):memories.append(sample(enc,pred[:,:,k:k+1].expand(-1,-1,16,-1)))
        fixed=torch.stack(memories,2)
        oracle=select(x['future_uv_teacher'][None].cuda().float())
        pack={'path':p,'group':list(group(p)),'hidden':hidden,'skip':skip,'base':baseline[0],
              'target':x['future_rgb'].cuda(),'appearance':appearance,'fixed':fixed,
              'uv':past_uv,'valid':good,'confidence':conf,'pred':pred,'oracle':oracle}
        # Keep the actual baseline latent/skip in float32: epoch0 must be exact.
        pack={k:(v.cpu().half() if k in ['appearance','fixed'] else v.cpu()) if torch.is_tensor(v) else v for k,v in pack.items()}
        temp=dest.with_suffix('.tmp');torch.save(pack,temp);temp.replace(dest)
        if i%8==0 or i+1==len(paths):
            progress={'stage':'trajectory_cache','done':i+1,'total':len(paths),'seconds':time.time()-started}
            dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)


def load(p):
    return torch.load(OUT/'trajectory/cache'/(key(p)+'.pt'),map_location='cpu',weights_only=False)


def device(d):return {k:v.cuda().float() if torch.is_tensor(v) and v.is_floating_point() else v.cuda() if torch.is_tensor(v) else v for k,v in d.items()}


def predict(model,net,d,arm):
    if arm=='fixed_attention':
        z,s=model.forward_fixed(d['hidden'],d['fixed'],d['pred'],d['valid'],d['confidence'])
    else:
        coords=d['oracle'] if arm=='oracle_attention' else d['pred']
        z,s=model(d['hidden'],d['appearance'],d['uv'],d['valid'],d['confidence'],coords,uniform=arm=='predicted_uniform')
    out=core.decode_frozen(net,z,d['skip'])
    return pixel_bound(out,d['base'][None])[0]


def loss(pred,target,lp):
    mse=(pred-target).square().mean()
    per=lp(pred*2-1,target*2-1).mean()
    temp=((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    return mse+.03*per+.1*temp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=6);a=ap.parse_args()
    seed();m=manifest();dest=OUT/'trajectory';dest.mkdir(parents=True,exist_ok=True)
    protocol={'manifest':m,'epochs':a.epochs,'seed':SEED,'tau_frozen':True,'trajectory_frozen':True,
      'arms':ARMS,'oracle':'future_uv_teacher only; historical track selection, masks, confidence shared; no future visibility',
      'oracle_limit':'teacher UV uses existing dataset clipped-normalized UV; not full oracle geometry or visibility',
      'sampling':'history track values vs fixed predicted endpoint values; one temporal attention layer per track, 4 heads, dim64',
      'uniform':'same projected appearance Value and output head; attention weights use historical reliability only',
      'budget':{'max_mean_abs':.002,'pixel_cap':.03,'feature_mean_abs_max':.002},
      'loss':'MSE+.03*AlexLPIPS+.1*temporal_MAE','selection':'dev composite includes exact identity epoch0',
      'training':'6 short adapter-only epochs, not formal backbone training','evaluation':'original validation one clip/video; exploratory',
      'hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('trajectory_attention.py'),ROOT/'tau128_full_fair_fast_20261009/best.pt',ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt']}}
    dump(dest/'protocol.json',protocol)
    net,traj=load_models();frozen_before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    prepare(net,traj,m);del traj;torch.cuda.empty_cache()
    # Reuse a small, bounded corpus in host RAM, not shared-fs each epoch.
    data={p:load(p) for p in m['fit']+m['dev']+m['evaluation']}
    ch=data[m['fit'][0]]['hidden'].shape[2];initial=TrackCrossAttention(ch).state_dict()
    models={n:TrackCrossAttention(ch).cuda() for n in ARMS}
    for model in models.values():model.load_state_dict(initial)
    optim={n:torch.optim.AdamW(v.parameters(),lr=3e-4,weight_decay=1e-4) for n,v in models.items()}
    lp=lpips_model();best={};history=[];start=time.time()
    @torch.no_grad()
    def evaluate(paths,keep_rows=False):
        scores={n:[] for n in ARMS};rows=[]
        for p in paths:
            d=device(data[p]);mets={'tau':metrics(d['base'],d['target'],d['base'],lp)}
            for n,model in models.items():
                model.eval();pred=predict(model,net,d,n)
                scores[n].append(float(loss(pred,d['target'],lp)))
                if keep_rows:mets[n]=metrics(pred,d['target'],d['base'],lp)
            if keep_rows:rows.append({'path':p,'group':d['group'],'metrics':mets})
        return {n:float(np.mean(v)) for n,v in scores.items()},rows
    base_scores,_=evaluate(m['dev'])
    for n,model in models.items():
        best[n]=base_scores[n];torch.save({'model':model.state_dict(),'epoch':0,'score':best[n]},dest/(n+'_best.pt'))
    for epoch in range(1,a.epochs+1):
        order=m['fit'].copy();random.Random(SEED+epoch).shuffle(order);tot={n:0. for n in ARMS}
        for p in order:
            d=device(data[p])
            for n,model in models.items():
                model.train();optim[n].zero_grad(set_to_none=True)
                pred=predict(model,net,d,n);l=loss(pred,d['target'],lp);assert torch.isfinite(l)
                l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim[n].step();tot[n]+=float(l.detach())
            assert not any(p.grad is not None for p in net.parameters())
        vals,_=evaluate(m['dev']);row={'stage':'trajectory_training','epoch':epoch,'train':{n:v/len(order) for n,v in tot.items()},'dev':vals,'seconds':time.time()-start}
        for n,model in models.items():
            if vals[n]<best[n]:
                best[n]=vals[n];torch.save({'model':model.state_dict(),'epoch':epoch,'score':best[n]},dest/(n+'_best.pt'))
        history.append(row);dump(dest/'history.json',history);dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    selected={}
    for n,model in models.items():
        ck=torch.load(dest/(n+'_best.pt'),map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);selected[n]=ck['epoch']
    _,rows=evaluate(m['evaluation'],True);dump(dest/'per_video.json',rows)
    result=summarize(rows);result['best_epochs']=selected
    result['comparisons']={n:{metric:__import__('common').ci([r['metrics']['predicted_attention'][metric]-r['metrics'][n][metric] for r in rows]) for metric in ['mse','lpips','temporal']} for n in ['predicted_uniform','fixed_attention','oracle_attention']}
    assert all(torch.equal(v.cpu(),frozen_before[k]) for k,v in net.state_dict().items())
    result['frozen_backbone_bitwise_verified']=True
    dump(dest/'summary.json',result);dump(dest/'COMPLETE.json',{'seconds':time.time()-start,'best_epochs':selected,'formal_training':False})
    print('TRAJECTORY_COMPLETE',flush=True)


if __name__=='__main__':main()
