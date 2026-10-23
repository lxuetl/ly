"""Four matched frozen-TAU propagation fusion diagnostics. No test tuning."""
import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import audit_coordinate_gt as old
e,prev,core=old.e,old.prev,old.core
ROOT=e.p.run.fit.ROOT
OUT=ROOT/'propagation_fusion_20261014'
MODES=['gt','pred','zero','wrong']
def key(p): return hashlib.sha256(p.encode()).hexdigest()[:24]
def splits(): return json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(11,16,3,padding=1),nn.GELU(),nn.Conv3d(16,16,3,padding=1),nn.GELU(),nn.Conv3d(16,3,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,tau,warp,gate):
        b,t,c,h,w=tau.shape
        horizon=torch.linspace(0,1,t,device=tau.device)[None,:,None,None,None].expand(b,t,1,h,w)
        inp=torch.cat([tau,warp,warp-tau,gate,horizon],2)
        inp=F.avg_pool3d(inp.permute(0,2,1,3,4),(1,2,2))
        delta=F.interpolate(self.net(inp),size=(t,h,w),mode='trilinear',align_corners=False).permute(0,2,1,3,4)
        delta=.08*torch.tanh(delta)*gate
        magnitude=delta.abs().mean((1,2,3,4),keepdim=True)
        delta=delta*(.002/magnitude.clamp_min(1e-8)).clamp(max=1)
        return (tau+delta).clamp(0,1)

@torch.no_grad()
def prepare(rank=0,world=1):
    OUT.mkdir(exist_ok=True);cache=OUT/'cache';cache.mkdir(exist_ok=True)
    ps=splits();gs={k:{tuple(prev.group(p)) for p in paths} for k,paths in ps.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]: assert not gs[a]&gs[b]
    if rank==0:prev.dump(OUT/'split_paths.json',{'paths':ps})
    prev.prior.seed_all(42);torch.set_num_threads(4)
    cfg=SimpleNamespace(backbone_checkpoint=str(ROOT/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(ROOT/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(ROOT/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    backbone,joint,adapter,_=prev.old.load_frozen(cfg,torch.device('cuda'))
    backbone.eval().requires_grad_(False);joint.eval().requires_grad_(False)
    raster=core.SparseDenseFlow(9).cuda();paths=ps['train']+ps['val'];ds=prev.VideoForecastDataset('','val',128,paths=paths);start=time.time()
    for i,p in enumerate(paths):
        if i%world!=rank:continue
        dest=cache/(key(p)+'.pt')
        if dest.exists():continue
        x=ds[i];obs={k:x[k][None].cuda() for k in prev.PAST};obs['context_rgb']=x['context_rgb'][None].cuda()
        src,hid,skip=core.extract_frozen_features(backbone,obs['context_rgb']);tau=core.decode_frozen(backbone,hid,skip)
        _,uv,rel,anchor=core.geometry_bundle('correct',joint,obs,8)
        last=obs['past_valid'][:,:,-1].bool();weight=rel*last[:,:,None]
        flow,support=raster(anchor,uv,weight,*hid.shape[-2:])
        gate=F.interpolate((support>1e-4).float().flatten(0,1),(128,128),mode='bilinear',align_corners=False).reshape(1,8,1,128,128)
        def warp(fl):
            fl=F.interpolate(fl.flatten(0,1),(128,128),mode='bilinear',align_corners=True).reshape(1,8,2,128,128)
            return core.backward_warp(obs['context_rgb'][:,-1],fl)
        pred=warp(flow);wrong=warp(-flow)
        z=np.load(p);meta=json.loads(str(z['metadata_json']));s=prev.load_worldtrack_sequence(Path(meta['source_path']),24);ids=np.asarray(z['query_ids'],dtype=np.int64)
        guv=np.asarray(s['tracks_uv'][:,ids],np.float32).transpose(1,0,2).copy();hh,ww=s['video_rgb'].shape[1:3];guv/=np.array([ww-1,hh-1],np.float32)
        valid=np.asarray(s['visibility'][:,ids]).T.astype(bool)&np.isfinite(guv).all(-1)&(guv>=0).all(-1)&(guv<=1).all(-1)
        guv=torch.tensor(np.nan_to_num(guv),device='cuda')[None];valid=torch.tensor(valid,device='cuda')[None]
        both=valid[:,:,15,None]&valid[:,:,16:24]&last[:,:,None]
        gtflow,_=raster(guv[:,:,15],guv[:,:,16:24],both.float(),*hid.shape[-2:])
        data={'path':p,'tau':tau[0],'pred':pred[0],'wrong':wrong[0],'gt':warp(gtflow)[0],
              'last':obs['context_rgb'][0,-1],'target':x['future_rgb'],'gate':gate[0]}
        data={k:v.cpu().half() if torch.is_tensor(v) else v for k,v in data.items()}
        assert all(torch.isfinite(v).all() for v in data.values() if torch.is_tensor(v))
        temp=dest.with_suffix('.tmp');torch.save(data,temp);os.replace(temp,dest)
        if (i//world+1)%20==0:
            progress={'stage':'cache','rank':rank,'position':i+1,'total':len(paths),'seconds':time.time()-start};prev.dump(OUT/f'cache_progress_{rank}.json',progress);print(json.dumps(progress),flush=True)
    prev.dump(OUT/f'CACHE_COMPLETE_{rank}.json',{'clips':len(paths[rank::world]),'seconds':time.time()-start})

def train(mode,epochs):
    torch.set_num_threads(2);prev.prior.seed_all(42);dest=OUT/mode;dest.mkdir(exist_ok=False);ps=splits()
    model=Fusion().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    import lpips
    lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def load(paths):
        items=[]
        for p in paths:
            d=torch.load(OUT/'cache'/(key(p)+'.pt'),map_location='cpu',weights_only=False)
            items.append({'path':p,'tau':d['tau'],'warp':d['last'][None].expand_as(d['tau']).clone() if mode=='zero' else d[mode], 'target':d['target'],'gate':d['gate']})
        return items
    trainset=load(ps['train']);valset=load(ps['val'])
    protocol={'mode':mode,'epochs':epochs,'batch_size':2,'lr':3e-4,'seed':42,'parameters':sum(p.numel() for p in model.parameters()),'counts':{k:len(v) for k,v in ps.items()},'tau_frozen':True,'loss':'MSE + .03 Alex LPIPS + .1 temporal MAE','pixel_cap':.08,'mean_budget_max':.002,'test_used':False,'wrong':'negative predicted dense flow, same predicted support','gate':'identical predicted support in all arms; zero is NOT geometry-free','gt':'future dataset projected trajectories and visibility, diagnostic upper bound only','selection':'best full validation composite including identity epoch 0'}
    prev.dump(dest/'protocol.json',protocol)
    def batch(items):return {k:torch.stack([d[k] for d in items]).cuda().float() for k in ['tau','warp','target','gate']}
    def losses(pred,target):
        mse=(pred-target).square().mean();per=lp(pred.flatten(0,1)*2-1,target.flatten(0,1)*2-1).mean();temp=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean()
        return mse+.03*per+.1*temp,{'mse':float(mse.detach()),'lpips':float(per.detach()),'temporal':float(temp.detach()),'mae':float((pred-target).abs().mean().detach())}
    @torch.no_grad()
    def evaluate():
        sums={};n=0;records=[];model.eval()
        for off in range(0,len(valset),2):
            items=valset[off:off+2];b=batch(items);pred=model(b['tau'],b['warp'],b['gate'])
            for j,item in enumerate(items):
                loss,met=losses(pred[j:j+1],b['target'][j:j+1]);met['cost']=float(loss);met['budget']=float((pred[j]-b['tau'][j]).abs().mean())
                records.append({'path':item['path'],**met});n+=1
                for k,v in met.items():sums[k]=sums.get(k,0)+v
        return {k:v/n for k,v in sums.items()},records
    best,records=evaluate();best_score=best['cost'];prev.dump(dest/'baseline.json',best)
    torch.save({'model':model.state_dict(),'epoch':0,'protocol':protocol},dest/'best.pt')
    start=time.time();history=[]
    for epoch in range(1,epochs+1):
        tick=time.time();model.train();order=list(range(len(trainset)));random.Random(epoch).shuffle(order);total=0;steps=0
        for off in range(0,len(order),2):
            b=batch([trainset[j] for j in order[off:off+2]]);pred=model(b['tau'],b['warp'],b['gate']);loss,_=losses(pred,b['target']);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach());steps+=1
            if steps==20 and epoch==1:print(json.dumps({'mode':mode,'stage':'first20','seconds':time.time()-tick,'steps_total':(len(order)+1)//2}),flush=True)
        met,records=evaluate();state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'protocol':protocol};torch.save(state,dest/'last.pt')
        if met['cost']<best_score:
            best_score=met['cost'];torch.save(state,dest/'best.pt');prev.dump(dest/'best_validation.json',{'epoch':epoch,'metrics':met});prev.dump(dest/'validation_per_clip.json',records)
        row={'epoch':epoch,'train_loss':total/steps,'validation':met,'best_cost':best_score,'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-start};history.append(row);prev.dump(dest/'progress.json',row);prev.dump(dest/'history.json',history);print(json.dumps({'mode':mode,**row}),flush=True)
    prev.dump(dest/'COMPLETE.json',{'seconds':time.time()-start,'test_used':False})

def launch(epochs):
    OUT.mkdir(exist_ok=True)
    prep=[subprocess.Popen([sys.executable,'-u',__file__,'prepare','--rank',str(i),'--world','4']) for i in range(4)]
    codes=[p.wait() for p in prep]
    assert codes==[0,0,0,0],codes
    prev.dump(OUT/'CACHE_COMPLETE.json',{'workers':4,'exit_codes':codes})
    procs={};handles=[]
    for mode in MODES:
        f=open(OUT/(mode+'.log'),'a');handles.append(f)
        procs[mode]=subprocess.Popen([sys.executable,'-u',__file__,'train','--mode',mode,'--epochs',str(epochs)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    prev.dump(OUT/'pids.json',{k:p.pid for k,p in procs.items()})
    statuses={k:p.wait() for k,p in procs.items()};prev.dump(OUT/'EXIT_STATUS.json',statuses)
    for f in handles:f.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','launch']);ap.add_argument('--mode',choices=MODES);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1);a=ap.parse_args()
    if a.action=='prepare':prepare(a.rank,a.world)
    elif a.action=='train':train(a.mode,a.epochs)
    else:launch(a.epochs)
