"""Train a small temporal region selector over frozen, observed-only candidates.

Two matched heads: geometry features vs zeroed explicit geometry channels.
Both share D4RT-derived candidates and history replay; the latter is NOT a
fully geometry-free system. Future data is only used in targets and evaluation.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import eval_tau128_candidate_bank as cb

prev=cb.previous
N=len(cb.NAMES)
VARIANTS=['geometry','no_explicit_geometry']

def pool(x):
    shape=x.shape
    return F.avg_pool2d(x.reshape(-1,shape[-3],128,128),16).reshape(*shape[:-2],8,8)

@torch.no_grad()
def input_features(bank,aux,observed,replay):
    # All arguments here are derived solely from observed frames/predictions.
    delta=bank-bank[0:1]
    t=bank.shape[1]
    rgb_delta=pool(delta).permute(1,0,2,3,4).reshape(t,N*3,8,8)*100
    last=pool(observed['context_rgb'][0,-1:]).expand(t,-1,-1,-1)
    velocity=pool(observed['context_rgb'][0,-1:]-observed['context_rgb'][0,-2:-1]).expand(t,-1,-1,-1)
    horizon=torch.linspace(0,1,t,device=bank.device)[:,None,None,None].expand(t,1,8,8)
    advantage=torch.tanh((replay[1:2]-replay)/1e-4).permute(1,0,2,3)
    magnitude=pool(delta.abs().mean(2,keepdim=True)).squeeze(2).permute(1,0,2,3)*100
    confidence=torch.nan_to_num(observed['past_confidence']).clamp(0,1)
    valid=observed['past_valid'].float()
    conf=((confidence*valid).sum()/valid.sum().clamp_min(1)).expand(t,1,8,8)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,8,device=bank.device),torch.linspace(-1,1,8,device=bank.device),indexing='ij')
    position=torch.stack([xx,yy])[None].expand(t,-1,-1,-1)
    features=torch.cat([rgb_delta,pool(bank[0]),last,velocity,horizon,advantage,magnitude,pool(aux),conf,position],1)
    assert features.shape==(8,57,8,8) and torch.isfinite(features).all()
    return features

class TemporalSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(57,32,3,padding=1),nn.GELU(),
                               nn.Conv3d(32,32,3,padding=1),nn.GELU(),nn.Conv3d(32,N+1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():self.net[-1].bias[1]=3.;self.net[-1].bias[-1]=-3.
    def forward(self,features,no_geometry=False,gain=1.):
        if no_geometry:
            features=features.clone();features[:,:,-7:-2]=0
        logits=self.net(features.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        choices=logits[:,:,:N].softmax(2)
        gate=logits[:,:,N:].sigmoid()*gain
        reference=torch.zeros_like(choices);reference[:,:,1]=1
        return reference*(1-gate)+gate*choices,gate

def synthesize(bank,weights,budget):
    # bank: B,N,T,C,H,W; weights: B,T,N,8,8
    b,n,t,c,h,w=bank.shape
    dense=F.interpolate(weights.reshape(b*t,n,8,8),(h,w),mode='bilinear',align_corners=False)
    dense=dense.reshape(b,t,n,h,w).permute(0,2,1,3,4)[:,:,:,None]
    pred=(bank*dense).sum(1)
    tau=bank[:,0];delta=pred-tau
    norm=delta.abs().mean((1,2,3,4),keepdim=True)
    factor=(budget.reshape(b,1,1,1,1)/norm.clamp_min(1e-9)).clamp(max=1.)
    return (tau+delta*factor).clamp(0,1)

@torch.no_grad()
def teacher_scores(bank,target,lpips):
    values=[]
    for pred in bank:
        err=(pred-target).square().mean(1,keepdim=True)
        difference=pred-target
        temporal=torch.cat([torch.zeros_like(difference[:1]),(difference[1:]-difference[:-1]).abs()],0).mean(1,keepdim=True)
        perceptual=lpips(pred*2-1,target*2-1)
        score=pool(err)+.03*pool(perceptual)+.1*pool(temporal)
        values.append(score[:,0])
    return torch.stack(values).permute(1,0,2,3)

class ExperimentCache:
    def __init__(self,out,cfg,lpips):
        self.root=out/'cache';self.root.mkdir(exist_ok=True)
        self.lpips=lpips;self.cfg=cfg;self.models=None
        self.counters=defaultdict(int);self.build_seconds=[];self.read_seconds=[]
    def initialize(self):
        if self.models is not None:return
        backbone,joint,adapter,_=prev.old.load_frozen(self.cfg,torch.device('cuda'))
        config=prev.load_yaml_config(str(prev.ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml'))
        d4rt=prev.build_model(config['model']).eval()
        state=prev._unwrap_state_dict(prev.load_checkpoint(str(prev.ROOT/'checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt'),map_location='cpu'))
        d4rt.load_state_dict(state,strict=True);del state
        d4rt.cuda().requires_grad_(False)
        self.models=(backbone,joint,adapter,d4rt,tuple(config.get_path('model.input.image_size',[256,256])))
    @torch.no_grad()
    def get(self,path,dataset,index):
        start=time.time();key=hashlib.sha256(str(path).encode()).hexdigest()[:24]
        file=self.root/(key+'.pt')
        if file.exists():
            value=torch.load(file,map_location='cpu',weights_only=False)
            assert value['path']==str(path)
            self.counters['read']+=1;self.read_seconds.append(time.time()-start)
            return value
        self.initialize()
        backbone,joint,adapter,d4rt,image_hw=self.models
        item=dataset[index]
        observed={k:item[k][None].cuda() for k in prev.PAST}
        observed['context_rgb']=item['context_rgb'][None].cuda()
        replay=cb.history_choices(path,observed,d4rt,image_hw,backbone,joint,adapter,return_scores=True)
        bank,budget,aux=cb.build_bank(observed,backbone,joint,adapter,return_aux=True)
        feature=input_features(bank,aux,observed,replay)
        # The feature tensor has already been constructed before labels/GT.
        target=item['future_rgb'].cuda()
        labels=teacher_scores(bank,target,self.lpips)
        batch=prev.core.move(prev.video_pad_collate([item]),torch.device('cuda'))
        mask,_=prev.prior.camera_compensated_dynamic_mask(batch,128,128,4,6)
        value={'path':str(path),'bank':bank.cpu(),'features':feature.cpu(),
               'target':target.cpu(),'teacher_scores':labels.cpu(),'mask':mask[0,:,None].cpu(),'budget':budget}
        torch.save(value,file.with_suffix('.tmp'));file.with_suffix('.tmp').replace(file)
        self.counters['built']+=1;self.build_seconds.append(time.time()-start)
        return value

def batchify(items):
    return {key:torch.stack([x[key] for x in items]).cuda() for key in ['bank','features','target','teacher_scores','mask']} | {'budget':torch.tensor([x['budget'] for x in items],device='cuda')}

def loss_function(model,batch,lpips,variant):
    weights,gate=model(batch['features'],variant=='no_explicit_geometry')
    pred=synthesize(batch['bank'],weights,batch['budget']);target=batch['target']
    mse=(pred-target).square().mean()
    perception=lpips(pred.flatten(0,1)*2-1,target.flatten(0,1)*2-1).mean()
    temporal=((pred[:,1:]-pred[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean()
    scores=batch['teacher_scores']
    temperature=scores.std(2,keepdim=True).clamp_min(1e-5)
    labels=(-((scores-scores.amin(2,keepdim=True))/temperature).clamp(max=30)).softmax(2)
    active=(batch['bank'][:,1]-batch['bank'][:,0]).abs().mean(2,keepdim=True)
    active=(pool(active)[:,:,0]>1e-6).float()
    ranking=(-(labels*weights.clamp_min(1e-8).log()).sum(2)*active).sum()/active.sum().clamp_min(1)
    beneficial=(scores[:,:,1]-scores.amin(2)>1e-5).float()
    risk=F.binary_cross_entropy(gate[:,:,0].clamp(1e-6,1-1e-6),beneficial,reduction='none')
    risk=(risk*active*(1+2*(1-beneficial))).sum()/active.sum().clamp_min(1)
    switching=(weights[:,1:]-weights[:,:-1]).abs().mean()
    spatial=(weights[...,1:,:]-weights[...,:-1,:]).abs().mean()+(weights[...,:,1:]-weights[...,:,:-1]).abs().mean()
    loss=mse+.03*perception+.1*temporal+.001*ranking+.0002*risk+.001*switching+.0001*spatial
    return loss,{'loss':loss.item(),'MSE':mse.item(),'LPIPS':perception.item(),'temporal':temporal.item(),'gate':gate.mean().item(),'switching':switching.item()}

@torch.no_grad()
def measure(pred,target,tau,mask,lpips):
    err=pred-target;mse=err.square().mean().item()
    expanded=mask.expand_as(err)
    return {'MSE':mse,'MAE':err.abs().mean().item(),'PSNR':-10*math.log10(max(mse,1e-12)),
            'LPIPS':lpips(pred*2-1,target*2-1).mean().item(),
            'temporal_MAE':((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean().item(),
            'dynamic_MSE':err.square()[expanded].mean().item() if expanded.any() else None,
            'pixel_delta':(pred-tau).abs().mean().item()}

def composite(m):return m['MSE']+.03*m['LPIPS']+.1*m['temporal_MAE']

@torch.no_grad()
def evaluate(models,cache,paths,dataset,lpips,gains=(1.,)):
    records=[]
    for index,path in enumerate(paths):
        item=cache.get(path,dataset,index);batch=batchify([item]);bank=batch['bank'][0]
        predictions={'tau':bank[0],'geometry':bank[1]};details={}
        for variant,model in models.items():
            model.eval()
            for gain in gains:
                name=f'{variant}_g{gain:g}';weights,gate=model(batch['features'],variant=='no_explicit_geometry',gain)
                pred=synthesize(batch['bank'],weights,batch['budget'])[0]
                assert (pred-bank[0]).abs().mean()<=item['budget']+1e-6
                predictions[name]=pred
                details[name]={'gate':gate.mean().item(),'switching':(weights[:,1:]-weights[:,:-1]).abs().mean().item()}
        record={'path':str(path),'group':prev.group(path),'metrics':{name:measure(pred,batch['target'][0],bank[0],batch['mask'][0],lpips) for name,pred in predictions.items()},'routing':details}
        records.append(record)
    means={}
    for name in records[0]['metrics']:
        means[name]={key:float(np.mean([r['metrics'][name][key] for r in records if r['metrics'][name][key] is not None])) for key in records[0]['metrics'][name] if any(r['metrics'][name][key] is not None for r in records)}
    return means,records

def bootstrap(rows,names):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row['group'])].append(row)
    rng=np.random.default_rng(42);result={}
    for name in names:
        for metric in ['MSE','LPIPS','temporal_MAE']:
            sums=[];counts=[]
            for rs in grouped.values():
                sums.append(sum(r['metrics']['geometry'][metric]-r['metrics'][name][metric] for r in rs));counts.append(len(rs))
            idx=rng.integers(0,len(sums),(5000,len(sums)))
            sample=np.array(sums)[idx].sum(1)/np.array(counts)[idx].sum(1)
            result[name+'/'+metric]={'gain':sum(sums)/sum(counts),'ci95':np.quantile(sample,[.025,.975]).tolist(),'clusters':len(sums)}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);prev.prior.seed_all(42)
    base=Path('/root/autodl-fs/d4rt_training')
    cfg=SimpleNamespace(backbone_checkpoint=str(base/'tau128_full_fair_fast_20261009/best.pt'),trajectory_checkpoint=str(base/'pointodyssey_full_uv_velocity_residual_60/best.pt'),geometry_checkpoint=str(base/'tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt'),image_size=128)
    paths=json.loads((base/'tau_v2_direction_rank_pb002_20261009/split_paths.json').read_text())['paths']
    paths={s:paths[s][:args.limit or None] for s in ['train','val','test']}
    sets={s:set(map(str,v)) for s,v in paths.items()}
    assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
    groups={s:{prev.group(p) for p in v} for s,v in paths.items()}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test'],'original-video split overlap'
    prev.dump(out/'split_paths.json',{'paths':paths})
    import lpips
    perception=lpips.LPIPS(net='alex',spatial=True,verbose=False).cuda().eval().requires_grad_(False)
    cache=ExperimentCache(out,cfg,perception)
    initial=TemporalSelector().cuda();models={VARIANTS[0]:initial,VARIANTS[1]:copy.deepcopy(initial)}
    opts={v:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for v,m in models.items()}
    datasets={s:prev.VideoForecastDataset('',s,128,paths=p) for s,p in paths.items()}
    protocol={'counts':{s:len(v) for s,v in paths.items()},'epochs':args.epochs,'batch_size':args.batch_size,'params_each':sum(p.numel() for p in initial.parameters()),'region_size':16,'candidates':cb.NAMES,'backbones_frozen':True,
              'ablation':'zeros 5 explicit geometry/confidence channels; retains same geometry-derived candidates and replay',
              'input_channels':57,'feature_label_separation':True,'replay_prefixes':[8,12],'padding':'repeat first frame to 16',
              'loss_weights':{'MSE':1,'LPIPS':.03,'temporal':.1,'preference':.001,'gate_risk':.0002,'weight_temporal':.001,'weight_spatial':.0001},
              'selection':'full validation composite MSE+.03 LPIPS+.1 temporal, then gain {0,.5,1}; test once',
              'lpips':'spatial AlexNet maps for teacher/training; scalar AlexNet for final calibration and test',
              'cache':'lossless float32 frozen candidates and observed-only features; future targets and teacher losses stored separately',
              'split_audit':'no exact-path or parsed original-video overlaps'}
    prev.dump(out/'protocol.json',protocol);print(json.dumps({'stage':'start',**protocol}),flush=True)
    best={v:float('inf') for v in models};history=[];start=time.time()
    for epoch in range(1,args.epochs+1):
        tick=time.time();totals={v:defaultdict(float) for v in models};batches=0
        order=list(range(len(paths['train'])));random.Random(42+epoch).shuffle(order)
        for offset in range(0,len(order),args.batch_size):
            indices=order[offset:offset+args.batch_size]
            items=[cache.get(paths['train'][i],datasets['train'],i) for i in indices]
            batch=batchify(items)
            for variant,model in models.items():
                model.train();opts[variant].zero_grad(set_to_none=True)
                loss,stats=loss_function(model,batch,perception,variant)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opts[variant].step()
                for key,value in stats.items():totals[variant][key]+=value
            batches+=1
            if batches%10==0 or args.limit:
                progress={'stage':'training','epoch':epoch,'epochs':args.epochs,'clips_epoch':min(offset+args.batch_size,len(order)),'train_clips':len(order),'batches':batches,'elapsed_seconds':time.time()-start,'epoch_seconds':time.time()-tick,'cache':dict(cache.counters),'recent_cache_build_seconds':float(np.mean(cache.build_seconds[-20:])) if cache.build_seconds else 0,'gpu_peak_GB':torch.cuda.max_memory_allocated()/2**30,'training':{v:{k:x/batches for k,x in d.items()} for v,d in totals.items()}}
                prev.dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        train_seconds=time.time()-tick
        print(json.dumps({'stage':'validation','epoch':epoch,'train_seconds':train_seconds}),flush=True)
        val,val_rows=evaluate(models,cache,paths['val'],datasets['val'],perception)
        row={'epoch':epoch,'train_seconds':train_seconds,'epoch_seconds':time.time()-tick,'val':val,'train':{v:{k:x/batches for k,x in d.items()} for v,d in totals.items()}}
        history.append(row);prev.dump(out/'history.json',history)
        for v,m in models.items():
            state={'model':m.state_dict(),'epoch':epoch,'protocol':protocol,'optimizer':opts[v].state_dict()}
            torch.save(state,out/f'{v}_last.pt')
            score=composite(val[v+'_g1'])
            if score<best[v]:best[v]=score;torch.save(state,out/f'{v}_best.pt')
        print(json.dumps({'stage':'epoch_complete',**row}),flush=True)
    for v,m in models.items():m.load_state_dict(torch.load(out/f'{v}_best.pt',map_location='cuda',weights_only=False)['model'])
    scalar_perception=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    val,val_rows=evaluate(models,cache,paths['val'],datasets['val'],scalar_perception,gains=(0.,.5,1.))
    selected={v:min([v+'_g0',v+'_g0.5',v+'_g1'],key=lambda n:composite(val[n])) for v in models}
    prev.dump(out/'validation_selection.json',{'selected':selected,'val':val})
    test,rows=evaluate(models,cache,paths['test'],datasets['test'],scalar_perception,gains=(0.,.5,1.))
    (out/'test_per_clip.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={'selected_on_validation':selected,'test':test,'paired_cluster_bootstrap':bootstrap(rows,[*selected.values(),*[v+'_g1' for v in models]]),'by_source':{d:{n:float(np.mean([r['metrics'][n]['MSE'] for r in rows if r['group'][0]==d])) for n in test} for d in sorted(set(r['group'][0] for r in rows))},'seconds':time.time()-start}
    prev.dump(out/'summary.json',summary);prev.dump(out/'COMPLETE.json',{'seconds':time.time()-start})
    print(json.dumps({'stage':'complete','selected':selected,'test':test}),flush=True)

if __name__=='__main__':main()
