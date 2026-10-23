"""Matched sparse tube supervision: old local scores vs rendered marginal utility."""
import copy,json,random,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import train_selector_utility_full as run
tr=run.tr
OUT=run.fit.ROOT/'selector_rendered_pilot_20261013'

def subset(paths,n,seed):
    buckets=defaultdict(list);rng=random.Random(seed)
    for p in paths:buckets[tr.prev.group(p)[0]].append(p)
    for ps in buckets.values():rng.shuffle(ps)
    out=[];used=set()
    while len(out)<n:
        for d in sorted(buckets):
            ps=buckets[d]
            if not ps:continue
            p=next((p for p in ps if tr.prev.group(p) not in used),ps[0])
            ps.remove(p);out.append(p);used.add(tr.prev.group(p))
            if len(out)==n:break
    return out

def tube_scores(m,features):
    z=m.net(features.permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
    return z.reshape(z.shape[0],4,2,8,8,8).mean(2)

def predict(m,b,kind):
    z=tube_scores(m,b['features']).repeat_interleave(2,dim=1)
    w=z.softmax(2) if kind.endswith('soft') else F.one_hot(z.argmax(2),8).permute(0,1,4,2,3).float()
    ref=torch.zeros_like(w);ref[:,:,1]=1
    return torch.where(((z.amax(2)-z[:,:,1])>.1)[:,:,None],w,ref)

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);tr.prev.prior.seed_all(20261013)
    paths=json.loads((run.fit.SOURCE/'split_paths.json').read_text())['paths']
    trainpaths=subset(paths['train'],64,13);valpaths=subset(paths['val'],32,14)
    assert not {tr.prev.group(p) for p in trainpaths}&{tr.prev.group(p) for p in valpaths}
    protocol={'train':trainpaths,'val':valpaths,'test_used':False,'epochs':60,'batch_size':2,'lr':.001,'seed':20261013,
      'tube':'same spatial cell, two consecutive forecast frames; 4 tubes per train clip, chosen from observed-derived active support without GT',
      'labels':'old=mean local teacher gain; rendered=full scalar evaluation loss reduction after real render, multiplied by 256 to normalize tube area-time',
      'matched':'same initialization, sparse supervision locations, candidates, tube decoding, optimizer, uncertainty loss in both arms',
      'uncertainty':'target magnitude <=0.1 in 1e-4 units uses interval loss, not hard beneficial/harmful classification',
      'selection':'validation composite MSE+.03 LPIPS+.1 temporal over 10-epoch checkpoints and soft/hard; baseline fallback',
      'warning':'isolated-tube marginal labels are not guaranteed additive when multiple tubes change together'}
    tr.prev.dump(OUT/'protocol.json',protocol)
    import lpips
    lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def cost(pred,target):
        return (pred-target).square().mean()+.03*lp(pred*2-1,target*2-1).mean()+.1*((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    data=[];start=time.time();rng=random.Random(15);pairs=[]
    with torch.no_grad():
        for i,p in enumerate(trainpaths):
            x=run.load(p);b=tr.batchify([x]);bank=b['bank'];base=bank[0,1];target=b['target'][0]
            active=tr.pool((base-bank[0,0]).abs().mean(1,keepdim=True))[:,0].reshape(4,2,8,8).mean(1)>1e-6
            loc=torch.nonzero(active).tolist();rng.shuffle(loc);loc=loc[:4]
            old=((x['teacher_scores'][:,1:2]-x['teacher_scores'])/1e-4).reshape(4,2,8,8,8).mean(1)
            new=torch.zeros_like(old);mask=torch.zeros_like(old)
            reference=cost(base,target)
            ref=torch.zeros_like(b['teacher_scores']);ref[:,:,1]=1
            for t,y,z in loc:
                mask[t,:,y,z]=1
                for k in range(8):
                    if k==1:continue
                    w=ref.clone();w[:,2*t:2*t+2,:,y,z]=0;w[:,2*t:2*t+2,k,y,z]=1
                    pred=tr.synthesize(bank,w,b['budget'])[0]
                    new[t,k,y,z]=(reference-cost(pred,target)).cpu()*256/1e-4
                    pairs.append([float(old[t,k,y,z]),float(new[t,k,y,z])])
            data.append({'features':x['features'],'old':old.clamp(-10,10),'rendered':new.clamp(-10,10),'mask':mask})
            progress={'stage':'labels','clips':i+1,'total':64,'seconds':time.time()-start,'remaining_label_seconds':(time.time()-start)/(i+1)*(63-i)}
            tr.prev.dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    torch.save(data,OUT/'training_labels.pt');tr.prev.dump(OUT/'label_pairs.json',pairs)
    val=[run.load(p) for p in valpaths]
    init=tr.TemporalSelector().cuda()
    with torch.no_grad():init.net[-1].bias.zero_()
    models={'old':init,'rendered':copy.deepcopy(init)}
    opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for k,m in models.items()}
    run.fit.predict=predict;best={k:float('inf') for k in models};history=[];training_start=time.time()
    for epoch in range(1,61):
        tick=time.time();order=list(range(64));random.Random(epoch).shuffle(order);losses=defaultdict(float)
        for off in range(0,64,2):
            items=[data[i] for i in order[off:off+2]]
            b={k:torch.stack([x[k] for x in items]).cuda() for k in items[0]}
            for kind,m in models.items():
                m.train();est=tube_scores(m,b['features']);target=b[kind]
                regular=F.smooth_l1_loss(est,target,reduction='none')
                interval=F.smooth_l1_loss((est.abs()-.1).clamp_min(0),torch.zeros_like(est),reduction='none')
                loss=(torch.where(target.abs()<=.1,interval,regular)*b['mask']).sum()/b['mask'].sum().clamp_min(1)
                assert torch.isfinite(loss)
                opts[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opts[kind].step();losses[kind]+=loss.item()/32
        progress={'stage':'training','epoch':epoch,'epochs':60,'seconds':time.time()-training_start,'loss':dict(losses)}
        if epoch%10==0 or epoch==1:
            modes={k+'_'+mode:m for k,m in models.items() for mode in ['soft','hard']}
            means,routing,rows=run.fit.evaluate(modes,val,lp)
            progress['metrics']=means
            for kind,m in models.items():
                mode=min([kind+'_soft',kind+'_hard'],key=lambda k:tr.composite(means[k]));score=tr.composite(means[mode])
                if score<best[kind]:
                    best[kind]=score;torch.save({'model':m.state_dict(),'epoch':epoch,'mode':mode,'protocol':protocol},OUT/(kind+'_best.pt'))
                    tr.prev.dump(OUT/(kind+'_selection.json'),{'epoch':epoch,'mode':mode,'selected_with_fallback':mode if score<tr.composite(means['baseline']) else 'baseline','metrics':means})
                    (OUT/(kind+'_val_rows.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        progress['epoch_seconds']=time.time()-tick;history.append(progress)
        tr.prev.dump(OUT/'history.json',history);tr.prev.dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start,'test_used':False})

if __name__=='__main__':main()
