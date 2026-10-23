"""Training-only 32-clip fit test. All comparisons are in-sample diagnostics."""
import copy,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import train_tau128_temporal_selector as tr

ROOT=Path('/root/autodl-fs/d4rt_training')
SOURCE=ROOT/'tau128_temporal_selector_20261011'
OUT=ROOT/'selector_fit32_20261012'

def active(batch):
    return (tr.pool((batch['bank'][:,1]-batch['bank'][:,0]).abs().mean(2,keepdim=True))[:,:,0]>1e-6).float()

def predict(model,batch,kind):
    if kind=='original':return model(batch['features'])[0]
    scores=model.net(batch['features'].permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
    # Predicted relative utility; use one candidate, fallback for small margins.
    best=scores.argmax(2)
    gain=scores.amax(2)-scores[:,:,1]
    best=torch.where(gain>.1,best,torch.ones_like(best))
    return F.one_hot(best,8).permute(0,1,4,2,3).float()

@torch.no_grad()
def evaluate(models,data,lpips):
    rows=[]
    for item in data:
        b=tr.batchify([item]);a=active(b)
        target=b['teacher_scores'].argmin(2)
        oracle=F.one_hot(target,8).permute(0,1,4,2,3).float()
        predictions={'baseline':b['bank'][0,1],'oracle_same_renderer':tr.synthesize(b['bank'],oracle,b['budget'])[0]}
        details={}
        for kind,model in models.items():
            model.eval();w=predict(model,b,kind)
            predictions[kind]=tr.synthesize(b['bank'],w,b['budget'])[0]
            details[kind]={'accuracy':(((w.argmax(2)==target).float()*a).sum()/a.sum().clamp_min(1)).item(),
                           'entropy':((-(w*w.clamp_min(1e-8).log()).sum(2)*a).sum()/a.sum().clamp_min(1)).item()}
        metrics={k:tr.measure(p,b['target'][0],b['bank'][0,0],b['mask'][0],lpips) for k,p in predictions.items()}
        rows.append({'path':item['path'],'metrics':metrics,'routing':details})
    means={k:{m:float(np.mean([r['metrics'][k][m] for r in rows if r['metrics'][k][m] is not None])) for m in rows[0]['metrics'][k] if any(r['metrics'][k][m] is not None for r in rows)} for k in rows[0]['metrics']}
    routing={k:{m:float(np.mean([r['routing'][k][m] for r in rows])) for m in ['accuracy','entropy']} for k in models}
    return means,routing,rows

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);tr.prev.prior.seed_all(20261012)
    paths=json.loads((SOURCE/'split_paths.json').read_text())['paths']['train']
    buckets=defaultdict(list)
    for p in paths:buckets[tr.prev.group(p)[0]].append(p)
    rng=random.Random(20261012)
    for v in buckets.values():rng.shuffle(v)
    chosen=[];used=set()
    while len(chosen)<32:
        for domain in sorted(buckets):
            remaining=[p for p in buckets[domain] if p not in chosen]
            if not remaining:continue
            p=next((p for p in remaining if tr.prev.group(p) not in used),remaining[0])
            chosen.append(p);used.add(tr.prev.group(p))
            if len(chosen)==32:break
    data=[]
    for p in chosen:
        key=hashlib.sha256(p.encode()).hexdigest()[:24]
        item=torch.load(SOURCE/'cache'/(key+'.pt'),map_location='cpu',weights_only=False)
        assert item['path']==p;data.append(item)
    counts={d:sum(tr.prev.group(p)[0]==d for p in chosen) for d in buckets}
    tr.prev.dump(OUT/'protocol.json',{'paths':chosen,'source_counts':counts,'epochs':80,'selection':'seeded source round robin, prefer distinct original videos','in_sample_only':True,'test_or_val_used':False,'original':'unchanged original loss and gated soft rendering','utility':'same 77257-param network, relative-utility regression and hard selection; ninth channel unused','oracle':'composite-score hard labels with identical bilinear renderer and budget','utility_units':'1e-4 composite score; Huber clip +-10; margin 0.1'})
    import lpips
    spatial=lpips.LPIPS(net='alex',spatial=True,verbose=False).cuda().eval().requires_grad_(False)
    scalar=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    init=tr.TemporalSelector().cuda();models={'original':init,'utility':copy.deepcopy(init)}
    opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for k,m in models.items()}
    # Utility head starts unbiased; network architecture and width are identical.
    with torch.no_grad():models['utility'].net[-1].bias.zero_()
    start=time.time();history=[]
    for epoch in range(81):
        loss_sum=defaultdict(float)
        if epoch:
            order=list(range(32));random.Random(epoch).shuffle(order)
            for off in range(0,32,2):
                b=tr.batchify([data[i] for i in order[off:off+2]])
                for kind,m in models.items():
                    m.train();opts[kind].zero_grad(set_to_none=True)
                    if kind=='original':loss,_=tr.loss_function(m,b,spatial,'geometry')
                    else:
                        est=m.net(b['features'].permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
                        labels=((b['teacher_scores'][:,:,1:2]-b['teacher_scores'])/1e-4).clamp(-10,10)
                        a=active(b)[:,:,None]
                        loss=(F.smooth_l1_loss(est,labels,reduction='none')*a).sum()/(a.sum()*8).clamp_min(1)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opts[kind].step();loss_sum[kind]+=loss.item()/16
        if epoch%10==0:
            metrics,routing,rows=evaluate(models,data,scalar)
            row={'epoch':epoch,'seconds':time.time()-start,'loss':dict(loss_sum),'metrics':metrics,'routing':routing}
            history.append(row);tr.prev.dump(OUT/'history.json',history)
            print(json.dumps(row),flush=True)
    for k,m in models.items():torch.save({'model':m.state_dict(),'epoch':80},OUT/(k+'_last.pt'))
    (OUT/'per_clip.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tr.prev.dump(OUT/'summary.json',{'in_sample_only':True,'source_counts':counts,'final':history[-1],'history':history})
    tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start})

if __name__=='__main__':main()
