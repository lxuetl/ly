"""Matched cached-action experiment: channel regressor vs shared candidate ranker."""
import json,random,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import spearmanr
import train_single_region_risk_expanded as e
OUT=e.p.run.fit.ROOT/'shared_ranker_comparison_20261013'

class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Conv3d(22,43,3,padding=1),nn.GELU(),nn.Conv3d(43,43,3,padding=1),nn.GELU(),nn.Conv3d(43,1,1))
        nn.init.zeros_(self.scorer[-1].weight);nn.init.zeros_(self.scorer[-1].bias)
    def forward(self,x):
        b,c,t,h,w=x.shape
        context=torch.cat([x[:,24:34],x[:,50:57]],1)
        inputs=[]
        for k in range(8):inputs.append(torch.cat([context,x[:,3*k:3*k+3],x[:,34+k:35+k],x[:,42+k:43+k]],1))
        z=self.scorer(torch.stack(inputs,1).reshape(b*8,22,t,h,w)).reshape(b,8,t,h,w)
        return torch.cat([z,torch.zeros_like(z[:,:1])],1)

class Shared(nn.Module):
    def __init__(self):super().__init__();self.net=SharedNet()

@torch.no_grad()
def evaluate(model,data):
    rows=[];corr=[];preds=[];labels=[]
    for off in range(0,len(data),8):
        ds=data[off:off+8];z=e.s.scores(model,torch.stack([d['features'] for d in ds]).cuda()).cpu()
        for d,zz in zip(ds,z):
            mask=d['mask'].bool().clone();mask[:,1]=False
            a=zz[mask].numpy();b=d['raw'][mask].numpy()
            if len(a)==0:
                rows.append({'path':d['path'],'group':e.group(d),'score':-1e30,'gain':0.,'oracle':0.});continue
            if a.std()>1e-9 and b.std()>1e-9:corr.append(float(spearmanr(a,b).statistic))
            preds.extend(a.tolist());labels.extend(b.tolist());k=int(a.argmax())
            rows.append({'path':d['path'],'group':e.group(d),'score':float(a[k]),'gain':float(b[k])*1e-4/256,'oracle':max(0.,float(b.max()))*1e-4/256})
    return {'within_clip_spearman':float(np.mean(corr)) if corr else 0.,'pooled_spearman':float(spearmanr(preds,labels).statistic),'rows':rows}

def decisions(rows,threshold):
    active=[r for r in rows if r['score']>threshold];gs=defaultdict(list)
    for r in rows:gs[tuple(r['group'])].append(r['gain'] if r['score']>threshold else 0.)
    vals=np.array([np.mean(v) for v in gs.values()]);rng=np.random.default_rng(42)
    ci=np.quantile(vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(1),[.025,.975]).tolist()
    return {'threshold':float(threshold),'coverage':len(active)/len(rows),'mean_clip_gain':sum(r['gain'] for r in active)/len(rows),
        'mean_video_gain':float(vals.mean()),'video_gain_ci95':ci,'harm_when_active':float(np.mean([r['gain'] < -1e-8 for r in active])) if active else None}

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);e.tr.prev.prior.seed_all(20261013);start=time.time()
    proto=json.loads((e.OUT/'protocol.json').read_text());parts={k:{tuple(g) for g in v} for k,v in proto['validation_groups'].items()}
    paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths'];data={k:[] for k in ['train','tune','risk','held']}
    for split in ['train','val']:
        for path in paths[split]:
            key=hashlib.sha256(path.encode()).hexdigest()[:24];d=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False)
            part='train' if split=='train' else next(k for k,v in parts.items() if e.group(d) in v);data[part].append(d)
    reg=e.tr.TemporalSelector().cuda()
    with torch.no_grad():reg.net[-1].bias.zero_()
    models={'regression':reg,'shared_rank':Shared().cuda()};opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for k,m in models.items()}
    protocol={'counts':{k:len(v) for k,v in data.items()},'params':{k:sum(x.numel() for x in m.parameters()) for k,m in models.items()},
      'epochs':60,'seed':20261013,'batch_size':8,'labels':'identical cached rendered marginal utility; same 16 cells and single-action renderer',
      'shared_inputs':'candidate RGB correction3, replay1, magnitude1, shared context17; no candidate ID',
      'rank_loss':'pairwise logistic for same-tube candidate gains differing >0.1 utility units, plus 0.1 regression anchor; fixed reference0',
      'checkpoint_selection':'mean within-clip Spearman on tuning videos only, evaluated every10 epochs',
      'comparison_caveat':'joint architecture+loss change, not an isolated attribution; single seed; previous validation pool reused',
      'decision':'fixed score>0 plus thresholds at tune top-score quantiles for nominal 25/50/75% coverage; not certified risk control','test_used':False}
    e.tr.prev.dump(OUT/'protocol.json',protocol);print(json.dumps({'stage':'start',**protocol}),flush=True)
    counts=defaultdict(int)
    for d in data['train']:counts[e.group(d)]+=1
    weights=[1/counts[e.group(d)] for d in data['train']];best={k:-float('inf') for k in models};history=[]
    for epoch in range(1,61):
        tick=time.time();order=random.Random(20261013+epoch).choices(range(len(data['train'])),weights=weights,k=len(data['train']));total=defaultdict(float);steps=0
        for off in range(0,len(order),8):
            ds=[data['train'][i] for i in order[off:off+8]];features=torch.stack([d['features'] for d in ds]).cuda()
            lab=torch.stack([d['raw'].clamp(-10,10) for d in ds]).cuda();mask=torch.stack([d['mask'] for d in ds]).cuda();mask[:,:,1]=0
            for kind,m in models.items():
                m.train();z=e.s.scores(m,features)
                point=F.smooth_l1_loss(z,lab,reduction='none');interval=F.smooth_l1_loss((z.abs()-.1).clamp_min(0),torch.zeros_like(z),reduction='none')
                anchor=(torch.where(lab.abs()<=.1,interval,point)*mask).sum()/mask.sum().clamp_min(1)
                loss=anchor
                if kind=='shared_rank':
                    diff=lab[:,:,:,None]-lab[:,:,None,:];zdiff=z[:,:,:,None]-z[:,:,None,:]
                    valid=(diff.abs()>.1)*mask[:,:,0:1,None].bool()
                    rank=(F.softplus(-diff.sign()*zdiff)*valid).sum()/valid.sum().clamp_min(1)
                    loss=rank+.1*anchor
                assert torch.isfinite(loss);opts[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opts[kind].step();total[kind]+=loss.item()
            steps+=1
        row={'epoch':epoch,'seconds':time.time()-tick,'elapsed':time.time()-start,'loss':{k:v/steps for k,v in total.items()}}
        if epoch%10==0:
            row['tune']={}
            for kind,m in models.items():
                m.eval();r=evaluate(m,data['tune']);score=r['within_clip_spearman'];row['tune'][kind]={k:v for k,v in r.items() if k!='rows'}
                if score>best[kind]:best[kind]=score;torch.save({'model':m.state_dict(),'epoch':epoch},OUT/(kind+'_best.pt'))
        for kind,m in models.items():torch.save({'model':m.state_dict(),'epoch':epoch},OUT/(kind+'_last.pt'))
        history.append(row);e.tr.prev.dump(OUT/'history.json',history);e.tr.prev.dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    summary={}
    for kind,m in models.items():
        ck=torch.load(OUT/(kind+'_best.pt'),weights_only=False);m.load_state_dict(ck['model']);m.eval()
        r={split:evaluate(m,ds) for split,ds in data.items()}
        thresholds={'positive':0.,**{str(c):float(np.quantile([a['score'] for a in r['tune']['rows']],1-c)) for c in [.25,.5,.75]}}
        summary[kind]={'epoch':ck['epoch'],'ranking':{split:{k:v for k,v in res.items() if k!='rows'} for split,res in r.items()},
          'decisions':{split:{name:decisions(r[split]['rows'],th) for name,th in thresholds.items()} for split in ['tune','risk','held']}}
        e.tr.prev.dump(OUT/(kind+'_rows.json'),r)
    e.tr.prev.dump(OUT/'summary.json',summary);e.tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps({'stage':'complete','seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
