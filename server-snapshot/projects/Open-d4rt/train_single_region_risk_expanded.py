"""Expanded observed-support marginal labels; isolated tuning, risk audit, holdout."""
import json,random,time,hashlib,math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta
import experiment_single_region as s
p=s.p;tr=s.tr
OUT=p.run.fit.ROOT/'single_region_risk_expanded_20261013'
THRESH=[0,.1,.5,1,2,5,10]

def group(d):return tuple(tr.prev.group(d['path']))

@torch.no_grad()
def results(model,data,threshold):
    rows=[]
    for off in range(0,len(data),16):
        ds=data[off:off+16];z=s.scores(model,torch.stack([d['features'] for d in ds]).cuda()).cpu()
        for d,zz in zip(ds,z):
            ix=s.select(zz,d,threshold)
            gain=float(d['raw'][ix])*1e-4/256 if ix is not None else 0.
            allow=d['mask'].bool().clone();allow[:,1]=False
            oracle=max(0.,float(d['raw'][allow].max()))*1e-4/256 if allow.any() else 0.
            rows.append({'path':d['path'],'group':list(group(d)),'active':ix is not None,'gain':gain,'oracle_gain':oracle})
    gs=defaultdict(list)
    for r in rows:gs[tuple(r['group'])].append(r)
    active=[rs for rs in gs.values() if any(r['active'] for r in rs)]
    # Conservative video event: any materially harmful accepted action in that video.
    failures=sum(any(r['active'] and r['gain'] < -1e-8 for r in rs) for rs in active)
    n=len(active);upper=float(beta.ppf(.95,failures+1,n-failures)) if n>failures else 1.
    means=np.array([np.mean([r['gain'] for r in rs]) for rs in gs.values()])
    rng=np.random.default_rng(20261013)
    ci=np.quantile(means[rng.integers(0,len(means),(5000,len(means)))].mean(1),[.025,.975]).tolist()
    return {'clips':len(rows),'videos':len(gs),'active_videos':n,'harmful_videos':failures,
      'video_risk_upper95':upper,'video_mean_gain':float(means.mean()),'gain_bootstrap_ci95':ci,
      'clip_mean_gain':float(np.mean([r['gain'] for r in rows])), 'coverage':float(np.mean([r['active'] for r in rows])),
      'oracle_clip_gain':float(np.mean([r['oracle_gain'] for r in rows])),'rows':rows}

def main():
    OUT.mkdir(exist_ok=False);(OUT/'labels').mkdir();torch.set_num_threads(4);tr.prev.prior.seed_all(20261013)
    start=time.time();paths=json.loads((p.run.fit.SOURCE/'split_paths.json').read_text())['paths']
    groups={k:{tuple(tr.prev.group(x)) for x in v} for k,v in paths.items()}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    vg=sorted(groups['val']);random.Random(20261013).shuffle(vg)
    nt=max(1,len(vg)//5);nr=(len(vg)-nt)//2
    partitions={'tune':set(vg[:nt]),'risk':set(vg[nt:nt+nr]),'held':set(vg[nt+nr:])}
    protocol={'counts':{k:len(v) for k,v in paths.items()},'videos':{k:len(v) for k,v in groups.items()},
     'validation_groups':{k:[list(x) for x in sorted(v)] for k,v in partitions.items()},
     'train':'all train clips, inverse-original-video-frequency sampling','regions_per_clip':16,'epochs':60,'batch_size':8,
     'risk':'fixed selected checkpoint/threshold; one-sided exact binomial 95% upper bound on any-harm video probability <=20%; at least14 active videos; video bootstrap gain lower95>0',
     'harm_epsilon':1e-8,'risk_limit':.2,'risk_caveat':'exploratory IID-video assumption; video pool used in prior experiments; not a safety guarantee',
     'threshold_selection':'only tuning videos, mean video gain maximized with empirical any-harm risk<=10% and >=5 active videos; otherwise abstain',
     'risk_failure':'abstain everywhere, never retune on risk or heldout','test_used':False,
     'reference':0,'units':'local-equivalent utility = full composite marginal gain *256/1e-4; unclipped evaluation, clipped+-10 training',
     'architecture':'one two-frame hard replacement among 16 observed-support sampled locations and 7 alternatives; or abstain',
     'init':'from scratch','lr':.001}
    tr.prev.dump(OUT/'protocol.json',protocol);print(json.dumps({'stage':'start',**protocol}),flush=True)
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def cost(pred,target):return (pred-target).square().mean()+.03*metric(pred*2-1,target*2-1).mean()+.1*((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    datasets={'train':[],'tune':[],'risk':[],'held':[]};todo=paths['train']+paths['val'];label_start=time.time()
    with torch.no_grad():
        for i,path in enumerate(todo):
            x=p.run.load(path);b=tr.batchify([x]);base=b['bank'][0,1];target=b['target'][0]
            active=tr.pool((base-b['bank'][0,0]).abs().mean(1,keepdim=True))[:,0].reshape(4,2,8,8).mean(1)>1e-6
            loc=torch.nonzero(active).tolist();key=hashlib.sha256(path.encode()).hexdigest()
            random.Random(int(key[:16],16)).shuffle(loc)
            raw=torch.zeros(4,8,8,8);mask=torch.zeros_like(raw);basecost=cost(base,target)
            ref=torch.zeros_like(b['teacher_scores']);ref[:,:,1]=1
            for t,y,z in loc[:16]:
                mask[t,:,y,z]=1
                for k in range(8):
                    if k==1:continue
                    w=ref.clone();w[:,2*t:2*t+2,:,y,z]=0;w[:,2*t:2*t+2,k,y,z]=1
                    raw[t,k,y,z]=(basecost-cost(tr.synthesize(b['bank'],w,b['budget'])[0],target)).cpu()*256/1e-4
            d={'path':path,'features':x['features'],'mask':mask,'raw':raw}
            torch.save(d,OUT/'labels'/(key[:24]+'.pt'))
            split='train' if i<len(paths['train']) else next(k for k,v in partitions.items() if group(d) in v)
            datasets[split].append(d)
            if (i+1)%10==0 or i==0:
                elapsed=time.time()-label_start
                progress={'stage':'labels','clips':i+1,'total':len(todo),'seconds':elapsed,'remaining_label_seconds':elapsed/(i+1)*(len(todo)-i-1)}
                tr.prev.dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    del metric;torch.cuda.empty_cache()
    m=tr.TemporalSelector().cuda()
    with torch.no_grad():m.net[-1].bias.zero_()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0);train=datasets['train']
    counts=defaultdict(int)
    for d in train:counts[group(d)]+=1
    weights=[1/counts[group(d)] for d in train]
    best=0.;selection={'epoch':0,'threshold':float('inf')};history=[];training_start=time.time()
    torch.save({'model':m.state_dict(),'epoch':0,'threshold':float('inf')},OUT/'best.pt')
    for epoch in range(1,61):
        m.train();order=random.Random(20261013+epoch).choices(range(len(train)),weights=weights,k=len(train));total=0.;steps=0
        for off in range(0,len(order),8):
            ds=[train[i] for i in order[off:off+8]];f=torch.stack([d['features'] for d in ds]).cuda()
            lab=torch.stack([d['raw'].clamp(-10,10) for d in ds]).cuda();mask=torch.stack([d['mask'] for d in ds]).cuda();mask[:,:,1]=0
            z=s.scores(m,f);regular=F.smooth_l1_loss(z,lab,reduction='none');interval=F.smooth_l1_loss((z.abs()-.1).clamp_min(0),torch.zeros_like(z),reduction='none')
            loss=(torch.where(lab.abs()<=.1,interval,regular)*mask).sum()/mask.sum().clamp_min(1)
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=loss.item();steps+=1
        row={'stage':'training','epoch':epoch,'loss':total/steps,'seconds':time.time()-training_start}
        if epoch%10==0:
            m.eval();evaluations={str(t):results(m,datasets['tune'],t) for t in THRESH}
            row['tuning']={k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in evaluations.items()}
            for threshold in THRESH:
                r=evaluations[str(threshold)]
                if r['active_videos']>=5 and r['harmful_videos']/r['active_videos']<=.1 and r['video_mean_gain']>best:
                    best=r['video_mean_gain'];selection={'epoch':epoch,'threshold':threshold};torch.save({'model':m.state_dict(),**selection},OUT/'best.pt')
        torch.save({'model':m.state_dict(),'epoch':epoch,'optimizer':opt.state_dict()},OUT/'last.pt')
        history.append(row);tr.prev.dump(OUT/'history.json',history);tr.prev.dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    ck=torch.load(OUT/'best.pt',weights_only=False);m.load_state_dict(ck['model']);m.eval()
    risk=results(m,datasets['risk'],ck['threshold'])
    passed=risk['active_videos']>=14 and risk['video_risk_upper95']<=.2 and risk['gain_bootstrap_ci95'][0]>0
    deployed=ck['threshold'] if passed else float('inf')
    held=results(m,datasets['held'],deployed)
    tr.prev.dump(OUT/'summary.json',{'selected':selection,'risk':risk,'risk_passed':passed,'deployed_threshold':str(deployed),'heldout':held,'seconds':time.time()-start,'test_used':False})
    tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start,'risk_passed':passed});print(json.dumps({'stage':'complete','risk_passed':passed}),flush=True)

if __name__=='__main__':main()
