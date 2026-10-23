"""Single two-frame region hard replacement with abstention; diagnostic only."""
import json,random,time
import numpy as np
import torch
import torch.nn.functional as F
import train_selector_rendered_pilot as p
tr=p.tr
OUT=p.run.fit.ROOT/'selector_single_region_20261013'
THRESHOLDS=[0,.1,.5,1,2,float('inf')]

def scores(m,f):
    z=p.tube_scores(m,f)
    ref=torch.zeros_like(z);ref[:,:,1]=1
    return z*(1-ref) # reference action exactly zero

def select(z,d,threshold):
    allow=d['mask'].bool().clone();allow[:,1]=False
    if not allow.any():return None
    zz=z.masked_fill(~allow,-float('inf'));idx=int(zz.flatten().argmax())
    if float(zz.flatten()[idx])<=threshold:return None
    return tuple(int(v) for v in np.unravel_index(idx,zz.shape))

@torch.no_grad()
def report(model,data,threshold,history=False):
    gains=[];fired=[];oracle=[];preds=[];targets=[]
    for d in data:
        z=scores(model,d['features'][None].cuda())[0].cpu()
        if history:
            z=d['features'][:,34:42].reshape(4,2,8,8,8).mean(1)
        ix=select(z,d,threshold)
        gains.append(float(d['raw'][ix])/256*1e-4 if ix is not None else 0.)
        fired.append(ix is not None)
        allowed=d['mask'].bool().clone();allowed[:,1]=False
        oracle.append(max(0.,float(d['raw'][allowed].max()))/256*1e-4 if allowed.any() else 0.)
        preds.extend(z[allowed].tolist());targets.extend(d['raw'][allowed].tolist())
    a=np.array(gains);fire=np.array(fired)
    return {'clips':len(data),'mean_gain':float(a.mean()),'activation_rate':float(fire.mean()),
      'harm_when_active':float((a[fire]<-1e-8).mean()) if fire.any() else None,
      'beneficial_when_active':float((a[fire]>1e-8).mean()) if fire.any() else None,
      'oracle_mean_gain':float(np.mean(oracle)), 'prediction_mae':float(np.mean(np.abs(np.array(preds)-targets))) if preds else None,
      'rows':[{'path':d['path'],'gain':g,'active':v,'oracle_gain':o} for d,g,v,o in zip(data,gains,fired,oracle)]}

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);tr.prev.prior.seed_all(20261013);start=time.time()
    proto=json.loads((p.OUT/'protocol.json').read_text())
    train=torch.load(p.OUT/'training_labels.pt',map_location='cpu',weights_only=False)
    for d,path in zip(train,proto['train']):d['path']=path;d['raw']=d['rendered'] # restore unclipped below
    pairs=json.loads((p.OUT/'label_pairs.json').read_text());ptr=0
    # Prior pairs stored in shuffled location order; regenerate exact sampling order.
    rng=random.Random(15)
    for d,path in zip(train,proto['train']):
        x=p.run.load(path);base=x['bank'][1];tau=x['bank'][0]
        active=tr.pool((base-tau).abs().mean(1,keepdim=True))[:,0].reshape(4,2,8,8).mean(1)>1e-6
        loc=torch.nonzero(active).tolist();rng.shuffle(loc);d['raw']=torch.zeros_like(d['rendered'])
        for t,y,xcoord in loc[:4]:
            for k in range(8):
                if k==1:continue
                d['raw'][t,k,y,xcoord]=pairs[ptr][1];ptr+=1
        assert torch.allclose(d['raw'].clamp(-10,10),d['rendered'])
    assert ptr==len(pairs)
    import lpips
    lp=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    def cost(pred,target):
        return (pred-target).square().mean()+.03*lp(pred*2-1,target*2-1).mean()+.1*((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    val=[];rng=random.Random(16)
    with torch.no_grad():
        for path in proto['val']:
            x=p.run.load(path);b=tr.batchify([x]);base=b['bank'][0,1];target=b['target'][0]
            active=tr.pool((base-b['bank'][0,0]).abs().mean(1,keepdim=True))[:,0].reshape(4,2,8,8).mean(1)>1e-6
            loc=torch.nonzero(active).tolist();rng.shuffle(loc)
            raw=torch.zeros(4,8,8,8);mask=torch.zeros_like(raw);reference=cost(base,target)
            ref=torch.zeros_like(b['teacher_scores']);ref[:,:,1]=1
            for t,y,z in loc[:4]:
                mask[t,:,y,z]=1
                for k in range(8):
                    if k==1:continue
                    w=ref.clone();w[:,2*t:2*t+2,:,y,z]=0;w[:,2*t:2*t+2,k,y,z]=1
                    raw[t,k,y,z]=(reference-cost(tr.synthesize(b['bank'],w,b['budget'])[0],target)).cpu()*256/1e-4
            val.append({'path':path,'features':x['features'],'mask':mask,'raw':raw})
    groups=sorted({tr.prev.group(d['path']) for d in val});random.Random(17).shuffle(groups)
    calgroups=set(groups[::2]);cal=[d for d in val if tr.prev.group(d['path']) in calgroups];held=[d for d in val if tr.prev.group(d['path']) not in calgroups]
    protocol={'train':len(train),'calibration':len(cal),'heldout_validation':len(held),'test_used':False,
     'actions':'4 observed-support randomly sampled two-frame cells per clip, 7 alternatives each; at most one hard replacement or abstain',
     'reference_score':0,'epochs':60,'thresholds':[str(v) for v in THRESHOLDS],
     'heldout_caveat':'subset of previously used validation pool, not a fresh final test set; disjoint original videos from calibration',
     'selection':'minimum calibration actual loss (maximum mean marginal gain); ties favor abstention; checkpoint every10 epochs',
     'cal_paths':[d['path'] for d in cal],'held_paths':[d['path'] for d in held]}
    tr.prev.dump(OUT/'protocol.json',protocol);torch.save({'train':train,'cal':cal,'held':held},OUT/'diagnostic_data.pt')
    model=tr.TemporalSelector().cuda()
    with torch.no_grad():model.net[-1].bias.zero_()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0);best=-float('inf');hist=[]
    for epoch in range(1,61):
        order=list(range(len(train)));random.Random(epoch).shuffle(order);total=0
        for off in range(0,len(train),2):
            ds=[train[i] for i in order[off:off+2]];f=torch.stack([d['features'] for d in ds]).cuda()
            label=torch.stack([d['rendered'] for d in ds]).cuda();mask=torch.stack([d['mask'] for d in ds]).cuda();mask[:,:,1]=0
            z=scores(model,f);regular=F.smooth_l1_loss(z,label,reduction='none');interval=F.smooth_l1_loss((z.abs()-.1).clamp_min(0),torch.zeros_like(z),reduction='none')
            loss=(torch.where(label.abs()<=.1,interval,regular)*mask).sum()/mask.sum().clamp_min(1)
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=loss.item()
        if epoch%10==0:
            results={str(t):report(model,cal,t) for t in THRESHOLDS}
            threshold=max(reversed(THRESHOLDS),key=lambda t:results[str(t)]['mean_gain']);gain=results[str(threshold)]['mean_gain']
            if gain>best:
                best=gain;torch.save({'model':model.state_dict(),'epoch':epoch,'threshold':threshold},OUT/'best.pt')
            hist.append({'epoch':epoch,'loss':total/32,'calibration':results,'elapsed':time.time()-start});tr.prev.dump(OUT/'history.json',hist)
            print(json.dumps({'epoch':epoch,'threshold':str(threshold),'cal_gain':gain,'elapsed':time.time()-start}),flush=True)
    torch.save({'model':model.state_dict(),'epoch':60},OUT/'last.pt')
    last_train=report(model,train,0)
    ck=torch.load(OUT/'best.pt',weights_only=False);model.load_state_dict(ck['model'])
    hr={str(t):report(model,cal,t,True) for t in THRESHOLDS};ht=max(reversed(THRESHOLDS),key=lambda t:hr[str(t)]['mean_gain'])
    summary={'selected_epoch':ck['epoch'],'threshold':str(ck['threshold']),'train_last_no_rejection':last_train,
      'calibration':report(model,cal,ck['threshold']),'heldout':report(model,held,ck['threshold']),
      'history_rule_threshold':str(ht),'history_rule_heldout':report(model,held,ht,True),
      'seconds':time.time()-start,'protocol':protocol}
    tr.prev.dump(OUT/'summary.json',summary);tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start})
    print(json.dumps({k:v for k,v in summary.items() if k in ['selected_epoch','threshold','seconds']}),flush=True)

if __name__=='__main__':main()
