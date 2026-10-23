"""Full train utility regression; validation-only checkpoint/decoder selection."""
import hashlib,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
import diagnose_selector_fit32 as fit
tr=fit.tr
OUT=fit.ROOT/'selector_utility_full_20261012'

def load(p):
    key=hashlib.sha256(p.encode()).hexdigest()[:24]
    x=torch.load(fit.SOURCE/'cache'/(key+'.pt'),map_location='cpu',weights_only=False)
    assert x['path']==p
    return x

def predict(m,b,kind):
    scores=m.net(b['features'].permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
    w=scores.softmax(2) if kind=='soft' else F.one_hot(scores.argmax(2),8).permute(0,1,4,2,3).float()
    ref=torch.zeros_like(w);ref[:,:,1]=1
    return torch.where(((scores.amax(2)-scores[:,:,1])>.1)[:,:,None],w,ref)

def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4);tr.prev.prior.seed_all(20261012)
    paths=json.loads((fit.SOURCE/'split_paths.json').read_text())['paths']
    groups={s:{tr.prev.group(p) for p in ps} for s,ps in paths.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(paths[a])&set(paths[b]) and not groups[a]&groups[b]
    protocol={'counts':{s:len(p) for s,p in paths.items()},'epochs':80,'batch_size':2,'lr':.001,'weight_decay':0,
       'seed':20261012,'from_scratch':True,'loss':'active-region Huber relative composite utility, units 1e-4, clip +-10',
       'selection':'minimum full validation MSE+.03 LPIPS+.1 temporal over epochs and soft/hard; baseline fallback',
       'soft_temperature':1,'margin':.1,'test_used':False,'cache_source':str(fit.SOURCE),'frozen_candidates':True}
    tr.prev.dump(OUT/'protocol.json',protocol);tr.prev.dump(OUT/'split_paths.json',{'paths':paths})
    print(json.dumps({'stage':'loading',**protocol}),flush=True)
    train=[]
    for i,p in enumerate(paths['train']):
        x=load(p)
        a=(tr.pool((x['bank'][1]-x['bank'][0]).abs().mean(1,keepdim=True))[:,0]>1e-6).float()
        train.append({'features':x['features'],'labels':((x['teacher_scores'][:,1:2]-x['teacher_scores'])/1e-4).clamp(-10,10),'active':a})
        if i%100==0:print(json.dumps({'stage':'load_train','clips':i+1}),flush=True)
    val=[load(p) for p in paths['val']]
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    m=tr.TemporalSelector().cuda()
    with torch.no_grad():m.net[-1].bias.zero_()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    fit.predict=predict
    best=float('inf');history=[];start=time.time()
    for epoch in range(1,81):
        tick=time.time();order=list(range(len(train)));random.Random(epoch).shuffle(order);total=0;steps=0
        m.train()
        for off in range(0,len(order),2):
            items=[train[i] for i in order[off:off+2]]
            b={k:torch.stack([x[k] for x in items]).cuda() for k in items[0]}
            est=m.net(b['features'].permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
            a=b['active'][:,:,None]
            loss=(F.smooth_l1_loss(est,b['labels'],reduction='none')*a).sum()/(a.sum()*8).clamp_min(1)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
            total+=loss.item();steps+=1
        means,routing,rows=fit.evaluate({'soft':m,'hard':m},val,metric)
        mode=min(['soft','hard'],key=lambda k:tr.composite(means[k]))
        score=tr.composite(means[mode])
        state={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'mode':mode,'protocol':protocol}
        torch.save(state,OUT/'last.pt')
        if score<best:
            best=score;torch.save(state,OUT/'best.pt')
            selected=mode if score<tr.composite(means['baseline']) else 'baseline'
            tr.prev.dump(OUT/'validation_selection.json',{'epoch':epoch,'mode':mode,'selected_with_fallback':selected,'metrics':means,'score':score})
            (OUT/'validation_per_clip.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        row={'epoch':epoch,'loss':total/steps,'metrics':means,'routing':routing,'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-start}
        history.append(row);tr.prev.dump(OUT/'history.json',history)
        tr.prev.dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start,'test_evaluated':False})

if __name__=='__main__':main()
