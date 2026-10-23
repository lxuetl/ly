"""Frozen local spatial descriptors vs original features; matched regression loss."""
import json,random,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import compare_shared_ranker as c
e=c.e;tr=e.tr
OUT=e.p.run.fit.ROOT/'local_feature_comparison_20261013'

class Regressor(nn.Module):
    def __init__(self,local):
        super().__init__();self.local=local
        if local:
            self.layers=nn.Sequential(nn.Conv3d(132,19,3,padding=1),nn.GELU(),nn.Conv3d(19,19,3,padding=1),nn.GELU(),nn.Conv3d(19,9,1))
            nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
        else:
            self.layers=tr.TemporalSelector().net
            with torch.no_grad():self.layers[-1].bias.zero_()
    def net(self,x):return self.layers(x if self.local else x[:,:57])

@torch.no_grad()
def descriptors(bank):
    # Only frozen candidate outputs; never labels, masks derived from GT, or target RGB.
    n,t,ch,h,w=bank.shape;delta=(bank-bank[1:2]).reshape(n*t,ch,h,w)
    gray=delta.mean(1,keepdim=True)
    quadrants=F.pixel_unshuffle(F.avg_pool2d(gray,8),2)
    meanabs=F.avg_pool2d(delta.abs().mean(1,keepdim=True),16)
    rms=(F.avg_pool2d(delta.square().mean(1,keepdim=True),16)+1e-12).sqrt()
    dx=F.pad((delta[:,:,:,1:]-delta[:,:,:,:-1]).abs().mean(1,keepdim=True),(0,1,0,0))
    dy=F.pad((delta[:,:,1:,:]-delta[:,:,:-1,:]).abs().mean(1,keepdim=True),(0,0,0,1))
    neighbor=F.avg_pool2d(meanabs,3,1,1,count_include_pad=False)
    candidate=torch.cat([quadrants,meanabs,rms,F.avg_pool2d(dx,16),F.avg_pool2d(dy,16),neighbor],1)
    candidate=candidate.reshape(n,t,9,8,8).permute(1,0,2,3,4).reshape(t,72,8,8)*100
    base=bank[1].mean(1,keepdim=True)
    variance=(F.avg_pool2d(base.square(),16)-F.avg_pool2d(base,16).square()).clamp_min(0).sqrt()
    dx=F.pad((base[:,:,:,1:]-base[:,:,:,:-1]).abs(),(0,1,0,0))
    dy=F.pad((base[:,:,1:,:]-base[:,:,:-1,:]).abs(),(0,0,0,1))
    context=torch.cat([variance,F.avg_pool2d(dx,16),F.avg_pool2d(dy,16)],1)
    out=torch.cat([candidate,context],1);assert out.shape==(8,75,8,8) and torch.isfinite(out).all()
    return out.cpu()

def main():
    OUT.mkdir(exist_ok=False);(OUT/'features').mkdir();torch.set_num_threads(4);tr.prev.prior.seed_all(20261013);start=time.time()
    proto=json.loads((e.OUT/'protocol.json').read_text());parts={k:{tuple(g) for g in v} for k,v in proto['validation_groups'].items()}
    paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths'];data={k:[] for k in ['train','tune','risk','held']};total=len(paths['train'])+len(paths['val']);i=0
    for split in ['train','val']:
        for path in paths[split]:
            key=hashlib.sha256(path.encode()).hexdigest()[:24];d=torch.load(e.OUT/'labels'/(key+'.pt'),map_location='cpu',weights_only=False)
            cached=e.p.run.load(path);local=descriptors(cached['bank'].cuda());torch.save(local,OUT/'features'/(key+'.pt'))
            d['features']=torch.cat([d['features'],local],1)
            part='train' if split=='train' else next(k for k,v in parts.items() if e.group(d) in v);data[part].append(d);i+=1
            if i%20==0 or i==1:
                elapsed=time.time()-start;row={'stage':'features','clips':i,'total':total,'seconds':elapsed,'eta_features_seconds':elapsed/i*(total-i)}
                tr.prev.dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    del cached;torch.cuda.empty_cache()
    models={'baseline':Regressor(False).cuda(),'local':Regressor(True).cuda()};opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for k,m in models.items()}
    protocol={'counts':{k:len(v) for k,v in data.items()},'params':{k:sum(x.numel() for x in m.parameters()) for k,m in models.items()},
      'epochs':60,'batch_size':8,'seed':20261013,'lr':.001,'labels_source':str(e.OUT/'labels'),'test_used':False,
      'features':'original57 + per-candidate9: signed2x2 quadrants, mean absolute, RMS, x/y gradient magnitudes, 3x3 neighborhood absolute correction; base texture std and x/y edges3',
      'safety':'derived only from frozen candidates; no future targets passed into feature extraction',
      'control':'same Huber+uncertainty interval loss, same sampling and rendered labels, fixed reference0, same16-region action bank',
      'capacity':'baseline width32 vs local width19 to match total parameter count; this trades hidden width for extra inputs',
      'selection':'mean within-clip Spearman on tune videos every10 epochs; risk/held not used for choice',
      'inference':'same single-action selector; positive threshold and tune-quantile thresholds; exploratory not risk-certified'}
    tr.prev.dump(OUT/'protocol.json',protocol);print(json.dumps({'stage':'train_start',**protocol}),flush=True)
    counts=defaultdict(int)
    for d in data['train']:counts[e.group(d)]+=1
    weights=[1/counts[e.group(d)] for d in data['train']];best={k:-float('inf') for k in models};history=[];trainstart=time.time()
    for epoch in range(1,61):
        tick=time.time();order=random.Random(20261013+epoch).choices(range(len(data['train'])),weights=weights,k=len(data['train']));losses=defaultdict(float);steps=0
        for off in range(0,len(order),8):
            ds=[data['train'][i] for i in order[off:off+8]];f=torch.stack([d['features'] for d in ds]).cuda();lab=torch.stack([d['raw'].clamp(-10,10) for d in ds]).cuda();mask=torch.stack([d['mask'] for d in ds]).cuda();mask[:,:,1]=0
            for kind,m in models.items():
                m.train();z=e.s.scores(m,f);reg=F.smooth_l1_loss(z,lab,reduction='none');unc=F.smooth_l1_loss((z.abs()-.1).clamp_min(0),torch.zeros_like(z),reduction='none')
                loss=(torch.where(lab.abs()<=.1,unc,reg)*mask).sum()/mask.sum().clamp_min(1)
                assert torch.isfinite(loss);opts[kind].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opts[kind].step();losses[kind]+=loss.item()
            steps+=1
        row={'stage':'training','epoch':epoch,'seconds':time.time()-tick,'elapsed_training':time.time()-trainstart,'loss':{k:v/steps for k,v in losses.items()}}
        if epoch%10==0:
            row['tune']={}
            for kind,m in models.items():
                m.eval();r=c.evaluate(m,data['tune']);score=r['within_clip_spearman'];row['tune'][kind]={k:v for k,v in r.items() if k!='rows'}
                if score>best[kind]:best[kind]=score;torch.save({'model':m.state_dict(),'epoch':epoch},OUT/(kind+'_best.pt'))
        for kind,m in models.items():torch.save({'model':m.state_dict(),'epoch':epoch},OUT/(kind+'_last.pt'))
        history.append(row);tr.prev.dump(OUT/'history.json',history);tr.prev.dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
    summary={}
    for kind,m in models.items():
        ck=torch.load(OUT/(kind+'_best.pt'),weights_only=False);m.load_state_dict(ck['model']);m.eval();r={split:c.evaluate(m,ds) for split,ds in data.items()}
        thresholds={'positive':0.,**{str(q):float(np.quantile([a['score'] for a in r['tune']['rows']],1-q)) for q in [.25,.5,.75]}}
        summary[kind]={'epoch':ck['epoch'],'ranking':{sp:{k:v for k,v in res.items() if k!='rows'} for sp,res in r.items()},'decisions':{sp:{name:c.decisions(r[sp]['rows'],th) for name,th in thresholds.items()} for sp in ['tune','risk','held']}}
        tr.prev.dump(OUT/(kind+'_rows.json'),r)
    tr.prev.dump(OUT/'summary.json',summary);tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps({'stage':'complete','seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
