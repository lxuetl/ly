"""Validation-only marginal interventions; no training or test access."""
import json,random,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
import train_selector_utility_full as run
tr=run.tr
OUT=run.OUT/'label_renderer_audit'

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    assert not (OUT/'COMPLETE.json').exists()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths=json.loads((run.OUT/'split_paths.json').read_text())['paths']['val']
    buckets=defaultdict(list)
    for p in paths:buckets[tr.prev.group(p)[0]].append(p)
    rng=random.Random(20261013)
    for ps in buckets.values():rng.shuffle(ps)
    chosen=[]
    while len(chosen)<32:
        for d in sorted(buckets):
            if buckets[d] and len(chosen)<32:chosen.append(buckets[d].pop())
    import lpips
    spatial=lpips.LPIPS(net='alex',spatial=True,verbose=False).to(device).eval().requires_grad_(False)
    scalar=lpips.LPIPS(net='alex',spatial=False,verbose=False).to(device).eval().requires_grad_(False)
    model=tr.TemporalSelector().to(device).eval()
    model.load_state_dict(torch.load(run.OUT/'best.pt',map_location='cpu',weights_only=False)['model'])
    rows=[];maps=[];start=time.time()
    for ci,p in enumerate(chosen):
        item=run.load(p)
        b={k:item[k][None].to(device) for k in ['bank','features','target','teacher_scores','mask']}
        b['budget']=torch.tensor([item['budget']],device=device)
        bank=b['bank'][0];target=b['target'][0]
        scores=b['teacher_scores'][0];base=bank[1];tau=bank[0]
        base_sp=tr.teacher_scores(base[None],target,spatial)[:,0].mean().item()
        base_sc=tr.composite(tr.measure(base,target,tau,b['mask'][0],scalar))
        def assess(pred):
            sp=tr.teacher_scores(pred[None],target,spatial)[:,0].mean().item()
            sc=tr.composite(tr.measure(pred,target,tau,b['mask'][0],scalar))
            return {'spatial_gain':base_sp-sp,'evaluation_gain':base_sc-sc}
        def render(w,mode):
            dense=F.interpolate(w,(128,128),mode=mode,**({'align_corners':False} if mode=='bilinear' else {}))
            return (bank.permute(1,0,2,3,4)*dense[:,:,None]).sum(1)
        ref=torch.zeros_like(scores);ref[:,1]=1
        gain=scores[:,1]-scores.amin(1);best=scores.argmin(1)
        active=(tr.pool((base-tau).abs().mean(1,keepdim=True))[:,0]>1e-6)
        locations=torch.nonzero((gain>1e-5)&active).cpu().tolist()
        rng.shuffle(locations)
        for t,y,x in locations[:8]:
            k=int(best[t,y,x]);w=ref.clone();w[t,:,y,x]=0;w[t,k,y,x]=1
            near=render(w,'nearest');bil=render(w,'bilinear')
            final=tr.synthesize(b['bank'],w[None],b['budget'])[0]
            norm=(bil-tau).abs().mean().item();factor=min(1.,float(b['budget'][0])/max(norm,1e-9))
            rows.append({'path':p,'source':tr.prev.group(p)[0],'cell':[t,y,x],'candidate':k,
                'label_local_gain':float(gain[t,y,x]),'label_global_gain':float(gain[t,y,x])/512,
                'nearest':assess(near),'bilinear':assess(bil),'budgeted':assess(final),'budget_factor':factor})
        oracle=F.one_hot(best,8).permute(0,3,1,2).float()
        for name,w in [('oracle',oracle),('learned_soft',run.predict(model,b,'soft')[0]),('learned_hard',run.predict(model,b,'hard')[0])]:
            implied=(scores[:,1]-(scores*w).sum(1)).mean().item()
            final=tr.synthesize(b['bank'],w[None],b['budget'])[0]
            maps.append({'path':p,'name':name,'label_implied_gain':implied,'actual':assess(final)})
        print(json.dumps({'clips':ci+1,'interventions':len(rows),'seconds':time.time()-start}),flush=True)
    result={'clips':len(chosen),'interventions':len(rows),'validation_only':True,'sample':'source round-robin; up to 8 randomly sampled beneficial active cells per clip; one cell/time changed at a time',
      'threshold':1e-5,'stages':{},'full_maps':{},'budget_active_fraction':float(np.mean([r['budget_factor']<.999999 for r in rows])),
      'note':'Spatial teacher temporal term averages T including initial zero; evaluation temporal averages T-1. Global intervention effects include neighboring space/time and LPIPS receptive fields.'}
    for stage in ['nearest','bilinear','budgeted']:
        result['stages'][stage]={key:{'mean_gain':float(np.mean([r[stage][key] for r in rows])),
           'harm_fraction':float(np.mean([r[stage][key]<-1e-10 for r in rows])),
           'label_correlation':float(np.corrcoef([r['label_global_gain'] for r in rows],[r[stage][key] for r in rows])[0,1])} for key in ['spatial_gain','evaluation_gain']}
    for name in ['oracle','learned_soft','learned_hard']:
        rs=[r for r in maps if r['name']==name]
        result['full_maps'][name]={'label_gain':float(np.mean([r['label_implied_gain'] for r in rs])),
          'spatial_gain':float(np.mean([r['actual']['spatial_gain'] for r in rs])),
          'evaluation_gain':float(np.mean([r['actual']['evaluation_gain'] for r in rs])),
          'harm_fraction':float(np.mean([r['actual']['evaluation_gain']<0 for r in rs]))}
    tr.prev.dump(OUT/'summary.json',result);tr.prev.dump(OUT/'interventions.json',rows);tr.prev.dump(OUT/'full_maps.json',maps)
    tr.prev.dump(OUT/'COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(result),flush=True)

if __name__=='__main__':main()
