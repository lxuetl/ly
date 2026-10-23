"""Merge unfiltered history repair and SDXL candidates; no heldout-label filtering."""
import json,time,os
from pathlib import Path
import numpy as np
import torch
import replay_preference_pilot as r
SRC=r.OUT
HIST=SRC.parent/'history_texture_candidates_20261015'
OUT=SRC.parent/'mixed_preference_20261015'
FIXED='history_align_texture_correct_a0.25'
def dump(n,d):
 p=OUT/n;p.with_suffix('.tmp').write_text(json.dumps(d,indent=2,allow_nan=False));os.replace(p.with_suffix('.tmp'),p)
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);start=time.time()
 dump('protocol.json',{'candidates':'TAU + all8 SDXL edits + all12 correct-flow history candidates; no GT-positive filtering; stress controls excluded from fit','primary':'same monotonic seven-feature pairwise NNLS ranker recalibrated on merged replay pool','split':'leave-original-video-out; fit and scale on other6 videos replay only','fixed_control':FIXED,'secondary':'old SDXL-only-calibrated ranker applied unchanged to merged pool','threshold':'relative-to-TAU score>0, same as prior; never tune on heldout','label':'LPIPS benefit supervision; safety assessed separately as LPIPS gain>0.0001,MSE/temporal<=1.01TAU','caveat':'same7 validation videos used in earlier development; exploratory, not fresh test evidence','family_identity_input':False})
 selection=json.loads((SRC/'selection.json').read_text());dump('selection.json',selection)
 for d in selection['selected']:
  j=d['index']
  for phase in ['replay','heldout']:
   for suffix in ['input','label']:
    dst=OUT/f'{phase}_{j}_{suffix}.pt'
    if not dst.exists():dst.symlink_to(SRC/dst.name)
   p=torch.load(SRC/f'{phase}_{j}_candidates.pt',weights_only=False)
   h=torch.load(HIST/f'{phase}_{j}.pt',weights_only=False)
   assert torch.equal(p['tau'],h['tau'])
   p.update({'history_'+n:v for n,v in h.items() if '_correct_' in n})
   assert len(p)==21
   torch.save(p,OUT/f'{phase}_{j}_candidates.pt')
 r.OUT=OUT
 if not (OUT/'features_no_labels.json').exists():r.features()
 r.evaluate()
 rows=json.loads((OUT/'labeled_records.json').read_text());decisions=json.loads((OUT/'heldout_decisions.json').read_text())
 old=json.loads((SRC/'fold_parameters.json').read_text());report={'videos':7,'comparisons':{},'within_history_ranking':{},'seconds':time.time()-start}
 tests=[x for x in rows if x['phase']=='heldout'];records=[]
 for test in tests:
  natural=[c for c in test['candidates'] if not c['name'].startswith('stress')];by={c['name']:c for c in natural};base=by['tau'];j=test['video']
  pars=next(p for p in old if p['video']==j and p['mode']=='combined');w=np.array(pars['weights']);scale=np.array(pars['scale'])
  oldscores={n:float((((np.array(c['x'])-np.array(base['x']))*np.array([1,-1,-1,-1,-1,-1,-1]))/scale)@w) for n,c in by.items()}
  pick=max(oldscores,key=oldscores.get);pick=pick if oldscores[pick]>1e-8 else 'tau'
  extra=[{'mode':'old_frozen_combined','selected':pick,'scores':oldscores},{'mode':'fixed_history','selected':FIXED,'scores':None},{'mode':'tau','selected':'tau','scores':None}]
  for a in [d for d in decisions if d['video']==j]+extra:
   c=by[a['selected']];scores=a['scores'];safe=lambda x: bool(x['safe_repair'])
   rec={'video':j,'group':test['group'],'mode':a['mode'],'selected':a['selected'],'safe':safe(c),'harmful':c['lpips']>base['lpips']+.0001 or c['mse']>base['mse']*1.01 or c['temporal']>base['temporal']*1.01,'metrics':{k:c[k] for k in ['mse','lpips','temporal']},'delta_tau':{k:c[k]-base[k] for k in ['mse','lpips','temporal']},'delta_fixed':{k:c[k]-by[FIXED][k] for k in ['mse','lpips','temporal']}}
   if scores is not None:
    pos=[x for x in natural if safe(x)];neg=[x for x in natural if x['name']!='tau' and not safe(x)]
    sdxl=[x for x in natural if x['name']!='tau' and not x['name'].startswith('history_')]
    rec.update(positive_total=len(pos),positive_accepted=sum(scores[x['name']]>1e-8 for x in pos),negative_total=len(neg),negative_accepted=sum(scores[x['name']]>1e-8 for x in neg),sdxl_total=len(sdxl),sdxl_accepted=sum(scores[x['name']]>1e-8 for x in sdxl),sdxl_selected=a['selected'] in [x['name'] for x in sdxl])
    hs=[x for x in natural if x['name'].startswith('history_')];pairs=[(scores[x['name']]-scores[y['name']])*(x['benefit']-y['benefit']) for i,x in enumerate(hs) for y in hs[:i] if abs(x['benefit']-y['benefit'])>1e-5]
    rec['history_pair_accuracy']=float(np.mean([.5 if v==0 else float(v>0) for v in pairs]))
   records.append(rec)
 for mode in sorted({x['mode'] for x in records}):
  rs=[x for x in records if x['mode']==mode];s={'choices':[x['selected'] for x in rs],'safe_selected':sum(x['safe'] for x in rs),'harmful_selected':sum(x['harmful'] for x in rs),'mean_metrics':{k:float(np.mean([x['metrics'][k] for x in rs])) for k in ['mse','lpips','temporal']},'delta_tau':{k:r.bootstrap([x['delta_tau'][k] for x in rs]) for k in ['mse','lpips','temporal']},'delta_fixed':{k:r.bootstrap([x['delta_fixed'][k] for x in rs]) for k in ['mse','lpips','temporal']}}
  if 'positive_total' in rs[0]:
   for k in ['positive_total','positive_accepted','negative_total','negative_accepted','sdxl_total','sdxl_accepted','sdxl_selected']:s[k]=sum(x[k] for x in rs)
   s['positive_recall']=s['positive_accepted']/max(s['positive_total'],1);s['sdxl_rejection']=1-s['sdxl_accepted']/max(s['sdxl_total'],1)
   s['history_pair_accuracy']=float(np.mean([x['history_pair_accuracy'] for x in rs]))
  report['comparisons'][mode]=s
 dump('comparison_per_video.json',records);dump('comparison_summary.json',report);dump('COMPLETE.json',{'seconds':time.time()-start});print(json.dumps(report),flush=True)
if __name__=='__main__':main()
