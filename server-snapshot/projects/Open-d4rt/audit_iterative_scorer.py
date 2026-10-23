"""Read-only scorer audit: no fitting, threshold selection or checkpoint changes."""
import json
from pathlib import Path
import numpy as np
import torch
import iterative_video_repair as ir
torch.set_num_threads(2)
root=Path('/root/autodl-fs/d4rt_training/iterative_preference_v1_20261017')
ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
m=ir.PreferenceEvaluator();m.load_state_dict(ck['model']);m.eval()
margin=np.array(json.loads((root/'calibration.json').read_text())['margins'])
rows=[]
with torch.no_grad():
 for file in sorted((root/'records').glob('val-*.json')):
  for r in json.loads(file.read_text()):
   names=list(r['features']);p=m(torch.tensor(list(r['features'].values()))).numpy();labs=r['labels']
   ids=[i for i,n in enumerate(names) if n!='keep'];top=max(ids,key=lambda i:p[i,0]);safe_top=max(ids,key=lambda i:p[i,5]);y=np.array([labs[n]['gain'] for n in names])
   pairs=[(p[i,0]-p[j,0])*(y[i]-y[j]) for i in ids for j in ids if i<j and abs(y[i]-y[j])>1e-5]
   tests=np.stack([p[:,0]>margin[0],p[:,1]+margin[1]<=0,p[:,2]+margin[2]<=0,p[:,3]+margin[3]<=0,p[:,4]+margin[4]<=0],1)[ids]
   rows.append({'step':r['step'],'top_safe':labs[names[top]]['safe_improvement'],'logit_top_safe':labs[names[safe_top]]['safe_improvement'],
                'pair_accuracy':np.mean([float(z>0) if z!=0 else .5 for z in pairs]) if pairs else None,
                'pass':tests.sum(0).tolist(),'n':len(ids),'gains':y[ids].tolist(),'pred':p[ids,0].tolist(),
                'raw_eligible':int(((p[ids,0]>0)&(p[ids,1]<=0)&(p[ids,2]<=0)&(p[ids,3]<=0)&(p[ids,4]<=0)).sum())})
out={'best_epoch':ck['epoch'],'target_scale':ck['target_scale'].tolist(),'steps':{}}
for s in range(3):
 rs=[r for r in rows if r['step']==s];acc=[r['pair_accuracy'] for r in rs if r['pair_accuracy'] is not None]
 out['steps'][s]={'states':len(rs),'gain_top1_safe':sum(r['top_safe'] for r in rs),'safe_logit_top1_safe':sum(r['logit_top_safe'] for r in rs),'pair_accuracy_state_mean':float(np.mean(acc)),
  'candidate_count':sum(r['n'] for r in rs),'pass_counts_gain_mse_time_base_mse_base_time':np.sum([r['pass'] for r in rs],0).tolist(),
  'raw_eligible_candidates':sum(r['raw_eligible'] for r in rs),
  'true_gain_quantiles':np.quantile([z for r in rs for z in r['gains']],[0,.5,.9,1]).tolist(),
  'pred_gain_quantiles':np.quantile([z for r in rs for z in r['pred']],[0,.5,.9,1]).tolist()}
print(json.dumps(out,indent=2))
