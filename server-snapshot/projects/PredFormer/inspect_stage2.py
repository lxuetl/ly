import json
from pathlib import Path
import numpy as np
import torch
base=Path('/root/autodl-tmp/predformer_native128_20261021')
root=Path('/root/autodl-tmp/predformer_queries_20261021')
report={};arrays={}
for name,path in [('native',base)]+[(n,root/n) for n in ['query','null','rgb']]:
    rows=json.loads((path/'best_validation.json').read_text());rows=sorted(rows,key=lambda r:r['path'])
    arrays[name]=rows
    ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
    report[name]={'epoch':ck['epoch'],'clips':len(rows),'metrics':{k:float(np.mean([r[k] for r in rows])) for k in ['mse','mae','psnr','temporal']}}
    if 'adapter.scale' in ck['model']:report[name]['adapter_scale']=float(ck['model']['adapter.scale'])
for name in ['query','null','rgb']:
    assert [r['path'] for r in arrays[name]]==[r['path'] for r in arrays['native']]
report['paired_clip_counts']={}
for a,b in [('query','native'),('rgb','null'),('rgb','query')]:
    d=np.array([x['mse']-y['mse'] for x,y in zip(arrays[a],arrays[b])]);report['paired_clip_counts'][a+'_minus_'+b]={'mean_mse_difference':float(d.mean()),'better':int((d<0).sum()),'worse':int((d>0).sum()),'note':'clips are not independent original videos; descriptive only'}
print(json.dumps(report,indent=2))
