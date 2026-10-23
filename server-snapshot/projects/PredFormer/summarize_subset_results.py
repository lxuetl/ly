import json
from pathlib import Path
r=Path('/root/autodl-tmp/predformer_subset_ablation_20261022')
keys=['mse','lpips','late_mse','late_lpips']
for d in sorted(r.iterdir()):
 if not d.is_dir():continue
 s=json.loads((d/'sensitivity.json').read_text());h=json.loads((d/'history.json').read_text())
 print(d.name,json.dumps({k:{m:v[m] for m in keys} for k,v in s.items()}))
 print('history',json.dumps([{k:a[k] for k in ['epoch','train_mse']}|{'val_mse':a['validation']['mse'],'lpips':a['validation']['lpips']} for a in h if a['epoch'] in [0,1,2,5,10,15]]))
