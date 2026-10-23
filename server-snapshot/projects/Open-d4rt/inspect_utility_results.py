import json
from pathlib import Path
import numpy as np
import train_tau128_temporal_selector as tr
p=Path('/root/autodl-fs/d4rt_training/selector_utility_full_20261012')
history=json.loads((p/'history.json').read_text())
rows=[json.loads(s) for s in (p/'validation_per_clip.jsonl').read_text().splitlines()]
for r in rows:
    r['group']=tr.prev.group(r['path'])
    r['metrics']['geometry']=r['metrics']['baseline']
print(json.dumps({'bootstrap_validation_post_selection':tr.bootstrap(rows,['soft','hard']),
 'by_source':{d:{'clips':len(rs),'mse_gain':float(np.mean([r['metrics']['baseline']['MSE']-r['metrics']['soft']['MSE'] for r in rs])),
 'lpips_gain':float(np.mean([r['metrics']['baseline']['LPIPS']-r['metrics']['soft']['LPIPS'] for r in rs])),
 'mse_win_fraction':float(np.mean([r['metrics']['soft']['MSE']<r['metrics']['baseline']['MSE'] for r in rs]))}
 for d in sorted({r['group'][0] for r in rows}) for rs in [[r for r in rows if r['group'][0]==d]]},
 'trajectory':[{'epoch':r['epoch'],'loss':r['loss'],'soft_score':tr.composite(r['metrics']['soft']),'hard_score':tr.composite(r['metrics']['hard']),'routing':r['routing']['soft']} for r in history if r['epoch'] in [1,10,20,40,52,60,80]]},indent=2))
