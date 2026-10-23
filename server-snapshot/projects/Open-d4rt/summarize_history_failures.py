import json
from collections import defaultdict
from pathlib import Path
import numpy as np
root=Path('/root/autodl-fs/d4rt_training')
old=[json.loads(x) for x in (root/'fixed_history_expanded_20261015/per_clip.jsonl').read_text().splitlines()]
diag=root/'history_regions_diagnostic_20261015'
rows=[json.loads(x) for x in (diag/'per_clip.jsonl').read_text().splitlines()]
lookup={r['path']:r for r in rows};assert len(rows)==347
out={'failure_reasons':{'insufficient_lpips_gain':0,'mse_gt1pct':0,'temporal_gt1pct':0},'source_failures':{},'examples':[],'reproduction_max':max(v for r in rows for v in r['reproduction'].values())}
sources=defaultdict(list)
for r in old:
 a=r['metrics']['fixed_history'];b=r['metrics']['tau'];z=r['metrics']['zero_motion']
 flags={'insufficient_lpips_gain':a['lpips']>=b['lpips']-.0001,'mse_gt1pct':a['mse']>b['mse']*1.01,'temporal_gt1pct':a['temporal']>b['temporal']*1.01}
 for k,v in flags.items():out['failure_reasons'][k]+=int(v)
 sources[r['group'][0]].append({'failed':any(flags.values()),'lpips_gap_zero':a['lpips']-z['lpips']})
 out['examples'].append({'path':r['path'],'group':r['group'],'flags':flags,'mse_gap_zero':a['mse']-z['mse'],'lpips_gap_zero':a['lpips']-z['lpips'],'lpips_gap_tau':a['lpips']-b['lpips']})
out['examples']=sorted(out['examples'],key=lambda x:x['lpips_gap_zero'],reverse=True)[:12]
out['source_failures']={k:{'clips':len(rs),'failed':sum(r['failed'] for r in rs),'lpips_gap_zero_clip_mean':float(np.mean([r['lpips_gap_zero'] for r in rs]))} for k,rs in sources.items()}
(diag/'failure_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
