import json
from pathlib import Path
import numpy as np
p=Path('/root/autodl-fs/d4rt_training/selector_utility_full_20261012/label_renderer_audit')
rs=json.loads((p/'interventions.json').read_text())
print(json.dumps({'threshold_sensitivity':{str(e):{s:sum(r[s]['evaluation_gain'] < -e for r in rs) for s in ['nearest','bilinear','budgeted']} for e in [1e-10,1e-8,1e-7,1e-6]},
 'source':{d:{'n':len(rr),'harm_fraction':np.mean([r['budgeted']['evaluation_gain'] < -1e-8 for r in rr])} for d in sorted({r['source'] for r in rs}) for rr in [[r for r in rs if r['source']==d]]},
 'mean_label_gain':np.mean([r['label_global_gain'] for r in rs])},indent=2))
