import json,numpy as np
from pathlib import Path
from stage1_baseline import ROOT
from internal_geometry import RUN
m=json.loads((RUN/'manifest.json').read_text());seen=set()
for p,r in m.items():
 if r['subset'] in seen:continue
 seen.add(r['subset']);pp=Path(p);pp=pp if pp.is_absolute() else ROOT/pp
 with np.load(pp) as z:meta=json.loads(str(z['metadata_json']))
 sp=Path(meta['source_path']);sp=sp if sp.is_absolute() else ROOT/sp
 with np.load(sp,allow_pickle=True) as z:
  print(r['subset'],str(sp),z.files)
  for k in z.files:
   if 'meta' in k or 'source' in k or 'sequence' in k:print(k,str(z[k])[:3000])
