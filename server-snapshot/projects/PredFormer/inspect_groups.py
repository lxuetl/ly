import json
import numpy as np
from pathlib import Path
from stage1_baseline import ROOT
p=Path('/root/autodl-tmp/predformer_token_audit_20261021/per_clip.json')
rows=json.loads(p.read_text());seen=set()
for r in rows:
    domain=str(Path(r['source']).parent)
    if domain in seen:continue
    seen.add(domain);pp=Path(r['path']);pp=pp if pp.is_absolute() else ROOT/pp
    with np.load(pp) as z:meta=json.loads(str(z['metadata_json']))
    print(domain,json.dumps(meta)[:4500])
