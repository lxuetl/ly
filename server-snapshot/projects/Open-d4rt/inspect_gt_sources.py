import json,numpy as np
import train_single_region_risk_expanded as e
paths=json.loads((e.p.run.fit.SOURCE/'split_paths.json').read_text())['paths']['val']
seen=set()
for p in paths:
 d=e.tr.prev.group(p)[0]
 if d in seen:continue
 seen.add(d);z=np.load(p);meta=json.loads(str(z['metadata_json']))
 print(json.dumps({'domain':d,'keys':z.files,'metadata':meta},default=str))
