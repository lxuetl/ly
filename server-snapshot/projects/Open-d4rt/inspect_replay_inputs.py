import json,inspect,numpy as np,torch
from pathlib import Path
import train_propagation_fusion as t
items=torch.load(t.ROOT/'sdxl_refiner_pilot_20261014/inputs.pt',weights_only=False)
for d in items:
 z=np.load(d['path']);m=json.loads(str(z['metadata_json']));s=np.load(m['source_path'],allow_pickle=True)
 print(json.dumps({'group':d['group'],'frames':len(s['images_jpeg_bytes']),'source_meta':str(s['metadata_json']) if 'metadata_json' in s.files else None}))
