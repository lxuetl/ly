import json
from pathlib import Path
import torch,numpy as np
from stage1_baseline import ROOT
base=Path('/root/autodl-tmp/predformer_native128_20261021')
splits=json.loads((base/'split_paths.json').read_text())['paths']
omega=Path('/root/autodl-fs/d4rt_datasets/vggt_omega_spatial_v4_20261002')
for split in ['train','val']:
    missing=[]
    for p in splits[split]:
        p=Path(p);f=omega/p.parent.name/(p.stem+'.pt')
        if not f.exists():missing.append(str(p))
    print(split,len(splits[split]),'omega_missing',len(missing),missing[:3])
p=Path(splits['train'][0]);p=p if p.is_absolute() else ROOT/p
with np.load(p) as z:
    print('D4RT',str(p),[(k,z[k].shape,str(z[k].dtype)) for k in z.files]);print(str(z['metadata_json']))
f=omega/p.parent.name/(p.stem+'.pt');q=torch.load(f,weights_only=True,map_location='cpu')
print('OMEGA',[(k,tuple(v.shape)) for k,v in q.items() if torch.is_tensor(v)])
print('manifest', (omega/'manifest.json').read_text()[:1200])
