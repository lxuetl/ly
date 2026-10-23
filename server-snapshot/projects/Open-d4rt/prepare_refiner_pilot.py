import json,hashlib,random
from collections import defaultdict
import torch
import train_propagation_fusion as t
out=t.ROOT/'sdxl_refiner_pilot_20261014';out.mkdir(exist_ok=True)
paths=json.loads((t.OUT/'split_paths.json').read_text())['paths']['val']
domains=defaultdict(lambda:defaultdict(list))
for p in paths:
 g=tuple(t.prev.group(p));domains[g[0]][g].append(p)
rng=random.Random(20261014);chosen=[]
for domain,groups in sorted(domains.items()):
 names=sorted(groups);rng.shuffle(names)
 for g in names[:2]:chosen.append((g,rng.choice(sorted(groups[g]))))
items=[]
for g,p in chosen:
 data=torch.load(t.OUT/'cache'/(t.key(p)+'.pt'),map_location='cpu',weights_only=False)
 items.append({'path':p,'group':g,'tau':data['tau'],'target':data['target']})
torch.save(items,out/'inputs.pt')
t.prev.dump(out/'selection.json',{'source':'validation only, <=2 original videos per source, 1 clip each, selected before metrics','clips':len(items),'items':[{'path':p,'group':g} for g,p in chosen],'no_geometry':'only tau and target fields copied; target evaluation only'})
print(json.dumps({'clips':len(items),'sources':list(domains),'out':str(out)}))
