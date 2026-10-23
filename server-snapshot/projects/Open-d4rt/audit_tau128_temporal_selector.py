import json
from pathlib import Path
import torch
from train_tau128_temporal_selector import TemporalSelector,synthesize,prev

base=Path('/root/autodl-fs/d4rt_training')
paths=json.loads((base/'tau_v2_direction_rank_pb002_20261009/split_paths.json').read_text())['paths']
groups={s:{prev.group(p) for p in v} for s,v in paths.items()}
assert not groups['train']&groups['val']
assert not groups['train']&groups['test']
assert not groups['val']&groups['test']
torch.manual_seed(123)
model=TemporalSelector().eval()
torch.nn.init.normal_(model.net[-1].weight,std=.02)
x=torch.randn(1,8,57,8,8)
changed=x.clone();changed[:,:,-7:-2]+=10
w,g=model(x,True);w2,g2=model(changed,True)
assert torch.equal(w,w2) and torch.equal(g,g2)
assert torch.allclose(w.sum(2),torch.ones_like(w[:,:,0]),atol=1e-6)
tau=torch.rand(1,1,8,3,128,128)
bank=(tau+torch.randn(1,8,8,3,128,128)*.001).clamp(0,1)
bank[:,0]=tau[:,0]
budget=(bank[:,1]-bank[:,0]).abs().mean().reshape(1)
zero,_=model(x,gain=0)
pred=synthesize(bank,zero,budget)
assert torch.allclose(pred,bank[:,1],atol=2e-7)
pred=synthesize(bank,w,budget)
assert (pred-bank[:,0]).abs().mean()<=budget+1e-7
pred.mean().backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
smoke=base/'tau128_temporal_selector_smoke_20261011'
assert (smoke/'COMPLETE.json').exists()
for variant in ['geometry','no_explicit_geometry']:
    state=torch.load(smoke/f'{variant}_last.pt',map_location='cpu',weights_only=False)['model']
    assert state['net.4.weight'].abs().sum()>0
    assert all(torch.isfinite(v).all() for v in state.values())
print(json.dumps({'PASS':['full original-video split separation','no-explicit-geometry input invariance','convex weights','exact gain-zero fallback','postmix budget','finite gradients','two trained finite smoke checkpoints','complete smoke evaluation'], 'counts':{s:len(v) for s,v in paths.items()},'params':sum(p.numel() for p in model.parameters())}))
