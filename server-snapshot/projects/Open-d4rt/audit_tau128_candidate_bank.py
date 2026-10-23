import json
from pathlib import Path
import torch
from eval_tau128_candidate_bank import route_oracle,cap_mix

bank=torch.stack([torch.zeros(2,3,4,4),torch.ones(2,3,4,4)])
target=torch.zeros(2,3,4,4);target[0,:,:,:2]=1;target[1,:,:,2:]=1
pred,diag=route_oracle(bank,target,2,True,1.)
assert torch.equal(pred,target)
assert diag['switch_rate']==1.
fixed,_=route_oracle(bank,target,2,False,1.)
assert (fixed-target).square().mean()>0
limited,factor=cap_mix(pred,bank[0],.1)
assert abs(limited.abs().mean().item()-.1)<1e-6
assert 0<factor<1

base=Path('/root/autodl-fs/d4rt_training')
rows=[json.loads(x) for x in (base/'tau128_candidate_bank_replay_smoke_20261011/test_per_clip.jsonl').read_text().splitlines()]
old=[json.loads(x) for x in (base/'tau128_ttpo_replay_20261011/test_per_clip.jsonl').read_text().splitlines()]
for row in rows:
    prior=next(x for x in old if x['path']==row['path'])
    for key,index in [('tau',0),('geometry',4)]:
        difference=abs(row['metrics'][key]['MSE']-prior['metrics'][index]['MSE'])
        assert difference<1e-7,(key,difference)
    assert len([x for x in row['metrics'] if x.startswith('replay_')])==4
    assert all(x['pixel_delta']<=row['budget']+1e-6 for x in row['metrics'].values())
print('PASS: spatial/horizon indexing, budget projection, baseline reproduction, four replay routes, all output budgets')
