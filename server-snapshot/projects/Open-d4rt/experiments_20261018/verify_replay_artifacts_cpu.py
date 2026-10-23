"""Byte checks on generated tensors only; never decode RGB or open GT."""
import json,hashlib
from pathlib import Path
import torch

ROOT=Path('/root/autodl-fs/d4rt_training/cpu_causal_replay_policy_20261019')


def digest(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1);records=[]
    for i in range(6):
        data=torch.load(ROOT/f'case_{i}/future_predictions.pt',map_location='cpu',weights_only=False)
        locked=json.loads((ROOT/f'case_{i}/prefix16_locked.json').read_text());decision=json.loads((ROOT/f'case_{i}/decision.json').read_text())
        assert data['decision']==decision['final_action']
        a8=torch.load(ROOT/f'case_{i}/allowed8.pt',map_location='cpu',weights_only=False)
        a12=torch.load(ROOT/f'case_{i}/allowed12.pt',map_location='cpu',weights_only=False)
        l8=json.loads((ROOT/f'case_{i}/prefix8_locked.json').read_text())
        l12=json.loads((ROOT/f'case_{i}/prefix12_locked.json').read_text())
        for motion in ['hold','cv']:
            assert a8[motion].dtype==torch.bool and a12[motion].dtype==torch.bool
            assert not (a12[motion]&~a8[motion]).any()
            assert int(a8[motion].sum())==l8['support'][motion]
            assert int(a12[motion].sum())==l12['support'][motion]
            assert locked['support'][motion]<=l12['support'][motion]
        for name,output in data['outputs'].items():
            assert digest(output)==locked['output_hashes'][name]
            assert torch.equal(output[4:],data['base'][4:])
            assert float((output[:4]-data['base'][:4]).abs().mean())<=.001001
            assert float((output-data['base']).abs().max())<=.015001
        records.append({'index':i,'output_count':len(data['outputs']),'verified':True})
    result={'output_hashes_match_locked':True,'last4_bitwise_keep':True,'cases':records,'GT_read':False}
    (ROOT/'artifact_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
