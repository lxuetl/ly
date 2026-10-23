import unittest
from unittest.mock import patch
import numpy as np
import torch
import reliability_cpu as r


class ReliabilityTests(unittest.TestCase):
    def test_stale_anchor_and_velocity(self):
        uv=torch.tensor([[[.1,.1],[.2,.1],[.3,.1],[.9,.9]]])
        valid=torch.tensor([[True,True,True,False]])
        a=r.alternatives(uv,valid,2)
        self.assertTrue(torch.allclose(a['hold'][0,0],torch.tensor([.3,.1])))
        # Velocity cap .08, horizon starts two frames after last valid observation.
        self.assertTrue(torch.allclose(a['cv'][0,0],torch.tensor([.46,.1])))

    def test_no_replay_no_future_writes(self):
        p=torch.full((3,8,2),.5);simple={'cv':p,'hold':p}
        self.assertFalse(r.gate_future(p,simple,{'available':False}).any())

    def test_gate_rejects_invalid_and_disagreement(self):
        p=torch.full((3,8,2),.5);q=p.clone();q[1,:,0]=.7;p[2,:,0]=1.1
        replay={'chosen_simple_model':'cv','median_error_per_horizon_px':{'cv':[0.]*7+[3.]}}
        g=r.gate_future(p,{'cv':q},replay)
        self.assertEqual(int(g.sum()),7)

    def test_prefix_forecast_not_changed_by_withheld_RGB(self):
        recorded=[];original=r.alternatives
        def capture(uv,valid,k=8):
            result=original(uv,valid,k);recorded.append({n:v.clone() for n,v in result.items()});return result
        def flow(a,b,p):
            return np.asarray(p).reshape(-1,2)+.1, np.ones(len(p),bool),np.zeros(len(p))
        rgb=torch.rand(16,3,32,32);changed=rgb.clone();changed[8:]=torch.rand_like(changed[8:])
        with patch.object(r,'lk',flow),patch.object(r,'alternatives',capture):
            r.strict_rgb_replay(rgb);r.strict_rgb_replay(changed)
        for name in recorded[0]:self.assertTrue(torch.equal(recorded[0][name],recorded[1][name]))

    def test_blank_history_abstains(self):
        rgb=torch.ones(16,3,32,32)*.5;uv=torch.ones(4,16,2)*.5
        _,mask,_=r.refine_history(rgb,uv,torch.ones(4,16,dtype=torch.bool))
        self.assertFalse(mask.any())


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
