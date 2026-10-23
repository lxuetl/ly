"""CPU factorial invariants, not model training."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from run_cpu_history_endpoint import history_data,memories,factor_effects,avg

class DummyTrajectory:
    def __call__(self,xyz,uv,valid,conf):
        return torch.zeros(*uv.shape[:2],8,3),uv[:,:,-1:].expand(-1,-1,8,-1).clamp(0,1)

def example():
    uv=torch.full((4,16,2),.3);uv[:,:,1]=.4
    return dict(past_uv=uv,past_xyz=torch.zeros(4,16,3),past_valid=torch.ones(4,16,dtype=torch.bool),
      past_confidence=torch.ones(4,16),history_uv_gt=uv-.05,history_valid_gt=torch.ones(4,16,dtype=torch.bool),
      future_uv_raw=torch.full((4,8,2),.5),future_visibility_gt=torch.ones(4,8,dtype=torch.bool),_path='unused')

class CPUFactorTests(unittest.TestCase):
    def test_selection_and_prediction_ignore_future_labels(self):
        x=example()
        with patch('run_cpu_history_endpoint.np.load',return_value={'query_ids':np.arange(4)}):
            h1=history_data(x,DummyTrajectory());x['future_uv_raw'].fill_(2);x['future_visibility_gt'].fill_(False)
            h2=history_data(x,DummyTrajectory())
        self.assertTrue(torch.equal(h1['ids'],torch.arange(4)))
        for k in ['ids','p','hu','gu','ns','nd']:self.assertTrue(torch.equal(h1[k],h2[k]))
        self.assertFalse(h2['fw'].any());self.assertTrue(h1['fw'].any())

    def test_same_history_weights_isolate_sampling_coordinates(self):
        x=example()
        with patch('run_cpu_history_endpoint.np.load',return_value={'query_ids':np.arange(4)}):h=history_data(x,DummyTrajectory())
        yy,xx=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),indexing='ij')
        features=torch.stack([xx,yy])[None,None].expand(1,16,-1,-1,-1)
        vd,vg,md,mg,own=memories(features,h)
        self.assertTrue(torch.allclose(md-mg,torch.full_like(md,.05),atol=1e-6))
        self.assertTrue(torch.equal(md,own))

    def test_factor_algebra(self):
        values={'D_P':5.,'G_P':4.,'D_G':3.,'G_G':2.,'D_P_own_history_mask':6.,'true_future_G':1.}
        mets={n:{k:v for k in ['mse','lpips','temporal']} for n,v in values.items()};e=factor_effects(mets)
        self.assertEqual(e['history_at_pred']['mse'],-1.)
        self.assertEqual(e['endpoint_with_D_history']['mse'],-2.)
        self.assertEqual(e['interaction']['mse'],0.)
        self.assertEqual(e['history_mask_only']['mse'],-1.)

if __name__=='__main__':unittest.main()
