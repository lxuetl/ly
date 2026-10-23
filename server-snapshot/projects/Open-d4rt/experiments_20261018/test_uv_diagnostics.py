import unittest
import torch
from uv_diagnostics import uv_baselines,last_index

class UVTimeTests(unittest.TestCase):
    def test_dense_linear_timestamps(self):
        uv=torch.zeros(1,1,16,2);uv[0,0,:,0]=torch.arange(16)*.01+.2;uv[...,1]=.4
        b,i=uv_baselines(uv,torch.ones(1,1,16,dtype=torch.bool))
        expected=torch.arange(16,24)*.01+.2
        self.assertTrue(torch.allclose(b['cv_elapsed'][0,0,:,0],expected,atol=1e-6))
        self.assertTrue(torch.allclose(b['cv_last2'],b['cv_elapsed'],atol=1e-6))
        self.assertTrue(torch.allclose(b['cv_model'],b['cv_elapsed'],atol=1e-6))
        self.assertEqual(i['age'].item(),0)

    def test_missing_tail_elapsed(self):
        uv=torch.zeros(1,1,16,2);uv[0,0,:,0]=torch.arange(16)*.01+.2;uv[...,1]=.4
        valid=torch.ones(1,1,16,dtype=torch.bool);valid[:,:,14:]=False
        b,i=uv_baselines(uv,valid)
        self.assertEqual(i['age'].item(),2)
        self.assertAlmostEqual(b['cv_elapsed'][0,0,0,0].item(),.36,places=6)
        self.assertAlmostEqual(b['cv_model'][0,0,0,0].item(),.34,places=6)
        valid[:]=False;valid[:,:,2]=True;valid[:,:,9]=True
        b,i=uv_baselines(uv,valid)
        self.assertAlmostEqual(b['cv_last2'][0,0,0,0].item(),.36,places=6)
        self.assertTrue(torch.equal(i['velocity'],torch.zeros_like(i['velocity'])))

    def test_single_valid_zero_speed(self):
        uv=torch.rand(1,1,16,2);v=torch.zeros(1,1,16,dtype=torch.bool);v[:,:,6]=True
        b,i=uv_baselines(uv,v)
        self.assertTrue(torch.equal(b['hold'],b['cv_last2']))

if __name__=='__main__':unittest.main()
