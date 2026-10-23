import unittest
import torch
from trajectory_attention import TrackCrossAttention,sample,splat
from local_risk import local_descriptor,RiskScorer,choose,supervised_loss
from common import pixel_bound


class PrototypeTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)

    def test_sampling_coordinates(self):
        feature=torch.arange(5).float()[None,None,None,None,:].expand(1,3,2,5,5)
        uv=torch.tensor([[[[.25,.5],[.5,.5],[.75,.5]]]])
        z=sample(feature,uv)
        self.assertTrue(torch.allclose(z[0,0,:,0],torch.tensor([1.,2.,3.]),atol=1e-6))

    def test_noop_invalid_and_backward(self):
        b,n,t,k,c=1,5,4,3,8
        hidden=torch.randn(b,k,c,8,8);values=torch.randn(b,n,t,c)
        past=torch.rand(b,n,t,2);future=torch.rand(b,n,k,2)
        valid=torch.ones(b,n,t,dtype=torch.bool);conf=torch.ones(b,n,t)
        model=TrackCrossAttention(c,dim=16,heads=4)
        out,_=model(hidden,values,past,valid,conf,future)
        self.assertTrue(torch.equal(out,hidden))
        (out.square().mean()).backward()
        self.assertGreater(float(model.out.weight.grad.abs().sum()),0.)
        with torch.no_grad():model.out.weight.normal_();model.out.bias.fill_(1.)
        out,_=model(hidden,values,past,valid*False,conf*0,future)
        self.assertTrue(torch.equal(out,hidden))
        out,_=model.forward_fixed(hidden,values[:,:,None].expand(-1,-1,k,-1,-1),future,valid,conf)
        self.assertEqual(out.shape,hidden.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_outside_never_writes_border(self):
        z,s=splat(torch.ones(1,2,3,4),torch.full((1,2,3,2),2.),torch.ones(1,2,3),8,8)
        self.assertEqual(float(z.abs().sum()),0.);self.assertEqual(float(s.sum()),0.)

    def test_fixed_vectorization_same_when_path_is_stationary(self):
        model=TrackCrossAttention(8,16,4)
        with torch.no_grad():model.out.weight.normal_();model.out.bias.fill_(.5)
        hidden=torch.randn(1,3,8,8,8);values=torch.randn(1,5,4,8)
        anchor=torch.rand(1,5,1,2);past=anchor.expand(-1,-1,4,-1);future=anchor.expand(-1,-1,3,-1)
        valid=torch.ones(1,5,4,dtype=torch.bool);conf=torch.ones(1,5,4)
        a,_=model(hidden,values,past,valid,conf,future)
        b,_=model.forward_fixed(hidden,values[:,:,None].expand(-1,-1,3,-1,-1),future,valid,conf)
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=0))

    def test_budget_and_zero_identity(self):
        base=torch.rand(2,8,3,16,16);proposal=base+torch.randn_like(base)
        x=pixel_bound(proposal,base)
        self.assertTrue(((x-base).abs().mean((1,2,3,4))<=.002001).all())
        self.assertTrue(((x-base).abs()<=.030001).all())
        self.assertTrue(torch.equal(pixel_bound(proposal,base,0.),base))
        tiny=base+1e-5
        self.assertTrue(torch.allclose(pixel_bound(tiny,base),tiny,atol=1e-7))

    def test_local_features_and_objective(self):
        base=torch.rand(8,3,128,128);sources={'align':torch.rand_like(base),'velocity':torch.rand_like(base)}
        local=local_descriptor(base,base+.001,sources)
        self.assertEqual(local.shape,(24,8,32,32));self.assertTrue(torch.isfinite(local).all())
        for use_local in [False,True]:
            for norm in [False,True]:
                m=RiskScorer(use_local,norm);g=torch.randn(4,27);l=local[None].expand(4,-1,-1,-1,-1)
                if norm:m.target_scale.copy_(torch.tensor([.001,.01,.01]))
                p=m(g,l);loss=supervised_loss(p,torch.randn_like(p),torch.ones(3));loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(m.head[-1].weight.grad)

    def test_risk_units_and_keep(self):
        p=torch.tensor([[.002,.005,.02],[.001,.005,.005]])
        self.assertEqual(choose(p,torch.zeros(3)),1)
        self.assertEqual(choose(p,torch.ones(3)),-1)
        self.assertEqual(choose(torch.zeros(4,3),torch.zeros(3)),-1)


if __name__=='__main__':unittest.main()
