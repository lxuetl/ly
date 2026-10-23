import unittest
import torch
import preference_guided_repair_pilot as p

class GuidanceTests(unittest.TestCase):
    def test_preference_gradient_direction(self):
        torch.manual_seed(42);x=torch.rand(8,3,32,32)*.5+.25
        positive=(x+.02*torch.randn_like(x)).clamp(0,1);negative=(2*x-positive).clamp(0,1)
        mask=torch.ones(8,1,32,32);z=x.clone().requires_grad_()
        loss=p.pref_loss(z,positive,negative,mask);g=torch.autograd.grad(loss,z)[0]
        reverse=p.pref_loss(z,negative,positive,mask);gr=torch.autograd.grad(reverse,z)[0]
        self.assertLess(float((g*gr).sum()),0)
        self.assertLess(float(p.pref_loss(x-.001*g/g.abs().mean(),positive,negative,mask)),float(loss))
    def test_limits_and_zero_pair(self):
        torch.manual_seed(1);b=torch.rand(8,3,32,32)
        v=p.project(b+1,b)
        self.assertLessEqual(float((v-b).abs().mean()),p.BUDGET+1e-7)
        src={'align':b,'velocity':b}
        z,_=p.optimize(b,b,b,b,src,'guided')
        self.assertTrue(torch.equal(z,b))

if __name__=='__main__':unittest.main()
