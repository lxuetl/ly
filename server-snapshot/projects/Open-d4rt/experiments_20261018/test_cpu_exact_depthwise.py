import unittest
import torch
from torch import nn
from cpu_exact_depthwise import channelwise,install


class DepthwiseTests(unittest.TestCase):
    def test_output_and_input_gradient(self):
        torch.manual_seed(91);layer=nn.Conv2d(96,96,7,padding=9,dilation=3,groups=96).eval().requires_grad_(False)
        x=torch.rand(1,96,12,12,requires_grad=True);y=x.detach().clone().requires_grad_(True)
        original=layer(x);chunked=channelwise(layer,y,32)
        self.assertTrue(torch.allclose(original,chunked,atol=2e-6,rtol=2e-6))
        a=torch.autograd.grad(original.square().sum(),x)[0];b=torch.autograd.grad(chunked.square().sum(),y)[0]
        self.assertTrue(torch.allclose(a,b,atol=3e-6,rtol=3e-6))

    def test_weights_unchanged_and_remainder_channels(self):
        layer=nn.Conv2d(65,65,3,padding=1,groups=65).eval().requires_grad_(False)
        before={k:v.clone() for k,v in layer.state_dict().items()};x=torch.rand(2,65,8,8);expected=layer(x)
        self.assertEqual(install(layer),1)
        self.assertTrue(torch.allclose(layer(x),expected,atol=2e-6,rtol=2e-6))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in layer.state_dict().items()))


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
