"""Small frozen-network equivalence checks; no real model fitting."""
import ast, json, types
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_num_threads(1);torch.manual_seed(42)
source=ast.parse(Path(__file__).with_name('three_stage_audit.py').read_text())
selected=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in
          ['encode_stream','decode_stream','feature_gradient_stream','latent_rgb_gradient']]
enc=torch.nn.Conv2d(3,4,1).eval().requires_grad_(False)
dec=torch.nn.Conv2d(6,3,1).eval().requires_grad_(False)
def encode(net,x):
    b,t,c,h,w=x.shape
    y=enc(x.reshape(b*t,c,h,w)).reshape(b,t,4,h,w)
    return y,y[:,:,:2]
def decode(net,h,s):
    b,t,c,y,x=h.shape
    return torch.sigmoid(dec(torch.cat([h,s],2).reshape(b*t,6,y,x))).reshape(b,t,3,y,x)
def sample(fe,uv):
    b,t,c,h,w=fe.shape;n=uv.shape[1]
    grid=uv.permute(0,2,1,3).reshape(b*t,n,1,2)*2-1
    return F.grid_sample(fe.reshape(b*t,c,h,w),grid,align_corners=True)[...,0].reshape(b,t,c,n).permute(0,3,1,2)
scope={'torch':torch,'encode':encode,'core':types.SimpleNamespace(decode_frozen=decode),'sample':sample}
exec(compile(ast.Module(body=selected,type_ignores=[]),'<stream>','exec'),scope)
h=torch.randn(1,8,4,5,5);s=torch.randn(1,8,2,5,5);target=torch.rand(8,3,5,5)
appearance=torch.rand(1,7,8,4);coords=torch.rand(1,7,8,2);ww=torch.rand(1,7,8);ww[:,:,3]=0
scale=torch.rand(1,1,1,4)+.1
hi=h.clone().requires_grad_(True);si=s.clone().requires_grad_(True)
features,_=encode(None,decode(None,hi,si))
loss=((((sample(features,coords)-appearance)/scale).square().mean(-1))*ww).sum()/ww.sum()
a,b=torch.autograd.grad(loss,(hi,si))
gh,gs,value=scope['feature_gradient_stream'](None,h,s,appearance,coords,ww,scale)
assert torch.allclose(gh,-a,atol=1e-7,rtol=1e-5)
assert torch.allclose(gs,-b,atol=1e-7,rtol=1e-5)
assert abs(value-float(loss.detach()))<1e-6
hi=h.clone().requires_grad_(True);si=s.clone().requires_grad_(True)
loss=(decode(None,hi,si)[0]-target).square().mean();a,b=torch.autograd.grad(loss,(hi,si))
gh,gs=scope['latent_rgb_gradient'](None,h,s,target)
assert torch.allclose(gh,a,atol=1e-7,rtol=1e-5) and torch.allclose(gs,b,atol=1e-7,rtol=1e-5)
rgb=torch.rand(1,16,3,5,5)
assert torch.allclose(encode(None,rgb)[0],scope['encode_stream'](None,rgb)[0],atol=1e-7)
assert torch.allclose(decode(None,h,s),scope['decode_stream'](None,h,s),atol=1e-7)
print(json.dumps({'passed':True,'checks':['encoder','decoder','feature_loss','feature_gradient_h','feature_gradient_skip','RGB_gradient_h','RGB_gradient_skip']}))
