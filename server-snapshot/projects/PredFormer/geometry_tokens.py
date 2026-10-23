"""Stage-2/3 building blocks. Not trained or connected to stage-1 baseline yet.
Geometry tokens must come exclusively from observed frames/predicted trajectories.
"""
import torch
from torch import nn

class FutureQueries(nn.Module):
    def __init__(self,dim=256,patches=256,horizons=8,heads=8):
        super().__init__();self.time=nn.Parameter(torch.randn(horizons,dim)*.02)
        self.position=nn.Parameter(torch.randn(patches,dim)*.02)
        self.norm=nn.LayerNorm(dim);self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,history):
        # history: B,T,P,D. Every output query explicitly represents horizon + patch.
        b,t,p,d=history.shape;assert p==len(self.position)
        mem=self.norm(history.reshape(b,t*p,d));out=[]
        for h in range(len(self.time)):
            q=history[:,-1]+self.position[None]+self.time[h]
            q=q+self.attn(self.norm(q),mem,mem,need_weights=False)[0];out.append(q+self.ff(q))
        return torch.stack(out,1)

class GeometryRead(nn.Module):
    def __init__(self,input_dim,dim=256,heads=8):
        super().__init__();self.project=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.norm=nn.LayerNorm(dim);self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.null=nn.Parameter(torch.zeros(1,1,dim));self.scale=nn.Parameter(torch.tensor(.1));self.heads=heads
    def forward(self,query,tokens,valid,patch_uv=None,trajectory_uv=None,sigma=.1):
        # B,H,P,D queries; B,H,N,C memory. Omega: aligned spatial/time features.
        # D4RT: per-horizon predicted point tokens, with normalized future UV.
        b,h,p,d=query.shape;n=tokens.shape[2];mem=self.project(tokens).reshape(b*h,n,d)
        mem=torch.cat([mem,self.null.expand(b*h,1,d)],1);q=query.reshape(b*h,p,d)
        bias=torch.zeros(b,h,p,n,device=q.device,dtype=q.dtype)
        if trajectory_uv is not None:
            assert patch_uv is not None and sigma>0
            bias=-((patch_uv[None,None,:,None,:]-trajectory_uv[:,:,None,:,:])**2).sum(-1)/(2*sigma*sigma)
        bias=bias.masked_fill(~valid[:,:,None,:].bool(),float('-inf'))
        bias=torch.cat([bias,torch.zeros(b,h,p,1,device=q.device,dtype=q.dtype)],-1)
        bias=bias.reshape(b*h,p,n+1).repeat_interleave(self.heads,0)
        delta=self.attn(self.norm(q),mem,mem,attn_mask=bias,need_weights=False)[0]
        return (q+self.scale*delta).reshape(b,h,p,d)

def smoke():
    torch.manual_seed(42);d=32
    x=torch.randn(1,3,4,d);future=FutureQueries(d,4,2,4)(x)
    m=GeometryRead(12,d,4);tokens=torch.randn(1,2,5,12);valid=torch.zeros(1,2,5,dtype=torch.bool)
    uv=torch.rand(1,2,5,2);patch=torch.rand(4,2)
    y=m(future,tokens,valid,patch,uv);assert y.shape==(1,2,4,d) and torch.isfinite(y).all()
    y.square().mean().backward();assert m.scale.grad is not None
    print('FUTURE_QUERY_AND_ALL_INVALID_GEOMETRY_SMOKE_OK')
if __name__=='__main__':smoke()
