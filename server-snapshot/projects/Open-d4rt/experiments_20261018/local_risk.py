"""One-shot scoring primitives. Features do not accept target RGB or labels."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def low(x):return F.avg_pool2d(F.pad(x,(2,2,2,2),mode='replicate'),5,1)


def local_descriptor(base,candidate,sources):
    delta=candidate-base;t,c,h,w=base.shape
    disagree=(sources['align']-sources['velocity']).abs().mean(1,keepdim=True)
    # Match existing max-disagreement tile exactly; only representation changes.
    selection=torch.stack([(s-base).abs().mean((0,1)) for s in sources.values()]).mean(0)
    tile=min(32,h,w);pooled=F.avg_pool2d(selection[None,None],tile,tile)[0,0]
    y,x=divmod(int(pooled.argmax()),pooled.shape[1]);mask=torch.zeros_like(disagree)
    mask[:,:,y*tile:(y+1)*tile,x*tile:(x+1)*tile]=1
    yy=max(0,min(h-64,y*tile-16));xx=max(0,min(w-64,x*tile-16))
    temporal=torch.cat([torch.zeros_like(delta[:1]),delta[1:]-delta[:-1]])
    horizon=torch.linspace(0,1,t,device=base.device)[:,None,None,None].expand(-1,1,h,w)
    features=torch.cat([base,candidate,delta,sources['align'],sources['velocity'],disagree,horizon,mask,delta-low(delta),temporal],1)
    assert features.shape[1]==24
    crop=features[:,:,yy:yy+64,xx:xx+64]
    return F.interpolate(crop,(32,32),mode='bilinear',align_corners=False).permute(1,0,2,3)


class RiskScorer(nn.Module):
    def __init__(self,local,normalized_outputs=False):
        super().__init__();self.local=local
        self.register_buffer('mean',torch.zeros(27));self.register_buffer('scale',torch.ones(27))
        self.normalized_outputs=normalized_outputs
        if normalized_outputs:
            self.register_buffer('target_mean',torch.zeros(3))
            self.register_buffer('target_scale',torch.ones(3))
        if local:
            self.encoder=nn.Sequential(nn.Conv3d(24,12,3,padding=1),nn.SiLU(),
                nn.Conv3d(12,12,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool3d((2,2,2)),nn.Flatten())
            self.head=nn.Sequential(nn.Linear(123,64),nn.SiLU(),nn.Linear(64,3))
        else:
            self.encoder=None
            self.head=nn.Sequential(nn.Linear(27,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))

    def forward(self,global_features,local_features):
        g=(global_features-self.mean)/self.scale.clamp_min(1e-5)
        out=self.head(torch.cat([g,self.encoder(local_features)],-1) if self.local else g)
        return out*self.target_scale+self.target_mean if self.normalized_outputs else out


def choose(predictions,margin,tolerance=.01,min_gain=1e-4):
    # gain is absolute LPIPS delta; risk predictions are dimensionless ratios.
    p=predictions.detach();margin=margin.to(p)
    gain=p[:,0]-margin[0];risk=p[:,1:]+margin[1:]
    safe=(gain>min_gain)&(risk<=tolerance).all(-1)
    return int(gain.masked_fill(~safe,float('-inf')).argmax()) if safe.any() else -1


def supervised_loss(pred,target,scales):
    normalized=(pred-target)/scales
    regression=F.smooth_l1_loss(normalized,torch.zeros_like(normalized))
    # Caller batches all four candidates for each state, in order.
    a=pred[:,0].reshape(-1,4)/scales[0];y=target[:,0].reshape(-1,4)
    dif=a[:,:,None]-a[:,None,:];sign=(y[:,:,None]-y[:,None,:]).sign()
    upper=torch.triu(torch.ones(4,4,dtype=torch.bool,device=pred.device),1)[None]
    valid=upper&(sign!=0)
    rank=F.softplus(-sign*dif)[valid].mean() if valid.any() else pred.sum()*0
    return regression+.1*rank
