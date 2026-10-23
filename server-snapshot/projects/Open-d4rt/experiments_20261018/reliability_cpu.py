"""Observed-RGB-only reliability heuristics. No labels, fitting or learned weights."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def inside(uv):
    return torch.isfinite(uv).all(-1) & (uv >= 0).all(-1) & (uv <= 1).all(-1)


def gray_video(rgb):
    a=(rgb.permute(0,2,3,1).numpy()*255).clip(0,255).astype(np.uint8)
    return [cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in a]


def lk(a,b,p):
    p=np.nan_to_num(p).astype(np.float32).reshape(-1,1,2)
    q,status,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(15,15),maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.01))
    if q is None:return p[:,0].copy(),np.zeros(len(p),bool),np.full(len(p),1e6)
    r,back,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(15,15),maxLevel=2)
    if r is None:return q[:,0],np.zeros(len(p),bool),np.full(len(p),1e6)
    e=np.linalg.norm(r[:,0]-p[:,0],axis=-1)
    return q[:,0],status[:,0].astype(bool)&back[:,0].astype(bool)&np.isfinite(q[:,0]).all(-1),e


def patches(rgb,uv):
    # uv [N,T,2] -> normalized grayscale patches [N,T,25].
    t,_,h,w=rgb.shape
    offsets=torch.tensor([(x,y) for y in range(-2,3) for x in range(-2,3)],dtype=uv.dtype)
    grid=uv.permute(1,0,2)[:,:,None,:]+offsets[None,None]/torch.tensor([w-1,h-1])
    gray=rgb.mean(1,keepdim=True)
    values=F.grid_sample(gray,grid*2-1,align_corners=True,padding_mode='border')[:,0].permute(1,0,2)
    return values


def ncc(a,b):
    a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-8)


def refine_history(rgb,uv,valid):
    """Last-frame anchored adjacent LK cycle plus bounded 3x3 patch search.

    Requires a valid last observed anchor. Occluded/reappearing tracks abstain;
    no assertion that last-frame identity itself has been verified.
    """
    gray=gray_video(rgb);n,t,_=uv.shape;factor=rgb.shape[-1]-1
    tracked=uv.clone();cycle=torch.full((n,t),1e6);alive=valid[:,-1].clone()&inside(uv[:,-1])
    ok=torch.zeros_like(valid);ok[:,-1]=alive;cycle[:,-1]=0
    for j in range(t-2,-1,-1):
        q,status,e=lk(gray[j+1],gray[j],tracked[:,j+1].numpy()*factor)
        tracked[:,j]=torch.from_numpy(q/factor)
        alive=alive&torch.from_numpy(status)&(torch.from_numpy(e)<=1)&inside(tracked[:,j])
        ok[:,j]=alive;cycle[:,j]=torch.from_numpy(e)
    original=patches(rgb,uv);template=original[:,-1:]
    best=ncc(original,template);bestuv=uv.clone()
    for dx,dy in [(x,y) for y in (-1,0,1) for x in (-1,0,1) if x or y]:
        candidate=uv+torch.tensor([dx,dy])/factor
        score=ncc(patches(rgb,candidate),template)
        take=(score>best+.03)&inside(candidate)
        bestuv=torch.where(take[...,None],candidate,bestuv);best=torch.where(take,score,best)
    # Constant appearance reference is conservative and will reject real rotations.
    texture=template.std(-1,unbiased=False)>.02
    mask=valid&inside(bestuv)&ok&((bestuv-tracked).norm(dim=-1)*factor<=2)&(best>=.6)&texture
    # Anchor is not moved by local search.
    bestuv[:,-1]=uv[:,-1]
    mask[:,-1]=valid[:,-1]&inside(uv[:,-1])&texture[:,0]
    return bestuv,mask,dict(cycle=cycle,ncc=best,tracked=tracked,
        moved=(bestuv-uv).norm(dim=-1)*factor)


def alternatives(uv,valid,k=8):
    n,t,_=uv.shape
    idx=(valid.long()*torch.arange(1,t+1)).argmax(-1)
    anchor=uv[torch.arange(n),idx]
    pair=valid[:,1:]&valid[:,:-1]
    pair[:,:max(0,t-1-4)]=False
    velocity=torch.nanmedian((uv[:,1:]-uv[:,:-1]).masked_fill(~pair[...,None],float('nan')),dim=1).values
    velocity=torch.nan_to_num(velocity).clamp(-.08,.08)
    # Account explicitly for a stale last valid observation.
    steps=torch.arange(t,t+k)[None,:]-idx[:,None]
    return {'hold':anchor[:,None].expand(-1,k,-1),
            'cv':anchor[:,None]+steps[...,None]*velocity[:,None]}


def strict_rgb_replay(rgb):
    """8->8 replay: prefix grid/corner seeds, all forecast inputs use frames0:8.

    Continued LK is a noisy observational reference, NOT GT trajectory labels.
    It is computed only after forecasts have been fixed.
    """
    gray=gray_video(rgb);h,w=gray[0].shape
    seeds=cv2.goodFeaturesToTrack(gray[0],maxCorners=128,qualityLevel=.01,minDistance=5)
    if seeds is None:return {'available':False,'reason':'no textured prefix seeds'}
    n=len(seeds);track=np.zeros((n,16,2),np.float32);valid=np.zeros((n,16),bool)
    track[:,0]=seeds[:,0];valid[:,0]=True
    for j in range(1,8):
        q,ok,e=lk(gray[j-1],gray[j],track[:,j-1]);track[:,j]=q
        valid[:,j]=valid[:,j-1]&ok&(e<=1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    forecast=alternatives(torch.from_numpy(track[:,:8]/(w-1)),torch.from_numpy(valid[:,:8]))
    # Holdout continuation cannot alter forecast or seed selection.
    for j in range(8,16):
        q,ok,e=lk(gray[j-1],gray[j],track[:,j-1]);track[:,j]=q
        valid[:,j]=valid[:,j-1]&ok&(e<=1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    target=torch.from_numpy(track[:,8:]/(w-1));mask=torch.from_numpy(valid[:,8:])
    error={name:(p-target).norm(dim=-1)*(w-1) for name,p in forecast.items()}
    horizon={name:[float(e[:,j][mask[:,j]].median()) if mask[:,j].any() else None for j in range(8)] for name,e in error.items()}
    # Historical replay evidence is a heuristic, not calibrated future probability.
    winner=min(error,key=lambda name:float(error[name][mask].mean()) if mask.any() else 1e9)
    return {'available':bool(mask.any()),'seeds':n,'valid_per_horizon':mask.sum(0).tolist(),
        'reference':'continued cycle-checked LK; no GT, not tracking accuracy proof',
        'mean_error_px':{name:float(e[mask].mean()) if mask.any() else None for name,e in error.items()},
        'median_error_per_horizon_px':horizon,'chosen_simple_model':winner}


def gate_future(pred,simple,replay):
    """Fixed prespecified heuristic; labels and validation outcomes unavailable."""
    name=replay.get('chosen_simple_model','cv');ref=simple[name]
    errs=replay.get('median_error_per_horizon_px',{}).get(name,[None]*8)
    reliable=torch.tensor([v is not None and v<=2 for v in errs])
    return inside(pred)&inside(ref)&((pred-ref).norm(dim=-1)*127<=2)&reliable[None]
