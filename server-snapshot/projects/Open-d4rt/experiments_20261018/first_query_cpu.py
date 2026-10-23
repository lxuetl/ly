"""First-query identity audit using observed RGB only; no labels or parameters."""
import torch
from reliability_cpu import gray_video,lk,inside,patches


def trace(gray,seed,start,stop):
    """Adjacent LK, cumulative validity: no resurrection after tracking failure."""
    h,w=gray[0].shape;factor=torch.tensor([w-1,h-1],dtype=seed.dtype)
    n=seed.shape[0];t=len(gray);uv=seed[:,None].expand(n,t,2).clone()
    ok=torch.zeros(n,t,dtype=torch.bool);cycle=torch.full((n,t),1e6)
    uv[:,start]=seed;ok[:,start]=inside(seed);cycle[:,start]=0
    direction=1 if stop>start else -1
    for j in range(start+direction,stop+direction,direction):
        q,status,e=lk(gray[j-direction],gray[j],(uv[:,j-direction]*factor).numpy())
        uv[:,j]=torch.from_numpy(q)/factor;cycle[:,j]=torch.from_numpy(e).float()
        ok[:,j]=ok[:,j-direction]&torch.from_numpy(status)&(cycle[:,j]<=1)&inside(uv[:,j])
    return uv,ok,cycle


def first_query_identity(rgb,query,raw_uv,raw_valid):
    """Three tracks: query-forward; forward-end-backward; D4RT-end-backward.

    The provided frame0 query is the inference input, not a GT replacement.
    The first and third tracks test different hypotheses: a forward/backward
    loop can be self-consistent even when the D4RT endpoint has changed identity.
    """
    gray=gray_video(rgb);factor=torch.tensor([rgb.shape[-1]-1,rgb.shape[-2]-1])
    f,fo,fc=trace(gray,query,0,len(gray)-1)
    b,bo,bc=trace(gray,f[:,-1],len(gray)-1,0)
    d,do,dc=trace(gray,raw_uv[:,-1],len(gray)-1,0)
    texture=patches(rgb,query[:,None].expand(-1,rgb.shape[0],-1))[:,0].std(-1,unbiased=False)>.02
    loop=(f-b).mul(factor).norm(dim=-1)
    return_error=(b[:,0]-query).mul(factor).norm(dim=-1)
    complete=fo[:,-1]&bo[:,0]&(return_error<=2)&texture&inside(query)
    # Original D4RT validity is preserved across all comparisons.
    memory_mask=raw_valid&fo&bo&(loop<=2)&complete[:,None]
    raw_gap=(raw_uv-f).mul(factor).norm(dim=-1)
    tail_gap=raw_gap[:,-1]
    d_return=(d[:,0]-query).mul(factor).norm(dim=-1)
    proxy_available=complete&raw_valid[:,-1]&do[:,0]
    proxy_drift=proxy_available&((tail_gap>2)|(d_return>2))
    identity_mask=memory_mask&do&(raw_gap<=2)&((f-d).mul(factor).norm(dim=-1)<=2)
    identity_mask=identity_mask&proxy_available[:,None]&~proxy_drift[:,None]
    delta=(f-raw_uv)*factor;length=delta.norm(dim=-1,keepdim=True)
    local=raw_uv+delta*(2/length.clamp_min(1e-12)).clamp(max=1)/factor
    local=torch.where(memory_mask[...,None],local,raw_uv)
    return {'forward':f,'forward_valid':fo,'reverse_from_forward':b,'reverse_from_D4RT':d,
        'reverse_D4RT_valid':do,'memory_mask':memory_mask,'identity_mask':identity_mask,
        'local':local,'proxy_available':proxy_available,'proxy_drift':proxy_drift,
        'tail_disagreement_px':tail_gap,'D4RT_return_error_px':d_return,
        'forward_loop_error_px':loop,'forward_return_error_px':return_error,
        'first_texture_ok':texture,'cycles':{'forward':fc,'forward_reverse':bc,'D4RT_reverse':dc}}
