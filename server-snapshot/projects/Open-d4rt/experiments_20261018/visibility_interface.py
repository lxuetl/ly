"""Future visibility is separate from observed reliability and UV bounds."""
import torch

def future_gate(uv, visibility=None, *, source='unknown', oracle_diagnostic=False):
    if source not in ('unknown','predicted','oracle_gt'):
        raise ValueError('visibility source must be explicit')
    if source=='oracle_gt' and not oracle_diagnostic:
        raise ValueError('future GT visibility is label-only; enable oracle diagnostic explicitly')
    if (visibility is None)!=(source=='unknown'):
        raise ValueError('unknown has no visibility tensor; known sources require a tensor')
    inside=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<=1).all(-1)
    gate=inside.to(uv.dtype)
    if visibility is not None:
        if visibility.shape!=uv.shape[:-1]:raise ValueError('visibility/UV shape mismatch')
        gate=gate*torch.nan_to_num(visibility.to(uv.dtype),nan=0.,posinf=0.,neginf=0.).clamp(0,1)
    return gate
