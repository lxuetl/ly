"""Controlled masks, finite-step MSE identity and factorial arithmetic."""
import torch
from reliability_cpu import inside

MAIN=['D_P','L_P','F_P','D_G','L_G','F_G']
ARMS=MAIN+['H_G','T_G','RGB_GT']


def common_support(old,identity,gt_history,gt_history_valid,oracle,oracle_visible):
    history=identity['memory_mask']&gt_history_valid&inside(gt_history)
    for uv in [old['uv'],identity['local'],identity['forward']]:history=history&inside(uv)
    cf=old['cf'];time=torch.exp((torch.arange(history.shape[1])-history.shape[1]+1)/4)
    w=cf*history*time;w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
    # Oracle visibility/coordinates are deliberately SHARED diagnostic controls.
    future=oracle_visible&inside(oracle)&inside(old['p'][0])&(history.sum(-1)>=3)[:,None]
    reliability=(cf*history).sum(-1)/history.sum(-1).clamp_min(1)
    return history,w,reliability[:,None]*future


def factor_effects(m):
    result={}
    for history in ['L','F']:
        result[history+'_history_at_P']={k:m[history+'_P'][k]-m['D_P'][k] for k in ['mse','lpips','temporal']}
        result[history+'_history_at_G']={k:m[history+'_G'][k]-m['D_G'][k] for k in ['mse','lpips','temporal']}
        result[history+'_interaction']={k:m[history+'_G'][k]-m[history+'_P'][k]-m['D_G'][k]+m['D_P'][k] for k in ['mse','lpips','temporal']}
    for history in ['D','L','F']:
        result[history+'_endpoint_G_minus_P']={k:m[history+'_G'][k]-m[history+'_P'][k] for k in ['mse','lpips','temporal']}
    return result


def direction_audit(step,rgb_gradient):
    dot=sum(float((s.double()*g.double()).sum()) for s,g in zip(step,rgb_gradient))
    ns=sum(float(s.double().square().sum()) for s in step)**.5
    ng=sum(float(g.double().square().sum()) for g in rgb_gradient)**.5
    return {'RGB_MSE_directional_derivative_latent':dot,
        'cosine_to_RGB_descent':-dot/max(ns*ng,1e-100)}


def mse_decomposition(pred,base,target):
    # Exact pixel-space identity, not a Taylor approximation of the decoder.
    delta=pred.double()-base.double();err=base.double()-target.double()
    linear=float((2*err*delta).mean());quadratic=float(delta.square().mean())
    exact=float(((err+delta).square()-err.square()).mean())
    return {'linear':linear,'quadratic':quadratic,'delta_MSE_double':exact,
        'identity_residual':exact-linear-quadratic,
        'classification':'wrong_pixel_direction' if linear>=0 else ('finite_step_overshoot' if exact>0 else 'MSE_descent')}
