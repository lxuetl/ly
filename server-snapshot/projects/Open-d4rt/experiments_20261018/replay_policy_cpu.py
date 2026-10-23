"""Causal RGB-prefix track reliability and discrete accept/keep rules."""
import hashlib
import cv2
import torch
from first_query_cpu import trace
from reliability_cpu import gray_video,inside,alternatives,patches

ALPHAS=(.125,.25,.5)
MOTIONS=('hold','cv')


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def padded_prefix(rgb):
    if len(rgb)>16 or len(rgb)<8:raise ValueError('prefix must contain8..16 observed frames')
    return torch.cat([rgb[:1].expand(16-len(rgb),-1,-1,-1),rgb],0).clone()


def key(motion,alpha):return motion+'_a'+str(alpha)


def improves(candidate,baseline):
    # Prespecified margins, not estimated from actual future labels.
    return (candidate['mse'] < baseline['mse']-max(1e-8,1e-4*baseline['mse'])
        and candidate['lpips'] < baseline['lpips']-1e-5)


def select(calibration,baseline,support):
    eligible=[(a,calibration[key(m,a)]['lpips'],calibration[key(m,a)]['mse'],key(m,a))
        for m in MOTIONS for a in ALPHAS if support[m]>0 and improves(calibration[key(m,a)],baseline)]
    return min(eligible)[-1] if eligible else 'keep'


def bounded_output(base,raw,alpha):
    out=base.clone();out[:4]=base[:4]+alpha*(raw[:4]-base[:4])
    return out


def track_prefix(rgb,allowed=None):
    """All inputs belong to current observed prefix; no D4RT/label/cache inputs.

    Inner endpoint backtest uses first(T-4) causal forward positions to forecast
    the last4 ALREADY OBSERVED positions. Future failures cannot resurrect tracks.
    """
    gray=gray_video(rgb);h,w=gray[0].shape
    corners=cv2.goodFeaturesToTrack(gray[0],maxCorners=64,qualityLevel=.01,minDistance=5)
    if corners is None:seed=torch.tensor([[.5,.5]])
    else:seed=torch.from_numpy(corners[:,0]).float()/torch.tensor([w-1,h-1])
    uv,valid,cycle=trace(gray,seed,0,len(rgb)-1)
    back,backvalid,_=trace(gray,uv[:,-1],len(rgb)-1,0)
    loop=(uv-back).mul(torch.tensor([w-1,h-1])).norm(dim=-1)
    texture=patches(rgb,seed[:,None].expand(-1,len(rgb),-1))[:,0].std(-1,unbiased=False)>.02
    complete=valid[:,-1]&backvalid[:,0]&(loop[:,0]<=2)&texture
    history=valid&backvalid&(loop<=2)&complete[:,None]
    cf=torch.exp(-cycle.clamp(max=100))*history
    tw=torch.exp((torch.arange(len(rgb))-len(rgb)+1)/4)
    hw=cf*tw;hw=hw/hw.sum(-1,keepdim=True).clamp_min(1e-12)
    endpoints=alternatives(uv,valid,4)
    cut=len(rgb)-4
    past_prediction=alternatives(uv[:,:cut],valid[:,:cut],4)
    # No complete-track or future-derived mask enters the inner forecast inputs.
    reference=uv[:,cut:];reference_valid=valid[:,cut:]&valid[:,cut-1:cut]
    weights={};signals={};raw_weights={}
    reliability=cf.sum(-1)/history.sum(-1).clamp_min(1)
    for motion in MOTIONS:
        error=(past_prediction[motion]-reference).mul(torch.tensor([w-1,h-1])).norm(dim=-1)
        reliable=reference_valid&(error<=2)
        raw=reliability[:,None]*(history.sum(-1)>=3)[:,None]*inside(endpoints[motion])
        weights[motion]=raw*reliable
        if allowed is not None:
            assert allowed[motion].shape==weights[motion].shape,'frame0 seed identity changed'
            weights[motion]=weights[motion]*allowed[motion]
        raw_weights[motion]=raw
        signals[motion]={'inner_observed_reference_samples':int(reference_valid.sum()),
            'inner_error_median_px':float(error[reference_valid].median()) if reference_valid.any() else None,
            'eligible_writes':int((weights[motion]>0).sum()),'ungated_writes':int((raw>0).sum())}
    return {'uv':uv,'history':history,'history_weights':hw,'endpoints':endpoints,
        'weights':weights,'raw_weights':raw_weights,'signals':signals,
        'prefix_sha256':digest(rgb),'seed_sha256':digest(seed),'prefix_frames':len(rgb)}
