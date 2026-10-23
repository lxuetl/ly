"""Read-only scorer feature-range audit and transport numerical invariants."""
import torch
import numpy as np
from common import OUT,manifest,key,dump
from diagnose_interfaces import DEST,stats
from trajectory_attention import sample,splat,TrackCrossAttention

def main():
    torch.set_num_threads(4);m=manifest();rows={sp:[] for sp in ['fit','dev','calibration','evaluation']}
    ck=torch.load(OUT/'local_risk/local_best.pt',map_location='cpu',weights_only=False)['model'];mean=ck['mean'];scale=ck['scale']
    for sp in rows:
        for p in m[sp]:
            d=torch.load(OUT/'local_risk/cache'/(key(p)+'.pt'),map_location='cpu',weights_only=False)
            rows[sp].append({'path':p,'group':d['group'],'global':d['global'],'local':d['local'].float()})
    fit=torch.cat([r['global'] for r in rows['fit']]);lo=fit.min(0).values;hi=fit.max(0).values
    result={}
    for sp,rr in rows.items():
        result[sp]=[{'path':r['path'],'group':r['group'],'max_abs_standardized_global':float(((r['global']-mean)/scale).abs().max()),'global_outside_fit_range_fraction':float(((r['global']<lo)|(r['global']>hi)).float().mean()),'local_abs':float(r['local'].abs().mean()),'local_abs_max':float(r['local'].abs().max())} for r in rr]
    # Coordinate order/align_corners: encoder ramp sampled at known normalized UV.
    yy,xx=torch.meshgrid(torch.linspace(0,1,7),torch.linspace(0,1,9),indexing='ij')
    feature=torch.stack([xx,yy])[None,None].repeat(1,3,1,1,1)
    uv=torch.tensor([[[[.2,.8],[.4,.6],[.9,.1]]]])
    assert torch.allclose(sample(feature,uv),uv,atol=1e-6)
    vals=torch.ones(1,1,2,3);coord=torch.tensor([[[[.5,.5],[2.,.5]]]])
    raster,sup=splat(vals,coord,torch.ones(1,1,2),9,9)
    assert torch.equal(raster[:,1],torch.zeros_like(raster[:,1]))
    assert torch.allclose(raster[:,0],sup[:,0].expand_as(raster[:,0]),atol=1e-6)
    model=TrackCrossAttention(3);model.out.bias.data.fill_(1.)
    hidden=torch.zeros(1,2,3,9,9);mem=torch.ones(1,1,3,3);hu=torch.full((1,1,3,2),.5)
    invalid=torch.zeros(1,1,3,dtype=torch.bool)
    h,s=model(hidden,mem,hu,invalid,torch.ones(1,1,3),coord)
    assert torch.equal(h,hidden) and not s.any()
    dump(DEST/'feature_ranges.json',result)
    dump(DEST/'numerical_checks.json',{'coordinate_xy_align_corners_ramp':True,'out_of_bounds_splat_zero':True,'constant_feature_support_consistency':True,'all_history_invalid_exact_identity_even_output_bias':True})
    print('RANGE_AND_NUMERICAL_CHECKS_COMPLETE',flush=True)

if __name__=='__main__':main()
