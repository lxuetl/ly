"""Post-generation, label-only endpoint metrics and illustrative history overlays."""
import torch
from common import dump,manifest,key
from run_first_query_cpu import OUT,OLD,VideoForecastDataset,nullable_avg
from reliability_cpu import inside
from render_first_query_audit import render


def main():
    torch.set_num_threads(1);paths=manifest()['evaluation'][:4];ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True);rows=[]
    for i,path in enumerate(paths):
        x=ds[i];old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False);r=torch.load(OUT/(key(path)+'_identity.pt'),weights_only=False)
        ids=old['ids'];p=old['p'][0];shift=r['forward'][:,-1]-old['uv'][:,-1];q=p+shift[:,None]
        g=x['future_uv_raw'][ids];visible=x['future_visibility_gt'][ids]
        allowed=r['memory_mask'][:,-1]&old['valid'][:,-1]&(r['memory_mask'].sum(-1)>=3)
        mask=visible&inside(g)&inside(p)&inside(q)&allowed[:,None]
        rows.append({'index':i,'path':path,'common_GT_visible_writes':int(mask.sum()),
            'raw_endpoint_EPE_px':nullable_avg((p-g).norm(dim=-1)*127,mask),
            'translated_endpoint_EPE_px':nullable_avg((q-g).norm(dim=-1)*127,mask)})
    dump(OUT/'endpoint_posthoc.json',{'label_only':True,'used_for_generation_or_selection':False,
        'scope':'4 frozen cases, same eligible GT-visible in-bounds samples within each pair','cases':rows})
    render(0);render(1)


if __name__=='__main__':main()
