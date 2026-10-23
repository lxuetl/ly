"""Label-only paired first/last audit, never used in normal decisions."""
import torch
from common import dump,manifest,key
from run_first_query_cpu import OUT,OLD,VideoForecastDataset,nullable_avg,mean_valid


def main():
    torch.set_num_threads(1);paths=manifest()['evaluation'];ds=VideoForecastDataset('','all',128,paths=paths,explicit_trajectory_interface=True);rows=[]
    for i,path in enumerate(paths):
        x=ds[i];old=torch.load(OLD/(key(path)+'_observed.pt'),weights_only=False);ids=old['ids'];gt=x['history_uv_gt'][ids];gv=x['history_valid_gt'][ids]
        first=(old['uv'][:,0]-gt[:,0]).norm(dim=-1)*127;last=(old['uv'][:,-1]-gt[:,-1]).norm(dim=-1)*127
        mask=gv[:,0]&gv[:,-1]&old['valid'][:,0]&old['valid'][:,-1]
        rows.append({'index':i,'path':path,'paired_points':int(mask.sum()),
            'first_EPE_px_same_points':nullable_avg(first,mask),'last_EPE_px_same_points':nullable_avg(last,mask),
            'new_gt2px_error':int((mask&(first<=2)&(last>2)).sum()),
            'gt2px_error_at_both':int((mask&(first>2)&(last>2)).sum()),
            'recovered_from_gt2px':int((mask&(first>2)&(last<=2)).sum())})
    dump(OUT/'paired_anchor_posthoc.json',{'label_only':True,'used_for_generation_or_selection':False,'videos':40,
        'first_EPE_mean_same_points':mean_valid([r['first_EPE_px_same_points'] for r in rows]),
        'last_EPE_mean_same_points':mean_valid([r['last_EPE_px_same_points'] for r in rows]),
        'pooled_counts':{k:sum(r[k] for r in rows) for k in ['paired_points','new_gt2px_error','gt2px_error_at_both','recovered_from_gt2px']},'per_video':rows})


if __name__=='__main__':main()
