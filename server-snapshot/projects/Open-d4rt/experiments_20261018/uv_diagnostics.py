"""No-fit UV baselines with explicit timestamps. No change to production model."""
import torch

def last_index(valid):
    t=valid.shape[-1]
    return (valid.long()*torch.arange(1,t+1,device=valid.device)).argmax(-1)

def gather_time(values,index):
    return values.gather(2,index[...,None,None].expand(-1,-1,1,values.shape[-1])).squeeze(2)

def uv_baselines(uv,valid,k=8):
    # Original model's validity contract; caller separately audits invalid UV.
    clean=torch.nan_to_num(uv,nan=0.,posinf=0.,neginf=0.)
    idx=last_index(valid);anchor=gather_time(clean,idx)
    pair=valid[:,:,1:]&valid[:,:,:-1]
    recent=pair.clone()
    if recent.shape[-1]>4:recent[:,:,:-4]=False
    velocity=torch.nan_to_num(torch.nanmedian((clean[:,:,1:]-clean[:,:,:-1]).masked_fill(~recent[...,None],float('nan')),dim=2).values).clamp(-.08,.08)
    steps=torch.arange(1,k+1,device=uv.device,dtype=uv.dtype)[None,None,:,None]
    elapsed=(uv.shape[2]-1-idx)[...,None,None]+steps
    older=valid & (torch.arange(uv.shape[2],device=uv.device)[None,None]<idx[...,None])
    prev=last_index(older);previous=gather_time(clean,prev)
    dt=(idx-prev).clamp_min(1).to(uv.dtype)
    last_velocity=(anchor-previous)/dt[...,None]
    last_velocity=torch.where(older.any(-1)[...,None],last_velocity,torch.zeros_like(last_velocity)).clamp(-.08,.08)
    hold=anchor[:,:,None].expand(-1,-1,k,-1)
    return {'hold':hold,'cv_model':(hold+velocity[:,:,None]*steps).clamp(0,1),
            'cv_elapsed':hold+velocity[:,:,None]*elapsed,'cv_last2':hold+last_velocity[:,:,None]*elapsed}, \
           {'index':idx,'anchor':anchor,'velocity':velocity,'last2_velocity':last_velocity,'age':uv.shape[2]-1-idx,
            'recent_valid_pairs':recent.sum(-1),'any_valid':valid.any(-1)}

def cached_model_parts(model,head_output,uv,valid):
    bases,info=uv_baselines(uv,valid,model.future_frames)
    steps=torch.arange(1,model.future_frames+1,device=uv.device,dtype=uv.dtype)[None,None,:,None]
    unbounded_baseline=info['anchor'][:,:,None]+info['velocity'][:,:,None]*steps
    residual=model.residual_scale*torch.tanh(head_output.reshape(*uv.shape[:2],model.future_frames,2))
    return unbounded_baseline+residual,residual,info
