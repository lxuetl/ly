"""Synthetic GPU installation check, not a dataset training run."""
import json,time,sys
from pathlib import Path
import torch
import openstl
from openstl.models import PredFormer_Model
torch.set_num_threads(2)
torch.manual_seed(42)
cfg=dict(height=128,width=128,num_channels=3,pre_seq=16,after_seq=8,
         patch_size=8,dim=256,heads=8,dim_head=32,dropout=.1,attn_dropout=.1,
         drop_path=.1,scale_dim=4,depth=1,Ndepth=3)
start=time.time()
model=PredFormer_Model(model_config=cfg).cuda().train()
opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
x=torch.rand(1,16,3,128,128,device='cuda')
target=torch.rand(1,8,3,128,128,device='cuda')
scaler=torch.amp.GradScaler('cuda')
with torch.autocast('cuda',dtype=torch.float16):
    raw=model(x)
    # Match official method for aft_seq_length < pre_seq_length.
    pred=raw[:,:8]
    loss=(pred.float()-target).square().mean()
assert pred.shape==target.shape and torch.isfinite(loss)
scaler.scale(loss).backward();scaler.unscale_(opt)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
scaler.step(opt);scaler.update();torch.cuda.synchronize()
result=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(),package=openstl.__file__,config=cfg,
    parameters=sum(p.numel() for p in model.parameters()),input=list(x.shape),raw_output=list(raw.shape),prediction=list(pred.shape),
    synthetic_loss=float(loss.detach()),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
    seconds=time.time()-start,forward_backward_optimizer='passed',pretrained_weights=False)
Path('installation_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
