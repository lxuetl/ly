"""Stage 2R: frozen native forecast plus zero-initialized future-query residual.
Validation MSE selects checkpoints (including epoch zero); LPIPS is report-only.
"""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lpips
from stage2_queries import QueryModel,BASE
from stage1_baseline import Videos,worker,save
from token_audit import encode,memory,image,metrics

class ResidualModel(QueryModel):
    def __init__(self,ck,arm):
        super().__init__(ck,arm)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        with torch.no_grad():
            # Match official in-place positional addition under AMP. An out-of-
            # place addition promotes fp16 to fp32 and changes the frozen path.
            z=self.base.to_patch_embedding(x)
            z+=self.base.pos_embedding.to(z.device)
            for block in self.base.blocks:z=block(z)
            b,t,p,d=z.shape
            native=self.base.mlp_head(z.reshape(-1,d)).reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)[:,:8]
        f=self.query(z)
        if self.adapter is not None:
            mem=memory(z)
            if self.arm=='null':mem=torch.zeros_like(mem)
            mem=mem[:,None].expand(-1,8,-1,-1)
            f=self.adapter(f,mem,torch.ones(mem.shape[:3],device=x.device,dtype=torch.bool))
        # No extra gate or clipping: keep attribution to residual-path change.
        return native+image(self,f)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['query','null','rgb'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=15);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    model=ResidualModel(ck,a.arm).cuda();net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    splits=json.loads((BASE/'split_paths.json').read_text())['paths'];out=Path(a.out)/a.arm;out.mkdir(parents=True,exist_ok=False)
    loaders={k:DataLoader(Videos(splits[k],BASE/'rgb_cache'),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,pin_memory=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    save(out/'protocol.json',{'arm':a.arm,'architecture':'frozen native forecast + future-query residual; zero last projection; no output clipping or extra gate','trainable':sum(p.numel() for p in params),'base_checkpoint':str(BASE/'best.pt'),'counts':{k:len(v) for k,v in splits.items()},'test_used':False,'geometry_used':False,'epochs':a.epochs,'batch':1,'seed':42,'lr':1e-4,'loss':'MSE only','selection':'validation MSE, epoch0 eligible','lpips':'AlexNet v0.1, clamp prediction[0,1], normalize[-1,1], report only','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def evaluate():
        model.eval();rows=[]
        with torch.inference_mode():
            for x,y,paths in loaders['val']:
                x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=model(x).float()
                r=metrics(p[0],y[0],net);r['path']=paths[0];rows.append(r)
        return rows,{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
    def checkpoint(epoch,summary):
        return {'model':{k:v for k,v in model.state_dict().items() if not k.startswith('base.')},'arm':a.arm,'epoch':epoch,'base_checkpoint':str(BASE/'best.pt'),'metrics':summary,'architecture':'residual_queries'}
    # Functional identity and frozen-base tests before any optimizer step.
    xx,_,_=loaders['val'].dataset[0];xx=xx[None].cuda();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        err=float((model(xx)-model.base(xx)[:,:8]).abs().max())
    assert err==0.,err
    rows,summary=evaluate();best=summary['mse'];save(out/'epoch0.json',summary)
    torch.save(checkpoint(0,summary),out/'best.pt');save(out/'best_validation.json',rows)
    save(out/'checks.json',{'initial_max_difference_from_native':err})
    start=time.time();history=[]
    for epoch in range(1,a.epochs+1):
        tick=time.time();model.train();total=0.
        for step,(x,y,_) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):p=model(x);loss=(p.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt)
            assert all(p.grad is None for p in model.base.parameters())
            nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();total+=float(loss.detach())
            if step%100==0:
                row={'epoch':epoch,'step':step,'steps':len(loaders['train']),'seconds':time.time()-tick};save(out/'progress.json',row);print(json.dumps(row),flush=True)
        rows,summary=evaluate();torch.save(checkpoint(epoch,summary),out/'last.pt')
        if summary['mse']<best:
            best=summary['mse'];torch.save(checkpoint(epoch,summary),out/'best.pt');save(out/'best_validation.json',rows)
        elapsed=time.time()-start
        row={'epoch':epoch,'train_mse':total/step,'validation':summary,'epoch_seconds':time.time()-tick,'elapsed_seconds':elapsed,'remaining_seconds':elapsed/epoch*(a.epochs-epoch)}
        history.append(row);save(out/'history.json',history);save(out/'progress.json',row);print(json.dumps(row),flush=True)
    save(out/'COMPLETE.json',{'best_val_mse':best,'seconds':time.time()-start,'test_used':False})
if __name__=='__main__':main()
