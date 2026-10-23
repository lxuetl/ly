"""Matched subset controls; no test data or future information in conditioning."""
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch,lpips
from torch.utils.data import DataLoader
from local_horizon import Model as Parent,Data,BASE,OLD
from stage1_baseline import save
from token_audit import metrics
RUN=Path('/root/autodl-tmp/predformer_subset_ablation_20261022')

class Model(Parent):
    def __init__(self,ck,arm,variant):
        super().__init__(ck,arm);self.variant=variant
        if variant=='no_decay':self.horizon_cap.fill_(1)
    def forward(self,x,g,mode='correct',shuffle_seed=42):
        if self.variant!='pre':return super().forward(x,g,mode=mode,shuffle_seed=shuffle_seed)
        prefix=self.encode(x);z=prefix;b,t,p,d=z.shape
        cap=self.history_cap(x).detach();self.last_cap_mean=float(cap.mean())
        self.last_outside_max=0.
        if mode!='bypass':
            mem=self.memory(prefix,g)
            if self.arm=='omega_shuffled':mem=mem[:,:,self.shuffle]
            if mode=='shuffle':
                idx=torch.randperm(16,generator=torch.Generator().manual_seed(shuffle_seed)).to(mem.device);mem=mem[:,:,idx]
            if mode=='zero':mem=torch.zeros_like(mem)
            mem=self.project(mem)+self.coord(self.spatial_pos.to(mem.dtype))[None]
            mem=mem.reshape(b,256,256)[:,None].expand(-1,8,-1,-1).reshape(b*8,256,256)
            q=z[:,:8];query=self.norm(q+self.horizon.weight[None,:,None,:])
            delta=self.read(query.reshape(b*8,p,d),mem,mem,attn_mask=self.local_bias.to(query.dtype),need_weights=False)[0].reshape(b,8,p,d)
            delta=delta*torch.sigmoid(self.gate(query))*cap[:,None]*self.horizon_cap[None,:,None,None]
            self.last_outside_max=float(delta.detach().masked_fill(cap[:,None]>0,0).abs().max())
            z=torch.cat([q+delta,z[:,8:]],1)
        z=self.base.blocks[-1](z)
        y=self.base.mlp_head(z.reshape(-1,d))
        return y.reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)[:,:8]

def prepare():
    RUN.mkdir(exist_ok=False)
    s=json.loads((OLD/'split_paths.json').read_text())['paths']
    rng=random.Random(20261022)
    chosen={'train':sorted(rng.sample(s['train'],128)), 'val':sorted(rng.sample(s['val'],64))}
    assert not set(chosen['train'])&set(chosen['val'])
    save(RUN/'subset.json',{'paths':chosen,'seed':20261022,'selection':'uniform clip sampling within existing locked train/val; exploratory subset, not full validation','test_used':False})

def main(variant,arm):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    out=RUN/(variant+'_'+arm);out.mkdir(exist_ok=False)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    model=Model(ck,arm,variant).cuda();net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    s=json.loads((RUN/'subset.json').read_text())['paths']
    loaders={k:DataLoader(Data(s[k],BASE/'rgb_cache','omega' if arm.startswith('omega') else arm),batch_size=1,shuffle=k=='train',num_workers=0,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    names={n for n,p in model.named_parameters() if p.requires_grad};params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    save(out/'protocol.json',{'variant':variant,'arm':arm,'epochs':15,'train_clips':128,'val_clips':64,'parameters':sum(p.numel() for p in params),'initial_checkpoint':str(BASE/'best.pt'),'train_loss':'MSE','selection':'val MSE including epoch0','lpips':'Alex clamp[0,1] then [-1,1]','subset_hash':hashlib.sha256((RUN/'subset.json').read_bytes()).hexdigest(),'code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_used':False,'pre_caveat':'only location changes; h embeddings remain on first8 latent slots before TSST, not guaranteed pure future slots; support bounds injection not final output','checkpoint':'trainable parameters only; reconstruct from initial_checkpoint and Model'})
    def evaluate(loader,mode='correct'):
        rows=[];model.eval()
        with torch.no_grad():
            for x,y,paths,g in loader:
                x=x.cuda();y=y.cuda();g=g.cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=model(x,g,mode=mode).float()[0]
                y=y[0];r=metrics(p,y,net);r['path']=paths[0]
                e=(p-y).square().mean(1);mask=(y-x[0,-1]).abs().mean(1)>.05
                r['late_mse']=float(e[4:].mean());r['late_lpips']=metrics(p[4:],y[4:],net)['lpips']
                for h in range(8):
                    r['h%d_mse'%(h+1)]=float(e[h].mean())
                    r['h%d_static_mse'%(h+1)]=float(e[h][~mask[h]].mean()) if (~mask[h]).any() else None
                rows.append(r)
        summary={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='path'}
        return rows,summary
    x,y,paths,g=next(iter(loaders['val']));x=x.cuda();g=g.cuda();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):identity=float((model(x,g)-model.base(x)[:,:8]).abs().max())
    assert identity==0,identity
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=(model(x,g).float()-y.cuda()).square().mean()
    loss.backward();assert model.read.out_proj.weight.grad is not None
    assert torch.isfinite(model.read.out_proj.weight.grad).all()
    assert all(p.grad is None for p in model.base.blocks[0].parameters())
    save(out/'checks.json',{'identity_error':identity,'prefix_frozen':True,'injection_gradient_finite':True,'outside_injection':model.last_outside_max})
    model.zero_grad(set_to_none=True);torch.manual_seed(2026)
    start=time.time();history=[];best=float('inf');best_epoch=0
    for epoch in range(16):
        tick=time.time();total=0
        if epoch:
            model.train()
            for x,y,paths,g in loaders['train']:
                x=x.cuda();y=y.cuda();g=g.cuda();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=(model(x,g).float()-y).square().mean()
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1);scaler.step(opt);scaler.update();total+=float(loss.detach())
                assert model.last_outside_max==0
        rows,summary=evaluate(loaders['val'])
        if summary['mse']<best:
            best=summary['mse'];best_epoch=epoch
            torch.save({'trainable':{n:p.detach().cpu() for n,p in model.named_parameters() if n in names},'epoch':epoch,'variant':variant,'arm':arm},out/'best_trainable.pt')
            save(out/'best_validation.json',rows)
        row={'epoch':epoch,'seconds':time.time()-tick,'elapsed':time.time()-start,'validation':summary,'train_mse':total/128 if epoch else None}
        history.append(row);save(out/'history.json',history);save(out/'progress.json',row);print(json.dumps(row),flush=True)
    weights=torch.load(out/'best_trainable.pt',map_location='cpu',weights_only=False)['trainable']
    missing,unexpected=model.load_state_dict(weights,strict=False);assert not unexpected
    sensitivity={}
    for split in ['train','val']:
        for mode in ['correct','shuffle','zero','bypass']:
            rows,summary=evaluate(loaders[split],mode);save(out/(split+'_'+mode+'_per_clip.json'),rows);sensitivity[split+'_'+mode]=summary
    save(out/'sensitivity.json',sensitivity)
    save(out/'COMPLETE.json',{'seconds':time.time()-start,'best_epoch':best_epoch,'metrics':sensitivity['val_correct'],'test_used':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--variant',choices=['reference','pre','no_decay']);p.add_argument('--arm',choices=['omega','rgb','null','omega_shuffled']);a=p.parse_args()
    if a.prepare:prepare()
    else:main(a.variant,a.arm)
