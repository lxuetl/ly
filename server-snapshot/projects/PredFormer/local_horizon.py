"""Future-feature-local, history-capped spatial injection. No future mask input."""
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np,torch,lpips
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from internal_geometry import Model as Parent,Data,RUN as OLD,BASE
from stage1_baseline import worker,save
from token_audit import metrics
RUN=Path('/root/autodl-tmp/predformer_local_horizon_20261022')

class Model(Parent):
    def __init__(self,ck,arm):
        super().__init__(ck,'omega' if arm.startswith('omega') else arm);self.arm=arm
        self.horizon=nn.Embedding(8,256);nn.init.normal_(self.horizon.weight,std=.02)
        self.gate=nn.Linear(256,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        q=(torch.arange(16)+.5)/16;k=(torch.arange(4)+.5)/4
        qy,qx=torch.meshgrid(q,q,indexing='ij');ky,kx=torch.meshgrid(k,k,indexing='ij')
        distance=torch.cdist(torch.stack([qx.flatten(),qy.flatten()],1),torch.stack([kx.flatten(),ky.flatten()],1)).square()
        allowed=torch.zeros(256,16,dtype=torch.bool).scatter_(1,distance.topk(4,largest=False).indices,True)
        bias=(-distance/(2*.2**2)).masked_fill(~allowed,float('-inf')).repeat(1,16)
        self.register_buffer('local_bias',bias)
        self.register_buffer('spatial_pos',self.pos[:,:16].clone())
        self.register_buffer('shuffle',torch.randperm(16,generator=torch.Generator().manual_seed(123)))
        self.register_buffer('horizon_cap',1/torch.sqrt(torch.arange(1,9).float()))
    def history_cap(self,x):
        # Fixed observed-frame change proxy, not a true object-motion segmentation.
        change=(x[:,9:16]-x[:,8:15]).abs().mean(2).amax(1,keepdim=True)
        change=F.avg_pool2d(change,8);change=F.max_pool2d(change,3,1,1).flatten(2).transpose(1,2)
        return ((change-.01)/.04).clamp(0,1)
    def memory(self,z,g):
        b,t,p,d=z.shape
        if self.arm.startswith('omega'):return g[:,:,:16].to(z.dtype)
        grid=F.adaptive_avg_pool2d(z.reshape(b*t,16,16,d).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2).reshape(b,t,16,d).repeat(1,1,1,8)
        return torch.zeros_like(grid) if self.arm=='null' else grid
    def forward(self,x,g,mode='correct',external=None,shuffle_seed=42):
        prefix=self.encode(x);z=self.base.blocks[-1](prefix);b,t,p,d=z.shape
        cap=self.history_cap(x).detach();self.last_cap_mean=float(cap.detach().mean())
        if mode!='bypass':
            mem=self.memory(prefix,g) if external is None else external
            if self.arm=='omega_shuffled' and mode!='unshuffle':mem=mem[:,:,self.shuffle]
            if mode=='shuffle':
                idx=torch.randperm(16,generator=torch.Generator().manual_seed(shuffle_seed)).to(mem.device);mem=mem[:,:,idx]
            if mode=='zero':mem=torch.zeros_like(mem)
            mem=self.project(mem)+self.coord(self.spatial_pos.to(mem.dtype))[None]
            mem=mem.reshape(b,256,256);q=z[:,:8]
            h=self.horizon.weight[None,:,None,:]
            query=self.norm(q+h)
            mem=mem[:,None].expand(-1,8,-1,-1).reshape(b*8,256,256)
            delta=self.read(query.reshape(b*8,p,d),mem,mem,attn_mask=self.local_bias.to(query.dtype),need_weights=False)[0].reshape(b,8,p,d)
            gate=torch.sigmoid(self.gate(query))*cap[:,None]*self.horizon_cap[None,:,None,None]
            delta=delta*gate
            # The injection is exactly zero outside the historical support, per patch.
            self.last_outside_max=float(delta.detach().masked_fill(cap[:,None]>0,0).abs().max())
            z=torch.cat([q+delta,z[:,8:]],1)
        y=self.base.mlp_head(z.reshape(-1,d))
        return y.reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)[:,:8]

def main(arm):
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    out=RUN/arm;out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);model=Model(ck,arm).cuda()
    net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    splits=json.loads((OLD/'split_paths.json').read_text())['paths']
    loaders={k:DataLoader(Data(splits[k],BASE/'rgb_cache','omega' if arm.startswith('omega') else arm),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');torch.manual_seed(2026)
    save(out/'protocol.json',{'arm':arm,'trainable':sum(p.numel() for p in params),'base':str(BASE/'best.pt'),'epochs':15,'batch':1,'lr':1e-4,'seed':42,'train':len(splits['train']),'val':len(splits['val']),'test_used':False,'loss':'MSE only','selection':'val MSE incl epoch0','location':'after trainable TSST3, before native head; first8 native forecast slots','tokens':'16 historical frames x16 spatial tokens; camera removed; learned 2048->256 projection','locality':'4 nearest 4x4 memory locations per query over all16 observed frames','history_cap':'max of last7 observed adjacent RGB differences; 8x8 average pool then3x3 dilation; clamp((change-.01)/.04,0,1)','horizon':'learned h1..h8 embedding in Q/gate; sigmoid gate capped by history_cap/sqrt(h)','caveat':'restriction applies to geometry delta only, not updates caused by trainable backbone/head; RGB change is not object motion truth','shuffle_control':'fixed seed123 content-only spatial permutation held across time during train+val','lpips':'Alex v0.1 clamp[0,1], normalize[-1,1]','code_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def evaluate():
        model.eval();rows=[]
        with torch.no_grad():
            for x,y,paths,g in loaders['val']:
                x=x.cuda();y=y.cuda();g=g.cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=model(x,g).float()
                r=metrics(p[0],y[0],net);r.update(path=paths[0],history_cap_mean=model.last_cap_mean);rows.append(r)
        return rows,{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
    def state(epoch,s):return {'model':model.state_dict(),'epoch':epoch,'metrics':s,'arm':arm,'model_config':ck['model_config']}
    x,y,p,g=next(iter(loaders['val']));x=x.cuda();y=y.cuda();g=g.cuda();model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):err=float((model(x,g)-model.base(x)[:,:8]).abs().max())
    assert err==0,err
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=(model(x,g).float()-y).square().mean()
    loss.backward();assert all(p.grad is None for p in model.base.blocks[0].parameters());assert model.read.out_proj.weight.grad is not None
    assert torch.isfinite(model.read.out_proj.weight.grad).all();model.zero_grad(set_to_none=True)
    save(out/'checks.json',{'identity_max_error':err,'history_input_frames':16,'geometry_delta_outside_support_max':model.last_outside_max,'prefix_frozen':True})
    rows,s=evaluate();best=s['mse'];torch.save(state(0,s),out/'best.pt');save(out/'best_validation.json',rows);save(out/'epoch0.json',s)
    start=time.time();history=[]
    for epoch in range(1,16):
        model.train();tick=time.time();total=0
        for step,(x,y,p,g) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();g=g.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x,g);loss=(pred.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();total+=float(loss.detach())
            assert model.last_outside_max==0
            if step%100==0:
                r={'epoch':epoch,'step':step,'steps':len(loaders['train']),'seconds':time.time()-tick};save(out/'progress.json',r);print(json.dumps(r),flush=True)
        rows,s=evaluate();torch.save(state(epoch,s),out/'last.pt')
        if s['mse']<best:best=s['mse'];torch.save(state(epoch,s),out/'best.pt');save(out/'best_validation.json',rows)
        r={'epoch':epoch,'train_mse':total/step,'validation':s,'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-start};history.append(r);save(out/'history.json',history);save(out/'progress.json',r);print(json.dumps(r),flush=True)
    bestck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(bestck['model']);model.eval();rows=[]
    ds=loaders['val'].dataset
    with torch.no_grad():
        for i in range(len(ds)):
            x,y,p,g=ds[i];x=x[None].cuda();y=y.cuda();g=g[None].cuda()
            for mode,seed in [('correct',42),('shuffle',42),('shuffle',123),('shuffle',3407),('zero',42),('bypass',42),('unshuffle',42)]:
                with torch.autocast('cuda',dtype=torch.float16):pred=model(x,g,mode,shuffle_seed=seed).float()[0]
                if mode=='correct':reference=pred
                r=metrics(pred,y,net);r.update(path=p,mode=mode+('_'+str(seed) if mode=='shuffle' else ''),output_l1_vs_correct=float((pred-reference).abs().mean()));rows.append(r)
    save(out/'sensitivity_per_clip.json',rows)
    save(out/'sensitivity.json',{mode:{k:float(np.mean([r[k] for r in rows if r['mode']==mode])) for k in ['mse','mae','psnr','lpips','temporal','output_l1_vs_correct']} for mode in sorted({r['mode'] for r in rows})})
    save(out/'COMPLETE.json',{'seconds':time.time()-start,'best_epoch':bestck['epoch'],'metrics':bestck['metrics'],'test_used':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['omega','rgb','null','omega_shuffled'],required=True);a=ap.parse_args();main(a.arm)
