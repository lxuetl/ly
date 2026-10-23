"""Native PredFormer: geometry before TSST3; train TSST3/head + adapter only."""
import argparse,json,time,hashlib,random
from pathlib import Path
import torch,numpy as np,lpips
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from openstl.models import PredFormer_Model
from stage1_baseline import Videos,worker,save
from stage2_queries import BASE
from geometry_experiment import OMEGA,positions,key
from token_audit import metrics
RUN=Path('/root/autodl-tmp/predformer_internal_20261022')
OLD=Path('/root/autodl-tmp/predformer_geometry_20261022')

def prepare():
    RUN.mkdir(exist_ok=False);(RUN/'omega_tokens').mkdir()
    splits=json.loads((BASE/'split_paths.json').read_text());save(RUN/'split_paths.json',splits)
    save(RUN/'manifest.json',json.loads((OLD/'manifest.json').read_text()))
    paths=splits['paths']['train']+splits['paths']['val'];start=time.time()
    for i,p in enumerate(paths):
        pp=Path(p);q=torch.load(OMEGA/pp.parent.name/(pp.stem+'.pt'),map_location='cpu',weights_only=True)
        spatial=F.adaptive_avg_pool2d(q['patch_tokens'].float(),(4,4)).flatten(2).transpose(1,2)
        tokens=torch.cat([spatial,q['tokens'][:,:1].float()],1).half()
        assert tokens.shape==(16,17,2048) and torch.isfinite(tokens).all()
        torch.save(tokens,RUN/'omega_tokens'/(key(p)+'.pt'))
        if (i+1)%100==0:print('prepare',i+1,len(paths),time.time()-start,flush=True)
    save(RUN/'PREPARED.json',{'clips':len(paths),'seconds':time.time()-start,'input':'context-only raw Omega spatial+camera tokens; no random feature projection; 4x4 spatial pooling retained'})

class Data(Videos):
    def __init__(self,paths,cache,arm):super().__init__(paths,cache);self.arm=arm
    def __getitem__(self,i):
        x,y,p=super().__getitem__(i)
        g=torch.load(RUN/'omega_tokens'/(key(p)+'.pt'),weights_only=True) if self.arm=='omega' else torch.zeros(1)
        return x,y,p,g

class Model(nn.Module):
    def __init__(self,ck,arm):
        super().__init__();self.arm=arm;self.base=PredFormer_Model(model_config=ck['model_config']);self.base.load_state_dict(ck['model']);self.base.requires_grad_(False)
        assert len(self.base.blocks)==3
        self.base.blocks[-1].requires_grad_(True);self.base.mlp_head.requires_grad_(True)
        if arm!='plain':
            self.project=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.GELU(),nn.Linear(256,256))
            self.coord=nn.Linear(4,256);self.norm=nn.LayerNorm(256);self.read=nn.MultiheadAttention(256,8,batch_first=True)
            nn.init.zeros_(self.read.out_proj.weight);nn.init.zeros_(self.read.out_proj.bias)
            self.register_buffer('pos',positions().reshape(16,17,4))
        self.train()
    def train(self,mode=True):
        super().train(mode);self.base.to_patch_embedding.eval()
        for block in self.base.blocks[:-1]:block.eval()
        return self
    def encode(self,x):
        with torch.no_grad():
            z=self.base.to_patch_embedding(x);z+=self.base.pos_embedding.to(z.device)
            for block in self.base.blocks[:-1]:z=block(z)
        return z
    def memory(self,z,g):
        if self.arm=='plain':return None
        b,t,p,d=z.shape
        if self.arm=='omega':return g.to(z.dtype)
        grid=F.adaptive_avg_pool2d(z.reshape(b*t,16,16,d).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2).reshape(b,t,16,d)
        mem=torch.cat([grid,z.mean(2,keepdim=True)],2).repeat(1,1,1,8)
        return torch.zeros_like(mem) if self.arm=='null' else mem
    def forward(self,x,g,mode='correct',external=None):
        z=self.encode(x);b,t,p,d=z.shape
        if self.arm!='plain' and mode!='bypass':
            mem=self.memory(z,g) if external is None else external
            if mode=='zero':mem=torch.zeros_like(mem)
            mem=self.project(mem)+self.coord(self.pos.to(mem.dtype))[None]
            # Same-frame spatial/camera reading, temporal integration in trainable TSST3.
            delta=self.read(self.norm(z.reshape(b*t,p,d)),mem.reshape(b*t,17,d),mem.reshape(b*t,17,d),need_weights=False)[0]
            z=z+delta.reshape(b,t,p,d)
        z=self.base.blocks[-1](z)
        y=self.base.mlp_head(z.reshape(-1,d))
        return y.reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)[:,:8]

def train(arm):
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    out=RUN/arm;out.mkdir(exist_ok=False);ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    model=Model(ck,arm).cuda();net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    splits=json.loads((RUN/'split_paths.json').read_text())['paths']
    loaders={k:DataLoader(Data(splits[k],BASE/'rgb_cache',arm),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    # Reset training RNG after model/metric construction: same dropout stream at start.
    torch.manual_seed(2026)
    protocol={'arm':arm,'trainable':sum(p.numel() for p in params),'base':str(BASE/'best.pt'),'unfrozen':['TSST3','native prediction head'],'injection':'before TSST3, same-frame cross-attention; no future query/residual image branch','adapter':'same 2048->256 learned projection in omega/rgb/null; RGB features repeated eight times, null zeros','epochs':15,'lr':1e-4,'batch':1,'loss':'MSE','selection':'val MSE including epoch0','train':len(splits['train']),'val':len(splits['val']),'test_used':False,'lpips':'Alex v0.1, clamp[0,1], normalize[-1,1]','source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out/'protocol.json',protocol)
    def evaluate():
        model.eval();rows=[]
        with torch.no_grad():
            for x,y,paths,g in loaders['val']:
                x=x.cuda();y=y.cuda();g=g.cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=model(x,g).float()
                r=metrics(p[0],y[0],net);r['path']=paths[0];rows.append(r)
        return rows,{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
    def state(epoch,s):return {'model':model.state_dict(),'epoch':epoch,'metrics':s,'arm':arm,'model_config':ck['model_config']}
    x,y,p,g=next(iter(loaders['val']));model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):err=float((model(x.cuda(),g.cuda())-model.base(x.cuda())[:,:8]).abs().max())
    assert err==0,err
    # Backward-only smoke: must reach the last block/head, never earlier layers.
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):p=model(x.cuda(),g.cuda());loss=(p.float()-y.cuda()).square().mean()
    loss.backward();assert all(p.grad is None for p in model.base.blocks[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.base.blocks[-1].parameters())
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    model.zero_grad(set_to_none=True);save(out/'checks.json',{'initial_identity_max_error':err,'frozen_prefix_gradients_absent':True,'last_block_gradients_present':True})
    rows,s=evaluate();best=s['mse'];torch.save(state(0,s),out/'best.pt');save(out/'best_validation.json',rows);save(out/'epoch0.json',s)
    start=time.time();history=[]
    for epoch in range(1,16):
        model.train();tick=time.time();total=0
        for step,(x,y,p,g) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();g=g.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x,g);loss=(pred.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();total+=float(loss.detach())
            if step%100==0:
                r={'epoch':epoch,'step':step,'steps':862,'seconds':time.time()-tick};save(out/'progress.json',r);print(json.dumps(r),flush=True)
        rows,s=evaluate();torch.save(state(epoch,s),out/'last.pt')
        if s['mse']<best:best=s['mse'];torch.save(state(epoch,s),out/'best.pt');save(out/'best_validation.json',rows)
        r={'epoch':epoch,'train_mse':total/step,'validation':s,'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-start};history.append(r);save(out/'history.json',history);save(out/'progress.json',r);print(json.dumps(r),flush=True)
    ckbest=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckbest['model']);model.eval()
    manifest=json.loads((RUN/'manifest.json').read_text());ds=loaders['val'].dataset;rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            x,y,p,g=ds[i];x=x[None].cuda();y=y.cuda();g=g[None].cuda()
            ids=[(i+k)%len(ds) for k in range(1,len(ds)) if manifest[ds.paths[(i+k)%len(ds)]]['source']!=manifest[p]['source']]
            j=next((j for j in ids if manifest[ds.paths[j]]['subset']==manifest[p]['subset']),ids[0])
            dx,_,dp,dg=ds[j]
            with torch.autocast('cuda',dtype=torch.float16):external=model.memory(model.encode(dx[None].cuda()),dg[None].cuda())
            for mode in ['correct','mismatch','zero','bypass']:
                with torch.autocast('cuda',dtype=torch.float16):pred=model(x,g,mode,external if mode=='mismatch' else None).float()
                if mode=='correct':reference=pred
                r=metrics(pred[0],y,net);r.update(path=p,source=manifest[p]['source'],donor=dp,mode=mode,output_l1_vs_correct=float((pred-reference).abs().mean()));rows.append(r)
    save(out/'sensitivity_per_clip.json',rows)
    save(out/'sensitivity.json',{mode:{k:float(np.mean([r[k] for r in rows if r['mode']==mode])) for k in ['mse','mae','psnr','lpips','temporal','output_l1_vs_correct']} for mode in ['correct','mismatch','zero','bypass']})
    save(out/'COMPLETE.json',{'seconds':time.time()-start,'best_epoch':ckbest['epoch'],'metrics':ckbest['metrics'],'test_used':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=['plain','rgb','null','omega']);a=ap.parse_args();torch.set_num_threads(2)
    if a.prepare:prepare()
    else:train(a.arm)
