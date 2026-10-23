"""Matched-capacity historical-token experiments. No future geometry is read."""
import argparse,json,time,random,hashlib,re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lpips
from stage1_baseline import Videos,ROOT,worker,save
from stage2_queries import BASE,QueryModel
from token_audit import metrics,image

RUN=Path('/root/autodl-tmp/predformer_geometry_20261022')
OMEGA=Path('/root/autodl-fs/d4rt_datasets/vggt_omega_spatial_v4_20261002')
def key(p):return hashlib.sha256(p.encode()).hexdigest()[:24]
def positions():
    v=(torch.arange(4)+.5)/4;yy,xx=torch.meshgrid(v,v,indexing='ij')
    uv=torch.cat([torch.stack([xx.flatten(),yy.flatten()],-1),torch.tensor([[.5,.5]])])
    t=torch.arange(16)/15
    pos=torch.zeros(16,17,4);pos[:,:,:2]=uv;pos[:,:,2]=t[:,None];pos[:,-1,3]=1
    return pos.reshape(272,4)
def prep():
    RUN.mkdir(exist_ok=False);cache=RUN/'tokens';cache.mkdir()
    splits=json.loads((BASE/'split_paths.json').read_text())['paths'];save(RUN/'split_paths.json',{'paths':splits})
    g=torch.Generator().manual_seed(913)
    proj=torch.randn(2048,256,generator=g)/np.sqrt(2048)
    dproj=torch.randn(12,256,generator=g)/np.sqrt(12)
    torch.save({'omega_projection':proj,'trajectory_projection':dproj},RUN/'fixed_projections.pt')
    manifest={};start=time.time();paths=splits['train']+splits['val']
    for i,p in enumerate(paths):
        pp=Path(p);pp=pp if pp.is_absolute() else ROOT/pp
        q=torch.load(OMEGA/pp.parent.name/(pp.stem+'.pt'),weights_only=True,map_location='cpu')
        spatial=F.adaptive_avg_pool2d(q['patch_tokens'].float(),(4,4)).flatten(2).transpose(1,2)
        # Camera token is index 0; register tokens are not spatial patch tokens.
        om=torch.cat([spatial,q['tokens'][:,:1].float()],1)@proj
        with np.load(pp) as z:
            meta=json.loads(str(z['metadata_json']))
            assert meta['context_frames']==16
            xyz=torch.from_numpy(z['past_xyz'].copy()).float();uv=torch.from_numpy(z['past_uv'].copy()).float()
            valid=torch.from_numpy(z['past_valid'].copy()).bool();vis=torch.from_numpy(z['past_visibility'].copy()).bool()
            conf=torch.from_numpy(z['past_confidence'].copy()).float()
        valid &= torch.isfinite(xyz).all(-1)&torch.isfinite(uv).all(-1)
        xyz=torch.nan_to_num(xyz);uv=torch.nan_to_num(uv);conf=torch.nan_to_num(conf).clamp_min(0)
        # History-only farthest-point sampling of reliable tracks, independent of labels.
        eligible=torch.where(valid.sum(1)>=8)[0]
        if len(eligible)<17:eligible=torch.arange(len(valid))
        last=(valid.long()*torch.arange(1,17)).argmax(1)
        anchor=uv[torch.arange(len(uv)),last]
        chosen=[int(eligible[valid[eligible].sum(1).argmax()])]
        for _ in range(16):
            dist=torch.cdist(anchor[eligible],anchor[chosen]).min(1).values
            dist[torch.isin(eligible,torch.tensor(chosen))]=-1
            chosen.append(int(eligible[dist.argmax()]))
        xyz=xyz[chosen];uv=uv[chosen];vv=valid[chosen];cf=conf[chosen];vs=vis[chosen]
        center=xyz[vv].median(0).values if vv.any() else torch.zeros(3)
        scale=(xyz[vv]-center).norm(dim=-1).median().clamp_min(1e-3) if vv.any() else torch.tensor(1.)
        xyz=((xyz-center)/scale).clamp(-20,20)
        dx=torch.cat([torch.zeros_like(xyz[:,:1]),xyz[:,1:]-xyz[:,:-1]],1)
        du=torch.cat([torch.zeros_like(uv[:,:1]),uv[:,1:]-uv[:,:-1]],1)
        pairs=vv[:,1:]&vv[:,:-1];dx[:,1:]*=pairs[:,:,None];du[:,1:]*=pairs[:,:,None]
        cf=torch.log1p(cf);cf=cf/cf.max().clamp_min(1)
        feat=torch.cat([xyz,uv,dx,du,cf[:,:,None],vs.float()[:,:,None]],-1)
        assert feat.shape[-1]==12
        tr=(feat@dproj).transpose(0,1).reshape(272,256)
        pos=torch.zeros(16,17,4);pos[:,:,:2]=uv.transpose(0,1);pos[:,:,2]=torch.arange(16)[:,None]/15
        payload={'omega':om.reshape(272,256).half(),'d4rt':tr.half(),'dpos':pos.reshape(272,4),'valid':vv.T.reshape(272)}
        assert all(torch.isfinite(v).all() for v in payload.values())
        torch.save(payload,cache/(key(p)+'.pt'))
        name=re.sub(r'_clip_\d+$','',meta['video_name'])
        if name==meta['video_name']:name=re.sub(r'_\d+$','',name)
        manifest[p]={'source':meta['subset']+'/'+name,'subset':meta['subset'],'geometry_input':'history only'}
        if (i+1)%50==0:print(json.dumps({'prepare':i+1,'total':len(paths),'seconds':time.time()-start}),flush=True)
    save(RUN/'manifest.json',manifest)
    save(RUN/'PREPARED.json',{'seconds':time.time()-start,'clips':len(paths),'test_used':False,'omega':'historical 4x4 pooled spatial + camera token per frame, fixed Gaussian projection 2048->256','d4rt':'17 history-selected tracks x 16 frames; XYZ UV velocity visibility confidence, fixed projection 12->256; NOT latent D4RT encoder features; no future teacher fields','capacity':'all arms 272x256 memory and identical trainable network','source_omega_manifest':str(OMEGA/'manifest.json')})

class Dataset(Videos):
    def __getitem__(self,i):
        x,y,p=super().__getitem__(i)
        return x,y,p,torch.load(RUN/'tokens'/(key(p)+'.pt'),weights_only=True)

class Model(QueryModel):
    def __init__(self,ck,arm):
        super().__init__(ck,'query');self.arm=arm
        self.project=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
        self.coord=nn.Sequential(nn.Linear(4,256),nn.GELU(),nn.Linear(256,256))
        self.read=nn.MultiheadAttention(256,8,batch_first=True);self.qnorm=nn.LayerNorm(256)
        self.register_buffer('pos',positions());self.register_buffer('nulltoken',torch.zeros(1,1,256))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def encode(self,x):
        with torch.no_grad():
            z=self.base.to_patch_embedding(x);z+=self.base.pos_embedding.to(z.device)
            for block in self.base.blocks:z=block(z)
        return z
    def memory(self,z,g):
        b,t,p,d=z.shape;pos=self.pos[None].expand(b,-1,-1);valid=torch.ones(b,272,device=z.device,dtype=torch.bool)
        if self.arm in ['rgb','null']:
            grid=F.adaptive_avg_pool2d(z.reshape(b*t,16,16,d).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2).reshape(b,t,16,d)
            mem=torch.cat([grid,z.mean(2,keepdim=True)],2).reshape(b,272,d)
            if self.arm=='null':mem=torch.zeros_like(mem)
        elif self.arm=='omega':mem=g['omega'].to(z.dtype)
        else:mem=g['d4rt'].to(z.dtype);pos=g['dpos'];valid=g['valid']
        return mem,pos,valid
    def forward(self,x,g,mode='correct',external=None):
        z=self.encode(x);b,t,p,d=z.shape
        with torch.no_grad():native=self.base.mlp_head(z.reshape(-1,d)).reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)[:,:8]
        mem,pos,valid=self.memory(z,g) if external is None else external
        if mode=='zero':mem=torch.zeros_like(mem)
        mem=self.project(mem)+self.coord(pos.to(mem.dtype))
        mem=torch.cat([mem,self.nulltoken.expand(b,-1,-1).to(mem.dtype)],1)
        mask=~torch.cat([valid,torch.ones(b,1,device=x.device,dtype=torch.bool)],1)
        history=self.query.norm(z.reshape(b,t*p,d));out=[]
        for h in range(8):
            q=z[:,-1]+self.query.position[None]+self.query.time[h]
            q=q+self.query.attn(self.query.norm(q),history,history,need_weights=False)[0]
            # Geometry participates BEFORE per-horizon nonlinear temporal decoding.
            if mode!='bypass':q=q+.1*self.read(self.qnorm(q),mem,mem,key_padding_mask=mask,need_weights=False)[0]
            out.append(q+self.query.ff(q))
        return native+image(self,torch.stack(out,1))

def train(arm):
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    out=RUN/arm;out.mkdir(exist_ok=False);ck=torch.load(BASE/'best.pt',weights_only=False,map_location='cpu')
    m=Model(ck,arm).cuda();net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    splits=json.loads((RUN/'split_paths.json').read_text())['paths']
    loaders={k:DataLoader(Dataset(splits[k],BASE/'rgb_cache'),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    def move(g):return {k:v.cuda() for k,v in g.items()}
    def evaluate():
        m.eval();rows=[]
        with torch.no_grad():
            for x,y,paths,g in loaders['val']:
                x=x.cuda();y=y.cuda();g=move(g)
                with torch.autocast('cuda',dtype=torch.float16):pred=m(x,g).float()
                r=metrics(pred[0],y[0],net);r['path']=paths[0];rows.append(r)
        return rows,{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
    def state(epoch,s):return {'model':{k:v for k,v in m.state_dict().items() if not k.startswith('base.')},'epoch':epoch,'metrics':s,'arm':arm}
    save(out/'protocol.json',{'trainable':sum(p.numel() for p in params),'arm':arm,'train':len(splits['train']),'val':len(splits['val']),'test_used':False,'seed':42,'epochs':15,'lr':1e-4,'loss':'MSE only','checkpoint_selection':'val MSE, includes epoch0','geometry':'history only','fusion':'cross-attention inside each future query, before FFN; fixed 0.1 residual strength; zero output projection','lpips':'Alex v0.1, clamp[0,1], normalize[-1,1]','source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    x,y,p,g=next(iter(loaders['val']));m.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        err=float((m(x.cuda(),move(g))-m.base(x.cuda())[:,:8]).abs().max())
    assert err==0
    rows,s=evaluate();best=s['mse'];torch.save(state(0,s),out/'best.pt');save(out/'best_validation.json',rows);save(out/'epoch0.json',s)
    start=time.time();history=[]
    for epoch in range(1,16):
        tick=time.time();m.train();total=0
        for step,(x,y,p,g) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();g=move(g);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=m(x,g);loss=(pred.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();total+=float(loss.detach())
            if step%100==0:
                r={'epoch':epoch,'step':step,'steps':len(loaders['train']),'seconds':time.time()-tick};save(out/'progress.json',r);print(json.dumps(r),flush=True)
        rows,s=evaluate();torch.save(state(epoch,s),out/'last.pt')
        if s['mse']<best:best=s['mse'];torch.save(state(epoch,s),out/'best.pt');save(out/'best_validation.json',rows)
        r={'epoch':epoch,'train_mse':total/step,'validation':s,'epoch_seconds':time.time()-tick,'elapsed_seconds':time.time()-start};history.append(r);save(out/'history.json',history);save(out/'progress.json',r);print(json.dumps(r),flush=True)
    # Fixed-checkpoint sensitivity: never select a checkpoint using mismatched inputs.
    state_=torch.load(out/'best.pt',weights_only=False,map_location='cpu');m.load_state_dict(state_['model'],strict=False);m.eval()
    manifest=json.loads((RUN/'manifest.json').read_text());ds=loaders['val'].dataset;rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            x,y,p,g=ds[i];x=x[None].cuda();y=y.cuda();g=move({k:v[None] for k,v in g.items()})
            candidates=[(i+k)%len(ds) for k in range(1,len(ds)) if manifest[ds.paths[(i+k)%len(ds)]]['source']!=manifest[p]['source']]
            j=next((j for j in candidates if manifest[ds.paths[j]]['subset']==manifest[p]['subset']),candidates[0])
            dx,_,dp,dg=ds[j];dg=move({k:v[None] for k,v in dg.items()})
            with torch.autocast('cuda',dtype=torch.float16):external=m.memory(m.encode(dx[None].cuda()),dg)
            for mode in ['correct','mismatch','zero','bypass']:
                with torch.autocast('cuda',dtype=torch.float16):pred=m(x,g,mode,external if mode=='mismatch' else None).float()
                if mode=='correct':reference=pred
                r=metrics(pred[0],y,net);r.update(path=p,source=manifest[p]['source'],donor=dp,mode=mode,output_l1_vs_correct=float((pred-reference).abs().mean()));rows.append(r)
    save(out/'sensitivity_per_clip.json',rows)
    summary={mode:{k:float(np.mean([r[k] for r in rows if r['mode']==mode])) for k in ['mse','mae','psnr','lpips','temporal','output_l1_vs_correct']} for mode in ['correct','mismatch','zero','bypass']}
    save(out/'sensitivity.json',summary);save(out/'COMPLETE.json',{'seconds':time.time()-start,'best_epoch':state_['epoch'],'metrics':state_['metrics'],'test_used':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=['rgb','null','omega','d4rt']);a=ap.parse_args();torch.set_num_threads(2)
    if a.prepare:prep()
    else:train(a.arm)
