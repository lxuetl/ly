"""Future queries / matched null and RGB memory adapters. Frozen stage1 encoder."""
import argparse,json,time,random,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from openstl.models import PredFormer_Model
from stage1_baseline import Videos,worker,save
from geometry_tokens import FutureQueries,GeometryRead
BASE=Path('/root/autodl-tmp/predformer_native128_20261021')

class QueryModel(nn.Module):
    def __init__(self,ck,arm):
        super().__init__();self.base=PredFormer_Model(model_config=ck['model_config']);self.base.load_state_dict(ck['model']);self.base.requires_grad_(False).eval()
        self.query=FutureQueries(256,256,8,8);self.head=copy.deepcopy(self.base.mlp_head).requires_grad_(True)
        self.adapter=GeometryRead(256,256,8) if arm!='query' else None;self.arm=arm
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def forward(self,x):
        with torch.no_grad():
            z=self.base.to_patch_embedding(x);z=z+self.base.pos_embedding.to(z.device)
            for block in self.base.blocks:z=block(z)
        future=self.query(z)
        if self.adapter is not None:
            # Same memory length/shape/projections for matched RGB/null arms.
            # Spatially pool history from16x16 to4x4 patches, preserve all16 timestamps.
            b,t,p,d=z.shape
            mem=torch.nn.functional.adaptive_avg_pool2d(z.reshape(b*t,16,16,d).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2).reshape(b,t*16,d)
            if self.arm=='null':mem=torch.zeros_like(mem)
            mem=mem[:,None].expand(-1,8,-1,-1)
            valid=torch.ones(mem.shape[:3],device=z.device,dtype=torch.bool)
            future=self.adapter(future,mem,valid)
        b,t,p,d=future.shape;y=self.head(future)
        # Same patch-to-image convention as official PredFormer.
        return y.reshape(b,t,16,16,3,8,8).permute(0,1,4,2,5,3,6).reshape(b,t,3,128,128)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['query','null','rgb'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False);model=QueryModel(ck,a.arm).cuda()
    if a.smoke:
        model.train();x=torch.rand(1,16,3,128,128,device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):y=model(x);loss=y.float().square().mean()
        loss.backward();assert y.shape==(1,8,3,128,128) and torch.isfinite(y).all()
        assert all(p.grad is None for p in model.base.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(json.dumps({'smoke':a.arm,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}));return
    out=Path(a.out)/a.arm;out.mkdir(parents=True,exist_ok=False)
    splits=json.loads((BASE/'split_paths.json').read_text())['paths'];loaders={k:DataLoader(Videos(splits[k],BASE/'rgb_cache'),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,pin_memory=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    save(out/'protocol.json',{'arm':a.arm,'base_checkpoint':str(BASE/'best.pt'),'base_epoch':ck['epoch'],'base_frozen':True,'counts':{k:len(v) for k,v in splits.items()},'epochs':a.epochs,'lr':1e-4,'batch':1,'seed':42,'loss':'MSE','test_used':False,'geometry_used':False,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'null_vs_rgb':'identical parameter count and memory shape; null uses zeros; not geometry experiment','query':'8 explicit future horizons; reads frozen16-frame visual tokens','checkpoint_selection':'val MSE','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    best=float('inf');start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();tick=time.time();total=0.;n=0
        for step,(x,y,_) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x);loss=(pred.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);scaler.step(opt);scaler.update();total+=float(loss.detach());n+=1
            if step%100==0:
                row={'arm':a.arm,'epoch':epoch,'step':step,'steps':len(loaders['train']),'train_mse':total/n,'epoch_seconds':time.time()-tick};save(out/'progress.json',row);print(json.dumps(row),flush=True)
        model.eval();rows=[]
        with torch.no_grad():
            for x,y,paths in loaders['val']:
                x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.float16):pred=model(x).float()
                for p,q,path in zip(pred,y,paths):
                    mse=float((p-q).square().mean());rows.append({'path':path,'mse':mse,'mae':float((p-q).abs().mean()),'psnr':float(-10*np.log10(max(mse,1e-12))),'temporal':float(((p[1:]-p[:-1])-(q[1:]-q[:-1])).abs().mean())})
        val=float(np.mean([r['mse'] for r in rows]));state={k:v for k,v in model.state_dict().items() if not k.startswith('base.')}
        checkpoint={'model':state,'epoch':epoch,'arm':a.arm,'base_checkpoint':str(BASE/'best.pt'),'val_mse':val}
        torch.save(checkpoint,out/'last.pt')
        if val<best:best=val;torch.save(checkpoint,out/'best.pt');save(out/'best_validation.json',rows)
        elapsed=time.time()-start;row={'arm':a.arm,'epoch':epoch,'epochs':a.epochs,'train_mse':total/n,'val_mse':val,'epoch_seconds':time.time()-tick,'elapsed_seconds':elapsed,'remaining_seconds':elapsed/epoch*(a.epochs-epoch),'peak_GiB':torch.cuda.max_memory_allocated()/2**30}
        save(out/'progress.json',row);print(json.dumps(row),flush=True)
    save(out/'COMPLETE.json',{'best_val_mse':best,'seconds':time.time()-start,'test_used':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
