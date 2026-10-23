"""Stage 1: native PredFormer, existing locked RGB split, no geometry/test tuning."""
import json,time,random,hashlib,argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from openstl.models import PredFormer_Model
ROOT=Path('/root/autodl-tmp/Open-d4rt')
SPLIT=Path('/root/autodl-fs/d4rt_training/propagation_fusion_20261014/split_paths.json')
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
class Videos(Dataset):
    def __init__(self,paths,cache):self.paths=paths;self.cache=cache
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        path=self.paths[i];key=hashlib.sha256(path.encode()).hexdigest()[:24];f=self.cache/(key+'.pt')
        if f.exists():rgb=torch.load(f,weights_only=True)
        else:
            p=Path(path);p=p if p.is_absolute() else ROOT/p
            with np.load(p) as z:meta=json.loads(str(z['metadata_json']))
            src=Path(meta['source_path']);src=src if src.is_absolute() else ROOT/src
            with np.load(src,allow_pickle=True) as z:
                frames=z['images_jpeg_bytes'][:24];assert len(frames)==24
                arr=[]
                for b in frames:
                    a=np.frombuffer(b,np.uint8) if isinstance(b,(bytes,bytearray)) else np.asarray(b,np.uint8)
                    im=cv2.imdecode(a,cv2.IMREAD_COLOR);assert im is not None
                    arr.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(128,128),interpolation=cv2.INTER_AREA))
            rgb=torch.from_numpy(np.stack(arr)).permute(0,3,1,2).contiguous();tmp=f.with_suffix('.tmp');torch.save(rgb,tmp);tmp.replace(f)
        rgb=rgb.float()/255
        return rgb[:16],rgb[16:24],path
def worker(_):cv2.setNumThreads(1);torch.set_num_threads(1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=15);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'rgb_cache';cache.mkdir()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    splits=json.loads(SPLIT.read_text())['paths'];assert not set(splits['train'])&set(splits['val'])
    save(out/'split_paths.json',{'paths':splits})
    cfg=dict(height=128,width=128,num_channels=3,pre_seq=16,after_seq=8,patch_size=8,dim=256,heads=8,dim_head=32,dropout=.1,attn_dropout=.1,drop_path=.1,scale_dim=4,depth=1,Ndepth=3)
    m=PredFormer_Model(model_config=cfg).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    loaders={k:DataLoader(Videos(splits[k],cache),batch_size=2,shuffle=k=='train',num_workers=2,pin_memory=True,persistent_workers=True,worker_init_fn=worker) for k in ['train','val']}
    save(out/'protocol.json',{'stage':'native PredFormer baseline','config':cfg,'counts':{k:len(v) for k,v in splits.items()},'batch_size':2,'epochs':a.epochs,'seed':42,'lr':3e-4,'loss':'MSE','best_selection':'validation MSE','test_used':False,'geometry_used':False,'output':'native16 sliced first8, official method','parameters':sum(p.numel() for p in m.parameters()),'split_sha256':hashlib.sha256(SPLIT.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    start=time.time();best=float('inf');stale=0
    for epoch in range(1,a.epochs+1):
        tick=time.time();m.train();total=0.;n=0
        for step,(x,y,paths) in enumerate(loaders['train'],1):
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=m(x)[:,:8];loss=(pred.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();total+=float(loss.detach())*len(x);n+=len(x)
            if step%25==0:
                progress={'epoch':epoch,'step':step,'steps':len(loaders['train']),'train_mse':total/n,'seconds_epoch':time.time()-tick,'elapsed_seconds':time.time()-start}
                save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        m.eval();rows=[]
        with torch.no_grad():
            for x,y,paths in loaders['val']:
                x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.float16):pred=m(x)[:,:8].float()
                for p,q,path in zip(pred,y,paths):
                    mse=float((p-q).square().mean());rows.append({'path':path,'mse':mse,'mae':float((p-q).abs().mean()),'psnr':float(-10*np.log10(max(mse,1e-12))),'temporal':float(((p[1:]-p[:-1])-(q[1:]-q[:-1])).abs().mean())})
        val=float(np.mean([r['mse'] for r in rows]));ck={'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'model_config':cfg,'val_mse':val}
        torch.save(ck,out/'last.pt')
        if val<best:best=val;stale=0;torch.save(ck,out/'best.pt');save(out/'best_validation.json',rows)
        else:stale+=1
        elapsed=time.time()-start;progress={'epoch':epoch,'epochs':a.epochs,'train_mse':total/n,'val_mse':val,'epoch_seconds':time.time()-tick,'elapsed_seconds':elapsed,'remaining_seconds_estimate':elapsed/epoch*(a.epochs-epoch),'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30}
        save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if stale>=5:break
    save(out/'COMPLETE.json',{'epochs':epoch,'best_val_mse':best,'seconds':time.time()-start,'test_used':False});print('COMPLETE',flush=True)
if __name__=='__main__':main()
