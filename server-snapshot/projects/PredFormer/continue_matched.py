"""Same additional optimizer-step budget as residual experiment; all native weights trainable."""
import json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
import lpips
from torch.utils.data import DataLoader
from openstl.models import PredFormer_Model
from stage1_baseline import Videos,worker,save
from stage2_queries import BASE
from token_audit import metrics

def main():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    out=Path('/root/autodl-tmp/predformer_continue_matched_20261021');out.mkdir(exist_ok=False)
    ck=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
    m=PredFormer_Model(model_config=ck['model_config']).cuda();m.load_state_dict(ck['model'])
    net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    splits=json.loads((BASE/'split_paths.json').read_text())['paths']
    loaders={k:DataLoader(Videos(splits[k],BASE/'rgb_cache'),batch_size=1,shuffle=k=='train',num_workers=2,persistent_workers=True,pin_memory=True,worker_init_fn=worker,generator=torch.Generator().manual_seed(42)) for k in ['train','val']}
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    save(out/'protocol.json',{'base':str(BASE/'best.pt'),'epochs':15,'additional_steps':15*len(loaders['train']),'batch':1,'lr':1e-4,'seed':42,'optimizer':'fresh AdamW, same as residual arm, not restoration of original optimizer','weight_decay':.01,'clip_grad':1.,'loss':'MSE','selection':'validation MSE, epoch0 eligible','counts':{k:len(v) for k,v in splits.items()},'test_used':False,'budget':'matched sample exposures and optimizer steps, NOT matched FLOPs or trainable parameter count','lpips':'Alex v0.1, clamp[0,1], normalize[-1,1], report only','trainable':sum(p.numel() for p in m.parameters()),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def evaluate():
        m.eval();rows=[]
        with torch.inference_mode():
            for x,y,paths in loaders['val']:
                x=x.cuda();y=y.cuda()
                with torch.autocast('cuda',dtype=torch.float16):p=m(x)[:,:8].float()
                r=metrics(p[0],y[0],net);r['path']=paths[0];rows.append(r)
        return rows,{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='path'}
    def state(epoch,s):return {'model':m.state_dict(),'model_config':ck['model_config'],'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'metrics':s}
    rows,s=evaluate();best=s['mse'];save(out/'epoch0.json',s);save(out/'best_validation.json',rows);torch.save(state(0,s),out/'best.pt')
    history=[];start=time.time()
    for epoch in range(1,16):
        m.train();tick=time.time();total=0
        for step,(x,y,_) in enumerate(loaders['train'],1):
            x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):p=m(x)[:,:8];loss=(p.float()-y).square().mean()
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();total+=float(loss.detach())
            if step%100==0:
                r={'epoch':epoch,'step':step,'steps':len(loaders['train']),'seconds':time.time()-tick};save(out/'progress.json',r);print(json.dumps(r),flush=True)
        rows,s=evaluate();torch.save(state(epoch,s),out/'last.pt')
        if s['mse']<best:best=s['mse'];torch.save(state(epoch,s),out/'best.pt');save(out/'best_validation.json',rows)
        elapsed=time.time()-start;r={'epoch':epoch,'train_mse':total/step,'validation':s,'epoch_seconds':time.time()-tick,'elapsed_seconds':elapsed,'remaining_seconds':elapsed/epoch*(15-epoch)}
        history.append(r);save(out/'history.json',history);save(out/'progress.json',r);print(json.dumps(r),flush=True)
    save(out/'COMPLETE.json',{'seconds':time.time()-start,'best_val_mse':best,'test_used':False})
if __name__=='__main__':main()
