import json,torch,numpy as np
from pathlib import Path
from local_horizon import Model,Data,BASE,OLD,RUN
from stage1_baseline import save
torch.set_num_threads(2)
splits=json.loads((OLD/'split_paths.json').read_text())['paths']
ds=Data(splits['val'],BASE/'rgb_cache','omega')
base=torch.load(BASE/'best.pt',map_location='cpu',weights_only=False)
results={}
for arm in ['omega','rgb','null','omega_shuffled']:
    model=Model(base,arm).cuda().eval()
    model.load_state_dict(torch.load(RUN/arm/'best.pt',map_location='cpu',weights_only=False)['model'])
    rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            x,y,path,g=ds[i];x=x[None].cuda();y=y.cuda()
            with torch.autocast('cuda',dtype=torch.float16):p=model(x,g[None].cuda()).float()[0]
            mask=(y-x[0,-1]).abs().mean(1)>.05
            err=(p-y).square().mean(1)
            for h in range(8):
                rows.append(dict(path=path,horizon=h+1,mse=err[h].mean().item(),static_mse=err[h][~mask[h]].mean().item() if (~mask[h]).any() else None,dynamic_mse=err[h][mask[h]].mean().item() if mask[h].any() else None))
    save(RUN/arm/'region_per_frame.json',rows)
    results[arm]={str(h):{k:float(np.mean([r[k] for r in rows if r['horizon']==h and r[k] is not None])) for k in ['mse','static_mse','dynamic_mse']} for h in range(1,9)}
    print(arm,flush=True)
    del model;torch.cuda.empty_cache()
save(RUN/'region_summary.json',results)
print(json.dumps(results),flush=True)
