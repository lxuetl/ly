"""Independent saved-output audit, no source/model imports from experiment."""
import argparse, hashlib, json
from pathlib import Path
import torch

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main(root,check_lpips=False):
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    spec=json.loads((root/'protocol.json').read_text());arms=spec['arms'];rows=[]
    if check_lpips:
        import lpips
        device='cuda' if torch.cuda.is_available() else 'cpu'
        lp=lpips.LPIPS(net='alex',verbose=False).to(device).eval().requires_grad_(False)
    for c in spec['cases']:
        i=c['index'];r=json.loads((root/f'case_{i}.json').read_text())
        assert r['path']==c['path'] and r['group']==c['group']
        assert r['weights_bitwise_verified']
        z=torch.load(root/f'predictions_{i}.pt',map_location='cpu',weights_only=False)
        if (root/f'rgb_{i}.pt').exists():
            x=torch.load(root/f'rgb_{i}.pt',weights_only=False)
            assert torch.equal(x['future_rgb'],z['target'])
        base=z['base'];target=z['target'];m=r['matched_metrics']
        if check_lpips:
            with torch.no_grad():
                for a,q in {'tau':base,**z['matched']}.items():
                    val=float(lp(q.to(device)*2-1,target.to(device)*2-1).mean())
                    assert abs(val-m[a]['lpips'])<2e-6,(i,a,val,m[a]['lpips'])
        assert abs(float((base-target).square().mean())-m['tau']['mse'])<2e-8
        for a in arms:
            q=z['matched'][a];d=q-base
            assert torch.isfinite(q).all()
            assert abs(float(d.abs().mean())-r['common_budget'])<2e-7
            assert float(d.abs().max())<=.030001 and float(d.abs().mean())<=.002001
            assert abs(float((q-target).square().mean())-m[a]['mse'])<2e-8
            assert abs(float((q-target).abs().mean())-m[a]['mae'])<2e-8
            ident=r['diagnostics'][a]['MSE_identity']
            assert abs(ident['identity_residual'])<1e-10
        for k in ['mse','lpips','temporal']:
            v={a:m[a][k] for a in arms}
            hist=((v['D_P']-v['H_P'])+(v['D_G']-v['H_G']))/2
            end=((v['D_P']-v['D_G'])+(v['H_P']-v['H_G']))/2
            third=v['H_G']-v['RGB_GT']
            assert abs(hist+end+third-v['D_P']+v['RGB_GT'])<1e-10
            d=r['matched_attribution'][k]
            assert abs(hist-d['history_shapley'])<1e-10
            assert abs(end-d['endpoint_shapley'])<1e-10
            assert abs(third-d['third_stage_total'])<1e-10
        rows.append(r)
    means={a:{k:sum(r['matched_metrics'][a][k] for r in rows)/len(rows) for k in ['mse','lpips','temporal']} for a in ['tau']+arms}
    harms={a:sum(any(r['matched_metrics'][a][k]>r['matched_metrics']['tau'][k] for k in ['mse','lpips']) for r in rows) for a in arms}
    root_weights=Path('/root/autodl-fs/d4rt_training')
    for name,p in [('tau',root_weights/'tau128_full_fair_fast_20261009/best.pt'),('trajectory',root_weights/'pointodyssey_full_uv_velocity_residual_60/best.pt')]:
        assert file_sha(p)==spec['checkpoint_sha256'][name]
    result={'passed':True,'cases':len(rows),'saved_MSE_MAE_budget_cap_verified':True,
        'LPIPS_independently_recomputed':check_lpips,
        'both_checkpoint_files_unchanged':True,'attribution_identities_verified':True,
        'matched_means':means,'strict_harm_cases':harms}
    reference=Path('/root/autodl-tmp/three_stage_audit_20261020/predictions_0.pt')
    if root.name=='three_stage_audit_gpu_20261021' and reference.exists():
        old=torch.load(reference,map_location='cpu',weights_only=False)
        new=torch.load(root/'predictions_0.pt',map_location='cpu',weights_only=False)
        assert torch.equal(old['target'],new['target'])
        diff=float((old['base']-new['base']).abs().max())
        assert diff<1e-4
        result['CPU_GPU_case0_baseline_max_abs']=diff
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--lpips',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);main(a.root,a.lpips)
