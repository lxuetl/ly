"""Shared, versioned protocol for two small frozen-backbone diagnostic pilots."""
from pathlib import Path
import sys, json, hashlib, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT), str(PROJECT/'forecasting'), '/root/autodl-tmp/OpenSTL']
ROOT = Path('/root/autodl-fs/d4rt_training')
OUT = ROOT/'trajectory_lhcr_pilot_20261018'
SEED = 20261018


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def key(path): return hashlib.sha256(str(path).encode()).hexdigest()[:24]


def group(path):
    # Reuse the established original-video grouping, not a clip-name heuristic.
    import train_propagation_fusion as t
    return tuple(t.prev.group(str(path)))


def balanced_one_per_video(paths, limit):
    videos={}
    for p in paths:videos.setdefault(group(p),p)
    sources=defaultdict(list)
    for g,p in sorted(videos.items()):sources[g[0]].append(p)
    rng=random.Random(SEED)
    for ps in sources.values():rng.shuffle(ps)
    result=[]
    while len(result)<limit and any(sources.values()):
        for source in sorted(sources):
            if sources[source] and len(result)<limit:result.append(sources[source].pop())
    return result


def manifest():
    destination=OUT/'manifest.json'
    if destination.exists():return json.loads(destination.read_text())
    split=json.loads((ROOT/'propagation_fusion_20261014/split_paths.json').read_text())['paths']
    gs={k:{group(p) for p in v} for k,v in split.items()}
    assert not gs['train']&gs['val'] and not gs['train']&gs['test'] and not gs['val']&gs['test']
    # Preserve the previous scorer's original-video fit/dev/calibration partition.
    prior=json.loads((ROOT/'iterative_preference_v1_20261017/protocol.json').read_text())['groups']
    selected={}
    for name,limit in [('fit',64),('dev',12),('calibration',12)]:
        allowed={tuple(g) for g in prior[name]}
        selected[name]=balanced_one_per_video([p for p in split['train'] if group(p) in allowed],limit)
    selected['evaluation']=balanced_one_per_video(split['val'],40)
    allgroups=[{group(p) for p in v} for v in selected.values()]
    for i in range(len(allgroups)):
        for j in range(i):assert not allgroups[i]&allgroups[j]
    selected['protocol']={'seed':SEED,'test_used':False,'TUM_used':False,
        'selection':'one clip per original video, balanced by source, outcome-independent',
        'counts':{k:len(v) for k,v in selected.items()},
        'evaluation':'reused original validation; exploratory pilot, not fresh confirmation',
        'parent_manifest_sha256':sha(ROOT/'propagation_fusion_20261014/split_paths.json')}
    dump(destination,selected);return selected


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)


def lpips_model():
    import lpips
    return lpips.LPIPS(net='alex',verbose=False).cuda().eval().requires_grad_(False)


def metrics(pred, target, baseline, lp):
    err=pred-target
    temporal=((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).abs().mean()
    return {'mse':float(err.square().mean()),'mae':float(err.abs().mean()),
            'lpips':float(lp(pred*2-1,target*2-1).mean()),'temporal':float(temporal),
            'budget':float((pred-baseline).abs().mean()),
            'horizon_mse':err.square().mean((1,2,3)).detach().cpu().tolist()}


def ci(values):
    x=np.asarray(values,dtype=float);r=np.random.default_rng(SEED)
    return {'mean':float(x.mean()),'ci95':np.quantile(x[r.integers(len(x),size=(4000,len(x)))].mean(1),[.025,.975]).tolist()}


def summarize(rows, reference='tau'):
    # The selected samples are one per original video; assert this explicitly.
    assert len({tuple(r['group']) for r in rows})==len(rows)
    names=list(rows[0]['metrics']);ms=['mse','mae','lpips','temporal','budget']
    means={n:{m:float(np.mean([r['metrics'][n][m] for r in rows])) for m in ms} for n in names}
    paired={n:{m:ci([r['metrics'][n][m]-r['metrics'][reference][m] for r in rows]) for m in ms[:4]} for n in names if n!=reference}
    return {'videos':len(rows),'video_equal':means,'paired_delta_vs_'+reference:paired,
            'per_source':{s:{n:{m:float(np.mean([r['metrics'][n][m] for r in rows if r['group'][0]==s])) for m in ms} for n in names} for s in sorted({r['group'][0] for r in rows})}}


def pixel_bound(pred, base, budget=.002, cap=.03):
    d=(pred.clamp(0,1)-base).clamp(-cap,cap)
    norm=d.abs().mean(tuple(range(1,d.ndim)),keepdim=True)
    return base+d*(budget/norm.clamp_min(1e-9)).clamp(max=1)
