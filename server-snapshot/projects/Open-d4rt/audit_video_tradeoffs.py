import json,re,os
from pathlib import Path
from collections import defaultdict
import numpy as np
import train_propagation_fusion as t
OUT=t.ROOT/'video_tradeoff_inventory_20261017'
def main():
 OUT.mkdir(exist_ok=True)
 vv=json.loads((t.ROOT/'fixed_blend_control_test_20261017/per_video.json').read_text());records=[]
 for v in vv:
  m=v['metrics'];x=m['velocity_fixed'];g=m['velocity_gate'];base=m['tau'];delta=x['mse']-g['mse']
  records.append({'group':v['group'],'clips':v['clips'],'fixed_mse':x['mse'],'gate_mse':g['mse'],'tau_mse':base['mse'],'fixed_minus_gate_mse':delta,'fixed_minus_gate_lpips':x['lpips']-g['lpips'],'fixed_minus_gate_temporal':x['temporal']-g['temporal'],'fixed_gain_vs_tau':base['mse']-x['mse'],'gate_gain_vs_tau':base['mse']-g['mse'],'opposite_vs_tau':(x['mse']-base['mse'])*(g['mse']-base['mse'])<0})
 records.sort(key=lambda r:r['fixed_minus_gate_mse'])
 split=json.loads((t.OUT/'split_paths.json').read_text())['paths'];used={tuple(t.prev.group(p)) for pp in split.values() for p in pp}
 inventory=[]
 roots=[Path('data/worldtrack_release'),Path('/root/autodl-fs/d4rt_datasets')]
 for root in roots:
  for directory,dirs,files in os.walk(root,followlinks=False):
   depth=len(Path(directory).relative_to(root).parts)
   dirs[:]=[d for d in dirs if d not in ['rgb','rgbs','images','depth','depths','normal','normals','masks'] and depth<5]
   packs=[n for n in files if n.endswith('.npz')]
   if not packs:continue
   inventory.append({'directory':directory,'npz_count':len(packs),'examples':packs[:3]})
 # TUM exists in worldtrack but not forecasting split: inspect every sequence's frame count.
 candidates=[]
 for p in Path('data/worldtrack_release/tum').glob('*.npz'):
  with np.load(p,allow_pickle=True) as z:
   keys=list(z.keys());framekey=next((k for k in ['images_jpeg_bytes','video_rgb','images','rgb'] if k in keys),None)
   n=len(z[framekey]) if framekey else None
  candidates.append({'path':str(p),'name':p.stem,'frames':n,'keys':keys,'current_split_member':tuple(t.prev.group(str(p))) in used,'history_hits':[]})
 scanned=0;skipped=[]
 for directory,dirs,files in os.walk(t.ROOT):
  depth=len(Path(directory).relative_to(t.ROOT).parts);dirs[:]=[d for d in dirs if d not in ['cache','__pycache__',OUT.name] and depth<3]
  for n in files:
   p=Path(directory)/n
   if not n.endswith(('.json','.jsonl','.py','.sh','.log')):continue
   if p.stat().st_size>20_000_000:skipped.append(str(p));continue
   try:txt=p.read_text(errors='replace')
   except OSError:continue
   scanned+=1
   for c in candidates:
    if c['name'] in txt:c['history_hits'].append(str(p))
 summary={'videos':len(records),'fixed_better_mse':sum(r['fixed_minus_gate_mse']<0 for r in records),'gate_better_mse':sum(r['fixed_minus_gate_mse']>0 for r in records),'opposite_vs_tau':[r for r in records if r['opposite_vs_tau']],'largest_fixed_wins':records[:6],'largest_gate_wins':records[-6:][::-1],'inventory':inventory,'tum_candidates':candidates,'historical_files_scanned':scanned,'skipped_large_files':skipped,'caveat':'Absence of string matches does not prove never used; search limited to retained training artifacts depth3 <=20MB. Raw sequence inspection is not certification of independent holdout.'}
 (OUT/'per_video.json').write_text(json.dumps(records,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
