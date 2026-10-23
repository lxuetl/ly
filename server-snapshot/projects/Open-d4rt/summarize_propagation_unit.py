import json
import numpy as np
from audit_propagation_unit import OUT, ci, prev
rows=[json.loads(x) for x in (OUT/'rows.jsonl').read_text().splitlines()]
held=[r for r in rows if r['partition']=='held']
out={'point_absolute':{},'rgb_absolute':{},'matched_feature':{},'baseline_held':{},'gain_scan':{},'rgb_pairs':{}}
for mode in ['zero','gt_exact','gt_reverse']:
 out['point_absolute'][mode]={region:float(np.mean([r['point'][mode][region] for r in rows if r['point'][mode][region] is not None])) for region in ['visible','moving_visible']}
for name in ['k9_zero','k9_gt','k33_gt','k9_pred_aligned']:
 out['rgb_absolute'][name]={region:float(np.mean([r['rgb'][name][region] for r in rows if r['rgb'][name][region] is not None])) for region in ['visible','moving_visible']}
for metric in ['mse','cost','moving_mse']:
 out['baseline_held'][metric]=float(np.mean([r['baseline'][metric] for r in held if r['baseline'][metric] is not None]))
for a,b in [('k9_gt','k33_gt'),('k9_pred_aligned','k9_pred_stale')]:
 out['rgb_pairs'][a+' minus '+b]={region:ci([r['rgb'][a][region]-r['rgb'][b][region] for r in rows if r['rgb'][a][region] is not None]) for region in ['visible','moving_visible']}
for k in [9,33]:
 for mode in ['gt','pred_aligned']:
  for gain in [.25,1.,4.]:
   name=f'k{k}_{mode}_g{gain}'
   out['gain_scan'][name]=ci([r['feature'][name]['cost']-r['baseline']['cost'] for r in held])
 for other in ['reverse','pred_aligned']:
  a=f'k{k}_gt_g0.25';b=f'k{k}_{other}_g0.25'
  out['matched_feature'][a+' minus '+b]={m:ci([r['feature'][a][m]-r['feature'][b][m] for r in held if r['feature'][a][m] is not None]) for m in ['mse','cost','moving_mse']}
prev.dump(OUT/'additional_summary.json',out)
print(json.dumps(out))
