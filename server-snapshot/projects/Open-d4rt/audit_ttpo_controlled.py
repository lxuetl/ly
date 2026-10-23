"""Post-hoc reporting only: no selection or generation based on future targets."""
import json,os,time
from pathlib import Path
import torch,numpy as np
import pyiqa,lpips
from run_ttpo_controlled import SRC,OUT,ci
@torch.no_grad()
def main():
 torch.set_num_threads(4)
 assert (OUT/'COMPLETE.json').exists()
 items=torch.load(SRC/'inputs.pt',weights_only=False);prefs=json.loads((OUT/'preferences.json').read_text())
 quality=pyiqa.create_metric('musiq',device='cuda').eval();lp=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
 rows=[]
 for i,item in enumerate(items):
  predictions=torch.load(OUT/f'clip_{i:02d}.pt',weights_only=False);cache=torch.load(SRC/f'clip_{i:02d}.pt',weights_only=False)
  predictions['nr_selected']=torch.stack([cache[prefs['names'][k]][f] for f,k in enumerate(prefs['clips'][i]['win'])])
  target=item['target'].cuda().float();metrics={}
  for n,p in predictions.items():
   p=p.cuda().float();d=p-target
   metrics[n]={'musiq':float(quality(p).mean()),'mse':float(d.square().mean()),'lpips':float(lp(p*2-1,target*2-1).mean()),'temporal':float(((p[1:]-p[:-1])-(target[1:]-target[:-1])).abs().mean())}
  rows.append({'group':item['group'],'path':item['path'],'metrics':metrics})
 report={'n':len(rows),'means':{},'paired':{},'iteration_checks':{}}
 for n in rows[0]['metrics']:
  report['means'][n]={m:float(np.mean([r['metrics'][n][m] for r in rows])) for m in metrics[n]}
 for a,b in [('pref1000','plain'),('pref1000','structure'),('pref1000','reverse1000'),('pref1000','tau'),('pref100','plain'),('pref1000_residual','plain_residual'),('pref1000_residual','tau'),('nr_selected','tau')]:
  report['paired'][a+'-vs-'+b]={m:ci([r['metrics'][a][m]-r['metrics'][b][m] for r in rows]) for m in metrics[a]}
 for arm in ['plain','structure','pref100','pref1000','reverse1000']:
  ts=[t for i in range(len(items)) for t in json.loads((OUT/f'trace_{i:02d}.json').read_text())['steps'] if t['arm']==arm]
  report['iteration_checks'][arm]={'steps':len(ts),'nonzero_updates':sum(t['update_rms']>0 for t in ts),'mean_update_rms':float(np.mean([t['update_rms'] for t in ts]))}
 (OUT/'audit_report.json').write_text(json.dumps(report,indent=2));(OUT/'audit_per_video.json').write_text(json.dumps(rows,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
