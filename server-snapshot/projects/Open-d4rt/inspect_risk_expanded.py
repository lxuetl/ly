import json,torch
import train_single_region_risk_expanded as e
p=e.OUT
h=json.loads((p/'history.json').read_text());ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
rows=[]
for r in h:
    if 'tuning' not in r:continue
    ts=r['tuning'];best=max(ts,key=lambda k:ts[k]['video_mean_gain'])
    rows.append({'epoch':r['epoch'],'loss':r['loss'],'best_mean_threshold':best,'stats':ts[best],
        'eligible':[k for k,v in ts.items() if v['active_videos']>=5 and v['harmful_videos']/v['active_videos']<=.1]})
proto=json.loads((p/'protocol.json').read_text())
print(json.dumps({'saved_best':{'epoch':ck['epoch'],'threshold':str(ck['threshold'])},'first_loss':h[0]['loss'],'last_loss':h[-1]['loss'],'history':rows,'validation_group_counts':{k:len(v) for k,v in proto['validation_groups'].items()}},indent=2))
