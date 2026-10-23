import json
from pathlib import Path
base=Path('/root/autodl-fs/d4rt_training')
smoke=base/'tau128_ttpo_replay_smoke_20261011'
rows=[json.loads(x) for x in (smoke/'test_per_clip.jsonl').read_text().splitlines()]
previous=[json.loads(x) for x in (base/'tau128_experience_router_extended_20261010/test_per_clip.jsonl').read_text().splitlines()]
for row in rows:
    for ix,variant in [(0,'tau'),(4,'all_geometry')]:
        reference=next(x for x in previous if x['name']==row['name'] and x['variant']==variant)
        diff=abs(reference['MSE']-row['metrics'][ix]['MSE'])
        print(json.dumps({'name':row['name'],'variant':variant,'old_MSE':reference['MSE'],'new_MSE':row['metrics'][ix]['MSE'],'abs_difference':diff}))
        assert diff < 1e-7
    assert all(w['geometry_frames']==w['cutoff'] for w in row['windows'])
print('SMOKE_AUDIT_PASS: baseline reproduction and prefix geometry length')
