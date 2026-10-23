import json
from pathlib import Path
r=Path('/root/autodl-tmp/predformer_frozen32_20261022')
for arm in ['omega','rgb','null','omega_shuffled']:
 for epoch in [0,10,30,50,100]:
  s=json.loads((r/arm/('epoch%d_summary.json'%epoch)).read_text())
  print(arm,epoch,json.dumps({mode:{k:s[mode][k] for k in ['mse','late_mse','lpips','late_lpips']} for mode in ['correct','shuffle']}))
