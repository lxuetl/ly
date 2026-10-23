#!/usr/bin/env python3
import json
from pathlib import Path

root=Path("/autodl-fs/data/d4rt_training/decisive_abc_20260828")
a=json.loads(Path("/autodl-fs/data/d4rt_training/simvp_fair_learning_curve_20260828/fraction100/summary.json").read_text())["metrics"]
b=json.loads((root/"B_composite_simvp/summary.json").read_text())["metrics"]
c=json.loads((root/"C_d4rt_on_B/summary.json").read_text())["metrics"]["structured"]
keys=("MAE","MSE","PSNR","SSIM","LPIPS","temporal_delta_MAE","dynamic_MAE","dynamic_MSE","dynamic_PSNR","Motion_IoU","Motion_F1")
def normalize(m):
    if "motion_mask_IoU" in m:
        m={**m,"Motion_IoU":m["motion_mask_IoU"],"Motion_F1":m["motion_mask_F1"]}
    return {k:m[k] for k in keys}
result={"A_pixel_simvp":normalize(a),"B_composite_simvp":normalize(b),"C_composite_simvp_plus_D4RT":normalize(c)}
(root/"decision_summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
