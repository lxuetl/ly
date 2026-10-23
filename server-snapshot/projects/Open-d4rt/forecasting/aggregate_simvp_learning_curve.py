#!/usr/bin/env python3
import csv
import json
from pathlib import Path

root = Path("/autodl-fs/data/d4rt_training/simvp_fair_learning_curve_20260828")
rows = []
for tag in ("fraction025", "fraction050", "fraction100"):
    summary = json.loads((root / tag / "summary.json").read_text())
    metrics = summary["metrics"]
    meta = summary["meta"]
    rows.append({
        "fraction": meta["train_fraction"],
        "train_clips": meta["train_clips"],
        "best_epoch": meta["best_epoch"],
        "MAE": metrics["MAE"], "MSE": metrics["MSE"],
        "PSNR": metrics["PSNR"], "SSIM": metrics["SSIM"],
        "LPIPS": metrics["LPIPS"],
        "temporal_delta_MAE": metrics["temporal_delta_MAE"],
        "dynamic_MAE": metrics["dynamic_MAE"],
        "dynamic_MSE": metrics["dynamic_MSE"],
        "dynamic_PSNR": metrics["dynamic_PSNR"],
        "Motion_IoU": metrics["motion_mask_IoU"],
        "Motion_F1": metrics["motion_mask_F1"],
        "trainable_params": meta["trainable_params"],
    })
with (root / "learning_curve.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(root / "learning_curve.json").write_text(json.dumps(rows, indent=2))
for row in rows:
    print(json.dumps(row))
