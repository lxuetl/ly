#!/usr/bin/env python3
import csv
import json
from pathlib import Path

root = Path("/autodl-fs/data/d4rt_training/unified_retest_20260828")
models = [
    "no_geometry",
    "capsr_v3",
    "capsr_v5",
    "capsr_v6",
    "router_v1",
    "stable_router_v2",
]
keys = [
    "MAE", "MSE", "RMSE", "PSNR", "SSIM", "LPIPS",
    "temporal_delta_MAE", "dynamic_MAE", "dynamic_MSE", "dynamic_PSNR",
    "Motion_IoU", "Motion_F1",
]

rows = []
base_added = False
for model in models:
    path = root / model / "summary.json"
    if not path.exists():
        continue
    summary = json.loads(path.read_text())
    metrics = summary["metrics"]
    if not base_added and "no_geometry_backbone" in metrics:
        row = {"model": "frozen_simvp_base"}
        row.update({key: metrics["no_geometry_backbone"].get(key) for key in keys})
        rows.append(row)
        base_added = True
    row = {"model": model}
    prediction_key = "no_geometry" if model == "no_geometry" else "structured"
    row.update({key: metrics[prediction_key].get(key) for key in keys})
    masks = metrics.get("masks", {})
    row.update({
        "camera_fraction": masks.get("camera_fraction"),
        "object_fraction": masks.get("object_fraction"),
        "synthesis_fraction": masks.get("synthesis_fraction"),
    })
    rows.append(row)

(root / "unified_metrics.json").write_text(json.dumps(rows, indent=2))
fieldnames = ["model", *keys, "camera_fraction", "object_fraction", "synthesis_fraction"]
with (root / "unified_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        row["model"],
        f"MAE={row.get('MAE', float('nan')):.6f}",
        f"MSE={row.get('MSE', float('nan')):.6f}",
        f"PSNR={row.get('PSNR', float('nan')):.3f}",
        f"SSIM={row.get('SSIM', float('nan')):.4f}",
        f"LPIPS={row.get('LPIPS', float('nan')):.4f}",
        f"tMAE={row.get('temporal_delta_MAE', float('nan')):.6f}",
        f"dynMSE={row.get('dynamic_MSE', float('nan')):.6f}",
        f"IoU={row.get('Motion_IoU', float('nan')):.4f}",
        f"F1={row.get('Motion_F1', float('nan')):.4f}",
    )
