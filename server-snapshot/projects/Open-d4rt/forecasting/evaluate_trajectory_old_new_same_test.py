#!/usr/bin/env python3
"""Evaluate old/new trajectory checkpoints on one identical test split."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, seed_all
from trajectory_metrics import last_uv
from trajectory_model import ImplicitConditionedJointForecaster, VelocityResidualImplicitForecaster
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


METRICS = (
    "XYZ_ADE", "XYZ_FDE", "UV_ADE", "UV_FDE", "dynamic_UV_ADE",
    "dynamic_UV_FDE", "static_UV_ADE", "static_UV_FDE",
    "warp_UV_ADE", "warp_UV_FDE", "large_UV_ADE", "large_UV_FDE",
)


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def domain(path: Path):
    text = str(path).replace("\\", "/")
    if "/po50/" in text:
        return "po50"
    for name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
        if f"/{name}/" in text:
            return name
    return path.parent.name


def sequence(path: Path):
    stem = path.stem
    if "_clip_" in stem:
        return re.sub(r"_clip_\d+$", "", stem)
    if domain(path) == "ds_mini":
        return stem.split("-")[0]
    return re.sub(r"_\d+$", "", stem)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    cls = VelocityResidualImplicitForecaster if checkpoint.get("model_variant") == "velocity_residual" else ImplicitConditionedJointForecaster
    model = cls(checkpoint["future_frames"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def masked_mean(value, mask):
    return float(value[mask].mean().item()) if mask.any() else float("nan")


def metrics(batch, xyz, uv):
    target_xyz = torch.nan_to_num(batch["future_xyz_teacher"].float())
    target_uv = torch.nan_to_num(batch["future_uv_teacher"].float())
    mask = batch["future_valid"].bool()
    mask &= torch.isfinite(batch["future_xyz_teacher"].float()).all(-1)
    mask &= torch.isfinite(batch["future_uv_teacher"].float()).all(-1)
    xyz_error = torch.linalg.vector_norm(xyz - target_xyz, dim=-1)
    uv_error = torch.linalg.vector_norm(uv - target_uv, dim=-1)
    final = mask[:, :, -1]
    anchor = last_uv(batch)
    gt_amp = torch.linalg.vector_norm(target_uv[:, :, -1] - anchor, dim=-1)
    pred_amp = torch.linalg.vector_norm(uv[:, :, -1] - anchor, dim=-1)
    point_ok = batch["past_valid"].bool().any(2)
    hist_conf = torch.nan_to_num(batch["past_confidence"].float()).amax(2).clamp(0, 1)
    strength = (pred_amp / 0.04).clamp(0, 1) * hist_conf * point_ok.float()
    warp_mask = mask.float() * strength[:, :, None]
    warp_final = final.float() * strength

    result = {
        "XYZ_ADE": masked_mean(xyz_error, mask),
        "XYZ_FDE": masked_mean(xyz_error[:, :, -1], final),
        "UV_ADE": masked_mean(uv_error, mask),
        "UV_FDE": masked_mean(uv_error[:, :, -1], final),
    }
    for label, selector in (("dynamic", gt_amp >= 0.05), ("static", gt_amp < 0.05), ("large", gt_amp >= 0.10)):
        ade_mask = mask & selector[:, :, None]
        fde_mask = final & selector
        result[f"{label}_UV_ADE"] = masked_mean(uv_error, ade_mask)
        result[f"{label}_UV_FDE"] = masked_mean(uv_error[:, :, -1], fde_mask)
    result["warp_UV_ADE"] = float((uv_error * warp_mask).sum().item() / max(warp_mask.sum().item(), 1e-8))
    result["warp_UV_FDE"] = float((uv_error[:, :, -1] * warp_final).sum().item() / max(warp_final.sum().item(), 1e-8))
    return result


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values))


@torch.no_grad()
def main():
    a = args(); seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    split_args = SimpleNamespace(**vars(a))
    splits, split_meta = combined_splits(split_args)
    paths = splits["test"] if a.max_test <= 0 else splits["test"][:a.max_test]
    dataset = VideoForecastDataset(a.cache_root, "test", 64, paths=paths, cache_items=False)
    loader = DataLoader(dataset, 1, collate_fn=video_pad_collate, num_workers=0)
    models = {}
    checkpoints = {}
    for label, path in (("old_trajectory", a.old_checkpoint), ("new_trajectory", a.new_checkpoint)):
        models[label], checkpoints[label] = load_model(path, device)

    rows = []
    for index, batch in enumerate(loader):
        path = Path(dataset.paths[index])
        batch = move(batch, device)
        for label, model in models.items():
            xyz, uv = model(batch["past_xyz"].float(), batch["past_uv"].float(), batch["past_valid"], batch["past_confidence"].float())
            rows.append({
                "model": label, "clip": path.stem, "domain": domain(path),
                "original_sequence": sequence(path), **metrics(batch, xyz, uv),
            })
        if (index + 1) % 25 == 0:
            print(json.dumps({"done": index + 1, "total": len(dataset)}), flush=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["domain"], row["original_sequence"])].append(row)
    sequence_rows = []
    for (model, source, name), items in grouped.items():
        sequence_rows.append({
            "model": model, "domain": source, "original_sequence": name,
            "num_clips": len(items), **{metric: finite_mean([x[metric] for x in items]) for metric in METRICS},
        })

    output = Path(a.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / "per_clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0])); writer.writeheader(); writer.writerows(sequence_rows)

    lookup = defaultdict(dict)
    for row in sequence_rows:
        lookup[row["model"]][f'{row["domain"]}::{row["original_sequence"]}'] = row
    names = sorted(set(lookup["old_trajectory"]) & set(lookup["new_trajectory"]))
    rng = np.random.default_rng(a.seed)
    bootstrap = []
    for metric in METRICS:
        old = np.asarray([lookup["old_trajectory"][name][metric] for name in names])
        new = np.asarray([lookup["new_trajectory"][name][metric] for name in names])
        valid = np.isfinite(old) & np.isfinite(new)
        old, new = old[valid], new[valid]
        difference = new - old
        indices = rng.integers(0, len(difference), size=(a.bootstrap, len(difference)))
        samples = difference[indices].mean(1)
        low, high = np.quantile(samples, (0.025, 0.975))
        bootstrap.append({
            "metric": metric, "new_minus_old": float(difference.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
            "significant": bool(low > 0 or high < 0),
            "new_sequence_win_rate": float(np.mean(difference < 0)),
            "num_sequences": int(len(difference)), "bootstrap_samples": a.bootstrap,
        })
    summary = {
        "split": split_meta, "clips": len(dataset), "original_sequences": len(names),
        "checkpoint_epochs": {k: v.get("epoch") for k, v in checkpoints.items()},
        "means": {
            model: {metric: finite_mean([row[metric] for row in sequence_rows if row["model"] == model]) for metric in METRICS}
            for model in models
        },
        "paired_bootstrap": bootstrap,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(dataset), "original_sequences": len(names), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
