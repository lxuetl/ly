"""Train trajectory and video predictors on ADT plus scene-disjoint DS clips."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from forecast_dataset import split_paths
from trajectory_model import JointTrajectoryForecaster, LiteTrajectoryForecaster
from train_joint_forecaster import evaluate, losses, move
from train_simvp_ablation import train_mode
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--output-dir", default="outputs/mixed_adt_ds")
    p.add_argument("--trajectory-checkpoint", default="outputs/trajectory_forecaster/best.pt")
    p.add_argument("--trajectory-epochs", type=int, default=80)
    p.add_argument("--video-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def ds_group(path):
    return Path(path).stem.split("-")[0]


def make_splits(cache_root, seed):
    root = Path(cache_root)
    adt_train = split_paths(root, "train", subset="adt_mini")
    adt_val = split_paths(root, "val", subset="adt_mini")
    ds = sorted((root / "ds_mini").glob("*.npz"))
    groups = sorted({ds_group(x) for x in ds})
    random.Random(seed).shuffle(groups)
    n_train = max(1, round(0.6 * len(groups)))
    n_val = max(1, round(0.2 * len(groups)))
    train_groups = set(groups[:n_train]); val_groups = set(groups[n_train:n_train+n_val])
    test_groups = set(groups[n_train+n_val:])
    ds_train = [x for x in ds if ds_group(x) in train_groups]
    ds_val = [x for x in ds if ds_group(x) in val_groups]
    ds_test = [x for x in ds if ds_group(x) in test_groups]
    return {
        "train": adt_train + ds_train, "val": adt_val + ds_val, "test": ds_test,
        "adt_train": adt_train, "adt_val": adt_val,
        "ds_train": ds_train, "ds_val": ds_val, "ds_test": ds_test,
        "ds_train_groups": sorted(train_groups), "ds_val_groups": sorted(val_groups),
        "ds_test_groups": sorted(test_groups),
    }


def manifest(splits, cache_root):
    root = Path(cache_root).resolve()
    out = {}
    for key, vals in splits.items():
        if key.endswith("groups"):
            out[key] = vals
        else:
            out[key] = [str(Path(x).resolve().relative_to(root)) for x in vals]
    return out


def train_joint(a, train_ds, val_ds, test_ds, device, out):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train = DataLoader(train_ds, 4, shuffle=True, generator=torch.Generator().manual_seed(a.seed),
                       collate_fn=video_pad_collate, num_workers=0)
    val = DataLoader(val_ds, 4, collate_fn=video_pad_collate, num_workers=0)
    test = DataLoader(test_ds, 1, collate_fn=video_pad_collate, num_workers=0)
    k = train_ds[0]["future_xyz_teacher"].shape[1]
    model = JointTrajectoryForecaster(k, 128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf"); history = []; joint_dir = out / "joint_trajectory"; joint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.trajectory_epochs + 1):
        model.train(); total = 0.0
        for batch in train:
            batch = move(batch, device); loss, *_ = losses(model, batch, 5.0)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
        metrics = evaluate(model, val, device, 5.0)
        row = {"epoch": epoch, "train_loss": total / len(train), **metrics}; history.append(row)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save({"model": model.state_dict(), "future_frames": k, "hidden_dim": 128, "uv_weight": 5.0}, joint_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(f"[joint] epoch={epoch:03d} train={row['train_loss']:.6f} val={metrics['loss']:.6f}", flush=True)
    (joint_dir / "history.json").write_text(json.dumps(history, indent=2))
    ck = torch.load(joint_dir / "best.pt", map_location=device); model.load_state_dict(ck["model"])
    metrics = evaluate(model, test, device, 5.0); (joint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    print("[joint] held-out DS", json.dumps(metrics), flush=True)
    return model, metrics


def main():
    a = parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    splits = make_splits(a.cache_root, a.seed)
    (out / "split_manifest.json").write_text(json.dumps(manifest(splits, a.cache_root), indent=2))
    train_ds = VideoForecastDataset(a.cache_root, "train", a.image_size, paths=splits["train"], cache_items=True)
    val_ds = VideoForecastDataset(a.cache_root, "val", a.image_size, paths=splits["val"], cache_items=True)
    test_ds = VideoForecastDataset(a.cache_root, "test", a.image_size, paths=splits["test"], cache_items=True)
    print(f"mixed train={len(train_ds)} (ADT={len(splits['adt_train'])}, DS={len(splits['ds_train'])}) "
          f"val={len(val_ds)} (ADT={len(splits['adt_val'])}, DS={len(splits['ds_val'])}) "
          f"heldout_DS_test={len(test_ds)}", flush=True)
    joint, joint_metrics = train_joint(a, train_ds, val_ds, test_ds, device, out)
    lite_ck = torch.load(a.trajectory_checkpoint, map_location="cpu")
    lite = LiteTrajectoryForecaster(lite_ck["future_frames"], lite_ck["hidden_dim"])
    lite.load_state_dict(lite_ck["model"]); lite.to(device).eval()
    for parameter in lite.parameters(): parameter.requires_grad_(False)
    sample = train_ds[0]; c = sample["context_rgb"].shape[0]; k = sample["future_rgb"].shape[0]
    common = dict(cache_root=a.cache_root, trajectory_checkpoint=a.trajectory_checkpoint,
                  joint_trajectory_checkpoint=str(out / "joint_trajectory" / "best.pt"), output_dir=str(out / "video"),
                  epochs=a.video_epochs, batch_size=a.batch_size, image_size=a.image_size, lr=3e-4,
                  seed=a.seed, device=a.device, injection="film", dynamic_loss_weight=0.0, run_name="")
    summary = {"joint_trajectory_heldout_DS": joint_metrics}
    for mode in ("no_geometry", "joint_future_geometry"):
        cfg = SimpleNamespace(**common); cfg.run_name = "baseline_no_geometry" if mode == "no_geometry" else "joint_geometry_film"
        summary[cfg.run_name] = train_mode(mode, cfg, train_ds, val_ds, test_ds, lite, joint, device, c, k)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
