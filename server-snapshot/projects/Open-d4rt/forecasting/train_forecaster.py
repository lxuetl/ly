"""Stage 3: train the lightweight D4RT future-trajectory forecaster."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_dataset import ForecastCacheDataset, pad_collate
from trajectory_model import LiteTrajectoryForecaster


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--output-dir", default="outputs/trajectory_forecaster")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def loss_fn(pred, batch):
    target = batch["future_xyz_teacher"].float()
    mask = batch["future_valid"].bool() & torch.isfinite(target).all(-1)
    weight = torch.nan_to_num(batch["future_confidence"].float(), nan=0.0).clamp(0, 1) * mask
    raw = F.smooth_l1_loss(pred, torch.nan_to_num(target), reduction="none").mean(-1)
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total = count = 0.0
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        p = model(b["past_xyz"].float(), b["past_uv"].float(), b["past_valid"], b["past_confidence"].float())
        total += float(loss_fn(p, b)); count += 1
    return total / max(count, 1)


def main():
    a = args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_ds, val_ds = ForecastCacheDataset(a.cache_root, "train"), ForecastCacheDataset(a.cache_root, "val")
    train = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=pad_collate)
    val = DataLoader(val_ds, a.batch_size, num_workers=0, pin_memory=True, collate_fn=pad_collate)
    sample = train_ds[0]; k = sample["future_xyz_teacher"].shape[1]
    model = LiteTrajectoryForecaster(k, a.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = float("inf"); history = []
    print(f"split: train={len(train_ds)} val={len(val_ds)} params={sum(x.numel() for x in model.parameters()):,}")
    for epoch in range(1, a.epochs + 1):
        model.train(); running = 0.0
        for b in train:
            b = {k0: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k0, v in b.items()}
            pred = model(b["past_xyz"].float(), b["past_uv"].float(), b["past_valid"], b["past_confidence"].float())
            loss = loss_fn(pred, b); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); running += float(loss)
        sched.step(); val_loss = evaluate(model, val, device); train_loss = running / len(train)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "future_frames": k, "hidden_dim": a.hidden_dim,
                        "epoch": epoch, "val_loss": best}, out / "best.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best={best:.6f}")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best checkpoint: {out/'best.pt'} val_loss={best:.6f}")


if __name__ == "__main__": main()
