"""Stage 4: evaluate, visualize and export geometry conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_dataset import ForecastCacheDataset, pad_collate
from trajectory_model import GeometryConditionEncoder, LiteTrajectoryForecaster, last_valid_xyz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--checkpoint", default="outputs/trajectory_forecaster/best.pt")
    p.add_argument("--output-dir", default="outputs/trajectory_forecaster/eval")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def constant_velocity(batch):
    xyz, valid = batch["past_xyz"].float(), batch["past_valid"].bool()
    anchor = last_valid_xyz(xyz, valid)
    # robust average of the last four valid one-step velocities
    pair = valid[:, :, 1:] & valid[:, :, :-1]
    vel = torch.nan_to_num(xyz[:, :, 1:] - xyz[:, :, :-1])
    pair[:, :, :-4] = False
    vel = (vel * pair[..., None]).sum(2) / pair.sum(2).clamp_min(1)[..., None]
    k = batch["future_xyz_teacher"].shape[2]
    steps = torch.arange(1, k + 1, device=xyz.device).view(1, 1, k, 1)
    return anchor[:, :, None] + vel[:, :, None] * steps


def errors(pred, target, mask):
    e = torch.linalg.vector_norm(pred - torch.nan_to_num(target), dim=-1)
    ade = e[mask].sum().item(); ade_n = mask.sum().item()
    fmask = mask[:, :, -1]; fde = e[:, :, -1][fmask].sum().item(); fde_n = fmask.sum().item()
    return ade, ade_n, fde, fde_n


def plot_example(path, past, target, pred, valid, n=12):
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ids = np.flatnonzero(valid.any(1))[:n]
    for i in ids:
        ax.plot(*past[i].T, color="0.65", linewidth=1)
        ax.plot(*target[i].T, color="tab:green", linewidth=1.5)
        ax.plot(*pred[i].T, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_title("gray: history | green: teacher | red: prediction")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    a = parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    model = LiteTrajectoryForecaster(ckpt["future_frames"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    # Adapter is intentionally untrained: it establishes the integration interface; train it with the video model.
    adapter = GeometryConditionEncoder(a.token_dim).to(device).eval()
    ds = ForecastCacheDataset(a.cache_root, a.split); loader = DataLoader(ds, 1, num_workers=0, collate_fn=pad_collate)
    sums = {"model": [0., 0, 0., 0], "constant_velocity": [0., 0, 0., 0]}
    with torch.no_grad():
        for j, b in enumerate(loader):
            name = b.pop("name")[0]; b = {k: v.to(device) for k, v in b.items()}
            pred = model(b["past_xyz"].float(), b["past_uv"].float(), b["past_valid"], b["past_confidence"].float())
            base = constant_velocity(b); target = b["future_xyz_teacher"].float()
            mask = b["future_valid"].bool() & torch.isfinite(target).all(-1)
            for key, value in (("model", pred), ("constant_velocity", base)):
                vals = errors(value, target, mask)
                sums[key] = [x + y for x, y in zip(sums[key], vals)]
            raw_geometry = adapter.raw_features(pred, b["query_uv"].float(), b["future_confidence"].float())
            # Exercise and shape-check the trainable downstream adapter, but do not
            # serialize its randomly initialized projection as meaningful features.
            tokens = adapter(pred, b["query_uv"].float(), b["future_confidence"].float())
            assert tokens.shape == (pred.shape[0], pred.shape[1] * pred.shape[2], a.token_dim)
            np.savez_compressed(out / f"{name}_prediction.npz", future_xyz_pred=pred[0].cpu().numpy(),
                                future_valid=mask[0].cpu().numpy(), query_uv=b["query_uv"][0].cpu().numpy(),
                                geometry_features_7d=raw_geometry[0].cpu().numpy())
            if j == 0:
                plot_example(out / "trajectories.png", b["past_xyz"][0].cpu().numpy(),
                             target[0].cpu().numpy(), pred[0].cpu().numpy(), mask[0].cpu().numpy())
    metrics = {}
    for key, (ade, an, fde, fn) in sums.items():
        metrics[key] = {"ADE_3D": ade / max(an, 1), "FDE_3D": fde / max(fn, 1)}
    metrics.update({"split": a.split, "num_sequences": len(ds), "token_dim": a.token_dim,
                    "note": "7D raw geometry is exported; adapter weights must be trained with the video model"})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2)); print(f"outputs: {out}")


if __name__ == "__main__": main()
