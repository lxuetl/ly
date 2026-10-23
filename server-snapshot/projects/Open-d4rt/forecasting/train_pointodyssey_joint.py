"""Train the implicit XYZ/UV forecaster on legacy domains plus PointOdyssey50."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from train_all_datasets import make_splits
from train_joint_forecaster import move
from train_simvp_ablation import seed_all
from trajectory_metrics import evaluate_detailed
from trajectory_model import ImplicitConditionedJointForecaster,VelocityResidualImplicitForecaster
from trajectory_sampling import clip_histograms, curriculum_loader, curriculum_losses
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--output-dir", default="outputs/pointodyssey50_curriculum")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model-variant", choices=("implicit","velocity_residual"), default="implicit")
    p.add_argument("--init-checkpoint", default="")
    return p.parse_args()


def main():
    a = args(); seed_all(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    subsets = [x.strip() for x in a.legacy_subsets.split(",") if x.strip()]
    splits, legacy_by = make_splits(a.cache_root, subsets, a.seed)
    po_root = Path(a.po_cache); po = json.loads(Path(a.po_split_manifest).read_text())
    po_counts = {}
    for split in ("train", "val", "test"):
        candidates = [po_root / "po50" / name for name in po["clips"][split]]
        usable = []
        for path in candidates:
            if not path.exists():
                raise FileNotFoundError(path)
            # Zero-valid future clips make masked trajectory loss undefined.
            with __import__("numpy").load(path) as z:
                if __import__("numpy").asarray(z["future_valid"]).any(): usable.append(path)
        splits[split].extend(usable); po_counts[split] = len(usable)
    manifest = {"seed": a.seed, "legacy_subsets": subsets, "legacy": {
        s: {k: (len(v) if k != "groups" else v) for k, v in d.items()}
        for s, d in legacy_by.items()}, "pointodyssey_usable": po_counts,
        "totals": {s: len(v) for s, v in splits.items()}}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    # PointOdyssey source RGB is 540x960. Caching decoded clips during the
    # curriculum histogram pass can exhaust host RAM, so keep loading on demand.
    mk = lambda s: VideoForecastDataset(a.cache_root, s, 64, paths=splits[s], cache_items=False)
    train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    val = DataLoader(val_ds, a.batch_size, collate_fn=video_pad_collate, num_workers=0)
    test = DataLoader(test_ds, 1, collate_fn=video_pad_collate, num_workers=0)
    future_frames = train_ds[0]["future_xyz_teacher"].shape[1]
    model_cls = VelocityResidualImplicitForecaster if a.model_variant == "velocity_residual" else ImplicitConditionedJointForecaster
    model = model_cls(future_frames, 128).to(device)
    if a.init_checkpoint and not a.resume:
        init = torch.load(a.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        print("warm_start", a.init_checkpoint, "missing", missing, "unexpected", unexpected, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    histograms = clip_histograms(train_ds)
    best = float("inf"); history = []; start_epoch = 1
    last_path, best_path = out / "last.pt", out / "best.pt"
    if a.resume and (last_path.exists() or best_path.exists()):
        resume_path = last_path if last_path.exists() else best_path
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck: optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        best = float(ck.get("best_score", ck.get("score", float("inf"))))
        history_path = out / "history.json"
        if history_path.exists():
            history = [x for x in json.loads(history_path.read_text())
                       if int(x["epoch"]) < start_epoch]
        print("resumed", resume_path, "start_epoch", start_epoch, "best", best, flush=True)
    print("split", manifest["totals"], "po", po_counts, flush=True)
    for epoch in range(start_epoch, a.epochs + 1):
        model.train(); total = 0.0; sampled = [0, 0, 0, 0]
        loader, natural_mix = curriculum_loader(train_ds, histograms, epoch, a.epochs,
                                                 a.batch_size, a.seed)
        for batch in loader:
            batch = move(batch, device)
            loss, *_, diag = curriculum_losses(model, batch, epoch, a.epochs, 5.0)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item(); sampled = [x+y for x, y in zip(sampled, diag["sampled_counts"])]
        metrics = evaluate_detailed(model, val, device, 5.0, 2.0)
        score = (metrics["loss"] + .02 * metrics["dynamic_UV_ADE_norm"]
                 + .03 * metrics["motion_bins"]["0.10-inf"]["UV_ADE_norm"]
                 + .02 * metrics["warp_UV_ADE_norm"])
        row = {"epoch": epoch, "train_loss": total/len(loader), "score": score,
               "sampled_counts": sampled, "natural_mix": natural_mix, **metrics}
        history.append(row); print("trajectory", epoch, "train", row["train_loss"],
                                   "val", metrics["loss"], "score", score, flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "future_frames": future_frames, "hidden_dim": 128, "epoch": epoch,
                 "score": score, "best_score": min(best, score), "curriculum_sampling": True,
                 "model_variant": a.model_variant}
        torch.save(state, last_path)
        if score < best:
            best = score; torch.save(state, best_path)
    ck = torch.load(out / "best.pt", map_location=device); model.load_state_dict(ck["model"])
    metrics = evaluate_detailed(model, test, device, 5.0, 2.0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("done best_epoch", ck["epoch"], json.dumps(metrics), flush=True)


if __name__ == "__main__": main()
