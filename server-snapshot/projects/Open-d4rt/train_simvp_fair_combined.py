#!/usr/bin/env python3
"""Train a pure SimVP baseline on the exact combined CaPSR split."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "forecasting"))

from simvp_geometry_model import build_video_predictor
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, prediction_loss, seed_all, test_metrics, validation_mse
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dynamic-loss-weight", type=float, default=0.0)
    parser.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def domain(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/po50/" in text:
        return "po50"
    for name in ("adt_mini", "ds_mini", "po_mini", "pstudio_mini"):
        if f"/{name}/" in text:
            return name
    return path.parent.name


def sequence_group(path: Path) -> str:
    stem = path.stem
    if "_clip_" in stem:
        return re.sub(r"_clip_\d+$", "", stem)
    if domain(path) == "ds_mini":
        return stem.split("-")[0]
    return re.sub(r"_\d+$", "", stem)


def fraction_paths(paths, fraction: float, seed: int):
    if fraction >= 1.0:
        return list(paths), {"fraction": 1.0, "groups": "all"}
    rng = random.Random(seed)
    selected = []
    manifest = {"fraction": fraction, "domains": {}}
    for source in sorted({domain(Path(path)) for path in paths}):
        source_paths = [Path(path) for path in paths if domain(Path(path)) == source]
        groups = sorted({sequence_group(path) for path in source_paths})
        rng.shuffle(groups)
        keep = max(1, round(len(groups) * fraction))
        chosen = set(groups[:keep])
        chosen_paths = [path for path in source_paths if sequence_group(path) in chosen]
        selected.extend(chosen_paths)
        manifest["domains"][source] = {
            "total_groups": len(groups), "selected_groups": sorted(chosen),
            "total_clips": len(source_paths), "selected_clips": len(chosen_paths),
        }
    return sorted(selected), manifest


def limit_paths(paths, maximum: int):
    paths = list(paths)
    return paths if maximum <= 0 else paths[:maximum]


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_args = SimpleNamespace(**vars(args))
    splits, split_meta = combined_splits(split_args)
    if args.sample_manifest:
        manifest = json.loads(Path(args.sample_manifest).read_text())
        selected = manifest.get("selected", manifest.get("paths", {}))
        chosen = {}
        for split in ("train", "val", "test"):
            allowed = {str(Path(path)): Path(path) for path in splits[split]}
            chosen[split] = [allowed[str(Path(path))] for path in selected.get(split, [])
                             if str(Path(path)) in allowed]
            if not chosen[split]:
                raise RuntimeError(f"sample manifest selected no valid {split} paths")
        train_paths = chosen["train"]
        splits["val"] = chosen["val"]
        splits["test"] = chosen["test"]
        fraction_manifest = {
            "fraction": args.train_fraction,
            "source_manifest": str(Path(args.sample_manifest)),
            "selected": {split: len(paths) for split, paths in chosen.items()},
        }
    else:
        train_paths, fraction_manifest = fraction_paths(
            splits["train"], args.train_fraction, args.seed
        )
    train_paths = limit_paths(train_paths, args.max_train)
    val_paths = limit_paths(splits["val"], args.max_val)
    test_paths = limit_paths(splits["test"], args.max_test)
    # Persist the exact paths so downstream geometry runs use the identical
    # examples rather than independently recreating an approximate fraction.
    (output / "split_paths.json").write_text(json.dumps({
        "selected": {
            "train": [str(path) for path in train_paths],
            "val": [str(path) for path in val_paths],
            "test": [str(path) for path in test_paths],
        },
        "fraction_selection": fraction_manifest,
    }, indent=2))
    train_dataset = VideoForecastDataset(
        args.cache_root, "train", args.image_size, paths=train_paths, cache_items=False
    )
    val_dataset = VideoForecastDataset(
        args.cache_root, "val", args.image_size, paths=val_paths, cache_items=False
    )
    test_dataset = VideoForecastDataset(
        args.cache_root, "test", args.image_size, paths=test_paths, cache_items=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, generator=generator,
        collate_fn=video_pad_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, args.batch_size, collate_fn=video_pad_collate, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 1, collate_fn=video_pad_collate, num_workers=0
    )
    sample = train_dataset[0]
    context_frames = sample["context_rgb"].shape[0]
    future_frames = sample["future_rgb"].shape[0]
    model = build_video_predictor(
        context_frames, future_frames, args.image_size, backbone="simvp", injection_type="add"
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch["context_rgb"], None, None, None)
                loss = prediction_loss(
                    prediction, batch["future_rgb"], batch["context_rgb"][:, -1],
                    args.dynamic_loss_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += loss.item()
        val_mse = validation_mse(
            model, None, None, val_loader, "no_geometry", device, future_frames
        )
        row = {"epoch": epoch, "train_loss": total / len(train_loader), "val_MSE": val_mse}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mse < best:
            best = val_mse; bad_epochs = 0
            torch.save({
                "model": model.state_dict(), "mode": "no_geometry",
                "context_frames": context_frames, "future_frames": future_frames,
                "image_size": args.image_size, "injection": "add", "backbone": "simvp",
                "backbone_config": model.backbone_config,
                "epoch": epoch, "train_fraction": args.train_fraction,
            }, output / "best.pt")
        else:
            bad_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        if epoch >= 10 and bad_epochs >= args.patience:
            print(json.dumps({"early_stop": epoch, "best_val_MSE": best}), flush=True)
            break

    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = test_metrics(
        model, None, None, test_loader, "no_geometry", device, future_frames, output
    )
    summary = {
        "metrics": metrics,
        "meta": {
            "best_epoch": checkpoint["epoch"], "train_fraction": args.train_fraction,
            "image_size": args.image_size,
            "train_clips": len(train_dataset), "val_clips": len(val_dataset),
            "test_clips": len(test_dataset), "split": split_meta,
            "fraction_selection": fraction_manifest,
            "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
