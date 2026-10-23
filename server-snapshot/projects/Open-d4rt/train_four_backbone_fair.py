#!/usr/bin/env python3
"""Fair no-geometry retraining for SimVP, TAU, DMVFN, and PredRNN++.

All backbones consume the same saved clip manifest and use the same image size,
loss, validation criterion, early stopping rule, and test metric implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "forecasting"))
sys.path.insert(0, "/root/autodl-tmp/OpenSTL")

from openstl.models import PredRNNpp_Model
from openstl.utils import reshape_patch, reshape_patch_back

from dmvfn_adapter import DMVFNVideoPredictor
from simvp_geometry_model import build_video_predictor
from train_pointodyssey_dmvfn_injection import combined_splits
from train_simvp_ablation import move, prediction_loss, seed_all, test_metrics
from video_forecast_dataset import VideoForecastDataset, video_pad_collate


class PredRNNppPredictor(nn.Module):
    """Small adapter around the official OpenSTL PredRNN++ implementation."""

    def __init__(self, context_frames: int, future_frames: int, image_size: int,
                 patch_size: int = 4, num_hidden: str = "64,64,64,64"):
        super().__init__()
        hidden = [int(value) for value in num_hidden.split(",")]
        self.context_frames = int(context_frames)
        self.future_frames = int(future_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.config = SimpleNamespace(
            in_shape=(self.context_frames, 3, self.image_size, self.image_size),
            pre_seq_length=self.context_frames,
            aft_seq_length=self.future_frames,
            total_length=self.context_frames + self.future_frames,
            patch_size=self.patch_size,
            filter_size=5,
            stride=1,
            # OpenSTL 1.0's GHU layer-normalization shape is inconsistent with
            # its 2x gate channels; the official non-LN path is stable here.
            layer_norm=0,
            reverse_scheduled_sampling=0,
        )
        self.backbone = PredRNNpp_Model(len(hidden), hidden, self.config)

    def _predict(self, context_rgb: torch.Tensor, future_rgb: torch.Tensor,
                 teacher_forcing_ratio: float) -> torch.Tensor:
        frames = torch.cat((context_rgb, future_rgb), dim=1)
        frames = frames.permute(0, 1, 3, 4, 2).contiguous()
        patched = reshape_patch(frames, self.patch_size)
        batch, _, height, width, channels = patched.shape
        mask = torch.zeros(
            batch, self.future_frames - 1, height, width, channels,
            device=patched.device, dtype=patched.dtype,
        )
        if teacher_forcing_ratio > 0 and self.future_frames > 1:
            use_truth = (
                torch.rand(batch, self.future_frames - 1, 1, 1, 1,
                           device=patched.device) < teacher_forcing_ratio
            )
            mask = use_truth.to(patched.dtype).expand_as(mask)
        generated, _ = self.backbone(patched, mask, return_loss=False)
        generated = reshape_patch_back(generated, self.patch_size)
        prediction = generated[:, -self.future_frames:].permute(0, 1, 4, 2, 3)
        return torch.sigmoid(prediction.contiguous())

    def teacher_forced(self, context_rgb: torch.Tensor, future_rgb: torch.Tensor,
                       ratio: float) -> torch.Tensor:
        return self._predict(context_rgb, future_rgb, ratio)

    def forward(self, context_rgb, geometry_7d=None, query_uv=None, point_valid=None):
        future = torch.zeros(
            context_rgb.shape[0], self.future_frames, *context_rgb.shape[2:],
            device=context_rgb.device, dtype=context_rgb.dtype,
        )
        return self._predict(context_rgb, future, 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True,
                        choices=("simvp", "tau", "dmvfn", "predrnnpp"))
    parser.add_argument("--cache-root", default="data/d4rt_forecasting_cache")
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--predrnn-hidden", default="64,64,64,64")
    parser.add_argument("--predrnn-patch-size", type=int, default=4)
    parser.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    return parser.parse_args()


def limited(paths, maximum):
    paths = list(paths)
    return paths if maximum <= 0 else paths[:maximum]


def exact_splits(args):
    available, split_meta = combined_splits(args)
    manifest = json.loads(Path(args.sample_manifest).read_text())
    selected = manifest.get("selected", manifest.get("paths", {}))
    chosen = {}
    for split in ("train", "val", "test"):
        valid = {str(Path(path)): Path(path) for path in available[split]}
        chosen[split] = [valid[str(Path(path))] for path in selected.get(split, [])
                         if str(Path(path)) in valid]
        if not chosen[split]:
            raise RuntimeError(f"Manifest selected no valid {split} clips")
    chosen["train"] = limited(chosen["train"], args.max_train)
    chosen["val"] = limited(chosen["val"], args.max_val)
    chosen["test"] = limited(chosen["test"], args.max_test)
    return chosen, split_meta


def build_model(args, context_frames, future_frames):
    if args.backbone in ("simvp", "tau"):
        return build_video_predictor(
            context_frames, future_frames, args.image_size,
            backbone=args.backbone, injection_type="add",
        )
    if args.backbone == "dmvfn":
        return DMVFNVideoPredictor(context_frames, future_frames)
    return PredRNNppPredictor(
        context_frames, future_frames, args.image_size,
        patch_size=args.predrnn_patch_size, num_hidden=args.predrnn_hidden,
    )


def training_prediction(model, batch, backbone, epoch, epochs):
    if backbone == "dmvfn":
        return model.teacher_forced_stages(
            batch["context_rgb"], batch["future_rgb"]
        )[-1]
    if backbone == "predrnnpp":
        # Standard scheduled-sampling direction: use less ground truth over time.
        ratio = max(0.0, 1.0 - (epoch - 1) / max(epochs - 1, 1))
        return model.teacher_forced(batch["context_rgb"], batch["future_rgb"], ratio)
    return model(batch["context_rgb"], None, None, None)


@torch.no_grad()
def validation_mse(model, loader, device):
    model.eval()
    squared = count = 0.0
    for batch in loader:
        batch = move(batch, device)
        prediction = model(batch["context_rgb"], None, None, None)
        squared += F.mse_loss(
            prediction, batch["future_rgb"], reduction="sum"
        ).item()
        count += batch["future_rgb"].numel()
    return squared / count


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths, split_meta = exact_splits(args)
    (output / "split_paths.json").write_text(json.dumps({
        "selected": {key: [str(path) for path in value] for key, value in paths.items()},
        "source_manifest": str(Path(args.sample_manifest)),
    }, indent=2))
    datasets = {
        split: VideoForecastDataset(
            args.cache_root, split, args.image_size, paths=paths[split], cache_items=False
        ) for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], args.batch_size, shuffle=True, generator=generator,
            collate_fn=video_pad_collate, num_workers=0,
        ),
        "val": DataLoader(
            datasets["val"], args.batch_size, collate_fn=video_pad_collate, num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"], 1, collate_fn=video_pad_collate, num_workers=0,
        ),
    }
    sample = datasets["train"][0]
    context_frames = sample["context_rgb"].shape[0]
    future_frames = sample["future_rgb"].shape[0]
    model = build_model(args, context_frames, future_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda" and args.backbone != "dmvfn"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best = float("inf")
    bad_epochs = 0
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        for batch in loaders["train"]:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction = training_prediction(
                    model, batch, args.backbone, epoch, args.epochs
                )
                loss = prediction_loss(
                    prediction, batch["future_rgb"], batch["context_rgb"][:, -1], 0.0
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        val_mse = validation_mse(model, loaders["val"], device)
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": total / len(loaders["train"]),
            "val_MSE": val_mse,
            "epoch_seconds": elapsed,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mse < best:
            best = val_mse
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "backbone": args.backbone,
                "context_frames": context_frames,
                "future_frames": future_frames,
                "image_size": args.image_size,
                "epoch": epoch,
                "val_MSE": val_mse,
                "args": vars(args),
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
        model, None, None, loaders["test"], "no_geometry", device,
        future_frames, output,
    )
    summary = {
        "metrics": metrics,
        "meta": {
            "backbone": args.backbone,
            "best_epoch": checkpoint["epoch"],
            "best_val_MSE": checkpoint["val_MSE"],
            "image_size": args.image_size,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "train_clips": len(datasets["train"]),
            "val_clips": len(datasets["val"]),
            "test_clips": len(datasets["test"]),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_seconds": time.perf_counter() - run_start,
            "split": split_meta,
            "args": vars(args),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
