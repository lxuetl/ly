#!/usr/bin/env python3
"""Precompute context-only VGGT-Omega depth, confidence, and camera features."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d4rt-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--output-size", type=int, default=64)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prepare_image(image, resolution):
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
    )
    image = _crop_to_supported_aspect_ratio(image.convert("RGB"))
    width, height = image.size
    target_h, target_w = _balanced_target_shape(height / width, resolution, 16)
    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return to_tensor(image)


def source_images(source_path, context_frames):
    with np.load(source_path, allow_pickle=True) as source:
        if "images_jpeg_bytes" in source.files:
            return [
                Image.open(io.BytesIO(raw)).convert("RGB")
                for raw in source["images_jpeg_bytes"][:context_frames]
            ]
    from eval_track3d_in_worldtrack import load_worldtrack_sequence
    sample = load_worldtrack_sequence(Path(source_path), context_frames)
    return [Image.fromarray(frame) for frame in sample["video_rgb"][:context_frames]]


def destination(output_root, cache_path):
    return output_root / cache_path.parent.name / f"{cache_path.stem}.npz"


def main():
    a = parse_args()
    d4rt_root = Path(a.d4rt_root)
    sys.path.insert(0, str(d4rt_root / "forecasting"))
    sys.path.insert(0, str(d4rt_root))
    from train_pointodyssey_dmvfn_injection import combined_splits
    from vggt_omega.models import VGGTOmega

    split_args = SimpleNamespace(**vars(a))
    splits, split_meta = combined_splits(split_args)
    split_for_path = {}
    for split in ("train", "val", "test"):
        for path in splits[split]:
            split_for_path[str(Path(path))] = split
    paths = []
    for index in range(max(len(splits[name]) for name in ("train", "val", "test"))):
        for name in ("train", "val", "test"):
            if index < len(splits[name]):
                path = Path(splits[name][index])
                if path not in paths:
                    paths.append(path)
    if a.max_items:
        paths = paths[: a.max_items]

    output_root = Path(a.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = VGGTOmega().eval()
    state = torch.load(a.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda()
    records = []
    started = time.time()

    for index, cache_path in enumerate(paths):
        out = destination(output_root, cache_path)
        if out.exists():
            records.append({"cache_path": str(cache_path), "output": str(out), "status": "existing"})
            continue
        with np.load(cache_path) as cached:
            metadata = json.loads(str(cached["metadata_json"]))
        context_frames = int(metadata["context_frames"])
        source_path = Path(metadata["source_path"])
        images = torch.stack([
            prepare_image(image, a.resolution)
            for image in source_images(source_path, context_frames)
        ]).cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            prediction = model(images)
        depth = prediction["depth"].squeeze(0).permute(0, 3, 1, 2)
        confidence = prediction["depth_conf"].squeeze(0)[:, None]
        depth = F.interpolate(depth, (a.output_size, a.output_size), mode="bilinear", align_corners=False)
        confidence = F.interpolate(confidence, (a.output_size, a.output_size), mode="bilinear", align_corners=False)
        pose = prediction["pose_enc"].squeeze(0)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            depth=depth[:, 0].float().cpu().numpy().astype(np.float16),
            depth_confidence=confidence[:, 0].float().cpu().numpy().astype(np.float16),
            pose_enc=pose.float().cpu().numpy(),
            metadata_json=json.dumps({
                "cache_path": str(cache_path), "source_path": str(source_path),
                "split": split_for_path[str(cache_path)],
                "context_frames": context_frames,
                "input_shape": list(images.shape),
                "checkpoint": a.checkpoint,
                "resolution": a.resolution,
            }),
        )
        records.append({"cache_path": str(cache_path), "output": str(out), "status": "created"})
        if (index + 1) % 10 == 0 or index == 0:
            elapsed = time.time() - started
            print(json.dumps({
                "done": index + 1, "total": len(paths),
                "elapsed_seconds": elapsed,
                "seconds_per_clip": elapsed / (index + 1),
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }), flush=True)

    manifest = {
        "split": split_meta, "num_paths": len(paths), "records": records,
        "checkpoint": a.checkpoint, "resolution": a.resolution,
        "output_size": a.output_size,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({"complete": True, "items": len(paths), "output": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
