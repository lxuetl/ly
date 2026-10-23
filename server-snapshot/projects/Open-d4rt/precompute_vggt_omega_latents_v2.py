#!/usr/bin/env python3
"""Cache leakage-safe VGGT-Omega context latents for the fair D4RT split."""

from __future__ import annotations

import argparse
import io
import json
import os
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
    p = argparse.ArgumentParser(description=__doc__)
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


def prepare_image(image: Image.Image, resolution: int) -> torch.Tensor:
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
    )

    image = _crop_to_supported_aspect_ratio(image.convert("RGB"))
    width, height = image.size
    target_h, target_w = _balanced_target_shape(height / width, resolution, 16)
    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    return to_tensor(image)


def source_images(source_path: Path, context_frames: int):
    with np.load(source_path, allow_pickle=True) as source:
        if "images_jpeg_bytes" in source.files:
            return [
                Image.open(io.BytesIO(raw)).convert("RGB")
                for raw in source["images_jpeg_bytes"][:context_frames]
            ]
    from eval_track3d_in_worldtrack import load_worldtrack_sequence

    sample = load_worldtrack_sequence(source_path, context_frames)
    return [Image.fromarray(frame) for frame in sample["video_rgb"][:context_frames]]


def destination(output_root: Path, cache_path: Path) -> Path:
    return output_root / cache_path.parent.name / f"{cache_path.stem}.pt"


def valid_cache(path: Path, context_frames: int) -> bool:
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
        return (
            tuple(item["tokens"].shape[:2]) == (context_frames, 17)
            and item["tokens"].shape[-1] == 2048
            and tuple(item["pose_enc"].shape) == (context_frames, 9)
            and tuple(item["depth"].shape[-2:]) == (64, 64)
            and torch.isfinite(item["tokens"].float()).all().item()
            and torch.isfinite(item["depth"].float()).all().item()
        )
    except Exception:
        return False


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    a = parse_args()
    d4rt_root = Path(a.d4rt_root)
    sys.path.insert(0, str(d4rt_root / "forecasting"))
    sys.path.insert(0, str(d4rt_root))
    from train_pointodyssey_dmvfn_injection import combined_splits
    from vggt_omega.models import VGGTOmega

    torch.manual_seed(a.seed)
    splits, split_meta = combined_splits(SimpleNamespace(**vars(a)))
    split_for_path = {
        str(Path(path)): split
        for split in ("train", "val", "test")
        for path in splits[split]
    }
    paths = []
    for index in range(max(len(splits[s]) for s in ("train", "val", "test"))):
        for split in ("train", "val", "test"):
            if index < len(splits[split]):
                path = Path(splits[split][index])
                if path not in paths:
                    paths.append(path)
    if a.max_items:
        paths = paths[: a.max_items]

    output_root = Path(a.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = VGGTOmega().eval()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = []
    started = time.time()
    created = skipped = 0
    for index, cache_path in enumerate(paths, 1):
        with np.load(cache_path) as cached:
            metadata = json.loads(str(cached["metadata_json"]))
        context_frames = int(metadata["context_frames"])
        out = destination(output_root, cache_path)
        if valid_cache(out, context_frames):
            skipped += 1
            status = "existing"
        else:
            source_path = Path(metadata["source_path"])
            images = torch.stack([
                prepare_image(image, a.resolution)
                for image in source_images(source_path, context_frames)
            ]).cuda(non_blocking=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images)
            tokens = prediction["camera_and_register_tokens"].squeeze(0)
            depth = prediction["depth"].squeeze(0).permute(0, 3, 1, 2)
            confidence = prediction["depth_conf"].squeeze(0)[:, None]
            depth = F.interpolate(
                depth, (a.output_size, a.output_size), mode="bilinear", align_corners=False
            )[:, 0]
            confidence = F.interpolate(
                confidence, (a.output_size, a.output_size), mode="bilinear", align_corners=False
            )[:, 0]
            pose = prediction["pose_enc"].squeeze(0)
            payload = {
                "tokens": tokens.to(device="cpu", dtype=torch.bfloat16).contiguous(),
                "depth": depth.to(device="cpu", dtype=torch.float16).contiguous(),
                "depth_confidence": confidence.to(device="cpu", dtype=torch.float16).contiguous(),
                "pose_enc": pose.to(device="cpu", dtype=torch.float32).contiguous(),
            }
            if not all(torch.isfinite(value.float()).all() for value in payload.values()):
                raise RuntimeError(f"non-finite VGGT-Omega feature for {cache_path}")
            atomic_save(payload, out)
            created += 1
            status = "created"
            del images, prediction, tokens, depth, confidence, pose, payload
        records.append({
            "cache_path": str(cache_path),
            "output": str(out),
            "split": split_for_path[str(cache_path)],
            "status": status,
        })
        if index == 1 or index % 10 == 0 or index == len(paths):
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-6)
            print(json.dumps({
                "stage": "vggt_omega_latent_cache_v2",
                "done": index,
                "total": len(paths),
                "created": created,
                "skipped": skipped,
                "seconds_per_clip": elapsed / index,
                "eta_seconds": (len(paths) - index) / max(rate, 1e-6),
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }), flush=True)

    manifest = {
        "format_version": 2,
        "leakage_policy": "observed context frames only",
        "split": split_meta,
        "num_paths": len(paths),
        "checkpoint": a.checkpoint,
        "resolution": a.resolution,
        "output_size": a.output_size,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({"complete": True, "items": len(paths), "output": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
