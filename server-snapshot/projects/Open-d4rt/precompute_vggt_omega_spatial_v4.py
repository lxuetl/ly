#!/usr/bin/env python3
"""Cache leakage-safe VGGT-Omega global and spatial tokens for semantic v4.

Only observed context frames are passed to VGGT-Omega.  In addition to the
camera/register tokens, this version keeps a compact 8x8 grid of final-layer
patch tokens so that the forecasting model can learn *where* geometry applies.
"""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument(
        "--vggt-root", default="/root/autodl-tmp/vggt-omega",
        help="VGGT-Omega source checkout (used when it is not pip-installed)",
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--po-cache", required=True)
    parser.add_argument("--po-split-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini"
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--patch-grid", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def valid_cache(path: Path, context_frames: int, patch_grid: int) -> bool:
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
        return (
            tuple(item["tokens"].shape[:2]) == (context_frames, 17)
            and item["tokens"].shape[-1] == 2048
            and tuple(item["patch_tokens"].shape)
            == (context_frames, 2048, patch_grid, patch_grid)
            and tuple(item["pose_enc"].shape) == (context_frames, 9)
            and tuple(item["depth"].shape[-2:]) == (64, 64)
            and all(
                torch.isfinite(item[key].float()).all().item()
                for key in (
                    "tokens", "patch_tokens", "depth", "depth_confidence", "pose_enc"
                )
            )
        )
    except Exception:
        return False


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def omega_forward_with_patches(model, images: torch.Tensor, patch_grid: int):
    """Run Omega once and expose its last-layer patch grid.

    `images` has shape [T, 3, H, W].  This mirrors VGGTOmega.forward instead of
    invoking the aggregator twice, which keeps preprocessing cost unchanged.
    """
    images = images.unsqueeze(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        aggregated_tokens, patch_token_start = model.aggregator(images)
    final_tokens = aggregated_tokens[-1]
    if final_tokens is None:
        raise RuntimeError("VGGT-Omega did not retain its final token layer")

    height, width = images.shape[-2:]
    patch_h, patch_w = height // 16, width // 16
    patch_tokens = final_tokens[:, :, patch_token_start:]
    expected = patch_h * patch_w
    if patch_tokens.shape[2] != expected:
        raise RuntimeError(
            f"patch count {patch_tokens.shape[2]} does not match {patch_h}x{patch_w}"
        )
    batch, frames, _, channels = patch_tokens.shape
    patch_tokens = patch_tokens.reshape(
        batch, frames, patch_h, patch_w, channels
    ).permute(0, 1, 4, 2, 3)
    patch_tokens = F.adaptive_avg_pool2d(
        patch_tokens.flatten(0, 1).float(), (patch_grid, patch_grid)
    ).reshape(batch, frames, channels, patch_grid, patch_grid)

    prediction = {
        "camera_and_register_tokens": final_tokens[
            :, :, :patch_token_start
        ].contiguous(),
        "patch_tokens": patch_tokens,
    }
    with torch.autocast(device_type="cuda", enabled=False):
        prediction["pose_enc"] = model.camera_head(
            aggregated_tokens, patch_token_start=patch_token_start
        )
        depth, depth_conf = model.dense_head(
            aggregated_tokens,
            images=images,
            patch_token_start=patch_token_start,
        )
        prediction["depth"] = depth
        prediction["depth_conf"] = depth_conf
    return prediction


def main():
    args = parse_args()
    d4rt_root = Path(args.d4rt_root)
    sys.path.insert(0, str(Path(args.vggt_root)))
    sys.path.insert(0, str(d4rt_root / "forecasting"))
    sys.path.insert(0, str(d4rt_root))
    from train_pointodyssey_dmvfn_injection import combined_splits
    from vggt_omega.models import VGGTOmega

    torch.manual_seed(args.seed)
    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
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
    if args.max_items:
        paths = paths[: args.max_items]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = VGGTOmega().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
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
        output_path = destination(output_root, cache_path)
        if valid_cache(output_path, context_frames, args.patch_grid):
            skipped += 1
            status = "existing"
        else:
            source_path = Path(metadata["source_path"])
            images = torch.stack([
                prepare_image(image, args.resolution)
                for image in source_images(source_path, context_frames)
            ]).cuda(non_blocking=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                prediction = omega_forward_with_patches(
                    model, images, args.patch_grid
                )
            tokens = prediction["camera_and_register_tokens"].squeeze(0)
            patches = prediction["patch_tokens"].squeeze(0)
            depth = prediction["depth"].squeeze(0).permute(0, 3, 1, 2)
            confidence = prediction["depth_conf"].squeeze(0)[:, None]
            depth = F.interpolate(
                depth, (args.output_size, args.output_size),
                mode="bilinear", align_corners=False,
            )[:, 0]
            confidence = F.interpolate(
                confidence, (args.output_size, args.output_size),
                mode="bilinear", align_corners=False,
            )[:, 0]
            pose = prediction["pose_enc"].squeeze(0)
            payload = {
                "tokens": tokens.to("cpu", torch.bfloat16).contiguous(),
                "patch_tokens": patches.to("cpu", torch.bfloat16).contiguous(),
                "depth": depth.to("cpu", torch.float16).contiguous(),
                "depth_confidence": confidence.to("cpu", torch.float16).contiguous(),
                "pose_enc": pose.to("cpu", torch.float32).contiguous(),
            }
            if not all(torch.isfinite(value.float()).all() for value in payload.values()):
                raise RuntimeError(f"non-finite VGGT-Omega feature for {cache_path}")
            atomic_save(payload, output_path)
            created += 1
            status = "created"
            del images, prediction, tokens, patches, depth, confidence, pose, payload
        records.append({
            "cache_path": str(cache_path),
            "output": str(output_path),
            "split": split_for_path[str(cache_path)],
            "status": status,
        })
        if index == 1 or index % 10 == 0 or index == len(paths):
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-6)
            print(json.dumps({
                "stage": "vggt_omega_spatial_cache_v4",
                "done": index,
                "total": len(paths),
                "created": created,
                "skipped": skipped,
                "seconds_per_clip": elapsed / index,
                "eta_seconds": (len(paths) - index) / max(rate, 1e-6),
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }), flush=True)

    manifest = {
        "format_version": 4,
        "leakage_policy": "observed context frames only",
        "split": split_meta,
        "num_paths": len(paths),
        "checkpoint": args.checkpoint,
        "resolution": args.resolution,
        "output_size": args.output_size,
        "patch_grid": args.patch_grid,
        "patch_channels": 2048,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({
        "complete": True, "items": len(paths), "output": str(output_root)
    }), flush=True)


if __name__ == "__main__":
    main()
