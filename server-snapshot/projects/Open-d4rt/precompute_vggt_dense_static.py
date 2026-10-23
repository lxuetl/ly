#!/usr/bin/env python3
"""Precompute dense, training-only VGGT static reprojection supervision.

VGGT is run on the complete ground-truth clip.  For every future frame, target
pixels are unprojected with target depth and reprojected into the final context
frame.  The resulting backward sampling grid is filtered by depth consistency,
VGGT confidence, image bounds, and ground-truth photometric consistency.  The
cache contains no learned scalar reward and is never needed at inference time.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
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
    p.add_argument("--omega-repo", default="/root/autodl-tmp/vggt-omega")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--po-cache", required=True)
    p.add_argument("--po-split-manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--legacy-subsets", default="adt_mini,ds_mini,po_mini,pstudio_mini")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--output-size", type=int, default=64)
    p.add_argument("--confidence-quantile", type=float, default=0.30)
    p.add_argument("--depth-log-threshold", type=float, default=0.10)
    p.add_argument("--photometric-threshold", type=float, default=0.15)
    p.add_argument("--max-train", type=int, default=216)
    p.add_argument("--max-val", type=int, default=80)
    p.add_argument("--max-test", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_limit(paths, limit, seed):
    paths = [Path(path) for path in paths]
    if not limit or limit >= len(paths):
        return paths
    groups = defaultdict(list)
    for path in paths:
        groups[path.parent.name].append(path)
    rng = np.random.default_rng(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected, keys = [], sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def source_images(source_path, frames):
    with np.load(source_path, allow_pickle=True) as source:
        if "images_jpeg_bytes" in source.files:
            return [
                Image.open(io.BytesIO(raw)).convert("RGB")
                for raw in source["images_jpeg_bytes"][:frames]
            ]
    from eval_track3d_in_worldtrack import load_worldtrack_sequence
    sample = load_worldtrack_sequence(Path(source_path), frames)
    return [Image.fromarray(frame).convert("RGB") for frame in sample["video_rgb"][:frames]]


def destination(root, cache_path):
    return Path(root) / Path(cache_path).parent.name / f"{Path(cache_path).stem}.npz"


def resize_geometry(depth, confidence, intrinsics, size):
    old_h, old_w = depth.shape[-2:]
    depth = F.interpolate(depth[:, None], (size, size), mode="bilinear", align_corners=False)[:, 0]
    confidence = F.interpolate(confidence[:, None], (size, size), mode="bilinear", align_corners=False)[:, 0]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= size / float(old_w)
    intrinsics[:, 1] *= size / float(old_h)
    return depth, confidence, intrinsics


def confidence_weight(value, quantile):
    flat = value.flatten()
    low = torch.quantile(flat, quantile)
    high = torch.quantile(flat, 0.90)
    return ((value - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


def backward_static_constraint(depth, confidence, extrinsics, intrinsics, rgb,
                               source, target, args):
    """Map each target pixel to the final context frame and estimate visibility."""
    size = depth.shape[-1]
    device, dtype = depth.device, depth.dtype
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype), indexing="ij",
    )
    pixels = torch.stack((xx.flatten(), yy.flatten(), torch.ones(size * size, device=device, dtype=dtype)))
    target_depth = depth[target].flatten()
    camera_target = (torch.linalg.inv(intrinsics[target]) @ pixels) * target_depth[None]
    rt, tt = extrinsics[target, :3, :3], extrinsics[target, :3, 3]
    world = rt.transpose(0, 1) @ (camera_target - tt[:, None])
    rs, ts = extrinsics[source, :3, :3], extrinsics[source, :3, 3]
    camera_source = rs @ world + ts[:, None]
    projected = intrinsics[source] @ camera_source
    z_source = camera_source[2]
    u = projected[0] / projected[2].clamp_min(1e-6)
    v = projected[1] / projected[2].clamp_min(1e-6)
    grid = torch.stack((2 * u / max(size - 1, 1) - 1, 2 * v / max(size - 1, 1) - 1), -1)
    grid = grid.reshape(1, size, size, 2)

    inside = (
        torch.isfinite(target_depth) & (target_depth > 1e-5) & (z_source > 1e-5)
        & torch.isfinite(u) & torch.isfinite(v)
        & (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
    ).reshape(size, size)
    sampled_source_depth = F.grid_sample(
        depth[source:source + 1, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, 0]
    sampled_source_conf = F.grid_sample(
        confidence[source:source + 1, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, 0]
    depth_error = (sampled_source_depth.clamp_min(1e-5).log() - z_source.reshape(size, size).clamp_min(1e-5).log()).abs()

    warped_source = F.grid_sample(
        rgb[source:source + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0]
    photo_error = (warped_source - rgb[target]).abs().mean(0)
    source_weight = confidence_weight(sampled_source_conf, args.confidence_quantile)
    target_weight = confidence_weight(confidence[target], args.confidence_quantile)
    weight = (source_weight * target_weight).sqrt()
    weight *= torch.exp(-depth_error / max(args.depth_log_threshold, 1e-6))
    weight *= torch.exp(-photo_error / 0.05)
    valid = inside & (depth_error < args.depth_log_threshold) & (photo_error < args.photometric_threshold)
    weight = torch.where(valid, weight, torch.zeros_like(weight)).clamp(0, 1)
    return grid[0], weight, depth_error, photo_error


def main():
    args = parse_args()
    root = Path(args.d4rt_root)
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "forecasting"))
    sys.path.insert(0, str(Path(args.omega_repo)))
    from train_pointodyssey_dmvfn_injection import combined_splits
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    splits, split_meta = combined_splits(SimpleNamespace(**vars(args)))
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    selected = {
        split: stratified_limit(splits[split], limits[split], args.seed + index)
        for index, split in enumerate(("train", "val", "test"))
    }
    jobs = [(split, path) for split in ("train", "val", "test") for path in selected[split]]
    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model = model.cuda()
    records, started = [], time.time()
    for index, (split, cache_path) in enumerate(jobs):
        out = destination(output_root, cache_path)
        if out.exists():
            records.append({"split": split, "cache_path": str(cache_path), "output": str(out), "status": "existing"})
            continue
        with np.load(cache_path) as cached:
            metadata = json.loads(str(cached["metadata_json"]))
        context_frames = int(metadata["context_frames"])
        future_frames = int(metadata["future_frames"])
        total_frames = context_frames + future_frames
        originals = source_images(Path(metadata["source_path"]), total_frames)
        # Square resize exactly matches the forecasting dataset's normalized UV/image convention.
        images = torch.stack([
            to_tensor(image.resize((args.resolution, args.resolution), Image.Resampling.BICUBIC))
            for image in originals
        ]).cuda()
        rgb_small = torch.stack([
            to_tensor(image.resize((args.output_size, args.output_size), Image.Resampling.BOX))
            for image in originals
        ]).cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            prediction = model(images)
            depth = prediction["depth"].squeeze(0).squeeze(-1).float()
            confidence = prediction["depth_conf"].squeeze(0).float()
            extrinsics, intrinsics = encoding_to_camera(prediction["pose_enc"], prediction["images"].shape[-2:])
            extrinsics, intrinsics = extrinsics.squeeze(0).float(), intrinsics.squeeze(0).float()
            depth, confidence, intrinsics = resize_geometry(depth, confidence, intrinsics, args.output_size)
            grids, weights, depth_errors, photo_errors = [], [], [], []
            source = context_frames - 1
            for target in range(context_frames, total_frames):
                grid, weight, depth_error, photo_error = backward_static_constraint(
                    depth, confidence, extrinsics, intrinsics, rgb_small,
                    source, target, args,
                )
                grids.append(grid); weights.append(weight)
                depth_errors.append(depth_error); photo_errors.append(photo_error)
        out.parent.mkdir(parents=True, exist_ok=True)
        weight_tensor = torch.stack(weights)
        np.savez_compressed(
            out,
            static_grid=torch.stack(grids).cpu().numpy().astype(np.float16),
            static_confidence=weight_tensor.cpu().numpy().astype(np.float16),
            depth_consistency_error=torch.stack(depth_errors).cpu().numpy().astype(np.float16),
            photometric_error=torch.stack(photo_errors).cpu().numpy().astype(np.float16),
            metadata_json=json.dumps({
                "cache_path": str(cache_path), "source_path": metadata["source_path"], "split": split,
                "context_frames": context_frames, "future_frames": future_frames,
                "valid_fraction": float((weight_tensor > 0).float().mean()),
                "mean_confidence": float(weight_tensor.mean()), "resolution": args.resolution,
                "output_size": args.output_size, "checkpoint": args.checkpoint,
            }),
        )
        records.append({"split": split, "cache_path": str(cache_path), "output": str(out), "status": "created"})
        del prediction, images, rgb_small, depth, confidence, extrinsics, intrinsics
        if (index + 1) % 5 == 0 or index == 0:
            elapsed = time.time() - started
            print(json.dumps({
                "stage": "dense_vggt", "done": index + 1, "total": len(jobs),
                "elapsed_seconds": elapsed, "seconds_per_clip": elapsed / (index + 1),
                "eta_seconds": elapsed / (index + 1) * (len(jobs) - index - 1),
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                "last_valid_fraction": float((weight_tensor > 0).float().mean()),
            }), flush=True)
    manifest = {
        "complete": True, "config": vars(args), "split_meta": split_meta,
        "selected": {key: [str(path) for path in value] for key, value in selected.items()},
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({"complete": True, "items": len(jobs), "output": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
