"""Full integrity and loader acceptance for converted WorldTrack-style clips."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


REQUIRED = {
    "images_jpeg_bytes",
    "tracks_XYZ",
    "fx_fy_cx_cy",
    "visibility",
    "extrinsics_w2c",
    "tracks_uv_source",
    "source_track_ids",
    "metadata_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo / "forecasting"))
    from prepare_pointodyssey_splits import RawClipDataset, collate

    started = time.time()
    paths = sorted(args.root.glob("*.npz"))
    manifest_path = args.root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[dict] = []
    warnings: list[dict] = []
    sequence_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    image_shapes: Counter[str] = Counter()
    frame_counts: Counter[int] = Counter()
    track_counts: list[int] = []
    visible_total = 0
    point_total = 0
    visible_positive_z = 0
    visible_in_bounds = 0
    intervals: dict[str, set[tuple[int, int]]] = defaultdict(set)
    clip_to_sequence: dict[str, str] = {}

    def fail(path: Path, message: str) -> None:
        errors.append({"file": path.name, "error": message})

    for index, path in enumerate(paths, 1):
        try:
            if path.is_symlink() and not path.resolve().is_file():
                raise ValueError("broken symlink")
            with np.load(path, allow_pickle=True) as z:
                missing = sorted(REQUIRED - set(z.files))
                if missing:
                    raise ValueError(f"missing keys {missing}")
                metadata = json.loads(str(z["metadata_json"]))
                images = z["images_jpeg_bytes"]
                xyz = np.asarray(z["tracks_XYZ"], dtype=np.float32)
                uv = np.asarray(z["tracks_uv_source"], dtype=np.float32)
                visibility = np.asarray(z["visibility"], dtype=bool)
                intrinsics = np.asarray(z["fx_fy_cx_cy"], dtype=np.float32)
                extrinsics = np.asarray(z["extrinsics_w2c"], dtype=np.float32)
                track_ids = np.asarray(z["source_track_ids"])

            if xyz.ndim != 3 or xyz.shape[-1] != 3:
                raise ValueError(f"invalid XYZ shape {xyz.shape}")
            frames, tracks = xyz.shape[:2]
            if uv.shape != (frames, tracks, 2):
                raise ValueError(f"UV shape {uv.shape} != {(frames, tracks, 2)}")
            if visibility.shape != (frames, tracks):
                raise ValueError(f"visibility shape {visibility.shape}")
            if extrinsics.shape != (frames, 4, 4):
                raise ValueError(f"extrinsics shape {extrinsics.shape}")
            if intrinsics.shape != (4,):
                raise ValueError(f"intrinsics shape {intrinsics.shape}")
            if track_ids.shape != (tracks,):
                raise ValueError(f"track id shape {track_ids.shape}")
            if len(images) != frames:
                raise ValueError(f"JPEG count {len(images)} != frames {frames}")
            if frames != 24:
                raise ValueError(f"expected 24 frames, got {frames}")
            if tracks <= 0 or tracks > 2048:
                raise ValueError(f"invalid track count {tracks}")
            if not np.isfinite(xyz).all():
                raise ValueError("XYZ contains non-finite values")
            if not np.isfinite(extrinsics).all():
                raise ValueError("extrinsics contain non-finite values")
            if not np.isfinite(intrinsics).all() or intrinsics[0] <= 0 or intrinsics[1] <= 0:
                raise ValueError(f"invalid intrinsics {intrinsics.tolist()}")
            if not np.allclose(extrinsics[:, 3], np.array([0, 0, 0, 1]), atol=2e-2):
                raise ValueError("invalid homogeneous extrinsics last row")
            if visibility.any() and not np.isfinite(uv[visibility]).all():
                raise ValueError("visible UV contains non-finite values")

            decoded_shape = None
            for frame_index, encoded in enumerate(images):
                image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"JPEG decode failed at frame {frame_index}")
                shape = tuple(int(x) for x in image.shape)
                if decoded_shape is None:
                    decoded_shape = shape
                elif shape != decoded_shape:
                    raise ValueError(f"inconsistent image shapes {decoded_shape} and {shape}")
            assert decoded_shape is not None
            height, width = decoded_shape[:2]

            sequence = str(metadata["source_sequence"])
            start = int(metadata["source_frame_start"])
            end = int(metadata["source_frame_end_exclusive"])
            if end - start != frames:
                raise ValueError(f"metadata frame span {start}:{end} != {frames}")
            if (start, end) in intervals[sequence]:
                raise ValueError(f"duplicate source interval {sequence}:{start}:{end}")
            intervals[sequence].add((start, end))
            clip_to_sequence[path.name] = sequence
            sequence_counts[sequence] += 1
            source_counts[str(metadata.get("source", "unknown"))] += 1
            image_shapes[str(list(decoded_shape))] += 1
            frame_counts[frames] += 1
            track_counts.append(tracks)

            visible_count = int(visibility.sum())
            visible_total += visible_count
            point_total += int(visibility.size)
            if visible_count:
                visible_positive_z += int((xyz[..., 2][visibility] > 0).sum())
                visible_uv = uv[visibility]
                in_bounds = (
                    (visible_uv[:, 0] >= -1.0)
                    & (visible_uv[:, 0] <= width)
                    & (visible_uv[:, 1] >= -1.0)
                    & (visible_uv[:, 1] <= height)
                )
                visible_in_bounds += int(in_bounds.sum())
        except Exception as exc:
            fail(path, str(exc))
        if index % 100 == 0 or index == len(paths):
            print(f"STRUCTURE {index}/{len(paths)} errors={len(errors)}", flush=True)

    expected_names = {p.name for p in paths}
    listed_names: list[str] = []
    split_sequences: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        names = list(manifest["clips"][split])
        listed_names.extend(names)
        split_sequences[split] = {clip_to_sequence[name] for name in names if name in clip_to_sequence}
        text_names = [x for x in (args.root / f"{split}.txt").read_text().splitlines() if x]
        if names != text_names:
            errors.append({"file": f"{split}.txt", "error": "text list differs from manifest"})
    if len(listed_names) != len(set(listed_names)):
        errors.append({"file": "split_manifest.json", "error": "duplicate clips across splits"})
    if set(listed_names) != expected_names:
        errors.append({
            "file": "split_manifest.json",
            "error": f"manifest mismatch missing={len(expected_names-set(listed_names))} extra={len(set(listed_names)-expected_names)}",
        })
    overlap = {
        "train_val": sorted(split_sequences["train"] & split_sequences["val"]),
        "train_test": sorted(split_sequences["train"] & split_sequences["test"]),
        "val_test": sorted(split_sequences["val"] & split_sequences["test"]),
    }
    if any(overlap.values()):
        errors.append({"file": "split_manifest.json", "error": f"sequence leakage {overlap}"})

    loader_checks = {}
    for split in ("train", "val", "test"):
        split_paths = [args.root / name for name in manifest["clips"][split]]
        loader = DataLoader(
            RawClipDataset(split_paths),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        sample_count = 0
        batch_count = 0
        valid_count = 0
        loader_points = 0
        loader_finite = True
        for batch in loader:
            batch_count += 1
            sample_count += int(batch["rgb"].shape[0])
            valid_count += int(batch["valid"].sum())
            loader_points += int(batch["valid"].numel())
            loader_finite &= bool(torch.isfinite(batch["xyz"]).all())
            loader_finite &= bool(torch.isfinite(batch["uv"]).all())
            loader_finite &= bool(torch.isfinite(batch["intrinsics"]).all())
            loader_finite &= bool(torch.isfinite(batch["extrinsics"]).all())
            if batch_count % 100 == 0:
                print(f"LOADER {split} batches={batch_count} samples={sample_count}", flush=True)
        loader_checks[split] = {
            "samples": sample_count,
            "batches": batch_count,
            "finite": loader_finite,
            "valid_fraction": valid_count / max(loader_points, 1),
        }
        if sample_count != len(split_paths) or not loader_finite:
            errors.append({"file": split, "error": f"loader acceptance failed: {loader_checks[split]}"})

    report = {
        "status": "pass" if not errors else "fail",
        "elapsed_seconds": time.time() - started,
        "clips_total": len(paths),
        "sequences_total": len(sequence_counts),
        "split_clip_counts": {s: len(manifest["clips"][s]) for s in ("train", "val", "test")},
        "split_sequence_counts": {s: len(split_sequences[s]) for s in split_sequences},
        "sequence_leakage": overlap,
        "frame_counts": dict(frame_counts),
        "track_count": {
            "min": min(track_counts) if track_counts else 0,
            "max": max(track_counts) if track_counts else 0,
            "mean": float(np.mean(track_counts)) if track_counts else 0.0,
        },
        "image_shapes": dict(image_shapes),
        "source_counts": dict(source_counts),
        "visible_fraction": visible_total / max(point_total, 1),
        "visible_positive_z_fraction": visible_positive_z / max(visible_total, 1),
        "visible_uv_in_bounds_fraction": visible_in_bounds / max(visible_total, 1),
        "loader_checks": loader_checks,
        "errors": errors,
        "warnings": warnings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
