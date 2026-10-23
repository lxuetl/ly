"""Convert PointOdyssey VLBM sequences into leakage-safe WorldTrack clips."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


REQUIRED = ("intrinsics.npy", "extrinsics.npy", "trajs_2d.npy", "trajs_3d.npy", "visibilities.npy")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--clip-frames", type=int, default=24)
    p.add_argument("--max-clips-per-sequence", type=int, default=10)
    p.add_argument("--max-tracks", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-") or "sequence"


def sequence_split(name: str) -> str:
    bucket = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else ("val" if bucket < 8 else "test")


def clip_starts(num_frames: int, clip_frames: int, maximum: int) -> list[int]:
    if num_frames < clip_frames:
        return []
    possible = num_frames - clip_frames + 1
    count = min(maximum, possible)
    return sorted(set(np.linspace(0, possible - 1, count, dtype=np.int64).tolist()))


def rgb_paths(sequence: Path) -> list[Path]:
    paths = sorted((sequence / "rgbs").glob("rgb_*.jpg"))
    if not paths:
        paths = sorted((sequence / "rgbs").glob("*.jpg"))
    return paths


def world_to_camera(world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    ones = np.ones((*world.shape[:-1], 1), dtype=np.float32)
    hom = np.concatenate((world.astype(np.float32), ones), axis=-1)
    return np.einsum("tij,tnj->tni", w2c.astype(np.float32), hom, optimize=True)[..., :3]


def convert_sequence(sequence: Path, out: Path, args) -> list[dict]:
    rgb = rgb_paths(sequence)
    intr = np.load(sequence / "intrinsics.npy", mmap_mode="r")
    extr = np.load(sequence / "extrinsics.npy", mmap_mode="r")
    uv = np.load(sequence / "trajs_2d.npy", mmap_mode="r")
    xyz = np.load(sequence / "trajs_3d.npy", mmap_mode="r")
    vis = np.load(sequence / "visibilities.npy", mmap_mode="r")
    lengths = [len(rgb), intr.shape[0], extr.shape[0], uv.shape[0], xyz.shape[0], vis.shape[0]]
    total = min(lengths)
    if uv.shape[1] != xyz.shape[1] or uv.shape[1] != vis.shape[1]:
        raise ValueError(f"trajectory count mismatch: uv={uv.shape}, xyz={xyz.shape}, vis={vis.shape}")
    # Sanitizing alone can collapse distinct source names (for example names
    # that differ only by a trailing underscore).  Keep a stable source-name
    # hash in every clip id so clips can never silently overwrite each other.
    sequence_hash = hashlib.sha1(sequence.name.encode()).hexdigest()[:8]
    seq_name = f"{safe_name(sequence.name)}-{sequence_hash}"
    rows = []
    for clip_i, start in enumerate(clip_starts(total, args.clip_frames, args.max_clips_per_sequence)):
        end = start + args.clip_frames
        name = f"{seq_name}_clip_{clip_i:03d}"
        target = out / f"{name}.npz"
        if target.exists() and not args.overwrite:
            rows.append({"status": "exists", "file": str(target), "sequence": sequence.name,
                         "start": start, "end": end, "split": sequence_split(sequence.name)})
            continue
        uv_clip = np.asarray(uv[start:end], dtype=np.float32)
        xyz_world = np.asarray(xyz[start:end], dtype=np.float32)
        visible = np.asarray(vis[start:end] > 0.5, dtype=bool)
        valid = visible[0] & np.isfinite(uv_clip[0]).all(-1) & np.isfinite(xyz_world[0]).all(-1)
        ids = np.flatnonzero(valid)
        if not len(ids):
            rows.append({"status": "skipped", "reason": "no frame-0 visible finite tracks",
                         "sequence": sequence.name, "start": start, "end": end})
            continue
        if args.max_tracks > 0 and len(ids) > args.max_tracks:
            local_seed = args.seed + int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
            ids = np.sort(np.random.default_rng(local_seed).choice(ids, args.max_tracks, replace=False))
        uv_clip = uv_clip[:, ids]
        xyz_world = xyz_world[:, ids]
        visible = visible[:, ids] & np.isfinite(uv_clip).all(-1) & np.isfinite(xyz_world).all(-1)
        extr_clip = np.asarray(extr[start:end], dtype=np.float32)
        xyz_cam = world_to_camera(xyz_world, extr_clip)
        visible &= np.isfinite(xyz_cam).all(-1) & (xyz_cam[..., 2] > 1e-6)
        images = np.empty(args.clip_frames, dtype=object)
        for i, path in enumerate(rgb[start:end]):
            images[i] = path.read_bytes()
        k = np.asarray(intr[start], dtype=np.float32)
        fx_fy_cx_cy = np.asarray((k[0, 0], k[1, 1], k[0, 2], k[1, 2]), dtype=np.float32)
        metadata = {
            "schema_version": 1,
            "source": "PointOdyssey_v1.2_VLBM",
            "source_sequence": sequence.name,
            "source_frame_start": int(start),
            "source_frame_end_exclusive": int(end),
            "split_hint": sequence_split(sequence.name),
            "image_size": [int(json.loads((sequence / "scene_info.json").read_text()).get("image_size", [0, 0])[0]),
                           int(json.loads((sequence / "scene_info.json").read_text()).get("image_size", [0, 0])[1])],
            "xyz_source": "world coordinates transformed by per-frame W2C extrinsics",
            "num_tracks": int(len(ids)),
        }
        payload = dict(images_jpeg_bytes=images, tracks_XYZ=xyz_cam.astype(np.float32),
                       fx_fy_cx_cy=fx_fy_cx_cy, visibility=visible,
                       extrinsics_w2c=extr_clip, tracks_uv_source=uv_clip.astype(np.float32),
                       source_track_ids=ids.astype(np.int64), metadata_json=np.asarray(json.dumps(metadata)))
        (np.savez_compressed if args.compressed else np.savez)(target, **payload)
        rows.append({"status": "ok", "file": str(target), "sequence": sequence.name,
                     "start": int(start), "end": int(end), "tracks": int(len(ids)),
                     "split": sequence_split(sequence.name)})
    return rows


def main():
    args = parse_args()
    source, out = Path(args.source_root), Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    sequences = sorted(p for p in source.iterdir() if p.is_dir() and
                       all((p / name).is_file() for name in REQUIRED) and (p / "rgbs").is_dir())
    rows = []
    for index, sequence in enumerate(sequences, 1):
        try:
            made = convert_sequence(sequence, out, args)
            rows.extend(made)
            print(f"[{index}/{len(sequences)}] {sequence.name}: {sum(x['status'] == 'ok' for x in made)} clips", flush=True)
        except Exception as exc:
            rows.append({"status": "error", "sequence": sequence.name, "error": repr(exc)})
            print(f"[{index}/{len(sequences)}] {sequence.name}: ERROR {exc}", flush=True)
    (out / "conversion_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"sequences_found": len(sequences), "clips_ok": sum(x["status"] == "ok" for x in rows),
               "clips_existing": sum(x["status"] == "exists" for x in rows),
               "clips_skipped": sum(x["status"] == "skipped" for x in rows),
               "errors": sum(x["status"] == "error" for x in rows)}
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
