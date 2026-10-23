"""Stream official PointOdyssey depth maps and recover missing 3D trajectories.

The official train tarball is a gzip stream split across four files.  This
script downloads one part at a time, concatenates them for tarfile, extracts
only depth PNGs for selected sequences, reconstructs world-space trajectories
at visible 2D track locations, writes compact WorldTrack clips, and deletes
temporary depth images after each sequence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import cv2
import numpy as np

from convert_pointodyssey_vlbm import convert_sequence
from stream_official_pointodyssey_targets import (
    DEFAULT_PARTS,
    DEFAULT_TARGETS,
    DownloadingConcatStream,
    member_sequence,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def is_depth(relative: tuple[str, ...]) -> bool:
    return (
        len(relative) == 2
        and relative[0] == "depths"
        and relative[1].lower().endswith(".png")
    )


def aligned_depth_paths(depth_dir: Path, frame_count: int) -> list[Path]:
    paths = sorted(depth_dir.glob("depth_*.png"))
    if len(paths) < frame_count:
        paths = sorted(depth_dir.glob("*.png"))
    if len(paths) < frame_count:
        raise RuntimeError(f"only {len(paths)} depth frames for {frame_count} annotations")
    return paths[:frame_count]


def fill_masked_tracks(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    """Interpolate hidden samples so tensors remain finite; masks stay unchanged."""
    frame_axis = np.arange(xyz.shape[0], dtype=np.float32)
    usable_tracks = 0
    for track in range(xyz.shape[1]):
        known = np.flatnonzero(valid[:, track])
        if known.size == 0:
            xyz[:, track] = 0.0
            continue
        usable_tracks += 1
        for coordinate in range(3):
            xyz[:, track, coordinate] = np.interp(
                frame_axis,
                known.astype(np.float32),
                xyz[known, track, coordinate],
            )
    return xyz, usable_tracks


def reconstruct_sequence(sequence_dir: Path, depth_dir: Path) -> dict:
    annotation_path = sequence_dir / "anno.npz"
    annotation = np.load(annotation_path, allow_pickle=True)
    uv = np.asarray(annotation["trajs_2d"], dtype=np.float32)
    valids = np.asarray(annotation["valids"], dtype=bool)
    visibs = np.asarray(annotation["visibs"], dtype=bool)
    intrinsics = np.asarray(annotation["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(annotation["extrinsics"], dtype=np.float32)

    if uv.ndim != 3 or uv.shape[-1] != 2:
        raise RuntimeError(f"invalid trajs_2d shape {uv.shape}")
    frames, tracks = uv.shape[:2]
    if valids.shape != (frames, tracks) or visibs.shape != (frames, tracks):
        raise RuntimeError(
            f"mask mismatch uv={uv.shape} valids={valids.shape} visibs={visibs.shape}"
        )
    if intrinsics.ndim == 2:
        intrinsics = np.repeat(intrinsics[None], frames, axis=0)
    if extrinsics.ndim == 2:
        extrinsics = np.repeat(extrinsics[None], frames, axis=0)
    if intrinsics.shape != (frames, 3, 3) or extrinsics.shape != (frames, 4, 4):
        raise RuntimeError(
            f"camera mismatch intrinsics={intrinsics.shape} extrinsics={extrinsics.shape}"
        )

    depth_paths = aligned_depth_paths(depth_dir, frames)
    xyz_world = np.zeros((frames, tracks, 3), dtype=np.float32)
    reconstructed = np.zeros((frames, tracks), dtype=bool)
    depth_min = np.inf
    depth_max = 0.0

    for frame, depth_path in enumerate(depth_paths):
        depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_u16 is None or depth_u16.ndim != 2:
            raise RuntimeError(f"cannot read depth image {depth_path}")
        height, width = depth_u16.shape
        u = np.rint(uv[frame, :, 0]).astype(np.int64)
        v = np.rint(uv[frame, :, 1]).astype(np.int64)
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        candidate = valids[frame] & visibs[frame] & in_image & np.isfinite(uv[frame]).all(1)
        ids = np.flatnonzero(candidate)
        if ids.size == 0:
            continue
        z = depth_u16[v[ids], u[ids]].astype(np.float32) / 65535.0 * 1000.0
        good_depth = np.isfinite(z) & (z > 1e-6) & (z < 999.0)
        ids = ids[good_depth]
        z = z[good_depth]
        if ids.size == 0:
            continue

        k_inv = np.linalg.inv(intrinsics[frame])
        pixels = np.stack(
            (uv[frame, ids, 0], uv[frame, ids, 1], np.ones(ids.size, dtype=np.float32)),
            axis=0,
        )
        xyz_cam = (k_inv @ pixels) * z[None]
        xyz_cam_h = np.concatenate(
            (xyz_cam, np.ones((1, ids.size), dtype=np.float32)), axis=0
        )
        xyz_w = (np.linalg.inv(extrinsics[frame]) @ xyz_cam_h)[:3].T
        finite = np.isfinite(xyz_w).all(1)
        ids = ids[finite]
        xyz_world[frame, ids] = xyz_w[finite]
        reconstructed[frame, ids] = True
        if ids.size:
            depth_min = min(depth_min, float(z[finite].min()))
            depth_max = max(depth_max, float(z[finite].max()))

    xyz_world, usable_tracks = fill_masked_tracks(xyz_world, reconstructed)
    effective_visibility = visibs & valids & reconstructed

    # Verify the recovered visible points project back to the source UVs.
    sample_t, sample_n = np.nonzero(reconstructed)
    if sample_t.size == 0:
        raise RuntimeError("no valid visible depth samples")
    stride = max(1, sample_t.size // 20000)
    sample_t, sample_n = sample_t[::stride], sample_n[::stride]
    residuals = []
    for frame in np.unique(sample_t):
        choose = sample_t == frame
        ids = sample_n[choose]
        points = xyz_world[frame, ids]
        hom = np.concatenate((points, np.ones((len(ids), 1), dtype=np.float32)), axis=1)
        cam = (extrinsics[frame] @ hom.T).T[:, :3]
        projected = (intrinsics[frame] @ cam.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        residuals.append(np.linalg.norm(projected - uv[frame, ids], axis=1))
    reprojection = np.concatenate(residuals)

    np.save(sequence_dir / "trajs_2d.npy", uv.astype(np.float16))
    np.save(sequence_dir / "trajs_3d.npy", xyz_world.astype(np.float16))
    np.save(sequence_dir / "intrinsics.npy", intrinsics.astype(np.float16))
    np.save(sequence_dir / "extrinsics.npy", extrinsics.astype(np.float16))
    np.save(sequence_dir / "visibilities.npy", effective_visibility.astype(np.float16))

    rgb_paths = sorted((sequence_dir / "rgbs").glob("*.jpg"))
    if not rgb_paths:
        rgb_paths = sorted((sequence_dir / "rgbs").glob("*.png"))
    if not rgb_paths:
        raise RuntimeError("RGB frames are missing")
    image = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read RGB image {rgb_paths[0]}")
    scene_info = {
        "num_frames": frames,
        "image_size": [int(image.shape[0]), int(image.shape[1])],
        "num_trajectories": tracks,
        "source": "PointOdyssey official depth backprojection",
        "depth_range_m": [float(depth_min), float(depth_max)],
        "reconstructed_visible_fraction": float(reconstructed.mean()),
        "usable_tracks": int(usable_tracks),
        "occluded_xyz_policy": "linear interpolation with original visibility kept false",
        "reprojection_error_px_mean": float(reprojection.mean()),
        "reprojection_error_px_p95": float(np.percentile(reprojection, 95)),
    }
    (sequence_dir / "scene_info.json").write_text(
        json.dumps(scene_info, indent=2), encoding="utf-8"
    )
    return scene_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--depth-staging-dir", required=True, type=Path)
    parser.add_argument("--spool-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default="aharley/pointodyssey")
    parser.add_argument("--hf-cli", type=Path, default=Path("/root/miniconda3/bin/hf"))
    parser.add_argument("--download-retries", type=int, default=30)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--max-clips-per-sequence", type=int, default=10)
    parser.add_argument("--max-tracks", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    args = parser.parse_args()

    targets = set(args.targets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.depth_staging_dir.mkdir(parents=True, exist_ok=True)
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    conversion_args = SimpleNamespace(
        clip_frames=args.clip_frames,
        max_clips_per_sequence=args.max_clips_per_sequence,
        max_tracks=args.max_tracks,
        seed=args.seed,
        compressed=False,
        overwrite=True,
    )
    stream = DownloadingConcatStream(
        repo_id=args.repo_id,
        parts=DEFAULT_PARTS,
        spool_dir=args.spool_dir,
        hf_cli=args.hf_cli,
        retries=args.download_retries,
    )
    recovered: set[str] = set()
    seen: set[str] = set()
    rows: list[dict] = []
    failures: dict[str, str] = {}
    quality: dict[str, dict] = {}

    def finish(sequence: str | None) -> None:
        if not sequence or sequence not in targets or sequence in recovered:
            return
        sequence_dir = args.work_dir / sequence
        depth_dir = args.depth_staging_dir / sequence / "depths"
        if not (sequence_dir / "anno.npz").is_file() or not depth_dir.is_dir():
            return
        try:
            info = reconstruct_sequence(sequence_dir, depth_dir)
            made = convert_sequence(sequence_dir, args.output_root, conversion_args)
            rows.extend(made)
            usable = sum(row.get("status") in {"ok", "exists"} for row in made)
            if usable <= 0:
                raise RuntimeError("conversion produced no usable clips")
            recovered.add(sequence)
            failures.pop(sequence, None)
            quality[sequence] = info
            log(
                f"SEQUENCE_RECOVERED name={sequence} clips={usable} "
                f"visible_fraction={info['reconstructed_visible_fraction']:.4f} "
                f"reprojection_p95={info['reprojection_error_px_p95']:.6f}"
            )
            shutil.rmtree(depth_dir, ignore_errors=True)
        except Exception as exc:
            failures[sequence] = repr(exc)
            log(f"SEQUENCE_ERROR name={sequence} error={exc!r}")

    current_sequence: str | None = None
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                sequence, relative = member_sequence(member.name)
                if sequence != current_sequence:
                    finish(current_sequence)
                    current_sequence = sequence
                if sequence not in targets or not is_depth(relative):
                    continue
                if ".." in relative or member.issym() or member.islnk():
                    continue
                seen.add(sequence)
                member.name = str(PurePosixPath(sequence, *relative))
                archive.extract(member, path=args.depth_staging_dir, set_attrs=False)
            finish(current_sequence)
    finally:
        stream.close()

    for sequence in sorted(targets - recovered):
        finish(sequence)
    missing = sorted(targets - recovered)
    for sequence in missing:
        failures.setdefault(sequence, "depth files were incomplete in the official archive")

    manifest_path = args.output_root / "official_depth_recovery_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "targets": len(targets),
        "targets_seen": sorted(seen),
        "recovered": len(recovered),
        "recovered_names": sorted(recovered),
        "missing": missing,
        "failures": failures,
        "clips_ready": sum(row.get("status") in {"ok", "exists"} for row in rows),
        "quality": quality,
    }
    (args.output_root / "official_depth_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"DEPTH_RECOVERY_SUMMARY {json.dumps(summary, ensure_ascii=False)}")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
