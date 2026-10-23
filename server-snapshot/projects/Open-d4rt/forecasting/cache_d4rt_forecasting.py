"""Build leakage-safe D4RT geometry caches for future-trajectory prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_track3d_in_worldtrack import load_worldtrack_sequence
from infer_track_3d import _infer_tracks, _resize_video, _resolve_device, _unwrap_state_dict
from src.core import load_checkpoint, load_yaml_config, seed_everything
from src.eval.tasks import _umeyama_sim3
from src.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    p.add_argument("--ckpt-path", default="checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    p.add_argument("--data-root", default="data/worldtrack_release")
    p.add_argument("--subsets", default="adt_mini")
    p.add_argument("--output-dir", default="data/d4rt_forecasting_cache")
    p.add_argument("--context-frames", type=int, default=16)
    p.add_argument("--future-frames", type=int, default=8)
    p.add_argument("--num-query-points", type=int, default=256)
    p.add_argument("--query-chunk-size", type=int, default=128)
    p.add_argument("--alignment-keep-ratio", type=float, default=0.85)
    p.add_argument("--limit-seqs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _select_queries(sample: dict[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    visible = np.asarray(sample["visibility"][0], dtype=bool)
    uv_all = np.asarray(sample["tracks_uv"][0], dtype=np.float64)
    z_all = np.asarray(sample["tracks_xyz_cam"][0, :, 2], dtype=np.float64)
    valid = visible & np.isfinite(uv_all).all(axis=1) & np.isfinite(z_all) & (np.abs(z_all) > 1e-8)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        raise RuntimeError("no finite frame-0 visible queries")
    if n > 0 and ids.size > n:
        ids = np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))
    h, w = sample["video_rgb"].shape[1:3]
    uv = uv_all[ids].astype(np.float32)
    uv[:, 0] /= float(max(w - 1, 1))
    uv[:, 1] /= float(max(h - 1, 1))
    return ids.astype(np.int64), np.clip(uv, 0.0, 1.0)


def _fit_context_to_teacher(
    ctx: dict[str, np.ndarray], full: dict[str, np.ndarray], c: int, keep_ratio: float
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.asarray(ctx["tracks_xyz_ref0"][:, :c], dtype=np.float64)
    y = np.asarray(full["tracks_xyz_ref0"][:, :c], dtype=np.float64)
    cx = np.asarray(ctx["tracks_confidence"][:, :c], dtype=np.float64)
    cy = np.asarray(full["tracks_confidence"][:, :c], dtype=np.float64)
    valid = np.isfinite(x).all(2) & np.isfinite(y).all(2) & np.isfinite(cx) & np.isfinite(cy)
    valid &= np.asarray(ctx["tracks_visibility"][:, :c], bool)
    valid &= np.asarray(full["tracks_visibility"][:, :c], bool)
    score = np.minimum(cx, cy)
    threshold = float("nan")
    if np.any(valid):
        ratio = float(np.clip(keep_ratio, 0.05, 1.0))
        threshold = float(np.quantile(score[valid], 1.0 - ratio))
        valid &= score >= threshold
    src, dst = x[valid], y[valid]
    fit = _umeyama_sim3(src, dst)
    if fit is None or fit[0] <= 1e-8:
        raise RuntimeError(f"Sim(3) alignment failed with {src.shape[0]} pairs")
    scale, rot, trans = fit
    residual = scale * (rot @ src.T).T + trans - dst
    diag = {
        "success": True,
        "num_pairs": int(src.shape[0]),
        "keep_ratio": float(keep_ratio),
        "confidence_threshold": threshold,
        "rmse": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "scale": float(scale),
    }
    return float(scale), rot, trans, diag


def _full_to_context(x: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    shape = x.shape
    flat = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    out = ((rot.T @ (flat - trans).T).T / scale).reshape(shape)
    return out.astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.context_frames <= 0 or args.future_frames <= 0:
        raise ValueError("context/future frame counts must be positive")
    total = args.context_frames + args.future_frames
    cfg = load_yaml_config(args.model_config)
    seed_everything(args.seed, deterministic=True)
    model = build_model(cfg["model"]).eval()
    state = _unwrap_state_dict(load_checkpoint(args.ckpt_path, map_location="cpu"))
    if not state:
        raise RuntimeError("checkpoint contains no state dict")
    loaded = model.load_state_dict(state, strict=False)
    device = _resolve_device(args.device)
    model.to(device).eval()
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    image_hw = (int(image_size[0]), int(image_size[1]))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    paths: list[tuple[str, Path]] = []
    for subset in [x.strip() for x in args.subsets.split(",") if x.strip()]:
        paths.extend((subset, p) for p in sorted((Path(args.data_root) / subset).glob("*.npz")))
    if args.limit_seqs > 0:
        paths = paths[: args.limit_seqs]
    print(f"model loaded: missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)}")
    for seq_i, (subset, path) in enumerate(paths):
        try:
            sample = load_worldtrack_sequence(path, num_frames=total)
            if sample["video_rgb"].shape[0] < total:
                raise RuntimeError(f"sequence has fewer than {total} frames")
            query_ids, query_uv = _select_queries(sample, args.num_query_points, args.seed + seq_i)
            video = _resize_video(sample["video_rgb"], image_hw=image_hw)
            ctx = _infer_tracks(model=model, video_model_rgb=video[: args.context_frames],
                                query_uv_norm=query_uv, query_chunk_size=args.query_chunk_size)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            full = _infer_tracks(model=model, video_model_rgb=video,
                                 query_uv_norm=query_uv, query_chunk_size=args.query_chunk_size)
            scale, rot, trans, align = _fit_context_to_teacher(
                ctx, full, args.context_frames, args.alignment_keep_ratio
            )
            c = args.context_frames
            future_full = np.asarray(full["tracks_xyz_ref0"][:, c:total], np.float32)
            target_ctx_gauge = _full_to_context(future_full, scale, rot, trans)
            past_xyz = np.asarray(ctx["tracks_xyz_ref0"][:, :c], np.float32)
            past_vis = np.asarray(ctx["tracks_visibility"][:, :c], bool)
            past_conf = np.asarray(ctx["tracks_confidence"][:, :c], np.float32)
            future_vis = np.asarray(full["tracks_visibility"][:, c:total], bool)
            future_conf = np.asarray(full["tracks_confidence"][:, c:total], np.float32)
            past_valid = past_vis & np.isfinite(past_xyz).all(2)
            future_valid = future_vis & np.isfinite(target_ctx_gauge).all(2)
            subset_dir = out_root / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            out_path = subset_dir / f"{path.stem}.npz"
            metadata = {
                "schema_version": 1, "subset": subset, "video_name": path.stem,
                "source_path": str(path), "context_frames": c, "future_frames": args.future_frames,
                "num_queries": int(query_uv.shape[0]), "coordinate_system": "D4RT context-only t_cam=0 gauge",
                "teacher": "full C+K D4RT; future labels only", "alignment": align,
            }
            np.savez_compressed(
                out_path, query_ids=query_ids, query_uv=query_uv,
                past_xyz=past_xyz, past_uv=np.asarray(ctx["tracks_uv_norm"][:, :c], np.float32),
                past_visibility=past_vis, past_confidence=past_conf, past_valid=past_valid,
                future_xyz_teacher=target_ctx_gauge, future_visibility=future_vis,
                future_confidence=future_conf, future_valid=future_valid,
                sim3_context_to_teacher_scale=np.float32(scale),
                sim3_context_to_teacher_rotation=rot.astype(np.float32),
                sim3_context_to_teacher_translation=trans.astype(np.float32),
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            row = {"status": "ok", "file": str(out_path), **metadata}
            print(f"[{seq_i + 1}/{len(paths)}] wrote {out_path} queries={query_uv.shape[0]} align_rmse={align['rmse']:.5f}")
        except Exception as exc:
            row = {"status": "error", "subset": subset, "video_name": path.stem, "error": str(exc)}
            print(f"[{seq_i + 1}/{len(paths)}] skipped {path.name}: {exc}")
        manifest.append(row)
    manifest_path = out_root / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in manifest), encoding="utf-8")
    ok = sum(x["status"] == "ok" for x in manifest)
    print(f"done: {ok}/{len(manifest)} caches; manifest={manifest_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
