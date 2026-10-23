"""Construct D4RT query tuples for downstream tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class QueryBatch:
    u: np.ndarray
    v: np.ndarray
    t_src: np.ndarray
    t_tgt: np.ndarray
    t_cam: np.ndarray

    def __post_init__(self) -> None:
        sizes = {np.asarray(x).reshape(-1).shape[0] for x in (self.u, self.v, self.t_src, self.t_tgt, self.t_cam)}
        if len(sizes) != 1:
            raise ValueError("all query fields must have equal length")

    @property
    def size(self) -> int:
        return int(np.asarray(self.u).size)

    def to_torch(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        return {
            "u": torch.as_tensor(self.u, dtype=torch.float32, device=device),
            "v": torch.as_tensor(self.v, dtype=torch.float32, device=device),
            "t_src": torch.as_tensor(self.t_src, dtype=torch.long, device=device),
            "t_tgt": torch.as_tensor(self.t_tgt, dtype=torch.long, device=device),
            "t_cam": torch.as_tensor(self.t_cam, dtype=torch.long, device=device),
        }


def _uv(candidate_uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(candidate_uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape [K,2], got {uv.shape}")
    return uv


def build_depth_queries(uv: np.ndarray, frame: int) -> QueryBatch:
    uv = _uv(uv)
    t = np.full((uv.shape[0],), int(frame), dtype=np.int64)
    return QueryBatch(uv[:, 0], uv[:, 1], t, t.copy(), t.copy())


def build_pose_queries(uv: np.ndarray, frame_i: int, frame_j: int) -> tuple[QueryBatch, slice, slice]:
    """Build the two point sets required by D4RT's Umeyama pose solver."""
    uv = _uv(uv)
    k = uv.shape[0]
    repeated = np.concatenate([uv, uv], axis=0)
    t_src = np.full((2 * k,), int(frame_i), dtype=np.int64)
    t_tgt = np.full((2 * k,), int(frame_i), dtype=np.int64)
    t_cam = np.concatenate([
        np.full((k,), int(frame_i), dtype=np.int64),
        np.full((k,), int(frame_j), dtype=np.int64),
    ])
    return QueryBatch(repeated[:, 0], repeated[:, 1], t_src, t_tgt, t_cam), slice(0, k), slice(k, 2 * k)


def build_track_queries(
    uv: np.ndarray,
    source_frames: np.ndarray,
    num_frames: int,
    world_reference_frame: int | None = None,
) -> QueryBatch:
    uv = _uv(uv)
    src = np.asarray(source_frames, dtype=np.int64).reshape(-1)
    if src.shape[0] != uv.shape[0]:
        raise ValueError("source_frames must have one value per query point")
    repeated = np.repeat(uv, int(num_frames), axis=0)
    t_src = np.repeat(src, int(num_frames))
    t_tgt = np.tile(np.arange(int(num_frames), dtype=np.int64), uv.shape[0])
    t_cam = t_tgt.copy() if world_reference_frame is None else np.full_like(t_tgt, int(world_reference_frame))
    return QueryBatch(repeated[:, 0], repeated[:, 1], t_src, t_tgt, t_cam)


def build_point_cloud_queries(uv: np.ndarray, source_frames: np.ndarray, reference_frame: int) -> QueryBatch:
    uv = _uv(uv)
    src = np.asarray(source_frames, dtype=np.int64).reshape(-1)
    if src.shape[0] != uv.shape[0]:
        raise ValueError("source_frames must have one value per query point")
    ref = np.full_like(src, int(reference_frame))
    return QueryBatch(uv[:, 0], uv[:, 1], src, src.copy(), ref)
