"""Pair cached D4RT geometry with source RGB clips."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_dataset import split_paths
from eval_track3d_in_worldtrack import load_worldtrack_sequence


class VideoForecastDataset(Dataset):
    def __init__(self, root, split, image_size=64, subset=None, paths=None, cache_items=False,
                 explicit_trajectory_interface=False):
        self.paths = ([Path(x) for x in paths] if paths is not None
                      else split_paths(root, split, subset=subset))
        self.image_size = int(image_size)
        # Opt-in prevents silently changing older, already evaluated protocols.
        self.explicit_trajectory_interface = bool(explicit_trajectory_interface)
        self._cache = {} if cache_items else None
        if not self.paths: raise RuntimeError(f"No {split} clips under {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        path = self.paths[index]; z = np.load(path)
        meta = json.loads(str(z["metadata_json"])); c, k = meta["context_frames"], meta["future_frames"]
        sample = load_worldtrack_sequence(Path(meta["source_path"]), c + k)
        frames = np.stack([cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                           for x in sample["video_rgb"]], 0)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        keys = ("query_uv", "past_xyz", "past_uv", "past_valid", "past_confidence",
                "future_xyz_teacher", "future_valid", "future_confidence")
        out = {x: torch.tensor(np.asarray(z[x])) for x in keys}
        if "future_uv_teacher" in z.files:
            future_uv = np.asarray(z["future_uv_teacher"], dtype=np.float32)
        else:
            ids = np.asarray(z["query_ids"], dtype=np.int64)
            future_uv = np.asarray(sample["tracks_uv"][c:c+k, ids], dtype=np.float32).transpose(1, 0, 2)
            h, w = sample["video_rgb"].shape[1:3]
            future_uv[..., 0] /= float(max(w - 1, 1)); future_uv[..., 1] /= float(max(h - 1, 1))
            future_uv = np.clip(future_uv, 0.0, 1.0)
        out["future_uv_teacher"] = torch.tensor(future_uv)
        if self.explicit_trajectory_interface:
            # GT labels only. Never substitute D4RT full-clip visibility for GT.
            ids = np.asarray(z["query_ids"], dtype=np.int64)
            raw_uv = np.asarray(sample["tracks_uv"][:, ids], dtype=np.float32).transpose(1, 0, 2).copy()
            h, w = sample["video_rgb"].shape[1:3]
            raw_uv /= np.asarray([max(w-1, 1), max(h-1, 1)], dtype=np.float32)
            inside = np.isfinite(raw_uv).all(-1) & (raw_uv >= 0).all(-1) & (raw_uv <= 1).all(-1)
            visibility = np.asarray(sample["visibility"][:, ids], dtype=bool).T
            out.update(future_uv_raw=torch.tensor(raw_uv[:, c:c+k]),
                       future_visibility_gt=torch.tensor(visibility[:, c:c+k]),
                       future_in_bounds=torch.tensor(inside[:, c:c+k]),
                       future_visibility_d4rt_teacher=torch.tensor(np.asarray(z["future_visibility"], dtype=bool)),
                       future_valid_d4rt_teacher=torch.tensor(np.asarray(z["future_valid"], dtype=bool)),
                       history_uv_gt=torch.tensor(raw_uv[:, :c]),
                       history_visibility_gt=torch.tensor(visibility[:, :c]),
                       history_in_bounds=torch.tensor(inside[:, :c]),
                       history_valid_gt=torch.tensor(visibility[:, :c] & inside[:, :c]))
            # Explicit-v2 callers get raw labels, not coordinates clamped to edges.
            out["future_uv_teacher"] = out["future_uv_raw"]
        out.update(context_rgb=frames[:c], future_rgb=frames[c:c+k], name=path.stem)
        if self._cache is not None:
            self._cache[index] = out
        return out


def video_pad_collate(items):
    out = {"name": [x["name"] for x in items],
           "context_rgb": torch.stack([x["context_rgb"] for x in items]),
           "future_rgb": torch.stack([x["future_rgb"] for x in items])}
    max_n = max(x["query_uv"].shape[0] for x in items)
    for key in ("query_uv", "past_xyz", "past_uv", "past_valid", "past_confidence",
                "future_xyz_teacher", "future_valid", "future_confidence", "future_uv_teacher"):
        vals = []
        for item in items:
            v = item[key]
            if v.shape[0] < max_n:
                v = torch.cat((v, torch.zeros((max_n-v.shape[0], *v.shape[1:]), dtype=v.dtype)))
            vals.append(v)
        out[key] = torch.stack(vals)
    extra = ("future_uv_raw", "future_visibility_gt", "future_in_bounds",
             "future_visibility_d4rt_teacher", "future_valid_d4rt_teacher", "history_uv_gt",
             "history_visibility_gt", "history_in_bounds", "history_valid_gt")
    for key in extra:
        if key not in items[0]:
            if any(key in item for item in items): raise ValueError("mixed trajectory interfaces")
            continue
        if any(key not in item for item in items): raise ValueError("mixed trajectory interfaces")
        vals = []
        for item in items:
            v = item[key]
            if v.shape[0] < max_n:
                v = torch.cat((v, torch.zeros((max_n-v.shape[0], *v.shape[1:]), dtype=v.dtype)))
            vals.append(v)
        out[key] = torch.stack(vals)
    return out
