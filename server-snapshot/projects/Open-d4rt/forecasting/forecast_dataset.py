"""Dataset utilities for leakage-safe D4RT caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def split_paths(root: str | Path, split: str, subset: str | None = None) -> list[Path]:
    paths = sorted((Path(root) / subset).glob("*.npz")) if subset else sorted(Path(root).glob("*/*.npz"))
    selected = []
    for path in paths:
        bucket = int(hashlib.sha1(path.stem.encode()).hexdigest()[:8], 16) % 10
        target = "train" if bucket < 8 else ("val" if bucket == 8 else "test")
        if split == "all" or target == split:
            selected.append(path)
    return selected


class ForecastCacheDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.paths = split_paths(root, split)
        if not self.paths:
            raise RuntimeError(f"No {split} samples under {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        z = np.load(self.paths[index])
        keys = ("query_uv", "past_xyz", "past_uv", "past_valid", "past_confidence",
                "future_xyz_teacher", "future_valid", "future_confidence")
        # Own the memory: compressed NPZ arrays may otherwise expose non-resizable
        # storage that PyTorch's multi-worker collator cannot batch safely.
        out = {k: torch.tensor(np.asarray(z[k])) for k in keys}
        out["name"] = self.paths[index].stem
        return out


def pad_collate(items):
    """Pad a variable number of queries; validity fields keep padding loss-free."""
    names = [x["name"] for x in items]
    max_n = max(x["query_uv"].shape[0] for x in items)
    out = {"name": names}
    for key in items[0]:
        if key == "name":
            continue
        values = []
        for item in items:
            value = item[key]
            if value.shape[0] < max_n:
                pad = torch.zeros((max_n - value.shape[0], *value.shape[1:]), dtype=value.dtype)
                value = torch.cat((value, pad), dim=0)
            values.append(value)
        out[key] = torch.stack(values)
    return out
