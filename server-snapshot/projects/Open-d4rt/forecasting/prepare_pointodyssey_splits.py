"""Create leakage-safe sequence splits and validate converted PointOdyssey clips."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=2)
    return p.parse_args()


def metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return json.loads(str(z["metadata_json"]))


def split_for(sequence: str, seed: int) -> str:
    # Stable 60/20/20 assignment at source-sequence granularity.
    bucket = int(hashlib.sha1(f"{seed}:{sequence}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else ("val" if bucket < 8 else "test")


class RawClipDataset(Dataset):
    def __init__(self, paths: list[Path]): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        with np.load(path, allow_pickle=True) as z:
            rgb = []
            for b in z["images_jpeg_bytes"]:
                im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if im is None: raise RuntimeError(f"JPEG decode failed: {path}")
                rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            xyz = np.asarray(z["tracks_XYZ"], np.float32)
            vis = np.asarray(z["visibility"], bool)
            uv = np.asarray(z["tracks_uv_source"], np.float32)
            intr = np.asarray(z["fx_fy_cx_cy"], np.float32)
            extr = np.asarray(z["extrinsics_w2c"], np.float32)
        valid = vis & np.isfinite(xyz).all(-1) & np.isfinite(uv).all(-1)
        return {"rgb": torch.from_numpy(np.stack(rgb)), "xyz": torch.from_numpy(xyz),
                "uv": torch.from_numpy(np.nan_to_num(uv)), "valid": torch.from_numpy(valid),
                "intrinsics": torch.from_numpy(intr), "extrinsics": torch.from_numpy(extr),
                "name": path.stem}


def collate(items):
    n = min(x["xyz"].shape[1] for x in items)
    return {"rgb": torch.stack([x["rgb"] for x in items]),
            "xyz": torch.stack([x["xyz"][:, :n] for x in items]),
            "uv": torch.stack([x["uv"][:, :n] for x in items]),
            "valid": torch.stack([x["valid"][:, :n] for x in items]),
            "intrinsics": torch.stack([x["intrinsics"] for x in items]),
            "extrinsics": torch.stack([x["extrinsics"] for x in items]),
            "name": [x["name"] for x in items]}


def main():
    a = arguments(); root = Path(a.root); paths = sorted(root.glob("*.npz"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths: groups[metadata(path)["source_sequence"]].append(path)
    splits = {x: [] for x in ("train", "val", "test")}
    sequence_splits = {}
    for sequence, clips in sorted(groups.items()):
        split = split_for(sequence, a.seed); sequence_splits[sequence] = split
        splits[split].extend(clips)
    # Avoid empty small-dataset partitions deterministically.
    if any(not splits[x] for x in splits):
        raise RuntimeError(f"empty partition: {Counter(sequence_splits.values())}")
    manifest = {"seed": a.seed, "policy": "source-sequence 60/20/20 SHA1",
                "sequences": sequence_splits,
                "clips": {s: [p.name for p in ps] for s, ps in splits.items()}}
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for s, ps in splits.items():
        (root / f"{s}.txt").write_text("".join(p.name + "\n" for p in ps), encoding="utf-8")
    # Read a real batch from every partition.
    checks = {}
    for s, ps in splits.items():
        batch = next(iter(DataLoader(RawClipDataset(ps), batch_size=min(a.batch_size, len(ps)),
                                     collate_fn=collate, num_workers=0)))
        checks[s] = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
        checks[s]["finite_xyz"] = bool(torch.isfinite(batch["xyz"]).all())
        checks[s]["valid_fraction"] = float(batch["valid"].float().mean())
    summary = {"clips_total": len(paths), "sequences_total": len(groups),
               "sequence_counts": dict(Counter(sequence_splits.values())),
               "clip_counts": {s: len(ps) for s, ps in splits.items()}, "batch_checks": checks}
    (root / "loader_acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
