#!/usr/bin/env python3
"""Read-only paired analysis of the completed 128px TAU experiment."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-fs/d4rt_training/tau128_v2_direction_rank_fast_20261009")
summary = json.loads((ROOT / "summary.json").read_text())
splits = json.loads((ROOT / "split_paths.json").read_text())
records = [json.loads(line) for line in (ROOT / "per_sequence.jsonl").open()]
variants = defaultdict(dict)
for record in records:
    variants[record["variant"]][record["name"]] = record

source = {}
for path_text in splits["paths"]["test"]:
    path = Path(path_text)
    source[path.stem] = "PointOdyssey" if "po50" in path.parts else path.parent.name

rng = np.random.default_rng(42)
for group in ["ALL", *sorted(set(source.values()))]:
    names = sorted(
        name for name in variants["correct"]
        if name in variants["frozen_tau"] and (group == "ALL" or source.get(name) == group)
    )
    if not names:
        continue
    print("SOURCE", group, "N", len(names))
    for key in ("MSE", "MAE", "dynamic_MSE", "dynamic_MAE"):
        pairs = [
            (variants["frozen_tau"][name].get(key), variants["correct"][name].get(key))
            for name in names
        ]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
        if not pairs:
            continue
        values = np.asarray(pairs)
        difference = values[:, 0] - values[:, 1]
        indices = rng.integers(0, len(difference), (3000, len(difference)))
        ci = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        print(key, "baseline", round(values[:, 0].mean(), 8), "v2", round(values[:, 1].mean(), 8),
              "improvement", round(difference.mean(), 8), "ci", [round(x, 8) for x in ci],
              "win", round(np.mean(difference > 0), 3))

baseline = summary["metrics"]["frozen_tau"]["per_horizon"]
correct = summary["metrics"]["correct"]["per_horizon"]
for horizon in baseline:
    print("HORIZON", horizon, "MSE", baseline[horizon]["MSE"], correct[horizon]["MSE"],
          "LPIPS", baseline[horizon]["LPIPS"], correct[horizon]["LPIPS"])

print("COUNTS", Counter(source.values()))
