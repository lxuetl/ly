#!/usr/bin/env python3
"""Original-video-clustered paired bootstrap for the 128px router pilot."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


root = Path("/root/autodl-fs/d4rt_training/tau128_experience_router_diagnostic_20261010")
result = json.loads((root / "summary.json").read_text())
paths = json.loads((root / "split_paths.json").read_text())["paths"]["test"]
cluster = {}
source = {}
for raw in paths:
    path = Path(raw)
    domain = "PointOdyssey" if "po50" in path.parts else path.parent.name
    original = re.sub(r"_clip_\d+$", "", path.stem)
    if domain != "PointOdyssey":
        original = re.sub(r"_\d+$", "", original)
    cluster[path.stem] = (domain, original)
    source[path.stem] = domain

records = defaultdict(dict)
for line in (root / "test_per_clip.jsonl").open():
    row = json.loads(line)
    records[row["variant"]][row["name"]] = row

rng = np.random.default_rng(42)
for reference, candidate in (
    ("tau", "all_geometry"),
    ("all_geometry", "geometry_0.50"),
    ("all_geometry", "geometry_soft"),
    ("rgb_0.50", "geometry_0.50"),
    ("all_geometry", "oracle_patch_not_deployable"),
):
    names = sorted(set(records[reference]) & set(records[candidate]))
    differences = np.asarray([
        records[reference][name]["MSE"] - records[candidate][name]["MSE"]
        for name in names
    ])
    grouped = defaultdict(list)
    for name, value in zip(names, differences):
        grouped[cluster[name]].append(value)
    sums = np.asarray([sum(v) for v in grouped.values()])
    counts = np.asarray([len(v) for v in grouped.values()])
    indices = rng.integers(0, len(sums), (5000, len(sums)))
    boot = sums[indices].sum(1) / counts[indices].sum(1)
    print(json.dumps({
        "reference": reference,
        "candidate": candidate,
        "gain_MSE": float(differences.mean()),
        "cluster_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "clip_win_rate": float((differences > 0).mean()),
        "clusters": len(grouped),
        "clips": len(names),
    }))

for domain in sorted(set(source.values())):
    names = [name for name in records["all_geometry"] if source.get(name) == domain]
    if not names:
        continue
    difference = np.mean([
        records["all_geometry"][name]["MSE"] - records["geometry_0.50"][name]["MSE"]
        for name in names
    ])
    print(json.dumps({"source": domain, "clips": len(names), "router_gain_MSE": float(difference)}))
