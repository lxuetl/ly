#!/usr/bin/env python3
"""Paired, source-wise and original-video-clustered 128px adapter comparisons."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


root = Path("/root/autodl-fs/d4rt_training/tau128_coverage_budget_capacity_20261009")
summary = json.loads((root / "summary.json").read_text())
paths = json.loads((root / "split_paths.json").read_text())["paths"]["test"]
source = {}
cluster = {}
for raw in paths:
    path = Path(raw)
    name = path.stem
    domain = "PointOdyssey" if "po50" in path.parts else path.parent.name
    original = re.sub(r"_clip_\d+$", "", name)
    if domain != "PointOdyssey":
        original = re.sub(r"_\d+$", "", original)
    source[name] = domain
    cluster[name] = (domain, original)

records = defaultdict(dict)
for line in (root / "per_sequence.jsonl").open():
    row = json.loads(line)
    records[row["variant"]][row["name"]] = row

names = (
    "frozen_tau", "original_exact", "scaled_exact", "correct", "rgb_control"
)
keys = ("MSE", "MAE", "PSNR", "SSIM", "LPIPS", "temporal_delta_MAE", "dynamic_MSE", "dynamic_MAE")
for name in names:
    values = summary["metrics"][name]
    print("METRICS", name, {key: values.get(key) for key in keys},
          "support", summary["budget_utilization"].get(name, {}).get("support"),
          "pixel_delta", summary["budget_utilization"].get(name, {}).get("pixel_delta"))

rng = np.random.default_rng(42)
comparisons = (
    ("frozen_tau", "original_exact"),
    ("original_exact", "scaled_exact"),
    ("scaled_exact", "correct"),
    ("rgb_control", "correct"),
    ("frozen_tau", "scaled_exact"),
    ("frozen_tau", "correct"),
    ("frozen_tau", "rgb_control"),
)
for reference, candidate in comparisons:
    shared = sorted(set(records[reference]) & set(records[candidate]))
    print("COMPARISON", reference, "to", candidate, "clips", len(shared))
    for key in ("MSE", "MAE", "dynamic_MSE", "dynamic_MAE"):
        valid = [name for name in shared
                 if np.isfinite(records[reference][name].get(key, np.nan))
                 and np.isfinite(records[candidate][name].get(key, np.nan))]
        difference = np.array([
            records[reference][name][key] - records[candidate][name][key]
            for name in valid
        ], dtype=np.float64)
        grouped = defaultdict(list)
        for name, value in zip(valid, difference):
            grouped[cluster[name]].append(value)
        sums = np.array([sum(group) for group in grouped.values()])
        counts = np.array([len(group) for group in grouped.values()])
        indices = rng.integers(0, len(sums), (3000, len(sums)))
        boot = sums[indices].sum(1) / counts[indices].sum(1)
        ci = np.quantile(boot, [0.025, 0.975])
        print("PAIRED", key, "gain", round(difference.mean(), 9),
              "cluster_ci", [round(float(x), 9) for x in ci],
              "win", round(float(np.mean(difference > 0)), 3),
              "clusters", len(grouped))
    for domain in sorted(set(source.values())):
        members = [name for name in shared if source.get(name) == domain]
        if not members:
            continue
        differences = [
            records[reference][name]["MSE"] - records[candidate][name]["MSE"]
            for name in members
        ]
        print("SOURCE", domain, "N", len(members), "MSE_gain", round(float(np.mean(differences)), 9))

for name in names:
    print("HORIZONS", name, [
        round(summary["metrics"][name]["per_horizon"][f"t+{step}"]["MSE"], 7)
        for step in range(1, 9)
    ])
