#!/usr/bin/env python3
"""Aggregate three spatial-v4.1 seeds into mean/std and pooled paired deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="42,123,3407")
    args = parser.parse_args()
    root = Path(args.root)
    seeds = [int(value) for value in args.seeds.split(",")]
    summaries = [
        json.loads((root / f"seed_{seed}" / "summary.json").read_text())
        for seed in seeds
    ]
    variants = list(summaries[0]["metrics"])
    metric_keys = (
        "MSE", "MAE", "PSNR", "SSIM", "LPIPS", "temporal_delta_MAE",
        "dynamic_MSE", "dynamic_MAE", "Motion_IoU", "Motion_F1",
    )
    aggregate = {}
    for variant in variants:
        aggregate[variant] = {}
        for key in metric_keys:
            values = np.array([
                summary["metrics"][variant][key] for summary in summaries
            ], dtype=np.float64)
            aggregate[variant][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
    output = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "metrics": aggregate,
        "best_epochs": [summary["meta"]["best_epoch"] for summary in summaries],
        "selection_eligible": [
            summary["meta"]["selection_eligible"] for summary in summaries
        ],
        "per_seed_bootstrap": {
            str(seed): summary["paired_bootstrap"]
            for seed, summary in zip(seeds, summaries)
        },
    }
    (root / "multiseed_summary.json").write_text(json.dumps(output, indent=2))
    (root / "COMPLETE").write_text("complete\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
