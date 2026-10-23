#!/usr/bin/env python3
"""Aggregate four-way video evaluation by original sequence and paired bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "MAE", "MSE", "PSNR", "SSIM", "LPIPS", "temporal_delta_MAE",
    "dynamic_MAE", "dynamic_MSE", "dynamic_PSNR", "Motion_IoU", "Motion_F1",
)
LOWER_IS_BETTER = {
    "MAE", "MSE", "LPIPS", "temporal_delta_MAE", "dynamic_MAE", "dynamic_MSE",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def aggregate(rows):
    result = {}
    for metric in (
        "MAE", "MSE", "SSIM", "LPIPS", "temporal_delta_MAE",
        "camera_fraction", "object_fraction", "synthesis_fraction",
    ):
        result[metric] = float(np.mean([row[metric] for row in rows]))
    result["PSNR"] = -10 * math.log10(max(result["MSE"], 1e-12))
    dynamic_count = sum(row["dynamic_count"] for row in rows)
    dynamic_abs = sum(row["dynamic_abs_sum"] for row in rows)
    dynamic_sq = sum(row["dynamic_sq_sum"] for row in rows)
    result["dynamic_MAE"] = dynamic_abs / max(dynamic_count, 1)
    result["dynamic_MSE"] = dynamic_sq / max(dynamic_count, 1)
    result["dynamic_PSNR"] = -10 * math.log10(max(result["dynamic_MSE"], 1e-12))
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    result["Motion_IoU"] = tp / max(tp + fp + fn, 1)
    result["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    result.update({
        "dynamic_count": dynamic_count, "dynamic_abs_sum": dynamic_abs,
        "dynamic_sq_sum": dynamic_sq, "tp": tp, "fp": fp, "fn": fn,
    })
    return result


def main():
    args = parse_args()
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    by_model = {}
    sequence_rows = []
    for payload in payloads:
        label = payload["variant"]
        grouped = defaultdict(list)
        for row in payload["sequences"]:
            key = f'{row["domain"]}::{row["original_sequence"]}'
            grouped[key].append(row)
        lookup = {}
        for key, rows in grouped.items():
            item = {
                "model": label,
                "sequence_key": key,
                "domain": rows[0]["domain"],
                "original_sequence": rows[0]["original_sequence"],
                **aggregate(rows),
                "num_clips": len(rows),
            }
            lookup[key] = item
            sequence_rows.append(item)
        by_model[label] = lookup

    names = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    if not names:
        raise RuntimeError("No common original sequences")
    if any(set(rows) != set(names) for rows in by_model.values()):
        raise RuntimeError("Models do not contain identical original sequences")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sequence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader(); writer.writerows(sequence_rows)

    domain_rows = []
    for model, lookup in by_model.items():
        rows = [lookup[name] for name in names]
        domain_rows.append({
            "model": model, "domain": "ALL", **aggregate(rows),
            "num_sequences": len(rows), "num_clips": sum(row["num_clips"] for row in rows),
        })
        for domain in sorted({row["domain"] for row in rows}):
            selected = [row for row in rows if row["domain"] == domain]
            domain_rows.append({
                "model": model, "domain": domain, **aggregate(selected),
                "num_sequences": len(selected),
                "num_clips": sum(row["num_clips"] for row in selected),
            })
    with (output / "per_domain.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]))
        writer.writeheader(); writer.writerows(domain_rows)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    models = list(by_model)
    for first_index, first in enumerate(models):
        for second in models[first_index + 1:]:
            for metric in METRICS:
                a = np.asarray([by_model[first][name][metric] for name in names])
                b = np.asarray([by_model[second][name][metric] for name in names])
                difference = a - b
                indices = rng.integers(0, len(names), size=(args.bootstrap, len(names)))
                samples = difference[indices].mean(1)
                low, high = np.quantile(samples, (0.025, 0.975))
                a_wins = difference < 0 if metric in LOWER_IS_BETTER else difference > 0
                bootstrap_rows.append({
                    "model_a": first, "model_b": second, "metric": metric,
                    "delta_a_minus_b": float(difference.mean()),
                    "ci95_low": float(low), "ci95_high": float(high),
                    "significant": bool(low > 0 or high < 0),
                    "model_a_sequence_win_rate": float(np.mean(a_wins)),
                    "num_sequences": len(names), "bootstrap_samples": args.bootstrap,
                })
    with (output / "paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader(); writer.writerows(bootstrap_rows)
    summary = {
        "models": models,
        "num_original_sequences": len(names),
        "per_domain": domain_rows,
        "paired_bootstrap": bootstrap_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"models": models, "original_sequences": len(names), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
