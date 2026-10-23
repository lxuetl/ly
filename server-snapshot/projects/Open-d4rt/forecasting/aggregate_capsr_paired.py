#!/usr/bin/env python3
"""Aggregate sequence metrics by domain and compute paired bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "MAE", "MSE", "PSNR", "SSIM", "LPIPS", "temporal_delta_MAE",
    "dynamic_MAE", "dynamic_MSE", "dynamic_PSNR", "Motion_IoU", "Motion_F1",
)
LOWER_IS_BETTER = {
    "MAE", "MSE", "LPIPS", "temporal_delta_MAE", "dynamic_MAE", "dynamic_MSE"
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def aggregate(rows):
    result = {}
    for metric in ("MAE", "MSE", "SSIM", "LPIPS", "temporal_delta_MAE",
                   "camera_fraction", "object_fraction", "synthesis_fraction"):
        result[metric] = float(np.mean([row[metric] for row in rows]))
    result["PSNR"] = -10 * math.log10(max(result["MSE"], 1e-12))
    dynamic_count = sum(row["dynamic_count"] for row in rows)
    dynamic_abs = sum(row["dynamic_abs_sum"] for row in rows)
    dynamic_sq = sum(row["dynamic_sq_sum"] for row in rows)
    result["dynamic_MAE"] = dynamic_abs / max(dynamic_count, 1)
    result["dynamic_MSE"] = dynamic_sq / max(dynamic_count, 1)
    result["dynamic_PSNR"] = -10 * math.log10(max(result["dynamic_MSE"], 1e-12))
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    result["Motion_IoU"] = tp / max(tp + fp + fn, 1)
    result["Motion_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    result["num_sequences"] = len(rows)
    return result


def main():
    args = parse_args()
    payloads = [json.loads(Path(path).read_text()) for path in args.inputs]
    by_model = {
        payload["variant"]: {row["name"]: row for row in payload["sequences"]}
        for payload in payloads
    }
    names = sorted(set.intersection(*(set(rows) for rows in by_model.values())))
    if any(len(rows) != len(names) for rows in by_model.values()):
        raise RuntimeError("Models do not contain identical sequence names")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sequence_rows = []
    domain_rows = []
    for model, lookup in by_model.items():
        rows = [lookup[name] for name in names]
        for row in rows:
            sequence_rows.append({"model": model, **row})
        domains = sorted({row["domain"] for row in rows})
        domain_rows.append({"model": model, "domain": "ALL", **aggregate(rows)})
        for domain in domains:
            selected = [row for row in rows if row["domain"] == domain]
            domain_rows.append({"model": model, "domain": domain, **aggregate(selected)})

    with (output / "per_sequence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader(); writer.writerows(sequence_rows)
    with (output / "per_domain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]))
        writer.writeheader(); writer.writerows(domain_rows)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    models = list(by_model)
    for first_index, first in enumerate(models):
        for second in models[first_index + 1:]:
            for metric in METRICS:
                first_values = np.asarray([by_model[first][name][metric] for name in names])
                second_values = np.asarray([by_model[second][name][metric] for name in names])
                difference = first_values - second_values
                indices = rng.integers(0, len(names), size=(args.bootstrap, len(names)))
                samples = difference[indices].mean(1)
                lower, upper = np.quantile(samples, (0.025, 0.975))
                if metric in LOWER_IS_BETTER:
                    first_win_rate = float(np.mean(difference < 0))
                else:
                    first_win_rate = float(np.mean(difference > 0))
                bootstrap_rows.append({
                    "model_a": first, "model_b": second, "metric": metric,
                    "delta_a_minus_b": float(difference.mean()),
                    "ci95_low": float(lower), "ci95_high": float(upper),
                    "significant": bool(lower > 0 or upper < 0),
                    "model_a_sequence_win_rate": first_win_rate,
                    "num_sequences": len(names), "bootstrap_samples": args.bootstrap,
                })
    with (output / "paired_bootstrap.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap_rows[0]))
        writer.writeheader(); writer.writerows(bootstrap_rows)
    (output / "summary.json").write_text(json.dumps({
        "per_domain": domain_rows,
        "paired_bootstrap": bootstrap_rows,
    }, indent=2))
    print(json.dumps({"models": models, "sequences": len(names), "output": str(output)}))


if __name__ == "__main__":
    main()
