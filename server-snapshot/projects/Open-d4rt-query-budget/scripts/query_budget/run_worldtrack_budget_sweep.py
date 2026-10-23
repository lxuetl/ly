#!/usr/bin/env python3
"""Run and aggregate D4RT WorldTrack query-budget ablations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", default="outputs/query_budget_worldtrack")
    p.add_argument("--budgets", default="4,8,16,32,64")
    p.add_argument("--selectors", default="random,fps,grid,task_aware_pose,task_aware_track")
    p.add_argument("--seeds", default="41,42,43", help="All seeds are used for random; deterministic methods use the first.")
    p.add_argument("--subsets", default="adt_mini,ds_mini")
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--query-chunk-size", type=int, default=4096)
    p.add_argument("--limit-seqs", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    runs = [(b, s, seed) for s in selectors for seed in (seeds if s == "random" else seeds[:1]) for b in budgets]
    for budget, selector, seed in runs:
        out_dir = output_root / f"k{budget}_{selector}_seed{seed}"
        summary_path = out_dir / "summary.json"
        if not (args.resume and summary_path.exists()):
            cmd = [
                sys.executable,
                str(root / "eval_track3d_in_worldtrack.py"),
                "--model-config", args.model_config,
                "--ckpt-path", args.ckpt_path,
                "--data-root", args.data_root,
                "--output-dir", str(out_dir),
                "--subsets", args.subsets,
                "--num-frames", str(args.num_frames),
                "--query-chunk-size", str(args.query_chunk_size),
                "--query-budget", str(budget),
                "--query-selector", selector,
                "--query-seed", str(seed),
                "--device", args.device,
                "--save-per-sequence",
            ]
            if args.limit_seqs > 0:
                cmd += ["--limit-seqs", str(args.limit_seqs)]
            subprocess.run(cmd, cwd=root, check=True)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for subset, metrics in payload.get("subsets", {}).items():
            rows.append({
                "budget": budget,
                "selector": selector,
                "seed": seed,
                "subset": subset,
                "num_sequences": metrics.get("num_sequences"),
                "avg_pts_global": metrics.get("avg_pts_global"),
                "epe_global": metrics.get("epe_global"),
                "avg_pts_global_dyn": metrics.get("avg_pts_global_dyn"),
                "epe_global_dyn": metrics.get("epe_global_dyn"),
                "camera_rot_deg": metrics.get("camera_rot_deg"),
                "camera_ate": metrics.get("camera_ate"),
                "camera_rpe_rot_deg": metrics.get("camera_rpe_rot_deg"),
                "camera_rpe_trans": metrics.get("camera_rpe_trans"),
                "inference_seconds": metrics.get("inference_seconds"),
                "milliseconds_per_query": metrics.get("milliseconds_per_query"),
                "peak_memory_mb": metrics.get("peak_memory_mb"),
                "total_queries": metrics.get("total_queries"),
            })
        (output_root / "ablation_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        if rows:
            with (output_root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    # Mean/std table makes the three-seed random baseline directly reportable.
    metric_names = ["avg_pts_global", "epe_global", "camera_rot_deg", "camera_ate", "inference_seconds", "peak_memory_mb"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["budget"], row["selector"], row["subset"])].append(row)
    aggregate: list[dict[str, object]] = []
    for (budget, selector, subset), items in sorted(groups.items()):
        record: dict[str, object] = {"budget": budget, "selector": selector, "subset": subset, "num_seeds": len(items)}
        for metric in metric_names:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            record[f"{metric}_mean"] = sum(values) / len(values) if values else None
            record[f"{metric}_std"] = (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5 if values else None
        aggregate.append(record)
    (output_root / "ablation_aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    if aggregate:
        with (output_root / "ablation_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    print(f"Wrote {len(rows)} rows to {output_root / 'ablation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
