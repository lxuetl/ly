#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
    "query_ids",
    "query_uv",
    "past_xyz",
    "past_uv",
    "past_visibility",
    "past_confidence",
    "past_valid",
    "future_xyz_teacher",
    "future_visibility",
    "future_confidence",
    "future_valid",
    "sim3_context_to_teacher_scale",
    "sim3_context_to_teacher_rotation",
    "sim3_context_to_teacher_translation",
    "metadata_json",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    raw = {x.name: x for x in args.raw_root.glob("*.npz")}
    cached = {x.name: x for x in args.cache_root.glob("*.npz")}
    errors = []
    warnings = []
    usable = 0
    total_queries = 0
    valid_values = 0
    possible_values = 0
    shape_counts = {}

    missing = sorted(set(raw) - set(cached))
    extra = sorted(set(cached) - set(raw))
    if missing:
        errors.append({"kind": "missing_cache", "count": len(missing), "examples": missing[:20]})
    if extra:
        warnings.append({"kind": "extra_cache", "count": len(extra), "examples": extra[:20]})

    for i, name in enumerate(sorted(set(raw) & set(cached)), 1):
        path = cached[name]
        try:
            with np.load(path, allow_pickle=False) as z:
                keys = set(z.files)
                absent = sorted(REQUIRED - keys)
                if absent:
                    raise ValueError(f"missing keys: {absent}")
                q = int(z["query_ids"].shape[0])
                expected = {
                    "query_uv": (q, 2),
                    "past_xyz": (q, 16, 3),
                    "past_uv": (q, 16, 2),
                    "past_visibility": (q, 16),
                    "past_confidence": (q, 16),
                    "past_valid": (q, 16),
                    "future_xyz_teacher": (q, 8, 3),
                    "future_visibility": (q, 8),
                    "future_confidence": (q, 8),
                    "future_valid": (q, 8),
                    "sim3_context_to_teacher_rotation": (3, 3),
                    "sim3_context_to_teacher_translation": (3,),
                }
                for key, shape in expected.items():
                    if z[key].shape != shape:
                        raise ValueError(f"{key} shape {z[key].shape}, expected {shape}")
                for key in ("query_uv", "past_xyz", "past_uv", "past_confidence",
                            "future_xyz_teacher", "future_confidence",
                            "sim3_context_to_teacher_scale", "sim3_context_to_teacher_rotation",
                            "sim3_context_to_teacher_translation"):
                    if not np.isfinite(z[key]).all():
                        raise ValueError(f"non-finite {key}")
                meta = json.loads(str(z["metadata_json"].item()))
                source_name = Path(meta["source_path"]).name
                if source_name != name:
                    raise ValueError(f"source mismatch: {source_name}")
                fv = z["future_valid"].astype(bool)
                if fv.any():
                    usable += 1
                total_queries += q
                valid_values += int(fv.sum())
                possible_values += int(fv.size)
                shape_counts[str(q)] = shape_counts.get(str(q), 0) + 1
        except Exception as exc:
            errors.append({"kind": "invalid_cache", "file": name, "error": repr(exc)})
        if i % 100 == 0 or i == len(set(raw) & set(cached)):
            print(f"CACHE_VALIDATE {i}/{len(set(raw) & set(cached))} errors={len(errors)}", flush=True)

    report = {
        "status": "pass" if not errors else "fail",
        "raw_files": len(raw),
        "cache_files": len(cached),
        "validated_files": len(set(raw) & set(cached)),
        "usable_files": usable,
        "total_queries": total_queries,
        "future_valid_fraction": valid_values / max(possible_values, 1),
        "query_count_distribution": shape_counts,
        "errors": errors,
        "warnings": warnings,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
