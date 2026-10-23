"""Audit camera/depth/visibility fields available to the forecasting pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from eval_track3d_in_worldtrack import load_worldtrack_sequence
def describe_array(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def main():
    cache_paths = (
        "data/d4rt_forecasting_cache/adt_mini/Apartment_release_clean_seq131_0.npz",
        "data/d4rt_forecasting_cache/ds_mini/01f258-3_obj_source_left_6.npz",
        "data/d4rt_forecasting_cache/po_mini/cab_e_3rd_1.npz",
        "data/d4rt_forecasting_cache/pstudio_mini/basketball_13.npz",
        "/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50/po50/ani-3f79ccd1_clip_000.npz",
    )
    report = []
    for cache_path in cache_paths:
        with np.load(cache_path, allow_pickle=True) as cache:
            cache_meta = json.loads(str(cache["metadata_json"]))
            cache_fields = {
                name: describe_array(cache[name])
                for name in cache.files
                if name != "metadata_json"
            }
        source_path = Path(cache_meta["source_path"])
        with np.load(source_path, allow_pickle=True) as raw:
            source_meta = json.loads(str(raw["metadata_json"])) if "metadata_json" in raw.files else {}
            raw_fields = {}
            for name in raw.files:
                if name in ("images_jpeg_bytes", "video_rgb"):
                    raw_fields[name] = {"skipped_payload": True}
                else:
                    raw_fields[name] = describe_array(raw[name])
        sample = load_worldtrack_sequence(source_path, 24)
        loaded_fields = {
            name: describe_array(value)
            for name, value in sample.items()
            if hasattr(value, "shape")
        }
        report.append({
            "cache_path": cache_path,
            "source_path": str(source_path),
            "cache_fields": cache_fields,
            "raw_fields": raw_fields,
            "loaded_fields": loaded_fields,
            "metadata": source_meta,
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
