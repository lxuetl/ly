"""Data-free smoke test for selectors and downstream query construction."""

from __future__ import annotations

import numpy as np

from src.query_budget import QueryBudgetConfig, QueryTask, build_pose_queries, build_selector, build_track_queries


def main() -> None:
    xs = np.linspace(0.05, 0.95, 24, dtype=np.float32)
    ys = np.linspace(0.05, 0.95, 16, dtype=np.float32)
    uv = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    cues = {
        "confidence": 1.0 - np.linalg.norm(uv - 0.5, axis=1),
        "static": np.ones((uv.shape[0],), dtype=np.float32),
        "motion": np.linalg.norm(uv - 0.5, axis=1),
        "visibility": np.ones((uv.shape[0],), dtype=np.float32),
    }

    for name in ("random", "uniform", "task_aware"):
        cfg = QueryBudgetConfig(task=QueryTask.POSE, budget=32, selector=name, seed=7)
        indices = build_selector(cfg).select(uv, cues)
        assert indices.shape == (32,)
        assert np.unique(indices).shape[0] == 32

    selected = uv[build_selector(QueryBudgetConfig(QueryTask.POSE, 32, "task_aware")).select(uv, cues)]
    pose_queries, left, right = build_pose_queries(selected, frame_i=0, frame_j=7)
    assert pose_queries.size == 64
    assert left.stop - left.start == right.stop - right.start == 32

    track_queries = build_track_queries(selected, np.zeros((32,), dtype=np.int64), num_frames=16)
    assert track_queries.size == 32 * 16
    print("query-budget smoke test passed")


if __name__ == "__main__":
    main()
