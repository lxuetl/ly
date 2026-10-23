"""Query selection baselines and a task-aware heuristic selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import QueryBudgetConfig, QueryTask


def _validate_uv(candidate_uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(candidate_uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"candidate_uv must have shape [N,2], got {uv.shape}")
    if uv.shape[0] == 0:
        raise ValueError("candidate_uv is empty")
    return uv


def _farthest_point_indices(uv: np.ndarray, budget: int, start: int = 0) -> np.ndarray:
    budget = min(int(budget), uv.shape[0])
    chosen = [int(start) % uv.shape[0]]
    min_dist = np.full((uv.shape[0],), np.inf, dtype=np.float64)
    for _ in range(1, budget):
        delta = uv - uv[chosen[-1]]
        min_dist = np.minimum(min_dist, np.sum(delta * delta, axis=1))
        min_dist[np.asarray(chosen, dtype=np.int64)] = -1.0
        chosen.append(int(np.argmax(min_dist)))
    return np.asarray(chosen, dtype=np.int64)


@dataclass
class BaseSelector:
    cfg: QueryBudgetConfig

    def select(self, candidate_uv: np.ndarray, cues: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        raise NotImplementedError


class RandomSelector(BaseSelector):
    def select(self, candidate_uv: np.ndarray, cues: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        uv = _validate_uv(candidate_uv)
        rng = np.random.default_rng(self.cfg.seed)
        return rng.choice(uv.shape[0], size=min(self.cfg.budget, uv.shape[0]), replace=False).astype(np.int64)


class UniformSelector(BaseSelector):
    def select(self, candidate_uv: np.ndarray, cues: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        uv = _validate_uv(candidate_uv)
        center = np.asarray([0.5, 0.5], dtype=np.float32)
        start = int(np.argmin(np.sum((uv - center) ** 2, axis=1)))
        return _farthest_point_indices(uv, self.cfg.budget, start=start)


_TASK_WEIGHTS: dict[QueryTask, dict[str, float]] = {
    QueryTask.POSE: {"confidence": 1.0, "static": 1.0, "coverage": 0.8, "edge": 0.2},
    QueryTask.INTRINSICS: {"confidence": 1.0, "radial": 1.0, "coverage": 0.5},
    QueryTask.DEPTH: {"uncertainty": 1.0, "edge": 0.8, "coverage": 0.5},
    QueryTask.TRACK: {"motion": 1.0, "visibility": 0.8, "uncertainty": 0.5, "coverage": 0.4},
    QueryTask.POINT_CLOUD: {"uncertainty": 0.8, "motion": 0.5, "edge": 0.5, "coverage": 0.8},
}


class TaskAwareHeuristicSelector(BaseSelector):
    """Greedy score-diversity selector used before the learned policy is trained."""

    def select(self, candidate_uv: np.ndarray, cues: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        uv = _validate_uv(candidate_uv)
        cues = {} if cues is None else cues
        score = np.zeros((uv.shape[0],), dtype=np.float64)
        weights = _TASK_WEIGHTS[self.cfg.task]

        derived = dict(cues)
        derived.setdefault("radial", np.linalg.norm(uv - 0.5, axis=1))
        derived.setdefault("coverage", np.ones((uv.shape[0],), dtype=np.float32))
        for name, weight in weights.items():
            value = np.asarray(derived.get(name, np.zeros_like(score)), dtype=np.float64).reshape(-1)
            if value.shape[0] != uv.shape[0]:
                raise ValueError(f"cue {name!r} has {value.shape[0]} values, expected {uv.shape[0]}")
            finite = np.isfinite(value)
            if np.any(finite):
                lo, hi = np.min(value[finite]), np.max(value[finite])
                normalized = np.zeros_like(value)
                normalized[finite] = (value[finite] - lo) / max(hi - lo, 1e-8)
                score += float(weight) * normalized

        budget = min(self.cfg.budget, uv.shape[0])
        chosen: list[int] = []
        min_distance = np.full((uv.shape[0],), np.inf, dtype=np.float64)
        for _ in range(budget):
            utility = score.copy()
            if chosen:
                delta = uv - uv[chosen[-1]]
                min_distance = np.minimum(min_distance, np.linalg.norm(delta, axis=1))
                utility += np.minimum(min_distance / max(self.cfg.min_distance, 1e-6), 1.0)
                utility[np.asarray(chosen, dtype=np.int64)] = -np.inf
            pick = int(np.argmax(utility))
            chosen.append(pick)
        return np.asarray(chosen, dtype=np.int64)


def build_selector(cfg: QueryBudgetConfig) -> BaseSelector:
    key = cfg.selector.strip().lower()
    if key == "random":
        return RandomSelector(cfg)
    if key in {"uniform", "fps"}:
        return UniformSelector(cfg)
    if key in {"task_aware", "heuristic"}:
        return TaskAwareHeuristicSelector(cfg)
    raise ValueError(f"Unknown selector: {cfg.selector}")
