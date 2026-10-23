"""Typed configuration for query-budget experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryTask(str, Enum):
    POSE = "pose"
    INTRINSICS = "intrinsics"
    DEPTH = "depth"
    TRACK = "track"
    POINT_CLOUD = "point_cloud"


@dataclass(frozen=True)
class QueryBudgetConfig:
    task: QueryTask
    budget: int
    selector: str = "uniform"
    exploration_ratio: float = 0.25
    min_distance: float = 0.03
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if not 0.0 <= self.exploration_ratio <= 1.0:
            raise ValueError("exploration_ratio must be in [0, 1]")
        if self.min_distance < 0.0:
            raise ValueError("min_distance must be non-negative")
