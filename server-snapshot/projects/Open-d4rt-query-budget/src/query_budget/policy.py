"""Learned task-conditioned policy for ranking candidate queries."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import QueryTask


class TaskConditionedQueryPolicy(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.tasks = list(QueryTask)
        self.task_embedding = nn.Embedding(len(self.tasks), hidden_dim)
        self.candidate_proj = nn.Linear(feature_dim, hidden_dim)
        self.budget_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor, task_ids: torch.Tensor, budget_ratio: torch.Tensor) -> torch.Tensor:
        """Return logits for candidate features shaped [B,N,C]."""
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B,N,C]")
        task = self.task_embedding(task_ids.long()).unsqueeze(1)
        budget = self.budget_proj(budget_ratio.float().view(-1, 1)).unsqueeze(1)
        return self.scorer(self.candidate_proj(candidate_features) + task + budget).squeeze(-1)

    def task_id(self, task: QueryTask) -> int:
        return self.tasks.index(task)
