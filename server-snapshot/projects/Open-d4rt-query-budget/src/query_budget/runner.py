"""Thin adapter that reuses the existing D4RT encode/decode interface."""

from __future__ import annotations

import torch

from src.eval.tasks import _encode_model_memory, _run_model_for_queries

from .task_queries import QueryBatch


@torch.inference_mode()
def run_query_batch(
    model: torch.nn.Module,
    video: torch.Tensor,
    query_batch: QueryBatch,
    *,
    aspect_ratio: torch.Tensor | None = None,
    chunk_size: int = 4096,
    memory: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("video must have shape [1,T,C,H,W]")
    if memory is None:
        memory = _encode_model_memory(model, video, aspect_ratio)
    query = query_batch.to_torch(video.device)
    prediction = _run_model_for_queries(
        model=model,
        video_b=video,
        aspect_b=aspect_ratio,
        query=query,
        chunk_size=chunk_size,
        memory_b=memory,
    )
    return prediction, memory
