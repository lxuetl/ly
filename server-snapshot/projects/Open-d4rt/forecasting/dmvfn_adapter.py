"""Adapter exposing official DMVFN as a context-to-future video backbone."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn


DEFAULT_REPO = Path('/root/autodl-tmp/CVPR2023-DMVFN')
if DEFAULT_REPO.exists() and str(DEFAULT_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO))
try:
    from model.arch import DMVFN
except ImportError as exc:
    raise ImportError(f"Official DMVFN repository not found at {DEFAULT_REPO}") from exc


class DMVFNVideoPredictor(nn.Module):
    """Use the final two context frames and autoregressively predict K frames."""
    scales = (4,4,4,2,2,2,1,1,1)

    def __init__(self, context_frames=16, future_frames=8):
        super().__init__();self.context_frames=int(context_frames);self.future_frames=int(future_frames)
        self.backbone=DMVFN()

    def pair_stages(self, first, second):
        # The official inference branch routes using only batch item zero. Using
        # the training branch here preserves correct per-sample routing.
        return self.backbone(torch.cat((first,second),1),list(self.scales),training=True)

    def teacher_forced_stages(self, context, future):
        sequence=torch.cat((context[:,-2:],future),1)
        first=sequence[:,:-2].reshape(-1,*sequence.shape[2:])
        second=sequence[:,1:-1].reshape(-1,*sequence.shape[2:])
        b,k=future.shape[:2]
        return [x.reshape(b,k,*x.shape[1:]) for x in self.pair_stages(first,second)]

    def forward(self, context_rgb, geometry_7d=None, query_uv=None, point_valid=None):
        first,second=context_rgb[:,-2],context_rgb[:,-1];predictions=[]
        for _ in range(self.future_frames):
            second_next=self.pair_stages(first,second)[-1]
            predictions.append(second_next);first,second=second,second_next
        return torch.stack(predictions,1)
