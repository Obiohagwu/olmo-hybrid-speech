"""
RMSNorm — Root Mean Square Layer Normalization.

Used instead of LayerNorm following Lee et al. (2026) SSM-TTM
and modern LLM practice (LLaMA, Mamba-2).

RMSNorm is cheaper than LayerNorm (no mean subtraction, no bias)
and empirically more stable with SSM architectures.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
