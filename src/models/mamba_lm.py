"""
Mamba (Selective SSM) language model for RVQ token prediction.

Implements both Mamba-1 and Mamba-2 (SSD) variants.
Pure Mamba — no attention layers. Linear-time sequence modeling.

Uses mamba-ssm CUDA kernels when available (20-40x faster).
Falls back to pure PyTorch sequential scan otherwise.

Key insight from the research:
- Mamba excels at long-range dependencies (musical structure over time)
- But may struggle with local harmonic detail (SMDIM finding)
- And in-context learning (Jamba finding)
This is why we also test hybrids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange

from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm

# Try to import fast CUDA kernels
try:
    from mamba_ssm import Mamba as MambaCuda
    MAMBA_CUDA_AVAILABLE = True
    print("Using mamba-ssm CUDA kernels (fast)")
except ImportError:
    MAMBA_CUDA_AVAILABLE = False
    print("WARNING: mamba-ssm not found, using pure PyTorch fallback (very slow)")


class MambaBlock(nn.Module):
    """
    Mamba-1 selective SSM block.

    Uses mamba-ssm CUDA kernels when available for fast parallel scan.
    Falls back to sequential PyTorch scan otherwise (very slow for training).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if MAMBA_CUDA_AVAILABLE:
            self.mamba = MambaCuda(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Pure PyTorch fallback (slow sequential scan)
            self.mamba = _MambaPurePytorch(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)


class _MambaPurePytorch(nn.Module):
    """Pure PyTorch fallback for Mamba-1. Very slow — use mamba-ssm for training."""

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dt_rank=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = rearrange(x_ssm, 'b t d -> b d t')
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]
        x_ssm = rearrange(x_ssm, 'b d t -> b t d')
        x_ssm = F.silu(x_ssm)

        x_dbl = self.x_proj(x_ssm)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        # Sequential scan (slow fallback)
        batch, seq_len, d_inner = x_ssm.shape
        dt_e = dt.unsqueeze(-1)
        A_e = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt_e * A_e)
        B_e = B.unsqueeze(2)
        dB = dt_e * B_e

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_ssm[:, t].unsqueeze(-1)
            ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))

        y = torch.stack(ys, dim=1)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    """
    Mamba-2 (SSD) block — simplified version.

    Key difference from Mamba-1:
    - Uses scalar-times-identity structure for A (not diagonal)
    - Enables matmul-based algorithms (tensor core friendly)
    - 2-8x faster than Mamba-1

    This is a simplified implementation. For full speed, use official mamba_ssm.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.d_head = self.d_inner // n_heads

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D depthwise conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSD parameters: per-head scalar A (log space)
        self.A_log = nn.Parameter(torch.randn(n_heads))

        # B, C projections (shared across heads)
        self.B_proj = nn.Linear(self.d_inner, n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, n_heads * d_state, bias=False)

        # dt projection (scalar per head)
        self.dt_proj = nn.Linear(self.d_inner, n_heads, bias=True)

        # D skip connection
        self.D = nn.Parameter(torch.ones(n_heads))

        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        B, T, _ = x.shape

        # Project and split gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv
        x_ssm = rearrange(x_ssm, 'b t d -> b d t')
        x_ssm = self.conv1d(x_ssm)[:, :, :T]
        x_ssm = rearrange(x_ssm, 'b d t -> b t d')
        x_ssm = F.silu(x_ssm)

        # SSD parameters
        dt = F.softplus(self.dt_proj(x_ssm))   # (B, T, n_heads)
        B_mat = self.B_proj(x_ssm)              # (B, T, n_heads * d_state)
        C_mat = self.C_proj(x_ssm)              # (B, T, n_heads * d_state)
        A = -torch.exp(self.A_log)               # (n_heads,)

        # Reshape for per-head processing
        B_mat = rearrange(B_mat, 'b t (h n) -> b h t n', h=self.n_heads)
        C_mat = rearrange(C_mat, 'b t (h n) -> b h t n', h=self.n_heads)
        x_heads = rearrange(x_ssm, 'b t (h d) -> b h t d', h=self.n_heads)

        # Sequential scan per head (simplified)
        h = torch.zeros(
            B, self.n_heads, self.d_state, self.d_head,
            device=x.device, dtype=x.dtype
        )
        ys = []

        for t in range(T):
            # Scalar discretization: dA = exp(dt * a)
            dA_t = torch.exp(dt[:, t, :] * A)  # (B, n_heads)
            dA_t = dA_t.unsqueeze(-1).unsqueeze(-1)  # (B, n_heads, 1, 1)

            h = dA_t * h + torch.einsum(
                'bhn,bhd->bhnd', B_mat[:, :, t], x_heads[:, :, t]
            ) * dt[:, t, :].unsqueeze(-1).unsqueeze(-1)

            y_t = torch.einsum('bhn,bhnd->bhd', C_mat[:, :, t], h)
            ys.append(y_t)

        y = torch.stack(ys, dim=2)  # (B, n_heads, T, d_head)

        # Add skip connection D
        y = y + x_heads * self.D.view(1, -1, 1, 1)

        # Merge heads
        y = rearrange(y, 'b h t d -> b t (h d)')

        # Gate
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class MambaLM(nn.Module):
    """
    Pure Mamba language model over RVQ tokens.

    Supports both Mamba-1 and Mamba-2 variants.
    No attention layers — linear-time sequence modeling.
    """

    def __init__(self, config, version: int = 1):
        super().__init__()
        self.config = config
        self.version = version
        c = config

        self.embed = RVQEmbedding(c.n_codebooks, c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        BlockClass = MambaBlock if version == 1 else Mamba2Block

        block_kwargs = dict(
            d_model=c.d_model,
            d_state=c.d_state,
            d_conv=c.d_conv,
            expand=c.expand,
            dropout=c.dropout,
        )
        if version == 1:
            block_kwargs["dt_rank"] = c.computed_dt_rank
        else:
            block_kwargs["n_heads"] = c.n_heads

        self.blocks = nn.ModuleList([
            BlockClass(**block_kwargs)
            for _ in range(c.n_layers)
        ])

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T) codebook indices
            mask: unused (Mamba doesn't need attention mask)

        Returns:
            logits: (B, K, T, vocab_size)
        """
        x = self.embed(codes)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)

    def compute_loss(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pad_token: int = 1024,
    ) -> torch.Tensor:
        logits = self.forward(codes, mask)
        B, K, T, V = logits.shape

        logits = logits[:, :, :-1, :].contiguous()
        targets = codes[:, :, 1:].contiguous()

        logits = logits.view(-1, V)
        targets = targets.view(-1)

        return F.cross_entropy(logits, targets, ignore_index=pad_token, label_smoothing=0.1)
