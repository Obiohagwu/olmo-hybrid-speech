"""
Transformer baseline for RVQ token language modeling.
MusicGen-style: autoregressive transformer with delay pattern over codebook tokens.

This is the control model — the architecture music generation is known to work with.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, d_head)
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # Using PyTorch 2.0+ SDPA for Flash Attention when available
        if mask is not None:
            # Combine causal mask with padding mask
            # causal: (1, 1, T, T), pad: (B, 1, 1, T) -> (B, 1, T, T)
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            pad_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_mask = causal.unsqueeze(0) & pad_mask  # (B, 1, T, T)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Autoregressive Transformer LM over RVQ tokens.

    Architecture matches MusicGen's core design:
    - Sum-of-embeddings input (one per codebook)
    - Causal transformer stack
    - Per-codebook output heads
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.embed = RVQEmbedding(c.n_codebooks, c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout)
            for _ in range(c.n_layers)
        ])

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)

        # Count params
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T) codebook indices
            mask: (B, T) boolean mask (True = valid)

        Returns:
            logits: (B, K, T, vocab_size)
        """
        x = self.embed(codes)   # (B, T, d_model)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        return self.output(x)   # (B, K, T, vocab_size)

    def compute_loss(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pad_token: int = 1024,
    ) -> torch.Tensor:
        """
        Compute next-token prediction loss across all codebooks.

        Input codes at time t predict codes at time t+1.
        """
        logits = self.forward(codes, mask)  # (B, K, T, V)

        B, K, T, V = logits.shape

        # Shift: predict t+1 from t
        logits = logits[:, :, :-1, :].contiguous()   # (B, K, T-1, V)
        targets = codes[:, :, 1:].contiguous()         # (B, K, T-1)

        # Flatten for cross-entropy
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=pad_token, label_smoothing=0.1)
        return loss
