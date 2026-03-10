"""
Hybrid Mamba-Attention language model for RVQ tokens.

Based on Jamba finding: even 1 attention layer out of 8 restores
in-context learning capabilities that pure Mamba lacks.

We test two ratios:
  - 1:7 (Jamba-style, compute-efficient)
  - 1:3 (more attention, potentially better for music's local harmonic structure)

Key insight from SMDIM: "Replacing self-attention entirely with Mamba
may result in the loss of local details essential for melodic and
harmonic nuances." The hybrid addresses this directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embeddings import RVQEmbedding, RVQOutputHead
from .transformer import TransformerBlock
from .mamba_lm import MambaBlock


class HybridLM(nn.Module):
    """
    Hybrid Mamba + Attention LM.

    Interleaves Mamba blocks with Transformer (attention) blocks
    at a specified ratio.

    For ratio 1:7 with 16 layers:
      layers 0-6: Mamba, layer 7: Attention, layers 8-14: Mamba, layer 15: Attention

    For ratio 1:3 with 16 layers:
      layers 0-2: Mamba, layer 3: Attn, 4-6: Mamba, 7: Attn, 8-10: Mamba, 11: Attn, 12-14: Mamba, 15: Attn
    """

    def __init__(self, config, attn_ratio: str = "1:7"):
        """
        Args:
            config: ModelConfig
            attn_ratio: "1:7" or "1:3" — ratio of attention to mamba layers
        """
        super().__init__()
        self.config = config
        self.attn_ratio = attn_ratio
        c = config

        self.embed = RVQEmbedding(c.n_codebooks, c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        # Parse ratio
        attn_part, mamba_part = map(int, attn_ratio.split(":"))
        cycle_len = attn_part + mamba_part

        # Build layer schedule
        self.blocks = nn.ModuleList()
        self.is_attn = []

        for i in range(c.n_layers):
            pos_in_cycle = i % cycle_len
            if pos_in_cycle >= mamba_part:
                # Attention layer
                self.blocks.append(
                    TransformerBlock(c.d_model, c.n_heads, c.d_ff, c.dropout)
                )
                self.is_attn.append(True)
            else:
                # Mamba layer
                self.blocks.append(
                    MambaBlock(
                        d_model=c.d_model,
                        d_state=c.d_state,
                        d_conv=c.d_conv,
                        expand=c.expand,
                        dt_rank=c.computed_dt_rank,
                        dropout=c.dropout,
                    )
                )
                self.is_attn.append(False)

        n_attn = sum(self.is_attn)
        n_mamba = len(self.is_attn) - n_attn
        print(f"Hybrid {attn_ratio}: {n_mamba} Mamba + {n_attn} Attention layers")

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T)
            mask: (B, T) — used only by attention layers

        Returns:
            logits: (B, K, T, vocab_size)
        """
        x = self.embed(codes)
        x = self.drop(x)

        for block, is_attn in zip(self.blocks, self.is_attn):
            if is_attn:
                x = block(x, mask=mask)
            else:
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
