"""
Shared embedding layer for multi-codebook RVQ tokens.

Each of the 9 codebooks gets its own embedding table. The embeddings
are summed to produce a single d_model vector per timestep.
This follows MusicGen's approach.
"""

import torch
import torch.nn as nn
import math

from .norms import RMSNorm


class RVQEmbedding(nn.Module):
    """
    Multi-codebook embedding: one embedding table per codebook,
    outputs are summed per timestep.

    Input:  (B, K, T) — K codebooks, T timesteps
    Output: (B, T, d_model)
    """

    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        d_model: int,
        pad_token: int = 1024,
        use_pos_embedding: bool = True,
        max_positions: int = 8192,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.d_model = d_model
        self.pad_token = pad_token
        self.use_pos_embedding = use_pos_embedding

        # One embedding per codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
            for _ in range(n_codebooks)
        ])

        # Learned positional encoding for non-RoPE architectures.
        if use_pos_embedding:
            self.pos_embedding = nn.Embedding(max_positions, d_model)
        else:
            self.pos_embedding = None

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)
            if emb.padding_idx is not None:
                nn.init.zeros_(emb.weight[emb.padding_idx])
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T) long tensor of codebook indices

        Returns:
            x: (B, T, d_model)
        """
        B, K, T = codes.shape

        # Sum embeddings from each codebook
        x = torch.zeros(B, T, self.d_model, device=codes.device, dtype=torch.float32)
        for k in range(K):
            x = x + self.embeddings[k](codes[:, k, :])  # (B, T, d_model)

        # Add positional encoding
        if self.pos_embedding is not None:
            positions = torch.arange(T, device=codes.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        return x


class RVQOutputHead(nn.Module):
    """
    Multi-codebook output head: predicts logits for each codebook independently.

    Input:  (B, T, d_model)
    Output: (B, K, T, vocab_size) — logits per codebook
    """

    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size

        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_codebooks)
        ])

        self.ln = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            logits: (B, K, T, vocab_size)
        """
        x = self.ln(x)
        logits = torch.stack(
            [head(x) for head in self.heads], dim=1
        )  # (B, K, T, vocab_size)
        return logits
