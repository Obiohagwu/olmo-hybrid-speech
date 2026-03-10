"""
MC-Mamba: Memory Caching for Mamba SSMs Applied to Music Generation.

Novel contribution: Applies the Memory Caching (MC) mechanism from
"Memory Caching: RNNs with Growing Memory" (arXiv:2602.24281) to Mamba
for the first time. The original paper only tested MC on linear attention
and Titans — never on selective state space models.

Key design decision: Output Activation Caching
  Rather than caching intermediate SSM hidden states (which the mamba-ssm
  CUDA kernel doesn't expose during training), we cache block output
  activations at segment boundaries:
    1. Run the full Mamba CUDA kernel over the entire sequence (preserving parallelism)
    2. After each block, extract output activations at positions [S, 2S, 3S, ...]
    3. Apply GRM (Gated Residual Memory) gating to inject cached context back via residual

  This is actually a better fit for Mamba than caching raw SSM state because:
    - Output activations live in d_model space (same as inputs) — no dimension mismatch
    - Cache entries are small: (B, d_model) vs SSM state (B, d_inner, d_state)
    - Full CUDA kernel parallelism preserved during training
    - Gradients flow through cache entries naturally

Architecture (GRM variant — best from the paper, Equations 8-10):
  Per MCMambaBlock, per forward pass:
    1. x_out = MambaBlock(x_in) — standard Mamba
    2. Extract boundary activations h_i = x_out[:, i*S-1, :] (values from output space)
    3. Compute segment means m_i from x_in (keys/gates from input space, per paper)
    4. Query: u_t = W_u @ x_in_t (query from input space, per paper Eq 10)
    5. Gate: γ_t^(i) = softmax( <u_t, m_i> / sqrt(d) ) over current + cached segments
    6. Output: y_t = γ_t^(current) * x_out_t + Σ γ_t^(cached_i) * h_i (gated, not additive)

New params per layer: W_u = d_model x d_model = ~590K (for d_model=768)
Total overhead: ~11.8M across 20 layers (~9% over base Mamba)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from .mamba_lm import MambaBlock
from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm


class SegmentCache:
    """
    Stateful buffer storing boundary activations and segment summaries.

    At each segment boundary (positions S-1, 2S-1, 3S-1, ...), we store:
      - h_i: the output activation at the boundary position (B, d_model)
      - m_i: the mean activation over the segment (B, d_model)

    The cache grows during a forward pass and can be reset between sequences.
    During training, entries are differentiable (gradients flow through them).
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self.boundary_activations: List[torch.Tensor] = []  # each (B, d_model)
        self.segment_means: List[torch.Tensor] = []          # each (B, d_model)

    def add(self, h_boundary: torch.Tensor, m_segment: torch.Tensor):
        """
        Add a new cache entry.

        Args:
            h_boundary: (B, d_model) activation at segment boundary
            m_segment: (B, d_model) mean activation over the segment
        """
        self.boundary_activations.append(h_boundary)
        self.segment_means.append(m_segment)

        # Evict oldest entries if over capacity
        if len(self.boundary_activations) > self.max_entries:
            # Detach evicted entries to free graph memory
            self.boundary_activations.pop(0)
            self.segment_means.pop(0)

    @property
    def num_entries(self) -> int:
        return len(self.boundary_activations)

    @property
    def is_empty(self) -> bool:
        return self.num_entries == 0

    def get_stacked(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack all cached entries into tensors.

        Returns:
            h_stack: (B, N, d_model) boundary activations
            m_stack: (B, N, d_model) segment means
            where N = num_entries
        """
        h_stack = torch.stack(self.boundary_activations, dim=1)  # (B, N, d)
        m_stack = torch.stack(self.segment_means, dim=1)          # (B, N, d)
        return h_stack, m_stack

    def reset(self):
        """Clear all cached entries."""
        self.boundary_activations.clear()
        self.segment_means.clear()


class GRMLayer(nn.Module):
    """
    Gated Residual Memory (GRM) layer (Equations 8-10 from the MC paper).

    For position t in segment s, the paper's GRM output is:
      y_t = γ_t^(s) * ℳ_t^(s)(q_t) + Σ_{i=1}^{s-1} γ_t^(i) * ℳ_L^(i)(q_t)

    Where gates γ are softmax-normalized over current segment + all cached segments:
      γ_t^(i) = softmax_i( <u_t, MeanPooling(S^(i))> )
      u_t = x_t @ W_u

    The softmax gates BOTH current and cached contributions (weights sum to 1),
    so the current segment can be gated down when cached context is more relevant.

    The only new learnable parameter is W_u (d_model x d_model).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)

        # Query projection — the only new parameter per layer
        self.W_u = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_u.weight)

    def forward(
        self,
        x: torch.Tensor,
        h_stack: torch.Tensor,
        m_stack: torch.Tensor,
        m_current: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve cached context via GRM gating with causal masking.

        Args:
            x: (B, T, d_model) block INPUT activations (queries from input space, per paper)
            h_stack: (B, N, d_model) cached boundary activations from OUTPUT (values)
            m_stack: (B, N, d_model) cached segment means from INPUT (keys for gating)
            m_current: (B, T, d_model) current segment mean from INPUT, broadcast per position
            causal_mask: (T, N) boolean mask — True where position t can attend to cache entry j

        Returns:
            h_retrieved: (B, T, d_model) gated retrieved context from cached segments
            gate_current: (B, T, 1) gate weight for current segment output
            attn_weights: (B, T, N+1) full attention distribution (cached + current)
        """
        B, T, D = x.shape
        N = h_stack.shape[1]

        # Query: (B, T, d_model)
        u = self.W_u(x)

        # Scores for cached segments: (B, T, N)
        scores_cached = torch.bmm(u, m_stack.transpose(1, 2)) * self.scale

        # Apply causal mask: positions can only attend to previous segments
        # causal_mask is (T, N), expand to (B, T, N)
        scores_cached = scores_cached.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        # Score for current segment: (B, T, 1) = dot product of u_t with current segment mean
        scores_current = (u * m_current).sum(dim=-1, keepdim=True) * self.scale

        # Concatenate and softmax over all segments (cached + current)
        # (B, T, N+1) — last entry is current segment
        all_scores = torch.cat([scores_cached, scores_current], dim=-1)
        attn_weights = F.softmax(all_scores, dim=-1)  # (B, T, N+1)

        # Split gates
        gates_cached = attn_weights[:, :, :N]   # (B, T, N)
        gate_current = attn_weights[:, :, N:]    # (B, T, 1)

        # Retrieve from cached segments: (B, T, d_model)
        h_retrieved = torch.bmm(gates_cached, h_stack)

        return h_retrieved, gate_current, attn_weights


class MCMambaBlock(nn.Module):
    """
    Memory-Cached Mamba block.

    Wraps a standard MambaBlock with:
      1. SegmentCache for storing boundary activations
      2. GRMLayer for retrieving cached context
      3. Residual injection of retrieved context

    Forward pass:
      1. x_out = MambaBlock(x_in)
      2. Extract boundary activations from x_out (values), segment means from x_in (keys)
      3. Build causal mask, GRM-gate current segment + cached segments (softmax, weights sum to 1)
      4. Return gated combination: γ_current * x_out + Σ γ_cached_i * h_i
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 32,
        dropout: float = 0.0,
        segment_size: int = 256,
        retrieval_scale: float = 1.0,
        max_cache_entries: int = 64,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.retrieval_scale = retrieval_scale

        # Standard Mamba block (handles its own pre-norm + residual)
        self.mamba_block = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )

        # MC components
        self.cache = SegmentCache(max_entries=max_cache_entries)
        self.grm = GRMLayer(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x_enhanced: (B, T, d_model)

        Paper's GRM (Eq 8-10) for position t in segment s:
          y_t = γ_t^(s) * ℳ_t^(s)(q_t) + Σ_{i=1}^{s-1} γ_t^(i) * ℳ_L^(i)(q_t)

        We implement this as:
          1. Run Mamba over full sequence (CUDA kernel, preserves parallelism)
          2. Extract segment boundaries and means from x_out
          3. Build causal mask so position t only retrieves from segments ending before t's segment
          4. GRM gates both current segment and cached segments via softmax (weights sum to 1)
          5. Output = gate_current * x_out + h_retrieved (where h_retrieved is already gate-weighted)
        """
        B, T, D = x.shape
        S = self.segment_size

        # Step 1: Standard Mamba forward (pre-norm + residual handled inside)
        x_out = self.mamba_block(x)

        # Step 2: Extract segment boundaries and means
        # Values (h_boundaries) from x_out (output space — proxy for memory state)
        # Keys/means (m_segments) from x (input space — per paper Eq 10)
        n_segments = T // S
        if n_segments == 0:
            # Sequence shorter than segment size — no caching possible
            return x_out

        boundary_positions = [(k + 1) * S - 1 for k in range(n_segments)]
        h_boundaries = []  # boundary activations from OUTPUT (values)
        m_segments = []    # segment means from INPUT (keys for gating)

        for k, pos in enumerate(boundary_positions):
            seg_start = k * S
            h_boundaries.append(x_out[:, pos, :])                        # (B, D) from output
            m_segments.append(x[:, seg_start:pos + 1, :].mean(dim=1))    # (B, D) from input

        # Stack: (B, N, D) where N = n_segments
        h_stack = torch.stack(h_boundaries, dim=1)
        m_stack = torch.stack(m_segments, dim=1)

        # Step 3: Build causal mask (T, N)
        # Position t is in segment k = t // S
        # Position t can attend to cache entry j only if segment j+1 ends before t's segment starts
        # i.e., boundary_positions[j] < k*S, i.e., (j+1)*S - 1 < k*S, i.e., j < k, i.e., j <= k-1
        # So position t in segment k can attend to entries 0..k-2 (segments 1..k-1 in paper notation)
        seg_idx = torch.arange(T, device=x.device) // S  # (T,) which segment each position belongs to
        cache_idx = torch.arange(n_segments, device=x.device)  # (N,)
        causal_mask = seg_idx.unsqueeze(1) > cache_idx.unsqueeze(0)  # (T, N)

        # Check if any position has at least one visible cache entry
        has_cache = causal_mask.any(dim=1).any()

        if not has_cache:
            # Only one segment — no previous segments to retrieve from
            # Cache boundaries for next call (inference) and return
            for k in range(n_segments):
                self.cache.add(h_boundaries[k], m_segments[k])
            return x_out

        # Step 4: Compute per-position CAUSAL current segment mean from INPUT
        # Each position t gets the cumulative mean of its segment's tokens up to t
        # (not the full segment mean, which would leak future info within the segment)
        positions = torch.arange(T, device=x.device)
        seg_starts = (positions // S) * S  # start of each position's segment
        within_seg_count = (positions - seg_starts + 1).unsqueeze(0).unsqueeze(-1).float()  # (1, T, 1)

        cumsum = torch.cumsum(x, dim=1)  # (B, T, D)
        padded_cumsum = torch.cat([torch.zeros(B, 1, D, device=x.device), cumsum], dim=1)  # (B, T+1, D)
        seg_offset = padded_cumsum[:, seg_starts, :]  # (B, T, D)
        m_current = (cumsum - seg_offset) / within_seg_count  # (B, T, D)

        # Step 5: GRM gated retrieval
        # Query u_t from INPUT x (per paper Eq 10: u_t = x_t @ W_u)
        # Values h_stack from OUTPUT x_out (our proxy for cached memory states)
        h_retrieved, gate_current, self._last_attn_weights = self.grm(
            x, h_stack, m_stack, m_current, causal_mask
        )

        # Step 6: Paper's Eq 8 — gated combination
        # y_t = γ_t^(current) * x_out_t + h_retrieved_t (already weighted by γ_t^(cached))
        x_out = gate_current * x_out + self.retrieval_scale * h_retrieved

        # Step 7: Cache boundaries for use in subsequent forward passes (inference)
        for k in range(n_segments):
            self.cache.add(h_boundaries[k], m_segments[k])

        return x_out

    def reset_cache(self):
        """Reset the segment cache (call between sequences)."""
        self.cache.reset()
        self._last_attn_weights = None

    @property
    def cache_stats(self) -> dict:
        """Return cache statistics for logging."""
        stats = {"num_entries": self.cache.num_entries}
        if hasattr(self, '_last_attn_weights') and self._last_attn_weights is not None:
            # Attention entropy: measures how spread the attention is over cache entries
            # Higher entropy = more uniform attention (less selective)
            attn = self._last_attn_weights  # (B, T, N)
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
            stats["grm_entropy"] = entropy.item()
        return stats


class MCMambaLM(nn.Module):
    """
    MC-Mamba Language Model for RVQ token prediction.

    Full model: embed → MCMambaBlocks → output head
    Same interface as MambaLM for drop-in replacement.

    Architecture:
      - RVQEmbedding: sum-of-embeddings input (one per codebook)
      - 20x MCMambaBlock: Mamba + Memory Cache + GRM
      - RVQOutputHead: per-codebook output logits

    Estimated params: ~140M (128M base + 11.8M MC overhead)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.embed = RVQEmbedding(c.n_codebooks, c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        # MC config
        mc = c.mc

        self.blocks = nn.ModuleList([
            MCMambaBlock(
                d_model=c.d_model,
                d_state=c.d_state,
                d_conv=c.d_conv,
                expand=c.expand,
                dt_rank=c.computed_dt_rank,
                dropout=c.dropout,
                segment_size=mc.segment_size,
                retrieval_scale=mc.retrieval_scale,
                max_cache_entries=mc.max_cache_entries,
            )
            for _ in range(c.n_layers)
        ])

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)

        # Weight tying: tie first codebook embedding with first output head
        if hasattr(self.output, 'heads') and hasattr(self.embed, 'embeddings'):
            self.output.heads[0].weight = self.embed.embeddings[0].weight

        # Scaled initialization for deeper layers
        self._init_weights()

        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        """Apply scaled initialization for stability in deep networks."""
        n_layers = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # Scale residual contributions by 1/sqrt(2*n_layers)
            # following GPT-2 / Mamba convention
            scale = 1.0 / math.sqrt(2.0 * n_layers)
            if hasattr(block.mamba_block.mamba, 'out_proj'):
                with torch.no_grad():
                    block.mamba_block.mamba.out_proj.weight *= scale

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
        # Reset caches at the start of each forward pass during training
        # (each training sample is an independent sequence)
        self.reset_all_caches()

        x = self.embed(codes)   # (B, T, d_model)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)   # (B, K, T, vocab_size)

    def compute_loss(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pad_token: int = 1024,
        codebook_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute next-token prediction loss across all codebooks.

        Args:
            codes: (B, K, T)
            mask: unused
            pad_token: padding token index
            codebook_weights: (K,) optional per-codebook loss weights

        Returns:
            scalar loss
        """
        logits = self.forward(codes, mask)  # (B, K, T, V)
        B, K, T, V = logits.shape

        if codebook_weights is not None:
            # Weighted per-codebook loss
            total_loss = 0.0
            for k in range(K):
                k_logits = logits[:, k, :-1, :].contiguous().view(-1, V)
                k_targets = codes[:, k, 1:].contiguous().view(-1)
                k_loss = F.cross_entropy(k_logits, k_targets, ignore_index=pad_token, label_smoothing=0.1)
                total_loss = total_loss + codebook_weights[k] * k_loss
            return total_loss / codebook_weights.sum()
        else:
            # Standard uniform loss
            logits = logits[:, :, :-1, :].contiguous()
            targets = codes[:, :, 1:].contiguous()

            logits = logits.view(-1, V)
            targets = targets.view(-1)

            return F.cross_entropy(logits, targets, ignore_index=pad_token, label_smoothing=0.1)

    def reset_all_caches(self):
        """Reset segment caches in all MCMambaBlocks."""
        for block in self.blocks:
            block.reset_cache()

    def get_mc_stats(self) -> dict:
        """Collect MC statistics from all blocks for logging."""
        stats = {}
        total_entries = 0
        total_entropy = 0.0
        n_entropy = 0

        for i, block in enumerate(self.blocks):
            block_stats = block.cache_stats
            total_entries += block_stats["num_entries"]
            if "grm_entropy" in block_stats:
                total_entropy += block_stats["grm_entropy"]
                n_entropy += 1

        stats["total_cache_entries"] = total_entries
        stats["avg_cache_entries"] = total_entries / len(self.blocks)
        if n_entropy > 0:
            stats["avg_grm_entropy"] = total_entropy / n_entropy

        return stats
