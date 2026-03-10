"""
MC-Linear-Attention: Faithful Memory Caching for Linear Attention RNNs.

This is the FAITHFUL implementation of the MC paper (arXiv:2602.24281) applied
to linear attention. Unlike MC-Mamba (which uses output-activation proxies),
linear attention maintains an explicit state matrix S_t in R^{d_k x d_v}
that we can cache directly and query with phi(q_t) @ S_i.

Key difference from MC-Mamba:
  - MC-Mamba caches boundary OUTPUT activations (d_model vectors) as proxy
  - MC-Linear-Attention caches actual STATE MATRICES (d_k x d_v per head)
  - Retrieval: phi(q_t) @ S_i (query-dependent, per-head) -- exactly as paper

Running both gives a controlled comparison: if MC works on linear attention
but not Mamba, the proxy was the bottleneck, not the mechanism itself.

Architecture (GRM variant -- Equations 8-10 from MC paper):
  Per MCLinearAttentionBlock, per forward pass:
    1. x_normed = RMSNorm(x_in)
    2. attn_out, boundary_states, phi_q = CausalLinearAttention(x_normed)
    3. x_out = x_in + dropout(attn_out)  -- standard block output
    4. Extract state matrices S_i at segment boundaries [S-1, 2S-1, ...]
    5. Compute segment means m_i from x_in (input space, per paper)
    6. Gating: gamma_i = softmax(<u_t, m_i>) where u_t = W_u @ x_in
    7. Retrieval: h_i = phi(q_t) @ S_i (per-head state query)
    8. Output: gamma_current * x_out + retrieval_scale * sum(gamma_i * h_i)

New params per layer: W_u = d_model x d_model = ~590K (for d_model=768)
Total MC overhead: ~11.8M across 20 layers (~9% over base)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from .linear_attention import CausalLinearAttention
from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm


class StateMatrixCache:
    """
    Stateful buffer storing state matrices and segment summaries.

    At each segment boundary (positions S-1, 2S-1, 3S-1, ...), we store:
      - S_i: the accumulated state matrix (B, n_heads, d_k, d_v)
      - m_i: the mean input activation over the segment (B, d_model)

    Unlike MC-Mamba's SegmentCache (which stores d_model vectors), this caches
    actual (d_k x d_v) state matrices per head -- the faithful MC approach.
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self.state_matrices: List[torch.Tensor] = []   # each (B, H, d_k, d_v)
        self.segment_means: List[torch.Tensor] = []     # each (B, d_model)

    def add(self, S_boundary: torch.Tensor, m_segment: torch.Tensor):
        """
        Add a new cache entry.

        Args:
            S_boundary: (B, H, d_k, d_v) state matrix at segment boundary
            m_segment: (B, d_model) mean input activation over the segment
        """
        self.state_matrices.append(S_boundary)
        self.segment_means.append(m_segment)

        if len(self.state_matrices) > self.max_entries:
            self.state_matrices.pop(0)
            self.segment_means.pop(0)

    @property
    def num_entries(self) -> int:
        return len(self.state_matrices)

    @property
    def is_empty(self) -> bool:
        return self.num_entries == 0

    def get_stacked(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack all cached entries into tensors.

        Returns:
            S_stack: (B, N, H, d_k, d_v) state matrices
            m_stack: (B, N, d_model) segment means
        """
        S_stack = torch.stack(self.state_matrices, dim=1)  # (B, N, H, dk, dv)
        m_stack = torch.stack(self.segment_means, dim=1)    # (B, N, d)
        return S_stack, m_stack

    def reset(self):
        """Clear all cached entries."""
        self.state_matrices.clear()
        self.segment_means.clear()


class MCGRMRetrieval(nn.Module):
    """
    Gated Residual Memory (GRM) retrieval for state matrices (Equations 8-10).

    This is the faithful MC retrieval: phi(q_t) @ S_i gives query-dependent,
    per-head retrieval from cached state matrices. Combined with softmax gating
    over segment means (same as MC-Mamba's GRM).

    The only new learnable parameter is W_u (d_model x d_model).
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.scale = 1.0 / math.sqrt(d_model)

        # Query projection for gating -- the only new parameter per layer
        self.W_u = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_u.weight)

    def forward(
        self,
        x: torch.Tensor,
        phi_q: torch.Tensor,
        state_stack: torch.Tensor,
        m_stack: torch.Tensor,
        m_current: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve cached context via GRM gating with state matrix queries.

        Args:
            x: (B, T, d_model) block INPUT activations (for gating, per paper)
            phi_q: (B, T, H, d_k) feature-mapped queries (for state retrieval)
            state_stack: (B, N, H, d_k, d_v) cached state matrices
            m_stack: (B, N, d_model) cached segment means (gating keys)
            m_current: (B, T, d_model) current segment mean per position
            causal_mask: (T, N) boolean -- True where position t can see entry j

        Returns:
            h_retrieved: (B, T, d_model) gated retrieved context from cached states
            gate_current: (B, T, 1) gate weight for current segment output
            attn_weights: (B, T, N+1) full attention distribution
        """
        B, T, D = x.shape
        N = state_stack.shape[1]
        H = self.n_heads

        # Gating scores (same mechanism as MC-Mamba's GRM)
        u = self.W_u(x)  # (B, T, D)

        # Scores for cached segments: (B, T, N)
        scores_cached = torch.bmm(u, m_stack.transpose(1, 2)) * self.scale

        # Apply causal mask
        scores_cached = scores_cached.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        # Score for current segment: (B, T, 1)
        scores_current = (u * m_current).sum(dim=-1, keepdim=True) * self.scale

        # Softmax over all segments (cached + current)
        all_scores = torch.cat([scores_cached, scores_current], dim=-1)  # (B, T, N+1)
        attn_weights = F.softmax(all_scores, dim=-1)

        gates_cached = attn_weights[:, :, :N]   # (B, T, N)
        gate_current = attn_weights[:, :, N:]    # (B, T, 1)

        # Retrieve from cached state matrices: phi(q_t) @ S_i
        # phi_q: (B, T, H, dk), state_stack: (B, N, H, dk, dv)
        # -> retrievals: (B, T, N, H, dv)
        retrievals = torch.einsum('bthk,bnhkv->btnhv', phi_q, state_stack)

        # Reshape to (B, T, N, d_model)
        retrievals = retrievals.reshape(B, T, N, H * self.d_v)

        # Apply gates: (B, T, N, 1) * (B, T, N, D) -> sum -> (B, T, D)
        h_retrieved = (gates_cached.unsqueeze(-1) * retrievals).sum(dim=2)

        return h_retrieved, gate_current, attn_weights


class MCLinearAttentionBlock(nn.Module):
    """
    Memory-Cached Linear Attention block.

    Contains its own RMSNorm + CausalLinearAttention (not wrapping LinearAttentionBlock)
    because we need access to intermediate state matrices via forward_with_states().

    Forward pass:
      1. x_normed = RMSNorm(x_in)
      2. attn_out, boundary_states, phi_q = attn.forward_with_states(x_normed, positions)
      3. x_out = x_in + dropout(attn_out)
      4. Extract segment means from x_in, build causal mask
      5. GRM retrieval using phi_q @ cached state matrices
      6. Return gamma_current * x_out + retrieval_scale * h_retrieved
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        use_deltanet: bool = False,
        dropout: float = 0.0,
        segment_size: int = 256,
        retrieval_scale: float = 1.0,
        max_cache_entries: int = 64,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.retrieval_scale = retrieval_scale

        # Core attention components (same as LinearAttentionBlock)
        self.norm = RMSNorm(d_model)
        self.attn = CausalLinearAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_deltanet=use_deltanet,
            dropout=dropout,
        )

        # MC components
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.cache = StateMatrixCache(max_entries=max_cache_entries)
        self.grm = MCGRMRetrieval(d_model, n_heads, d_k, d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            x_enhanced: (B, T, d_model)
        """
        B, T, D = x.shape
        S = self.segment_size

        n_segments = T // S
        if n_segments == 0:
            # Sequence shorter than segment size -- standard forward, no caching
            return x + self.attn(self.norm(x))

        # Step 1-2: Forward with intermediate state capture
        boundary_positions = [(k + 1) * S - 1 for k in range(n_segments)]
        x_normed = self.norm(x)
        attn_out, boundary_states, phi_q = self.attn.forward_with_states(
            x_normed, boundary_positions
        )

        # Step 3: Standard block output (residual)
        x_out = x + attn_out

        # Step 4: Extract segment means from INPUT (per paper Eq 10)
        m_segments = []
        for k in range(n_segments):
            seg_start = k * S
            seg_end = (k + 1) * S
            m_segments.append(x[:, seg_start:seg_end, :].mean(dim=1))  # (B, D)

        # Stack: (B, N, D)
        m_stack = torch.stack(m_segments, dim=1)

        # State stack: (B, N, H, dk, dv)
        state_stack = torch.stack(boundary_states, dim=1)

        # Step 5: Build causal mask (T, N)
        # Position t in segment k can attend to cache entry j only if j < k
        seg_idx = torch.arange(T, device=x.device) // S  # (T,)
        cache_idx = torch.arange(n_segments, device=x.device)  # (N,)
        causal_mask = seg_idx.unsqueeze(1) > cache_idx.unsqueeze(0)  # (T, N)

        has_cache = causal_mask.any(dim=1).any()

        if not has_cache:
            # Only one segment -- no previous segments to retrieve from
            for k in range(n_segments):
                self.cache.add(boundary_states[k], m_segments[k])
            return x_out

        # Step 6: Per-position CAUSAL current segment mean from INPUT
        # Each position t gets the cumulative mean of its segment's tokens up to t
        # (not the full segment mean, which would leak future info within the segment)
        positions = torch.arange(T, device=x.device)
        seg_starts = (positions // S) * S  # start of each position's segment
        within_seg_count = (positions - seg_starts + 1).unsqueeze(0).unsqueeze(-1).float()  # (1, T, 1)

        cumsum = torch.cumsum(x, dim=1)  # (B, T, D)
        padded_cumsum = torch.cat([torch.zeros(B, 1, D, device=x.device), cumsum], dim=1)  # (B, T+1, D)
        seg_offset = padded_cumsum[:, seg_starts, :]  # (B, T, D)
        m_current = (cumsum - seg_offset) / within_seg_count  # (B, T, D)

        # Step 7: GRM gated retrieval with state matrices
        h_retrieved, gate_current, self._last_attn_weights = self.grm(
            x, phi_q, state_stack, m_stack, m_current, causal_mask
        )

        # Step 8: Gated combination (Eq 8 from paper)
        x_out = gate_current * x_out + self.retrieval_scale * h_retrieved

        # Cache boundaries for subsequent forward passes (inference)
        for k in range(n_segments):
            self.cache.add(boundary_states[k], m_segments[k])

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
            attn = self._last_attn_weights  # (B, T, N+1)
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
            stats["grm_entropy"] = entropy.item()
        return stats


class MCLinearAttentionLM(nn.Module):
    """
    MC-Linear-Attention Language Model for RVQ token prediction.

    Full model: embed -> MCLinearAttentionBlocks -> output head
    Same interface as MCMambaLM for drop-in replacement.

    Architecture:
      - RVQEmbedding: sum-of-embeddings input (one per codebook)
      - 20x MCLinearAttentionBlock: linear attention + MC state caching + GRM
      - RVQOutputHead: per-codebook output logits

    Estimated params: ~130M (119M base + 11.8M MC overhead)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        la = getattr(c, 'la', None)
        n_heads = la.n_heads if la else c.n_heads
        use_deltanet = la.use_deltanet if la else False

        self.embed = RVQEmbedding(c.n_codebooks, c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)

        mc = c.mc

        self.blocks = nn.ModuleList([
            MCLinearAttentionBlock(
                d_model=c.d_model,
                n_heads=n_heads,
                use_deltanet=use_deltanet,
                dropout=c.dropout,
                segment_size=mc.segment_size,
                retrieval_scale=mc.retrieval_scale,
                max_cache_entries=mc.max_cache_entries,
            )
            for _ in range(c.n_layers)
        ])

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)

        # Weight tying
        if hasattr(self.output, 'heads') and hasattr(self.embed, 'embeddings'):
            self.output.heads[0].weight = self.embed.embeddings[0].weight

        self._init_weights()
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        """Scaled initialization for stability in deep networks."""
        n_layers = len(self.blocks)
        for block in self.blocks:
            scale = 1.0 / math.sqrt(2.0 * n_layers)
            with torch.no_grad():
                block.attn.W_o.weight *= scale

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            codes: (B, K, T) codebook indices
            mask: unused

        Returns:
            logits: (B, K, T, vocab_size)
        """
        # Reset caches at start of each forward (training: independent sequences)
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
            total_loss = 0.0
            for k in range(K):
                k_logits = logits[:, k, :-1, :].contiguous().view(-1, V)
                k_targets = codes[:, k, 1:].contiguous().view(-1)
                k_loss = F.cross_entropy(k_logits, k_targets, ignore_index=pad_token, label_smoothing=0.1)
                total_loss = total_loss + codebook_weights[k] * k_loss
            return total_loss / codebook_weights.sum()
        else:
            logits = logits[:, :, :-1, :].contiguous()
            targets = codes[:, :, 1:].contiguous()

            logits = logits.view(-1, V)
            targets = targets.view(-1)

            return F.cross_entropy(logits, targets, ignore_index=pad_token, label_smoothing=0.1)

    def reset_all_caches(self):
        """Reset segment caches in all blocks."""
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
