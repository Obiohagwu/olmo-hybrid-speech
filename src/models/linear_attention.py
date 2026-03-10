"""
Causal Linear Attention language model for RVQ token prediction.

Implements causal linear attention with explicit state matrix S_t in R^{d_k x d_v}:
  S_t = S_{t-1} + phi(k_t) outer v_t^T
  y_t = phi(q_t) @ S_t / (phi(q_t) @ z_t)

where phi(x) = elu(x) + 1 ensures non-negative queries/keys.

The state matrix is directly queryable, enabling faithful MC implementation
(vs MC-Mamba's output-activation proxy). This is THE key advantage of linear
attention for the MC mechanism: we can cache actual state matrices and do
query-dependent retrieval q_t @ S_i, exactly as the MC paper describes.

Supports DeltaNet variant (use_deltanet=True):
  S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T phi(k_t)) outer phi(k_t)^T
  where beta_t is a learned per-head gate (sigmoid).

Pure PyTorch -- no FLA/Triton dependency. Sequential scan for correctness.
Can add FLA kernels later as optimization if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict

from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm


def elu_plus_1(x: torch.Tensor) -> torch.Tensor:
    """Feature map phi(x) = elu(x) + 1. Ensures non-negative outputs."""
    return F.elu(x) + 1.0


class CausalLinearAttention(nn.Module):
    """
    Causal linear attention with explicit state matrix.

    Each head maintains state S_t in R^{d_k x d_v}:
      S_t = S_{t-1} + phi(k_t) outer v_t^T
      y_t = phi(q_t) @ S_t / (phi(q_t) @ z_t)
    where z_t = z_{t-1} + phi(k_t) is the normalizer.

    DeltaNet variant (use_deltanet=True):
      S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T phi(k_t)) outer phi(k_t)^T
      Error-correction update gives better retrieval, same state shape.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        use_deltanet: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.use_deltanet = use_deltanet

        # Q, K, V, O projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # DeltaNet: learnable per-head beta gate
        if use_deltanet:
            self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
            # Conservative init: sigmoid(-2) ≈ 0.12, prevents state explosion early
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.constant_(self.beta_proj.bias, -2.0)

        self.dropout = nn.Dropout(dropout)

        # State: stored after forward for external access
        self._last_state = None  # (B, H, d_k, d_v)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(proj.weight)

    def _scan(
        self,
        x: torch.Tensor,
        capture_positions: Optional[set] = None,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:
        """
        Dispatches to chunk-parallel or sequential scan.

        Chunk-parallel (~30-60x faster) is used for standard linear attention.
        Sequential fallback is used for DeltaNet (state-dependent updates).

        Args:
            x: (B, T, d_model)
            capture_positions: if set, capture state matrices at these positions
            chunk_size: chunk size for parallel scan (default 64)

        Returns:
            y: (B, T, d_model) attention output (after W_o + dropout)
            captured_states: dict mapping position -> (B, H, d_k, d_v) state
            phi_q: (B, T, H, d_k) feature-mapped queries
        """
        B, T, D = x.shape
        H = self.n_heads
        dk = self.d_k
        dv = self.d_v

        # Project Q, K, V and reshape to multi-head
        q = self.W_q(x).view(B, T, H, dk)
        k = self.W_k(x).view(B, T, H, dk)
        v = self.W_v(x).view(B, T, H, dv)

        # Apply feature map
        phi_q = elu_plus_1(q)  # (B, T, H, dk)
        phi_k = elu_plus_1(k)  # (B, T, H, dk)

        if self.use_deltanet:
            phi_k = F.normalize(phi_k, dim=-1)
            beta = torch.sigmoid(self.beta_proj(x))  # (B, T, H)
            y_raw, captured = self._sequential_scan(
                phi_q, phi_k, v, beta, capture_positions
            )
        else:
            # Check if capture positions align with chunk boundaries
            use_chunks = T >= chunk_size
            if use_chunks and capture_positions is not None:
                for pos in capture_positions:
                    if (pos + 1) % chunk_size != 0:
                        use_chunks = False
                        break

            if use_chunks:
                y_raw, captured = self._chunk_parallel_scan(
                    phi_q, phi_k, v, chunk_size, capture_positions
                )
            else:
                y_raw, captured = self._sequential_scan(
                    phi_q, phi_k, v, None, capture_positions
                )

        y = self.W_o(y_raw)
        y = self.dropout(y)

        return y, captured, phi_q

    def _sequential_scan(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        beta: Optional[torch.Tensor],
        capture_positions: Optional[set],
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Sequential scan fallback. Required for DeltaNet (state-dependent updates).

        Args:
            phi_q: (B, T, H, dk) feature-mapped queries
            phi_k: (B, T, H, dk) feature-mapped keys
            v: (B, T, H, dv) values
            beta: (B, T, H) DeltaNet gates, or None for standard
            capture_positions: positions to capture state at

        Returns:
            y: (B, T, H*dv) raw output (before W_o)
            captured: dict mapping position -> (B, H, dk, dv) state
        """
        B, T, H, dk = phi_q.shape
        dv = v.shape[-1]

        S = torch.zeros(B, H, dk, dv, device=phi_q.device, dtype=phi_q.dtype)
        z = torch.zeros(B, H, dk, device=phi_q.device, dtype=phi_q.dtype)
        captured = {}
        outputs = []

        for t in range(T):
            k_t = phi_k[:, t]
            v_t = v[:, t]
            q_t = phi_q[:, t]

            if beta is not None:
                beta_t = beta[:, t]
                retrieved = torch.einsum('bhkv,bhk->bhv', S, k_t)
                error = v_t - retrieved
                update = torch.einsum('bhk,bhv->bhkv', k_t, error)
                S = S + beta_t[:, :, None, None] * update
            else:
                S = S + torch.einsum('bhk,bhv->bhkv', k_t, v_t)

            z = z + k_t

            numerator = torch.einsum('bhk,bhkv->bhv', q_t, S)
            denominator = (q_t * z).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            y_t = numerator / denominator

            outputs.append(y_t)

            if capture_positions is not None and t in capture_positions:
                captured[t] = S

        self._last_state = S

        y = torch.stack(outputs, dim=1)  # (B, T, H, dv)
        y = y.reshape(B, T, H * dv)
        return y, captured

    def _chunk_parallel_scan(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
        capture_positions: Optional[set],
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Chunk-parallel linear attention scan. ~30-60x faster than sequential.

        Splits the sequence into chunks of size C. Within each chunk, uses
        parallel matmuls (GPU-friendly). Between chunks, passes state sequentially.

        Only T/C sequential steps instead of T. Same math, same outputs.

        Args:
            phi_q: (B, T, H, dk) feature-mapped queries
            phi_k: (B, T, H, dk) feature-mapped keys
            v: (B, T, H, dv) values
            chunk_size: number of positions per chunk
            capture_positions: positions to capture state at (must be at chunk boundaries)

        Returns:
            y: (B, T, H*dv) raw output (before W_o)
            captured: dict mapping position -> (B, H, dk, dv) state
        """
        B, T, H, dk = phi_q.shape
        dv = v.shape[-1]
        C = chunk_size

        # Pad to multiple of C
        n_chunks = (T + C - 1) // C
        T_padded = n_chunks * C
        pad = T_padded - T

        if pad > 0:
            phi_q = F.pad(phi_q, (0, 0, 0, 0, 0, pad))
            phi_k = F.pad(phi_k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))

        # Reshape into chunks: (B, n_chunks, C, H, dk/dv)
        qc = phi_q.view(B, n_chunks, C, H, dk)
        kc = phi_k.view(B, n_chunks, C, H, dk)
        vc = v.view(B, n_chunks, C, H, dv)

        # Causal mask for intra-chunk attention (created once, reused)
        causal_mask = torch.tril(
            torch.ones(C, C, device=phi_q.device, dtype=phi_q.dtype)
        )

        S = torch.zeros(B, H, dk, dv, device=phi_q.device, dtype=phi_q.dtype)
        z = torch.zeros(B, H, dk, device=phi_q.device, dtype=phi_q.dtype)
        captured = {}
        all_outputs = []

        for c in range(n_chunks):
            q_c = qc[:, c]  # (B, C, H, dk)
            k_c = kc[:, c]  # (B, C, H, dk)
            v_c = vc[:, c]  # (B, C, H, dv)

            # Inter-chunk: contribution from accumulated state
            # phi(Q) @ S_prev: (B, C, H, dk) @ (B, H, dk, dv) -> (B, C, H, dv)
            inter = torch.einsum('bchk,bhkv->bchv', q_c, S)
            z_inter = torch.einsum('bchk,bhk->bch', q_c, z)  # (B, C, H)

            # Intra-chunk: causal attention within chunk (parallel matmuls)
            # A[i,j] = phi(q_i) . phi(k_j), masked to j <= i
            attn = torch.einsum('bihk,bjhk->bhij', q_c, k_c)  # (B, H, C, C)
            attn = attn * causal_mask

            # Intra output and normalizer
            intra = torch.einsum('bhij,bjhv->bihv', attn, v_c)  # (B, C, H, dv)
            z_intra = attn.sum(dim=-1).permute(0, 2, 1)  # (B, H, C) -> (B, C, H)

            # Combine with normalizer
            numerator = inter + intra
            denominator = (z_inter + z_intra).unsqueeze(-1).clamp(min=1e-6)
            y_c = numerator / denominator

            all_outputs.append(y_c)

            # Update accumulated state for next chunk
            S = S + torch.einsum('bchk,bchv->bhkv', k_c, v_c)
            z = z + k_c.sum(dim=1)

            # Capture state at chunk boundary (last position of chunk)
            last_pos = (c + 1) * C - 1
            if capture_positions is not None and last_pos in capture_positions:
                captured[last_pos] = S

        self._last_state = S

        # Concatenate chunks: (B, T_padded, H, dv) -> trim -> (B, T, H*dv)
        y = torch.cat(all_outputs, dim=1)
        if pad > 0:
            y = y[:, :T, :, :]
        y = y.reshape(B, T, H * dv)

        return y, captured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Args:
            x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        y, _, _ = self._scan(x)
        return y

    def forward_with_states(
        self,
        x: torch.Tensor,
        positions: List[int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass that also captures intermediate state matrices.

        Used by MCLinearAttentionBlock to extract boundary states for caching.

        Args:
            x: (B, T, d_model)
            positions: list of timestep positions to capture state at

        Returns:
            y: (B, T, d_model) attention output
            boundary_states: list of (B, H, d_k, d_v) states at requested positions
            phi_q: (B, T, H, d_k) feature-mapped queries (for MC retrieval)
        """
        y, captured, phi_q = self._scan(x, capture_positions=set(positions))
        boundary_states = [captured[p] for p in positions]
        return y, boundary_states, phi_q

    def get_state(self) -> torch.Tensor:
        """
        Return the final state matrix (after last forward call).

        Returns:
            S: (B, n_heads, d_k, d_v)
        """
        if self._last_state is None:
            raise RuntimeError("No state available -- call forward() first")
        return self._last_state


class LinearAttentionBlock(nn.Module):
    """
    Linear attention block: pre-norm -> CausalLinearAttention -> residual.

    Same interface as MambaBlock: forward(x) -> x
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        use_deltanet: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = CausalLinearAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_deltanet=use_deltanet,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm -> attention -> residual."""
        return x + self.attn(self.norm(x))


class LinearAttentionLM(nn.Module):
    """
    Linear Attention language model for RVQ token prediction.

    Same interface as MambaLM: forward(codes, mask) -> logits.
    Drop-in replacement for baseline comparison.

    Architecture:
      - RVQEmbedding: sum-of-embeddings input (one per codebook)
      - N x LinearAttentionBlock: causal linear attention
      - RVQOutputHead: per-codebook output logits
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

        self.blocks = nn.ModuleList([
            LinearAttentionBlock(
                d_model=c.d_model,
                n_heads=n_heads,
                use_deltanet=use_deltanet,
                dropout=c.dropout,
            )
            for _ in range(c.n_layers)
        ])

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)

        # Weight tying: first codebook embedding <-> first output head
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
            mask: unused (linear attention doesn't need mask)

        Returns:
            logits: (B, K, T, vocab_size)
        """
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
    ) -> torch.Tensor:
        """Next-token prediction loss across all codebooks."""
        logits = self.forward(codes, mask)
        B, K, T, V = logits.shape

        logits = logits[:, :, :-1, :].contiguous()
        targets = codes[:, :, 1:].contiguous()

        logits = logits.view(-1, V)
        targets = targets.view(-1)

        return F.cross_entropy(logits, targets, ignore_index=pad_token, label_smoothing=0.1)
