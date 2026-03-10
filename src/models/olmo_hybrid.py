"""
OLMo-style hybrid decoder for RVQ token language modeling.

This adapts the public OLMo Hybrid design to the local audio-token stack:
  - 3 Gated DeltaNet recurrent mixer blocks
  - 1 full attention block
  - repeated throughout the network

Attention uses PyTorch SDPA so A100 training can hit the FlashAttention
kernel on the fast path, while padding-aware batches fall back to masked SDPA.

The recurrent mixer follows the released OLMo-core GatedDeltaNet structure
(q/k/v/a/b projections, A_log + dt_bias, short depthwise convs, gated RMSNorm),
but executes the recurrent scan in plain PyTorch instead of the fused FLA kernel.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RVQEmbedding, RVQOutputHead
from .norms import RMSNorm

try:
    import torch.distributed.tensor as _torch_dist_tensor
    from torch.distributed._tensor import Partial as _DTensorPartial
    from torch.distributed._tensor import Placement as _DTensorPlacement
    from torch.distributed._tensor import Replicate as _DTensorReplicate
    from torch.distributed._tensor import Shard as _DTensorShard
    from torch.distributed._tensor import distribute_module as _DTensorDistributeModule
    from torch.distributed._tensor import distribute_tensor as _DTensorDistributeTensor

    if not hasattr(_torch_dist_tensor, "Replicate"):
        _torch_dist_tensor.Replicate = _DTensorReplicate
    if not hasattr(_torch_dist_tensor, "Shard"):
        _torch_dist_tensor.Shard = _DTensorShard
    if not hasattr(_torch_dist_tensor, "Partial"):
        _torch_dist_tensor.Partial = _DTensorPartial
    if not hasattr(_torch_dist_tensor, "Placement"):
        _torch_dist_tensor.Placement = _DTensorPlacement
    if not hasattr(_torch_dist_tensor, "distribute_module"):
        _torch_dist_tensor.distribute_module = _DTensorDistributeModule
    if not hasattr(_torch_dist_tensor, "distribute_tensor"):
        _torch_dist_tensor.distribute_tensor = _DTensorDistributeTensor
except Exception:
    pass

FLA_DISABLE = os.environ.get("OLMO_DISABLE_FLA", "").strip().lower() in {"1", "true", "yes", "on"}

if not FLA_DISABLE:
    try:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

        HAS_FLA = True
    except ImportError:
        HAS_FLA = False
        fla_causal_conv1d = None
        fla_chunk_gated_delta_rule = None
else:
    HAS_FLA = False
    fla_causal_conv1d = None
    fla_chunk_gated_delta_rule = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class HeadRMSNorm(nn.Module):
    """RMSNorm over the per-head channel dimension."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GatedRMSNorm(nn.Module):
    """Per-head RMSNorm with a SiLU gate, matching OLMo-core GDN structure."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        gate = gate.float()
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * norm * self.weight
        x = x * F.silu(gate)
        return x.to(dtype=x_dtype)


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal conv over the sequence dimension with SiLU activation."""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_FLA and x.device.type == "cuda":
            return fla_causal_conv1d(
                x=x,
                weight=self.conv.weight.squeeze(1),
                bias=self.conv.bias,
                activation="silu",
            )[0]

        seq_len = x.size(1)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :seq_len]
        x = x.transpose(1, 2).contiguous()
        return F.silu(x)


class RotaryEmbedding(nn.Module):
    """Standard RoPE cache for attention heads."""

    def __init__(self, dim: int, base: int = 500_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :].to(dtype=dtype)
        self._sin_cached = emb.sin()[None, None, :, :].to(dtype=dtype)
        self._cached_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._cached_seq_len < seq_len
            or self._cos_cached.device != q.device
            or self._cos_cached.dtype != q.dtype
        ):
            self._build_cache(seq_len, q.device, q.dtype)

        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class SwiGLUFeedForward(nn.Module):
    """Bias-free SwiGLU MLP used in OLMo-style blocks."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class OLMoAttentionMixer(nn.Module):
    """RoPE + QK-norm attention with optional grouped-query attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = HeadRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, base=rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat_factor = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat_factor, dim=1)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        use_gqa = self.n_kv_heads != self.n_heads

        # Preserve the FlashAttention fast path for the common full-length case.
        if mask is not None and bool(mask.all().item()):
            mask = None

        if mask is None:
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    dropout_p=dropout_p,
                    enable_gqa=use_gqa,
                )
            except TypeError:
                if use_gqa:
                    k = self._repeat_kv(k)
                    v = self._repeat_kv(v)
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                    dropout_p=dropout_p,
                )

        seq_len = q.size(-2)
        causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        key_mask = mask[:, None, None, :].to(dtype=torch.bool)
        attn_mask = causal[None, None, :, :] & key_mask

        try:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                enable_gqa=use_gqa,
            )
        except TypeError:
            if use_gqa:
                k = self._repeat_kv(k)
                v = self._repeat_kv(v)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)

        out = self._sdpa(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class GatedDeltaNetMixer(nn.Module):
    """Paper-style Gated DeltaNet mixer with a plain PyTorch recurrent scan."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_v_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_v_heads is not None and (n_v_heads < n_heads or n_v_heads % n_heads != 0):
            raise ValueError(
                f"n_v_heads={n_v_heads} must be >= n_heads={n_heads} and divisible by it"
            )

        self.n_heads = n_heads
        self.n_v_heads = n_v_heads or n_heads
        self.head_k_dim = head_dim or (d_model // n_heads)
        self.head_v_dim = int(self.head_k_dim * expand_v)
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.allow_neg_eigval = allow_neg_eigval

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.a_proj = nn.Linear(d_model, self.n_v_heads, bias=False)
        self.b_proj = nn.Linear(d_model, self.n_v_heads, bias=False)
        self.gate_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)

        self.A_log = nn.Parameter(torch.empty(self.n_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.n_v_heads))

        self.q_conv1d = CausalDepthwiseConv1d(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.k_conv1d = CausalDepthwiseConv1d(self.key_dim, kernel_size=conv_size, bias=conv_bias)
        self.v_conv1d = CausalDepthwiseConv1d(
            self.value_dim, kernel_size=conv_size, bias=conv_bias
        )
        self.o_norm = GatedRMSNorm(self.head_v_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.a_proj,
            self.b_proj,
            self.gate_proj,
            self.out_proj,
        ):
            nn.init.xavier_uniform_(proj.weight)
        for conv in (self.q_conv1d.conv, self.k_conv1d.conv, self.v_conv1d.conv):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        with torch.no_grad():
            self.A_log.uniform_(0.0, 16.0).log_()
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                torch.empty_like(self.dt_bias).uniform_(math.log(dt_min), math.log(dt_max))
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        state_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

        beta = torch.sigmoid(self.b_proj(x))
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -torch.exp(self.A_log.float()).view(1, 1, -1) * F.softplus(
            self.a_proj(x).float() + self.dt_bias.float().view(1, 1, -1)
        )

        q = self.q_conv1d(self.q_proj(x)).view(bsz, seq_len, self.n_heads, self.head_k_dim)
        k = self.k_conv1d(self.k_proj(x)).view(bsz, seq_len, self.n_heads, self.head_k_dim)
        v = self.v_conv1d(self.v_proj(x)).view(bsz, seq_len, self.n_v_heads, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=2)
            k = k.repeat_interleave(repeat_factor, dim=2)

        if HAS_FLA and x.device.type == "cuda":
            out, _ = fla_chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            alpha = torch.exp(g).to(dtype=state_dtype)
            beta = beta.to(dtype=state_dtype)
            q = F.normalize(q.to(dtype=state_dtype), dim=-1, eps=1e-6)
            k = F.normalize(k.to(dtype=state_dtype), dim=-1, eps=1e-6)
            v = v.to(dtype=state_dtype)

            state = torch.zeros(
                bsz,
                self.n_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=x.device,
                dtype=state_dtype,
            )
            outputs = []

            # Plain recurrent scan fallback for machines without the fused FLA kernel.
            for t in range(seq_len):
                q_t = q[:, t]
                k_t = k[:, t]
                v_t = v[:, t]
                alpha_t = alpha[:, t][:, :, None, None]
                beta_t = beta[:, t][:, :, None, None]

                retained = torch.einsum("bhvk,bhk->bhv", state, k_t)
                state = alpha_t * (
                    state - beta_t * torch.einsum("bhv,bhk->bhvk", retained, k_t)
                ) + beta_t * torch.einsum("bhv,bhk->bhvk", v_t, k_t)
                outputs.append(torch.einsum("bhvk,bhk->bhv", state, q_t))

            out = torch.stack(outputs, dim=1)

        gate = self.gate_proj(x).view(bsz, seq_len, self.n_v_heads, self.head_v_dim)
        out = self.o_norm(out, gate.to(dtype=state_dtype))
        out = out.reshape(bsz, seq_len, -1).to(dtype=x.dtype)
        out = self.out_proj(out)
        return self.dropout(out)


class OLMoAttentionBlock(nn.Module):
    """Pre-norm attention + SwiGLU block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: int,
        qk_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.mixer_norm = RMSNorm(d_model)
        self.mixer = OLMoAttentionMixer(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            dropout=dropout,
        )
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.mixer(self.mixer_norm(x), mask=mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class OLMoDeltaNetBlock(nn.Module):
    """Pre-norm paper-style GDN + SwiGLU block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        gdn_qk_ratio: float,
        gdn_value_factor: int,
        gdn_conv_size: int,
        allow_neg_eigval: bool,
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.mixer_norm = RMSNorm(d_model)
        self.mixer = GatedDeltaNetMixer(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=max(1, int(gdn_qk_ratio * (d_model // n_heads))),
            expand_v=float(gdn_value_factor),
            allow_neg_eigval=allow_neg_eigval,
            conv_size=gdn_conv_size,
            norm_eps=norm_eps,
            dropout=dropout,
        )
        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class OLMoHybridLM(nn.Module):
    """
    OLMo-style hybrid language model over RVQ tokens.

    Block schedule follows the public OLMo Hybrid recipe:
    three recurrent GDN blocks, then one full attention block, with the
    final layer optionally forced to attention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.embed = RVQEmbedding(
            c.n_codebooks,
            c.vocab_size,
            c.d_model,
            use_pos_embedding=False,
        )
        self.drop = nn.Dropout(c.dropout)

        attention_period = c.olmo.attention_period
        if attention_period < 2:
            raise ValueError("olmo.attention_period must be >= 2")

        self.blocks = nn.ModuleList()
        self.is_attention = []
        for layer_idx in range(c.n_layers):
            is_attention = (layer_idx + 1) % attention_period == 0
            if c.olmo.force_final_attention and layer_idx == c.n_layers - 1:
                is_attention = True

            if is_attention:
                self.blocks.append(
                    OLMoAttentionBlock(
                        d_model=c.d_model,
                        d_ff=c.d_ff,
                        n_heads=c.n_heads,
                        n_kv_heads=c.resolved_n_kv_heads,
                        rope_theta=c.olmo.rope_theta,
                        qk_norm=c.olmo.qk_norm,
                        dropout=c.dropout,
                    )
                )
                self.is_attention.append(True)
            else:
                self.blocks.append(
                    OLMoDeltaNetBlock(
                        d_model=c.d_model,
                        d_ff=c.d_ff,
                        n_heads=c.n_heads,
                        gdn_qk_ratio=c.olmo.gdn_qk_ratio,
                        gdn_value_factor=c.olmo.gdn_value_factor,
                        gdn_conv_size=c.olmo.gdn_conv_size,
                        allow_neg_eigval=c.olmo.use_negative_eigvals,
                        norm_eps=c.layer_norm_eps,
                        dropout=c.dropout,
                    )
                )
                self.is_attention.append(False)

        n_attn = sum(self.is_attention)
        n_delta = len(self.is_attention) - n_attn
        print(f"OLMo Hybrid: {n_delta} DeltaNet + {n_attn} Attention layers")

        self.output = RVQOutputHead(c.n_codebooks, c.vocab_size, c.d_model)

        self._scale_residual_projections()
        self.n_params = sum(p.numel() for p in self.parameters())

    def _scale_residual_projections(self):
        scale = 1.0 / math.sqrt(2.0 * max(len(self.blocks), 1))
        with torch.no_grad():
            for block in self.blocks:
                block.mixer.out_proj.weight.mul_(scale)
                block.ff.down_proj.weight.mul_(scale)

    def forward(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(codes)
        x = self.drop(x)

        for block, is_attention in zip(self.blocks, self.is_attention):
            if is_attention:
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
        _, _, _, vocab_size = logits.shape

        logits = logits[:, :, :-1, :].contiguous().view(-1, vocab_size)
        targets = codes[:, :, 1:].contiguous().view(-1)
        return F.cross_entropy(
            logits,
            targets,
            ignore_index=pad_token,
            label_smoothing=0.1,
        )
