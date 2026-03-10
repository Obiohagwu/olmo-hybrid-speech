"""
Model factory: instantiate any architecture from config.

Extends the ssm-mamba factory with mc_mamba, linear_attention,
and mc_linear_attention dispatch.
"""

from .transformer import TransformerLM
from .mamba_lm import MambaLM
from .hybrid import HybridLM
from .mc_mamba import MCMambaLM
from .linear_attention import LinearAttentionLM
from .mc_linear_attention import MCLinearAttentionLM
from .olmo_hybrid import OLMoHybridLM


def build_model(config):
    """
    Build model from config.arch string.

    Returns:
        model: nn.Module
    """
    arch = config.arch

    if arch == "transformer":
        model = TransformerLM(config)
    elif arch == "mamba1":
        model = MambaLM(config, version=1)
    elif arch == "mamba2":
        model = MambaLM(config, version=2)
    elif arch == "hybrid_1_7":
        model = HybridLM(config, attn_ratio="1:7")
    elif arch == "hybrid_1_3":
        model = HybridLM(config, attn_ratio="1:3")
    elif arch == "mc_mamba":
        model = MCMambaLM(config)
    elif arch == "linear_attention":
        model = LinearAttentionLM(config)
    elif arch == "mc_linear_attention":
        model = MCLinearAttentionLM(config)
    elif arch == "olmo_hybrid":
        model = OLMoHybridLM(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    print(f"Built {arch} model: {model.n_params / 1e6:.1f}M parameters")
    return model
