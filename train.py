"""
Training script for MC-Mamba: Memory Caching for Mamba SSMs.

Usage:
    python train.py --arch mc_mamba                              # MC-Mamba (novel)
    python train.py --arch mc_mamba --segment_size 128           # Ablation: smaller segments
    python train.py --arch mamba1                                # Baseline (control)
    python train.py --arch transformer                           # Transformer baseline
    python train.py --arch olmo_hybrid --preset music_olmo_hybrid_300m_a100
    python train.py --arch olmo_hybrid --preset music_olmo_hybrid_m4_smoke --device mps
    python train.py --arch olmo_hybrid --preset music_olmo_hybrid_m4_20m_ctx512 --device mps
    python train.py --arch transformer --preset music_transformer_m4_20m_ctx512 --device mps
    python train.py --arch olmo_hybrid --preset speech_olmo_hybrid_m4_20m_ctx512 --device mps
    python train.py --arch transformer --preset speech_transformer_m4_20m_ctx512 --device mps
    python train.py --arch hybrid_1_7                            # Hybrid baseline
    python train.py --arch hybrid_1_3                            # Hybrid baseline
    python train.py --arch mc_linear_attention --preset speech_960h_pilot
    python train.py --arch mc_linear_attention --preset speech_scaled

Features:
  - EMA (Exponential Moving Average) model for evaluation
  - Codebook loss weighting (coarse codebooks weighted higher)
  - Weight tying (first codebook embedding ↔ first output head)
  - Scaled initialization for deep networks
  - MC-specific logging: GRM attention entropy, cache utilization
"""

import os
import sys
import time
import math
import json
import copy
import argparse
import re
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, CodecConfig, ExperimentConfig, MCConfig, LinearAttentionConfig
from src.models.factory import build_model
from src.data.tokenizer import PreTokenizedDataset, collate_fn


EXPERIMENT_PRESETS = {
    "speech_960h_pilot": {
        "train": {
            "dataset_name": "librispeech_960h_encodec24",
            "data_dir": "./data/speech/librispeech_960h_encodec24",
            "output_dir": "./runs/speech",
            "batch_size": 12,
            "grad_accum_steps": 4,
            "lr": 3e-4,
            "warmup_steps": 1000,
            "max_steps": 30_000,
            "eval_every": 1000,
            "save_every": 2000,
            "log_every": 50,
            "max_duration_sec": 20.0,
        },
        "model": {
            "max_seq_len": 2048,
            "dropout": 0.1,
        },
        # Fallback defaults if codec_meta.json is absent.
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "speech_scaled": {
        "train": {
            "dataset_name": "speech_scaled",
            "data_dir": "./data/speech/scaled_encodec24",
            "output_dir": "./runs/speech",
            "batch_size": 16,
            "grad_accum_steps": 8,
            "lr": 2e-4,
            "warmup_steps": 4000,
            "max_steps": 200_000,
            "eval_every": 2000,
            "save_every": 5000,
            "log_every": 100,
            "max_duration_sec": 20.0,
        },
        "model": {
            "max_seq_len": 2048,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "speech_olmo_hybrid_m4_20m_ctx512": {
        "train": {
            "dataset_name": "speech_encodec24_8s",
            "data_dir": "./data/speech/encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 300,
            "max_steps": 30_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 0,
            "mixed_precision": True,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 512,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "speech_olmo_hybrid_m4_20m_ctx1024": {
        "train": {
            "dataset_name": "speech_encodec24_8s",
            "data_dir": "./data/speech/encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 1,
            "grad_accum_steps": 8,
            "lr": 3e-4,
            "warmup_steps": 500,
            "max_steps": 12_000,
            "eval_every": 200,
            "save_every": 500,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 0,
            "mixed_precision": True,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "speech_olmo_hybrid_a100_20m_ctx1024": {
        "train": {
            "dataset_name": "ljspeech_encodec24_8s",
            "data_dir": "./data/speech/ljspeech_encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 32,
            "grad_accum_steps": 1,
            "lr": 3e-4,
            "warmup_steps": 500,
            "max_steps": 12_000,
            "eval_every": 200,
            "save_every": 500,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 8,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking": True,
            "mixed_precision": True,
            "compile_model": True,
            "compile_mode": "max-autotune-no-cudagraphs",
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "fused_optimizer": True,
            "enable_flash_sdp": True,
            "enable_mem_efficient_sdp": True,
            "enable_math_sdp": True,
            "enable_cudnn_sdp": True,
            "float32_matmul_precision": "high",
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "speech_transformer_a100_20m_ctx1024": {
        "train": {
            "dataset_name": "ljspeech_encodec24_8s",
            "data_dir": "./data/speech/ljspeech_encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 48,
            "grad_accum_steps": 1,
            "lr": 3e-4,
            "warmup_steps": 500,
            "max_steps": 12_000,
            "eval_every": 200,
            "save_every": 500,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 8,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking": True,
            "mixed_precision": True,
            "compile_model": True,
            "compile_mode": "max-autotune-no-cudagraphs",
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "fused_optimizer": True,
            "enable_flash_sdp": True,
            "enable_mem_efficient_sdp": True,
            "enable_math_sdp": True,
            "enable_cudnn_sdp": True,
            "float32_matmul_precision": "high",
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "speech_transformer_m4_20m_ctx512": {
        "train": {
            "dataset_name": "speech_encodec24_8s",
            "data_dir": "./data/speech/encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 300,
            "max_steps": 30_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 512,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "speech_transformer_m4_20m_ctx1024": {
        "train": {
            "dataset_name": "speech_encodec24_8s",
            "data_dir": "./data/speech/encodec24_8s",
            "output_dir": "./runs/speech",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 300,
            "max_steps": 30_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
        "codec": {
            "sample_rate": 24000,
            "n_codebooks": 8,
            "codebook_size": 1024,
            "pad_token": 1024,
            "bos_token": 1025,
            "eos_token": 1026,
            "vocab_size": 1027,
        },
    },
    "music_olmo_hybrid_300m_a100": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/music",
            "batch_size": 4,
            "grad_accum_steps": 8,
            "lr": 2e-4,
            "warmup_steps": 2000,
            "max_steps": 200_000,
            "eval_every": 2000,
            "save_every": 5000,
            "log_every": 50,
            "max_duration_sec": 24.0,
        },
        "model": {
            "d_model": 1024,
            "n_layers": 22,
            "n_heads": 16,
            "d_ff": 2816,
            "max_seq_len": 2048,
            "dropout": 0.1,
        },
        "olmo": {
            "n_kv_heads": 4,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "music_olmo_hybrid_m4_smoke": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/local",
            "batch_size": 1,
            "grad_accum_steps": 4,
            "lr": 3e-4,
            "warmup_steps": 100,
            "max_steps": 2000,
            "eval_every": 200,
            "save_every": 500,
            "log_every": 20,
            "max_duration_sec": 8.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 12,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 512,
            "dropout": 0.1,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "music_olmo_hybrid_m4_20m_ctx512": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/local",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 200,
            "max_steps": 20_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 6.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 512,
            "dropout": 0.1,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "music_olmo_hybrid_m4_20m_ctx1024": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/local",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 200,
            "max_steps": 20_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 12.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
        "olmo": {
            "n_kv_heads": 2,
            "qk_norm": True,
            "rope_theta": 500_000,
            "attention_period": 4,
        },
    },
    "music_transformer_m4_20m_ctx512": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/local",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 200,
            "max_steps": 20_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 6.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 512,
            "dropout": 0.1,
        },
    },
    "music_transformer_m4_20m_ctx1024": {
        "train": {
            "dataset_name": "fma_large",
            "data_dir": "./data/dac_tokens",
            "output_dir": "./runs/local",
            "batch_size": 1,
            "grad_accum_steps": 16,
            "lr": 3e-4,
            "warmup_steps": 200,
            "max_steps": 20_000,
            "eval_every": 500,
            "save_every": 1000,
            "log_every": 20,
            "max_duration_sec": 12.0,
            "num_workers": 0,
            "mixed_precision": False,
        },
        "model": {
            "d_model": 384,
            "n_layers": 8,
            "n_heads": 6,
            "d_ff": 1024,
            "max_seq_len": 1024,
            "dropout": 0.1,
        },
    },
}


def compute_olmo_ffn_dim(d_model: int, multiple_of: int = 256) -> int:
    """LLaMA/OLMo-style SwiGLU hidden width rounded to a hardware-friendly multiple."""
    hidden = int((8 * d_model) / 3)
    return multiple_of * math.ceil(hidden / multiple_of)


def apply_codec_settings(config: ExperimentConfig, codec_settings: dict):
    """Apply codec settings to both codec and model token geometry."""
    codec_keys = {
        "sample_rate",
        "n_codebooks",
        "codebook_size",
        "pad_token",
        "bos_token",
        "eos_token",
        "vocab_size",
    }
    for key in codec_keys:
        if key in codec_settings:
            setattr(config.codec, key, codec_settings[key])

    # Keep model tokenization geometry aligned with codec settings.
    config.model.n_codebooks = config.codec.n_codebooks
    config.model.codebook_size = config.codec.codebook_size
    config.model.vocab_size = config.codec.vocab_size


def apply_preset(config: ExperimentConfig, preset_name: str):
    """Apply a named experiment preset to model/train config."""
    preset = EXPERIMENT_PRESETS[preset_name]
    for key, value in preset.get("train", {}).items():
        setattr(config.train, key, value)
    for key, value in preset.get("model", {}).items():
        setattr(config.model, key, value)
    for key, value in preset.get("olmo", {}).items():
        setattr(config.model.olmo, key, value)
    if "codec" in preset:
        apply_codec_settings(config, preset["codec"])


def maybe_apply_codec_metadata(config: ExperimentConfig, codec_meta_path: Path) -> bool:
    """Load codec metadata from disk and apply it if present."""
    if not codec_meta_path.exists():
        return False

    with open(codec_meta_path) as f:
        meta = json.load(f)
    apply_codec_settings(config, meta)
    print(f"Loaded codec metadata from {codec_meta_path}: {meta}")
    return True


def update_config_section(target, values: dict):
    """Recursively apply a saved JSON config section onto a dataclass-like object."""
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dict__"):
            update_config_section(current, value)
        else:
            setattr(target, key, value)


def maybe_apply_resume_config(config: ExperimentConfig, resume_from: Optional[str]) -> bool:
    """Load config.json next to a checkpoint when resuming."""
    if resume_from is None:
        return False

    resume_path = Path(resume_from).expanduser()
    config_path = resume_path.parent / "config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        payload = json.load(f)

    if "codec" in payload:
        apply_codec_settings(config, payload["codec"])
    if "model" in payload:
        update_config_section(config.model, payload["model"])
    if "train" in payload:
        update_config_section(config.train, payload["train"])
    if "arch" in payload:
        config.model.arch = payload["arch"]

    print(f"Loaded resume config from {config_path}")
    return True


def get_lr(step: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def count_parameters(model: nn.Module) -> dict:
    """Count total and per-component parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for p in model.embed.parameters())
    output = sum(p.numel() for p in model.output.parameters())
    backbone = total - embed - output

    result = {
        "total": total,
        "trainable": trainable,
        "embed": embed,
        "backbone": backbone,
        "output_head": output,
    }

    # MC-specific: count GRM parameters
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        if hasattr(first_block, 'grm'):
            grm_params = sum(
                sum(p.numel() for p in block.grm.parameters())
                for block in model.blocks
            )
            result["mc_grm"] = grm_params
            result["mc_overhead_pct"] = 100.0 * grm_params / total

    return result


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model params with EMA params (for eval)."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after eval."""
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = None


class NullGradScaler:
    """No-op scaler for non-CUDA training paths."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def is_mps_available() -> bool:
    """Return True when PyTorch can actually use Apple's MPS backend."""
    return bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve a user-requested device string to an available torch.device."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not is_mps_available():
            raise RuntimeError(
                "Requested device 'mps' but MPS is unavailable in this PyTorch runtime."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{requested}'. Use one of: auto, cuda, mps, cpu.")


def safe_torch_save(obj, path: Path) -> None:
    """Save checkpoints atomically and fall back if zip serialization breaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(obj, tmp_path)
    except RuntimeError:
        if tmp_path.exists():
            tmp_path.unlink()
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)

    os.replace(tmp_path, path)


class Trainer:
    def __init__(self, config: ExperimentConfig, arch: str, resume_from: Optional[str] = None):
        self.config = config
        self.arch = arch
        self.resume_from = Path(resume_from).expanduser() if resume_from else None

        config.model.arch = arch

        # Setup device
        self.device = resolve_device(config.train.device)
        if config.train.mixed_precision and self.device.type == "cuda":
            self.dtype = torch.bfloat16
        elif config.train.mixed_precision and self.device.type == "mps":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if self.dtype != torch.float32:
            self.ctx = torch.amp.autocast(device_type=self.device.type, dtype=self.dtype)
        else:
            self.ctx = nullcontext()

        if self.device.type == "cuda" and self.dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = NullGradScaler()

        # Seed
        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.train.seed)

        if self.device.type == "cuda":
            self._configure_cuda_runtime()

        # Build model
        self.model = build_model(config.model).to(self.device)
        self.runtime_model = self.model
        self.param_counts = count_parameters(self.model)

        # Compile if requested
        if config.train.compile_model and hasattr(torch, "compile"):
            print(f"Compiling model with torch.compile(mode={config.train.compile_mode})...")
            try:
                self.runtime_model = torch.compile(self.model, mode=config.train.compile_mode)
            except TypeError:
                self.runtime_model = torch.compile(self.model)

        # EMA
        self.ema = EMA(self.model, decay=config.train.ema_decay) if config.train.use_ema else None

        # Codebook loss weights
        if config.train.codebook_loss_weights is not None:
            self.codebook_weights = torch.tensor(
                config.train.codebook_loss_weights, dtype=torch.float32, device=self.device
            )
        else:
            self.codebook_weights = None

        # Optimizer
        optimizer_kwargs = {
            "lr": config.train.lr,
            "betas": config.train.betas,
            "weight_decay": config.train.weight_decay,
        }
        if self.device.type == "cuda" and config.train.fused_optimizer:
            optimizer_kwargs["fused"] = True
        try:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
            )
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_kwargs,
            )

        # Output directory
        if self.resume_from is not None:
            self.output_dir = self.resume_from.parent
            run_name = self.output_dir.name
            self.config.train.output_dir = str(self.output_dir.parent)
            self.config.train.run_name = run_name
        else:
            run_name = config.train.run_name or f"{arch}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.output_dir = Path(config.train.output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        log_mode = "a" if self.resume_from is not None else "w"
        self.log_file = open(self.output_dir / "train.log", log_mode)
        self.metrics_log = []

        # Save config (copy dicts to avoid mutating original config)
        config_dict = {
            "arch": arch,
            "params": self.param_counts,
            "codec": dict(vars(config.codec)),
            "model": dict(vars(config.model)),
            "train": dict(vars(config.train)),
        }
        # Serialize nested configs properly
        if hasattr(config.model, 'mc'):
            config_dict["model"]["mc"] = dict(vars(config.model.mc))
        if hasattr(config.model, 'la'):
            config_dict["model"]["la"] = dict(vars(config.model.la))
        if hasattr(config.model, 'olmo'):
            config_dict["model"]["olmo"] = dict(vars(config.model.olmo))

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        self.start_step = 0
        self.best_val_loss = float("inf")
        if self.resume_from is not None:
            self._resume_state()

    def _configure_cuda_runtime(self):
        tc = self.config.train
        torch.backends.cuda.matmul.allow_tf32 = tc.allow_tf32
        torch.backends.cudnn.allow_tf32 = tc.allow_tf32
        torch.backends.cudnn.benchmark = tc.cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(tc.float32_matmul_precision)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(tc.enable_flash_sdp)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(tc.enable_mem_efficient_sdp)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(tc.enable_math_sdp)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(tc.enable_cudnn_sdp)
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    def _compute_loss(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute loss, handling codebook_weights for any model architecture."""
        pad_token = self.config.codec.pad_token

        if self.codebook_weights is not None:
            # Weighted codebook loss — works for any model
            logits = self.runtime_model(codes, mask)  # (B, K, T, V)
            B, K, T, V = logits.shape
            total_loss = 0.0
            for k in range(K):
                k_logits = logits[:, k, :-1, :].contiguous().view(-1, V)
                k_targets = codes[:, k, 1:].contiguous().view(-1)
                k_loss = F.cross_entropy(k_logits, k_targets, ignore_index=pad_token, label_smoothing=0.1)
                total_loss = total_loss + self.codebook_weights[k] * k_loss
            return total_loss / self.codebook_weights.sum()
        else:
            return self.runtime_model.compute_loss(codes, mask, pad_token=pad_token)

    def log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def _infer_best_val_loss_from_log(self) -> float:
        """Recover the best validation loss seen so far from the existing log."""
        log_path = self.output_dir / "train.log"
        if not log_path.exists():
            return float("inf")

        best = float("inf")
        pattern = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)")
        with open(log_path) as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    best = min(best, float(match.group(1)))
        return best

    def _resume_state(self):
        """Load model/optimizer/EMA state from an existing checkpoint."""
        ckpt = torch.load(self.resume_from, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.ema is not None and "ema_shadow" in ckpt:
            self.ema.shadow = {
                name: tensor.to(self.device)
                for name, tensor in ckpt["ema_shadow"].items()
            }

        self.start_step = int(ckpt.get("step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", self._infer_best_val_loss_from_log()))
        self.log(f"Resumed from checkpoint: {self.resume_from}")
        self.log(f"Resume step: {self.start_step}")
        if math.isfinite(self.best_val_loss):
            self.log(f"Best val loss so far: {self.best_val_loss:.4f}")

    def build_dataloader(self, split: str = "train") -> DataLoader:
        """Build dataloader for pre-tokenized data."""
        data_dir = Path(self.config.train.data_dir) / split
        dataset = PreTokenizedDataset(
            data_dir=str(data_dir),
            max_seq_len=self.config.model.max_seq_len,
            n_codebooks=self.config.codec.n_codebooks,
            pad_token=self.config.codec.pad_token,
            use_delay_pattern=self.config.model.use_delay_pattern,
        )
        self.log(f"Loaded {split} dataset: {len(dataset)} samples from {data_dir}")

        loader_kwargs = {
            "batch_size": self.config.train.batch_size,
            "shuffle": (split == "train"),
            "num_workers": self.config.train.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": bool(self.config.train.pin_memory and self.device.type == "cuda"),
            "drop_last": (split == "train"),
        }
        if self.config.train.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.config.train.persistent_workers
            loader_kwargs["prefetch_factor"] = self.config.train.prefetch_factor

        return DataLoader(
            dataset,
            **loader_kwargs,
        )

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> dict:
        """Run evaluation on validation set."""
        # Use EMA params for evaluation if available
        if self.ema is not None:
            self.ema.apply(self.model)

        self.runtime_model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_batches = 0
        transfer_non_blocking = bool(self.config.train.non_blocking and self.device.type == "cuda")

        for batch in val_loader:
            if n_batches >= max_batches:
                break

            codes = batch["codes"].to(self.device, non_blocking=transfer_non_blocking)
            mask = batch["mask"].to(self.device, non_blocking=transfer_non_blocking)

            with self.ctx:
                loss = self._compute_loss(codes, mask)

            total_loss += loss.item() * codes.shape[0]
            total_tokens += codes.shape[0]
            n_batches += 1

        self.runtime_model.train()

        # Restore original params after EMA eval
        if self.ema is not None:
            self.ema.restore(self.model)

        avg_loss = total_loss / max(total_tokens, 1)
        return {
            "val_loss": avg_loss,
            "val_perplexity": math.exp(min(avg_loss, 20)),
        }

    def train(self):
        """Main training loop."""
        tc = self.config.train

        self.log(f"Architecture: {self.arch}")
        self.log(
            f"Codec: sr={self.config.codec.sample_rate}, "
            f"n_codebooks={self.config.codec.n_codebooks}, "
            f"codebook_size={self.config.codec.codebook_size}, "
            f"vocab={self.config.codec.vocab_size}"
        )
        self.log(f"Parameters: {self.param_counts}")
        self.log(f"Device: {self.device}, dtype: {self.dtype}")
        if self.device.type == "cuda":
            self.log(f"CUDA device: {torch.cuda.get_device_name(self.device)}")
            self.log(
                "CUDA runtime: "
                f"compile={tc.compile_model}({tc.compile_mode}), "
                f"tf32={tc.allow_tf32}, "
                f"fused_adamw={tc.fused_optimizer}, "
                f"flash_sdp={tc.enable_flash_sdp}, "
                f"mem_efficient_sdp={tc.enable_mem_efficient_sdp}, "
                f"math_sdp={tc.enable_math_sdp}, "
                f"cudnn_sdp={tc.enable_cudnn_sdp}"
            )
            self.log(
                "Loader: "
                f"workers={tc.num_workers}, "
                f"prefetch={tc.prefetch_factor}, "
                f"persistent_workers={tc.persistent_workers}, "
                f"pin_memory={tc.pin_memory}, "
                f"non_blocking={tc.non_blocking}"
            )
        self.log(f"Effective batch size: {tc.batch_size * tc.grad_accum_steps}")
        if self.arch in ("mc_mamba", "mc_linear_attention"):
            mc = self.config.model.mc
            self.log(f"MC Config: segment_size={mc.segment_size}, "
                     f"retrieval_scale={mc.retrieval_scale}, "
                     f"max_cache_entries={mc.max_cache_entries}")
        if self.arch in ("linear_attention", "mc_linear_attention"):
            la = self.config.model.la
            self.log(f"LA Config: n_heads={la.n_heads}, "
                     f"use_deltanet={la.use_deltanet}, "
                     f"feature_map={la.feature_map}")
        if self.arch == "olmo_hybrid":
            olmo = self.config.model.olmo
            self.log(
                f"OLMo Config: n_kv_heads={self.config.model.resolved_n_kv_heads}, "
                f"rope_theta={olmo.rope_theta}, "
                f"qk_norm={olmo.qk_norm}, "
                f"attention_period={olmo.attention_period}"
            )
        if self.ema is not None:
            self.log(f"EMA decay: {tc.ema_decay}")
        self.log("-" * 60)

        # Build dataloaders
        train_loader = self.build_dataloader("train")
        val_loader = self.build_dataloader("val")

        # Training state
        step = self.start_step
        best_val_loss = self.best_val_loss
        train_iter = iter(train_loader)

        self.runtime_model.train()
        t0 = time.time()
        transfer_non_blocking = bool(tc.non_blocking and self.device.type == "cuda")

        while step < tc.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(tc.grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                codes = batch["codes"].to(self.device, non_blocking=transfer_non_blocking)
                mask = batch["mask"].to(self.device, non_blocking=transfer_non_blocking)

                with self.ctx:
                    loss = self._compute_loss(codes, mask)
                    loss = loss / tc.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Gradient clipping
            if tc.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.max_grad_norm
                )
            else:
                grad_norm = 0.0

            # Update LR
            lr = get_lr(step, tc)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)

            step += 1

            # Logging
            if step % tc.log_every == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    tc.batch_size * tc.grad_accum_steps
                    * self.config.model.n_codebooks
                    * self.config.model.max_seq_len
                    * tc.log_every / dt
                )
                metrics = {
                    "step": step,
                    "train_loss": accum_loss,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": time.time() - t0,
                }

                # MC-specific logging
                if hasattr(self.model, 'get_mc_stats'):
                    mc_stats = self.model.get_mc_stats()
                    metrics.update(mc_stats)

                self.metrics_log.append(metrics)

                log_msg = (
                    f"step {step:6d} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | grad_norm {metrics['grad_norm']:.2f} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )

                # Add MC stats to log line
                if "avg_grm_entropy" in metrics:
                    log_msg += f" | grm_ent {metrics['avg_grm_entropy']:.3f}"
                if "avg_cache_entries" in metrics:
                    log_msg += f" | cache {metrics['avg_cache_entries']:.0f}"

                self.log(log_msg)
                t0 = time.time()

            # Evaluation
            if step % tc.eval_every == 0:
                eval_metrics = self.evaluate(val_loader)
                ema_tag = " (EMA)" if self.ema is not None else ""
                self.log(
                    f"  EVAL step {step}{ema_tag}: val_loss={eval_metrics['val_loss']:.4f} "
                    f"ppl={eval_metrics['val_perplexity']:.2f}"
                )

                if eval_metrics["val_loss"] < best_val_loss:
                    best_val_loss = eval_metrics["val_loss"]
                    self.best_val_loss = best_val_loss
                    self.save_checkpoint(step, is_best=True)

            # Save checkpoint
            if step % tc.save_every == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(step, is_best=False)
        self.log(f"Training complete. Best val loss: {best_val_loss:.4f}")

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_log, f, indent=2)

        self.log_file.close()

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config.model),
            "best_val_loss": self.best_val_loss,
        }

        # Save EMA state if available
        if self.ema is not None:
            ckpt["ema_shadow"] = self.ema.shadow

        path = self.output_dir / f"checkpoint_{step}.pt"
        safe_torch_save(ckpt, path)

        if is_best:
            best_path = self.output_dir / "best.pt"
            safe_torch_save(ckpt, best_path)
            self.log(f"  Saved best checkpoint at step {step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MC-Mamba / baselines over RVQ tokens")
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default=None,
        help="Optional experiment preset (dataset + training hyperparameters)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["transformer", "mamba1", "mamba2", "hybrid_1_7", "hybrid_1_3", "mc_mamba",
                 "linear_attention", "mc_linear_attention", "olmo_hybrid"],
        help="Model architecture to train",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from an existing checkpoint path. Reuses that checkpoint's run directory.",
    )
    parser.add_argument(
        "--codec_meta",
        type=str,
        default=None,
        help="Path to codec metadata JSON (default: <data_dir>/codec_meta.json)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cuda, mps, or cpu")
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile_mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--non_blocking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused_optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_flash_sdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_mem_efficient_sdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_math_sdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable_cudnn_sdp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--float32_matmul_precision", type=str, default=None,
                        choices=["highest", "high", "medium"])

    # MC-specific arguments
    parser.add_argument("--segment_size", type=int, default=256,
                        help="MC segment size S (cache boundary every S timesteps)")
    parser.add_argument("--retrieval_scale", type=float, default=1.0,
                        help="Scale factor for GRM retrieval residual")
    parser.add_argument("--max_cache_entries", type=int, default=64,
                        help="Maximum number of cached segments per layer")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--codebook_weights", type=str, default=None,
                        help="Comma-separated codebook loss weights (e.g. '2,1.5,1,1,1,0.8,0.8,0.5,0.5')")

    # Linear attention arguments
    parser.add_argument("--use_deltanet", action="store_true",
                        help="Use DeltaNet variant for linear attention (error-correction update)")
    parser.add_argument("--la_n_heads", type=int, default=None,
                        help="Number of heads for linear attention (default: same as n_heads)")
    parser.add_argument("--n_kv_heads", type=int, default=None,
                        help="Grouped-query KV head count for OLMo Hybrid attention")
    parser.add_argument("--rope_theta", type=int, default=None,
                        help="RoPE theta/base for OLMo Hybrid attention")

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig()

    # Apply preset first, then let explicit CLI args override.
    if args.preset:
        apply_preset(config, args.preset)

    # When resuming, prefer the original run config before applying CLI overrides.
    maybe_apply_resume_config(config, args.resume_from)

    config.model.arch = args.arch
    config.train.seed = args.seed
    config.train.device = args.device

    if args.data_dir:
        config.train.data_dir = args.data_dir
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.run_name:
        config.train.run_name = args.run_name

    # Align codec/model settings from preprocessed dataset metadata when available.
    codec_meta_path = Path(args.codec_meta) if args.codec_meta else Path(config.train.data_dir) / "codec_meta.json"
    maybe_apply_codec_metadata(config, codec_meta_path)

    # Override optional args
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.grad_accum_steps:
        config.train.grad_accum_steps = args.grad_accum_steps
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.eval_every:
        config.train.eval_every = args.eval_every
    if args.save_every:
        config.train.save_every = args.save_every
    if args.log_every:
        config.train.log_every = args.log_every
    if args.lr:
        config.train.lr = args.lr
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        config.train.prefetch_factor = args.prefetch_factor
    if args.d_model:
        config.model.d_model = args.d_model
    if args.n_layers:
        config.model.n_layers = args.n_layers
    if args.n_heads:
        config.model.n_heads = args.n_heads
    if args.d_ff:
        config.model.d_ff = args.d_ff

    # MC config
    config.model.mc.segment_size = args.segment_size
    config.model.mc.retrieval_scale = args.retrieval_scale
    config.model.mc.max_cache_entries = args.max_cache_entries

    # Linear attention config
    if args.use_deltanet:
        config.model.la.use_deltanet = True
    if args.la_n_heads:
        config.model.la.n_heads = args.la_n_heads
    if args.n_kv_heads:
        config.model.olmo.n_kv_heads = args.n_kv_heads
    if args.rope_theta:
        config.model.olmo.rope_theta = args.rope_theta
    if args.compile_model is not None:
        config.train.compile_model = args.compile_model
    if args.compile_mode is not None:
        config.train.compile_mode = args.compile_mode
    if args.persistent_workers is not None:
        config.train.persistent_workers = args.persistent_workers
    if args.pin_memory is not None:
        config.train.pin_memory = args.pin_memory
    if args.non_blocking is not None:
        config.train.non_blocking = args.non_blocking
    if args.allow_tf32 is not None:
        config.train.allow_tf32 = args.allow_tf32
    if args.cudnn_benchmark is not None:
        config.train.cudnn_benchmark = args.cudnn_benchmark
    if args.fused_optimizer is not None:
        config.train.fused_optimizer = args.fused_optimizer
    if args.enable_flash_sdp is not None:
        config.train.enable_flash_sdp = args.enable_flash_sdp
    if args.enable_mem_efficient_sdp is not None:
        config.train.enable_mem_efficient_sdp = args.enable_mem_efficient_sdp
    if args.enable_math_sdp is not None:
        config.train.enable_math_sdp = args.enable_math_sdp
    if args.enable_cudnn_sdp is not None:
        config.train.enable_cudnn_sdp = args.enable_cudnn_sdp
    if args.float32_matmul_precision is not None:
        config.train.float32_matmul_precision = args.float32_matmul_precision

    if args.arch == "olmo_hybrid" and not args.d_ff:
        config.model.d_ff = compute_olmo_ffn_dim(
            config.model.d_model,
            config.model.olmo.ffn_multiple_of,
        )
    elif args.d_model and not args.d_ff:
        config.model.d_ff = args.d_model * 4

    # EMA
    if args.no_ema:
        config.train.use_ema = False

    # Codebook weights
    if args.codebook_weights:
        weights = tuple(float(w) for w in args.codebook_weights.split(","))
        assert len(weights) == config.model.n_codebooks, \
            f"Expected {config.model.n_codebooks} weights, got {len(weights)}"
        config.train.codebook_loss_weights = weights

    trainer = Trainer(config, args.arch, resume_from=args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
