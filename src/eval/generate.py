"""
Autoregressive generation from trained models.

Handles:
  - Temperature sampling with top-k/top-p
  - Multi-codebook generation with delay pattern
  - Decoding back to audio via DAC
  - MC-Mamba cache reset at generation start
"""

import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np


@torch.no_grad()
def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    Sample from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: (..., vocab_size)
        temperature: sampling temperature
        top_k: keep top-k tokens (0 = disabled)
        top_p: nucleus sampling threshold (0 = disabled)

    Returns:
        sampled token indices: (...)
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])


@torch.no_grad()
def generate(
    model,
    n_codebooks: int = 9,
    max_steps: int = 860,       # ~10 seconds at 86Hz
    temperature: float = 0.9,
    top_k: int = 250,
    top_p: float = 0.0,
    bos_token: int = 1025,
    pad_token: int = 1024,
    device: str = "cuda",
    prompt_codes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Autoregressive generation of RVQ tokens.

    For MC-Mamba models, resets all segment caches before generation
    so the cache builds up fresh for the generated sequence.

    Args:
        model: trained model
        n_codebooks: number of RVQ codebooks
        max_steps: number of steps to generate (in delay-pattern space)
        temperature: sampling temperature
        top_k: top-k filtering
        top_p: nucleus sampling
        bos_token: beginning-of-sequence token
        pad_token: padding token
        device: device
        prompt_codes: optional prompt for continuation (already delay-patterned)

    Returns:
        codes: (K, T) generated codebook indices (delay pattern already applied)
    """
    model.eval()

    # Reset MC caches if model supports it
    if hasattr(model, 'reset_all_caches'):
        model.reset_all_caches()

    if prompt_codes is not None:
        generated = prompt_codes.unsqueeze(0).to(device)
    else:
        start = torch.full((1, n_codebooks, 1), pad_token, dtype=torch.long, device=device)
        start[0, 0, 0] = bos_token
        generated = start

    for step in range(max_steps):
        # Reset caches before each full forward pass since we pass the
        # entire generated sequence each time (no KV cache yet)
        if hasattr(model, 'reset_all_caches'):
            model.reset_all_caches()

        logits = model(generated)  # (1, K, T_current, V)
        next_logits = logits[0, :, -1, :]  # (K, V)

        next_tokens = []
        for k in range(n_codebooks):
            current_t = generated.shape[2]
            if current_t <= k:
                next_tokens.append(pad_token)
            else:
                token = sample_top_k_top_p(
                    next_logits[k].unsqueeze(0),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ).item()
                next_tokens.append(token)

        next_col = torch.tensor(next_tokens, dtype=torch.long, device=device).view(1, n_codebooks, 1)
        generated = torch.cat([generated, next_col], dim=-1)

    return generated.squeeze(0)


@torch.no_grad()
def generate_audio(
    model,
    tokenizer,
    duration_sec: float = 10.0,
    temperature: float = 0.9,
    top_k: int = 250,
    top_p: float = 0.0,
    device: str = "cuda",
    prompt_codes: Optional[torch.Tensor] = None,
    n_codebooks: int = 9,
    frame_rate: int = 86,
) -> np.ndarray:
    """
    Generate audio waveform end-to-end.

    Args:
        model: trained model
        tokenizer: DACTokenizer instance
        duration_sec: target duration in seconds
        temperature: sampling temperature
        top_k: top-k filtering
        device: device
        n_codebooks: number of RVQ codebooks (9 for DAC, 8 for EnCodec)
        frame_rate: codec frame rate in Hz (86 for DAC, 75 for EnCodec)

    Returns:
        audio: numpy array at codec sample rate
    """
    from src.data.tokenizer import DelayPattern

    max_steps = int(duration_sec * frame_rate) + n_codebooks - 1

    codes_delayed = generate(
        model=model,
        n_codebooks=n_codebooks,
        max_steps=max_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        prompt_codes=prompt_codes,
    )

    delay = DelayPattern(n_codebooks, pad_token=1024)
    codes = delay.revert(codes_delayed)

    audio = tokenizer.decode(codes)
    return audio
