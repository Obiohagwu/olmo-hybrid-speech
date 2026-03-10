"""
Evaluation metrics for the SSM vs Transformer experiment.

Metrics from research_paper_compendium.md §8.3:
  - FAD (Fréchet Audio Distance)
  - KL divergence
  - Long-range coherence (MI between distant frames)
  - Training efficiency (loss per FLOP)
  - Per-codebook loss breakdown
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path


def per_codebook_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token: int = 1024,
) -> Dict[str, float]:
    """
    Compute cross-entropy loss broken down by codebook.

    This reveals whether certain architectures handle
    specific codebook levels better (e.g., coarse vs fine).

    Args:
        logits: (B, K, T, V)
        targets: (B, K, T)

    Returns:
        dict mapping codebook index to loss value
    """
    B, K, T, V = logits.shape
    losses = {}

    for k in range(K):
        k_logits = logits[:, k, :-1, :].contiguous().view(-1, V)
        k_targets = targets[:, k, 1:].contiguous().view(-1)
        loss = F.cross_entropy(k_logits, k_targets, ignore_index=pad_token)
        losses[f"codebook_{k}"] = loss.item()

    losses["mean"] = np.mean(list(losses.values()))
    return losses


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token: int = 1024,
) -> Dict[str, float]:
    """
    Compute top-1 and top-5 accuracy per codebook.
    """
    B, K, T, V = logits.shape
    results = {}

    for k in range(K):
        k_logits = logits[:, k, :-1, :].contiguous()  # (B, T-1, V)
        k_targets = targets[:, k, 1:].contiguous()     # (B, T-1)

        # Mask out padding
        valid = k_targets != pad_token
        if valid.sum() == 0:
            continue

        k_logits = k_logits[valid]
        k_targets = k_targets[valid]

        # Top-1
        preds = k_logits.argmax(dim=-1)
        top1 = (preds == k_targets).float().mean().item()

        # Top-5
        _, top5_preds = k_logits.topk(5, dim=-1)
        top5 = (top5_preds == k_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

        results[f"codebook_{k}_top1"] = top1
        results[f"codebook_{k}_top5"] = top5

    return results


@torch.no_grad()
def compute_long_range_coherence(
    codes: torch.Tensor,
    window_frames: int = 344,     # ~4 sec at 86Hz
    hop_frames: int = 172,        # ~2 sec at 86Hz
) -> float:
    """
    Estimate long-range coherence by computing cosine similarity
    between codebook histograms at different time windows.

    Higher coherence = model maintains consistent musical character.

    This is a proxy for musical structure preservation.
    True evaluation would use human listeners or genre classifiers.

    Args:
        codes: (K, T) generated codebook indices
        window_frames: window size in frames
        hop_frames: hop size in frames

    Returns:
        mean coherence score (0 to 1)
    """
    K, T = codes.shape

    if T < 2 * window_frames:
        return 0.0

    # Compute per-window codebook histograms
    histograms = []
    for start in range(0, T - window_frames + 1, hop_frames):
        window = codes[:, start : start + window_frames]  # (K, W)
        # Histogram: count frequency of each code per codebook
        hist = torch.zeros(K * 1024, dtype=torch.float32)
        for k in range(K):
            for code in window[k]:
                if code < 1024:  # skip special tokens
                    hist[k * 1024 + code] += 1
        hist = hist / hist.sum().clamp(min=1)
        histograms.append(hist)

    if len(histograms) < 2:
        return 0.0

    # Compute pairwise cosine similarity between distant windows
    similarities = []
    for i in range(len(histograms)):
        for j in range(i + 2, len(histograms)):  # skip adjacent
            sim = F.cosine_similarity(
                histograms[i].unsqueeze(0),
                histograms[j].unsqueeze(0),
            ).item()
            similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def compute_fad(
    generated_dir: str,
    reference_dir: str,
    model_name: str = "vggish",
) -> float:
    """
    Compute Fréchet Audio Distance between generated and reference audio.

    Requires: pip install fadtk

    Args:
        generated_dir: path to directory of generated .wav files
        reference_dir: path to directory of reference .wav files
        model_name: embedding model ("vggish", "clap", "encodec")

    Returns:
        FAD score (lower is better)
    """
    try:
        from fadtk import FrechetAudioDistance
        fad = FrechetAudioDistance(
            model_name=model_name,
            sample_rate=44100,
        )
        score = fad.score(reference_dir, generated_dir)
        return score
    except ImportError:
        print("FAD computation requires `pip install fadtk`")
        return float("nan")


class ExperimentTracker:
    """
    Track and compare metrics across all 5 architectures.
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def add_result(self, arch: str, metrics: Dict):
        """Add metrics for an architecture."""
        self.results[arch] = metrics

    def summary_table(self) -> str:
        """Generate a comparison table across architectures."""
        if not self.results:
            return "No results yet."

        architectures = list(self.results.keys())
        all_keys = set()
        for m in self.results.values():
            all_keys.update(m.keys())

        lines = []
        header = f"{'Metric':<30}" + "".join(f"{a:<15}" for a in architectures)
        lines.append(header)
        lines.append("-" * len(header))

        for key in sorted(all_keys):
            row = f"{key:<30}"
            for arch in architectures:
                val = self.results[arch].get(key, "N/A")
                if isinstance(val, float):
                    row += f"{val:<15.4f}"
                else:
                    row += f"{str(val):<15}"
            lines.append(row)

        return "\n".join(lines)

    def save(self):
        """Save results to disk."""
        import json
        with open(self.output_dir / "comparison.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        with open(self.output_dir / "comparison.txt", "w") as f:
            f.write(self.summary_table())
