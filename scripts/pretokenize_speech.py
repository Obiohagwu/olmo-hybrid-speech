"""
Batched speech tokenization for local EnCodec workflows.

This script is optimized for small local machines:
  - converts audio to mono 24 kHz
  - chunks into fixed-duration windows
  - batches equal-length chunks for EnCodec
  - skips outputs that already exist
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import Encodec24kTokenizer


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def is_mps_available() -> bool:
    return bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        if is_mps_available():
            return "mps"
        return "cpu"
    if requested not in {"cpu", "mps"}:
        raise ValueError("Use one of: auto, mps, cpu")
    if requested == "mps" and not is_mps_available():
        raise RuntimeError("Requested MPS but it is unavailable in this PyTorch runtime.")
    return requested


def chunk_waveform(
    wav: torch.Tensor,
    chunk_samples: int,
    hop_samples: int,
    keep_tail: bool,
    min_tail_samples: int,
) -> list[tuple[int, int, torch.Tensor]]:
    total_samples = int(wav.shape[-1])
    chunks: list[tuple[int, int, torch.Tensor]] = []
    chunk_idx = 0
    start = 0

    while start + chunk_samples <= total_samples:
        chunks.append((chunk_idx, start, wav[:, start : start + chunk_samples]))
        chunk_idx += 1
        start += hop_samples

    tail = total_samples - start
    if keep_tail and tail >= min_tail_samples:
        padded = F.pad(wav[:, start:], (0, chunk_samples - tail))
        chunks.append((chunk_idx, start, padded))

    if total_samples < chunk_samples and not chunks:
        padded = F.pad(wav, (0, chunk_samples - total_samples))
        chunks.append((0, 0, padded))

    return chunks


def flush_batch(
    tokenizer: Encodec24kTokenizer,
    batch_wavs: list[torch.Tensor],
    batch_targets: list[Path],
) -> None:
    codes_batch = tokenizer.encode_wavs_batch(batch_wavs)
    for codes, out_path in zip(codes_batch, batch_targets):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(codes.to(torch.uint16), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretokenize speech audio with batched EnCodec.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", help="auto, mps, or cpu")
    parser.add_argument("--chunk_sec", type=float, default=8.0)
    parser.add_argument("--hop_sec", type=float, default=None)
    parser.add_argument("--keep_tail", action="store_true")
    parser.add_argument("--min_tail_sec", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--encodec_bandwidth", type=float, default=6.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    root_output_dir = Path(args.output_dir)
    split_output_dir = root_output_dir / args.split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer = Encodec24kTokenizer(device=device, bandwidth=args.encodec_bandwidth)

    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS])
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Found {len(files)} audio files in {input_dir}")
    print(f"Using device={device}, batch_size={args.batch_size}, chunk_sec={args.chunk_sec}")

    if not files:
        return

    chunk_samples = int(round(args.chunk_sec * tokenizer.sample_rate))
    hop_sec = args.hop_sec if args.hop_sec is not None else args.chunk_sec
    hop_samples = int(round(hop_sec * tokenizer.sample_rate))
    min_tail_samples = int(round(args.min_tail_sec * tokenizer.sample_rate))

    codec_meta = {
        "codec": "encodec_24khz",
        "sample_rate": int(tokenizer.sample_rate),
        "n_codebooks": int(tokenizer.n_codebooks),
        "codebook_size": int(tokenizer.codebook_size),
        "pad_token": int(tokenizer.codebook_size),
        "bos_token": int(tokenizer.codebook_size) + 1,
        "eos_token": int(tokenizer.codebook_size) + 2,
        "vocab_size": int(tokenizer.codebook_size) + 3,
        "encodec_bandwidth": float(args.encodec_bandwidth),
        "chunk_sec": float(args.chunk_sec),
        "hop_sec": float(hop_sec),
    }
    with open(root_output_dir / "codec_meta.json", "w") as f:
        json.dump(codec_meta, f, indent=2)

    batch_wavs: list[torch.Tensor] = []
    batch_targets: list[Path] = []
    written = 0

    for audio_file in tqdm(files, desc=f"Tokenizing {args.split}"):
        wav = tokenizer.load_audio(str(audio_file))
        chunks = chunk_waveform(
            wav=wav,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            keep_tail=args.keep_tail,
            min_tail_samples=min_tail_samples,
        )

        rel = audio_file.relative_to(input_dir)
        stem = rel.with_suffix("")
        for chunk_idx, start_sample, chunk in chunks:
            out_path = split_output_dir / stem.parent / f"{stem.name}__chunk{chunk_idx:04d}.pt"
            if out_path.exists():
                continue

            batch_wavs.append(chunk)
            batch_targets.append(out_path)

            if len(batch_wavs) >= args.batch_size:
                flush_batch(tokenizer, batch_wavs, batch_targets)
                written += len(batch_targets)
                batch_wavs = []
                batch_targets = []

    if batch_wavs:
        flush_batch(tokenizer, batch_wavs, batch_targets)
        written += len(batch_targets)

    print(f"Wrote {written} token files to {split_output_dir}")


if __name__ == "__main__":
    main()
