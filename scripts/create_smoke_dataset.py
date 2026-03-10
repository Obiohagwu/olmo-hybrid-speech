"""
Create a tiny deterministic token dataset for end-to-end training smoke tests.

The generated samples are structured enough that a small model can overfit and
show a real loss drop, which is useful for local M4 verification.
"""

import argparse
import json
from pathlib import Path

import torch


def make_codes(
    sample_idx: int,
    length: int,
    n_codebooks: int,
    codebook_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    period = int(torch.randint(16, 33, (1,), generator=generator).item())
    motif = torch.randint(
        0,
        min(codebook_size, 128),
        (n_codebooks, period),
        generator=generator,
        dtype=torch.long,
    )

    codes = torch.empty((n_codebooks, length), dtype=torch.long)
    sample_shift = (sample_idx * 17) % codebook_size

    for t in range(length):
        motif_idx = t % period
        phrase_idx = t // period
        phrase_shift = (phrase_idx * 5) % codebook_size
        for k in range(n_codebooks):
            coarse = motif[k, motif_idx]
            tied_offset = (k * 23 + (t // max(1, period // 2))) % codebook_size
            if k == 0:
                value = coarse + sample_shift + phrase_shift
            else:
                value = motif[0, motif_idx] + coarse + tied_offset + sample_shift
            codes[k, t] = value % codebook_size

    return codes


def write_split(
    split_dir: Path,
    split_name: str,
    num_samples: int,
    min_length: int,
    max_length: int,
    n_codebooks: int,
    codebook_size: int,
    generator: torch.Generator,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        length = int(
            torch.randint(min_length, max_length + 1, (1,), generator=generator).item()
        )
        codes = make_codes(idx, length, n_codebooks, codebook_size, generator)
        torch.save(codes, split_dir / f"{split_name}_{idx:04d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small token dataset for smoke tests.")
    parser.add_argument("--output_dir", type=str, default="./data/smoke_tokens")
    parser.add_argument("--train_samples", type=int, default=64)
    parser.add_argument("--val_samples", type=int, default=16)
    parser.add_argument("--n_codebooks", type=int, default=9)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--min_length", type=int, default=96)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generator = torch.Generator().manual_seed(args.seed)

    write_split(
        output_dir / "train",
        "train",
        args.train_samples,
        args.min_length,
        args.max_length,
        args.n_codebooks,
        args.codebook_size,
        generator,
    )
    write_split(
        output_dir / "val",
        "val",
        args.val_samples,
        args.min_length,
        args.max_length,
        args.n_codebooks,
        args.codebook_size,
        generator,
    )

    codec_meta = {
        "sample_rate": args.sample_rate,
        "n_codebooks": args.n_codebooks,
        "codebook_size": args.codebook_size,
        "pad_token": args.codebook_size,
        "bos_token": args.codebook_size + 1,
        "eos_token": args.codebook_size + 2,
        "vocab_size": args.codebook_size + 3,
    }
    with open(output_dir / "codec_meta.json", "w") as f:
        json.dump(codec_meta, f, indent=2)

    print(f"Wrote smoke dataset to {output_dir}")
    print(f"Train samples: {args.train_samples}")
    print(f"Val samples: {args.val_samples}")


if __name__ == "__main__":
    main()

