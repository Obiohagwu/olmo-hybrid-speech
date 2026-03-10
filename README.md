# OLMo-Hybrid Speech

A small-scale speech codec language modeling with an OLMo-Hybrid / Gated
DeltaNet-style decoder.

If you have the compute, try to train it to the full 12k steps and lmk the results haha...

This repo contains the code, sampling pipeline, and report source for a
21.9M-parameter pilot trained on LJ Speech tokenized with EnCodec 24 kHz. The
goal was primarily to test whether a small hybrid recurrent-attention backbone can
learn enough speech-token structure to generate clearly speech-like audio
without text conditioning.

The best preserved A100 checkpoint in the pilot reached:

- `EMA val_loss = 4.2847`
- `perplexity = 72.58`
- `step = 1800`

To be clear though, this is a more so a pilot result than an actual benchmark claim.

## Project page

- Samples and run summary:
  - https://research.nani-inc.com/2026/03/09/olmo-hybrid-ljspeech-a100-pilot.html

## What is in this repo

- `src/`: model, data, and eval code
- `scripts/`: tokenization, sampling, and A100 training helpers
- `train.py`: main training entry point
- `config.py`: presets and config plumbing
- `paper/`: LaTeX and markdown report sources

This public repo intentionally excludes (for now):

- raw audio datasets
- tokenized data
- training runs and checkpoints
- local scratch/sample directories 

## Model summary

The stable pilot model uses:

- `21,904,648` parameters
- `d_model = 384`
- `n_layers = 8`
- `d_ff = 1024`
- `n_heads = 6`
- `n_kv_heads = 2`
- `8` EnCodec codebooks
- `max_seq_len = 1024`
- a `3:1` recurrent-to-attention schedule
- `6` Gated DeltaNet blocks and `2` attention blocks

Training used delay-patterned EnCodec 24 kHz tokens from 8-second LJ Speech
chunks.

## Results snapshot

On an A100-SXM4-80GB, the best preserved run improved validation cleanly through
step `1800`. The stable run used CUDA flash attention and fused AdamW, but not
the intended fused Flash Linear Attention recurrent kernel, due Triton kernel
failures on the available pod stack. So the reported systems performance should
be read as a conservative no-FLA result.

## Install

Typical CUDA setup:

```bash
python3 -m pip install -U pip wheel setuptools ninja packaging soundfile
```

If you want to try the fused recurrent path:

```bash
python3 -m pip install --no-build-isolation flash-linear-attention
```

## Train

Main entry point:

```bash
python3 train.py \
  --arch olmo_hybrid \
  --preset speech_olmo_hybrid_a100_20m_ctx1024 \
  --device cuda \
  --run_name ljspeech_olmo_hybrid_a100
```

Helper launcher:

```bash
scripts/train_ljspeech_a100.sh --run_name ljspeech_olmo_hybrid_a100
```

## Sample

The sampler rolls out in delayed-token space to match the training objective.

```bash
python3 scripts/sample_speech.py \
  --run_dir runs/speech/<run_name> \
  --checkpoint best \
  --device cuda \
  --duration_sec 6 \
  --num_samples 4
```

## Report

The technical report sources are in:

- `paper/main.tex`
- `paper/references.bib`

## Status

Currently:

- unconditional speech codec LM training
- OLMo-Hybrid / Gated DeltaNet backbone
- CUDA-first A100 training path
- local delayed-space speech sampling

Not yet claiming these but upon more experimentation, goals are:

- superiority over a matched transformer baseline
- text-conditioned TTS
- speaker cloning
- full fused recurrent-kernel training stability
