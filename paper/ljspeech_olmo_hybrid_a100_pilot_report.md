# A Small-Scale OLMo-Hybrid Speech Codec LM Pilot on LJ Speech

## Abstract

This report documents a small-scale speech codec language modeling experiment using an OLMo-hybrid style recurrent-attention decoder trained on `LJ Speech` tokenized with `EnCodec 24 kHz`. The central question was whether a compact `~21.9M` parameter hybrid model could learn enough local speech structure to produce voice-like audio rather than static or degenerate decoder artifacts. On an `A100-SXM4-80GB`, the model reached an EMA validation loss of `4.2847` and perplexity `72.58` by step `1800`, and qualitative samples contained a clear speaking voice despite substantial babble and weak semantic coherence. The run did not use the intended fused Flash Linear Attention Gated DeltaNet kernel due upstream Triton/kernel instability, so the reported system performance is conservative relative to the architecture's intended fast path.

## 1. Motivation

The goal of this pilot was not to build a production TTS model. The goal was to answer a narrower architectural question:

> Can a small OLMo-hybrid style recurrent-attention language model learn speech codec structure well enough to emit non-noisy, clearly voice-like samples?

This is a useful threshold because it separates:

- pipeline failures
- tokenizer failures
- architecture failures

from a model that is genuinely learning speech dynamics, even if it has not yet learned semantics or strong long-range structure.

## 2. Model

The model was a compact speech adaptation of the OLMo-hybrid pattern.

### 2.1 Architecture

| Item | Value |
|---|---|
| Total parameters | `21,904,648` |
| Decoder depth | `8` layers |
| Width | `d_model=384` |
| FFN size | `d_ff=1024` |
| Attention heads | `6` |
| KV heads | `2` |
| Hybrid schedule | attention every `4th` block |
| Block mix | `6` Gated DeltaNet + `2` attention |
| Context | `1024` delayed steps |
| Codebooks | `8` |
| Vocab | `1027` |

The recurrent blocks used a paper-aligned Gated DeltaNet-style structure, but the A100 training run described here did **not** use the fused FLA kernel. The recurrent scan executed in the plain PyTorch fallback path.

### 2.2 Data Representation

Audio was tokenized into `8` EnCodec residual codebooks and trained with a delay-pattern objective. The model predicted next delayed tokens autoregressively rather than generating waveform samples directly.

## 3. Dataset and Tokenization

### 3.1 Corpus

`LJ Speech` was used because:

- it is clean
- single-speaker
- sentence segmented
- easy to tokenize cheaply
- well matched to an early proof-of-learning objective

### 3.2 Tokenization

| Item | Value |
|---|---|
| Codec | EnCodec |
| Sample rate | `24 kHz` |
| Codebooks | `8` |
| Codebook size | `1024` |
| Chunk size | `8.0s` |
| Split | `12,624` train / `666` val |

Token files, checkpoints, and sample bundles were preserved in a local artifact
bundle outside the public repository.

## 4. Training Setup

### 4.1 Hardware

| Item | Value |
|---|---|
| GPU | `NVIDIA A100-SXM4-80GB` |
| Precision | `bfloat16` |
| Attention backend | CUDA SDPA flash path enabled |
| Optimizer | fused AdamW |
| FLA recurrent kernel | disabled during stable run |

### 4.2 Stable Run Configuration

The stable A100 run used:

| Item | Value |
|---|---|
| True batch size | `24` |
| Grad accumulation | `1` |
| Effective batch | `24` |
| Max steps | `12000` planned |
| Warmup | `500` |
| Peak LR | `3e-4` |
| Eval cadence | every `200` steps |
| Save cadence | every `200` steps |
| Throughput | about `17.9k tok/s` |

An earlier `batch_size=32` attempt OOMed. Smaller batch/accum combinations were also tried. The `24x1` configuration was the best stable operating point found under the time budget.

## 5. Systems Notes

### 5.1 Fused Kernel Failure

The intended fast path for the recurrent mixer was the fused Flash Linear Attention Gated DeltaNet kernel. In practice, the pod stack failed in the fused backward kernel with Triton/codegen errors. As a result, the stable run used:

- fused CUDA SDPA for attention
- fused AdamW
- bf16
- pinned-memory loaders
- non-blocking transfers

but **not** the fused recurrent GDN kernel.

This matters because the model was still materially underutilizing the A100 relative to what a fully fused recurrent path should achieve.

### 5.2 Pod Volatility

The run environment was operationally fragile:

- pod restarts changed SSH endpoints
- pod-local storage could not be trusted
- one pod likely died during or near the `2000` save boundary

Because of that, important checkpoints were copied off-pod incrementally and preserved locally.

## 6. Results

### 6.1 Validation Trajectory

| Step | EMA Val Loss | PPL |
|---|---:|---:|
| 200 | 6.7878 | 886.99 |
| 400 | 6.1530 | 470.13 |
| 600 | 5.6474 | 283.54 |
| 800 | 5.1973 | 180.78 |
| 1000 | 4.8510 | 127.87 |
| 1200 | 4.6169 | 101.18 |
| 1400 | 4.4626 | 86.72 |
| 1600 | 4.3585 | 78.14 |
| 1800 | 4.2847 | 72.58 |

The best saved checkpoint available locally is `step 1800`.

### 6.2 Training Behavior

The training run was notably cleaner than the earlier M4 pilot:

- no NaNs
- no late validation collapse through `1800`
- low and stable gradient norms after warmup
- steady throughput around `17.9k tok/s`

### 6.3 Sample Quality

Qualitatively, the model crossed the key threshold:

- outputs were voice-like rather than static
- there was an obvious speaking timbre
- failures were babble and weak articulation rather than total collapse

That is the main success criterion for this pilot.

At the same time, the outputs remained limited:

- semantics were weak or absent
- pacing could still sound off
- sample variance was high
- some later checkpoints produced better individual samples even before they were clearly dominant on validation

## 7. Interpretation

The main conclusion is modest but real:

> A `~21.9M` OLMo-hybrid style recurrent-attention codec LM can learn speech structure on a clean single-speaker corpus well before full convergence, producing clear voice-like audio by step `1800`.

This is important because the model achieved this despite three major handicaps:

1. no text conditioning
2. no fused recurrent GDN kernel
3. only a partial fraction of the planned `12k` training budget

In other words, the experiment succeeded on the core architectural question even though the systems path was compromised.

## 8. Limitations

This report should not be overclaimed.

Main limitations:

- no matched transformer baseline in this report
- single-speaker corpus only
- no human evaluation protocol beyond direct listening
- no formal intelligibility metric
- no semantic content metric
- no full `12k` completion
- no final `2000+` checkpoint preserved from the pod

So the right framing is not "state-of-the-art speech generation." The right framing is "small hybrid codec LM pilot that clearly learned speech-like structure."

## 9. Artifact Inventory

Saved local checkpoints:

- `checkpoint_1200.pt`
- `checkpoint_1400.pt`
- `checkpoint_1600.pt`
- `checkpoint_1800.pt`
- `best.pt`

Saved local sample bundles:

- `checkpoint_1200_local_cpu_speech5`
- `checkpoint_1400_local_cpu_speech3`
- `checkpoint_1600_local_cpu_speech3`
- `checkpoint_1800_local_cpu_speech3`

## 10. Next Steps

The highest-value next steps are:

1. add a matched transformer baseline
2. move to text-conditioned speech generation
3. revisit fused GDN kernels on a more stable CUDA/Triton stack
4. run a more systematic perceptual comparison across checkpoints

If the present result is the end of the budget, that is still acceptable. The pilot already answered the main viability question.
