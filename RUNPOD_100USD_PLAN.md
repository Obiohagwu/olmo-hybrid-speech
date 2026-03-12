# RunPod $100 Plan

This plan assumes roughly `$100` of A100-80GB time and prioritizes extracting
the most publishable signal from the existing project.

## Priority order

1. Resume the stable hybrid run from `checkpoint_1800.pt` and push it toward
   `12000` steps, stopping early if validation clearly peaks.
2. If budget remains after the hybrid run, start a short matched transformer
   baseline (`2k-4k` steps).
3. Use any leftover budget for extra sampling from the best hybrid and baseline
   checkpoints.

Do **not** spend the first tranche of time on:

- MLX
- fixing FLA
- a new dataset
- text conditioning
- voice cloning

Those are follow-ups. The highest-value use of the next GPU spend is finishing
the hybrid story and getting the baseline started.

## Why this plan

The A100 no-FLA hybrid run improved monotonically through the best preserved
checkpoint at step `1800`:

| Step | EMA val loss | PPL |
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

This is already your strongest result. The next logical step is to continue the
same run until it either:

- peaks and starts degrading on validation, or
- reaches the planned `12000` budget.

## Non-negotiable infrastructure rule

Never trust pod-local disk again.

Use one of:

- a mounted persistent / network volume, or
- direct checkpoint sync off-pod every `200` steps

The safest path is to place the resumed run **directly on the mounted volume**
and resume from a checkpoint copied there in advance.

## Target operating point

Use the same stable no-FLA operating point that produced the best run so far:

- architecture: `olmo_hybrid`
- preset: `speech_olmo_hybrid_a100_20m_ctx1024`
- true batch size: `24`
- grad accumulation: `1`
- device: `cuda`
- mixed precision: `bf16`
- flash attention / CUDA SDPA: on
- fused optimizer: on
- compile: off for this stable continuation

This is the best practical no-FLA regime found so far.

## Hybrid continuation plan

Resume from:

- `checkpoint_1800.pt`

Continue to:

- `max_steps = 12000`

Cadence:

- `eval_every = 200`
- `save_every = 200`
- `log_every = 20`

Stopping rule:

- If EMA validation loss worsens at **two consecutive evals after the best
  checkpoint**, stop early and keep `best.pt`.

This is conservative and avoids burning budget on a clearly peaked run.

## Transformer baseline plan

Only start this **after** the hybrid continuation is either finished or clearly
peaked.

Run:

- architecture: `transformer`
- preset: `speech_transformer_a100_20m_ctx1024`
- true batch size: start at the preset value and reduce only if required
- target: `2000-4000` steps
- same `eval_every = 200`, `save_every = 200`, `log_every = 20`

Goal:

- establish a matched baseline for the paper / report
- not to finish the entire baseline budget in one sitting

## What to pull locally during the run

At minimum, pull these whenever they change:

- `best.pt`
- `checkpoint_*.pt` at each save boundary
- `train.log`
- `config.json`

If the pod dies again, the current best checkpoint should already exist locally.

## Suggested budget split

For roughly `$100`:

- `~70-80%`: hybrid continuation
- `~20-30%`: transformer baseline start

If the hybrid is still improving cleanly and budget remains, it is reasonable to
give the hybrid more time before starting the baseline.
