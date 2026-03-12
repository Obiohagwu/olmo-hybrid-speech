#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUNS_ROOT="${RUNS_ROOT:-/runpod-volume/olmo-hybrid-speech/runs/speech}"
DATA_DIR="${DATA_DIR:-$ROOT/data/speech/ljspeech_encodec24_8s}"
RUN_NAME="${RUN_NAME:-ljspeech_transformer_a100_20m_ctx1024_4k}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUNS_ROOT"

cd "$ROOT"

python3 train.py \
  --arch transformer \
  --preset speech_transformer_a100_20m_ctx1024 \
  --device cuda \
  --data_dir "$DATA_DIR" \
  --output_dir "$RUNS_ROOT" \
  --run_name "$RUN_NAME" \
  --max_steps 4000 \
  --eval_every 200 \
  --save_every 200 \
  --log_every 20 \
  --compile_model
