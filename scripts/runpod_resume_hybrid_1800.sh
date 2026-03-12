#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Recommended persistent layout:
#   /runpod-volume/olmo-hybrid-speech/runs/speech/<run_name>/checkpoint_1800.pt
RUNS_ROOT="${RUNS_ROOT:-/runpod-volume/olmo-hybrid-speech/runs/speech}"
RUN_NAME="${RUN_NAME:-ljspeech_olmo_hybrid_a100_no_fla_12k_b24a1}"
RESUME_FROM="${RESUME_FROM:-$RUNS_ROOT/$RUN_NAME/checkpoint_1800.pt}"
DATA_DIR="${DATA_DIR:-$ROOT/data/speech/ljspeech_encodec24_8s}"

export PYTHONUNBUFFERED=1
export OLMO_DISABLE_FLA="${OLMO_DISABLE_FLA:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$ROOT"

if [[ ! -f "$RESUME_FROM" ]]; then
  echo "Missing checkpoint: $RESUME_FROM" >&2
  exit 1
fi

python3 train.py \
  --arch olmo_hybrid \
  --preset speech_olmo_hybrid_a100_20m_ctx1024 \
  --device cuda \
  --data_dir "$DATA_DIR" \
  --resume_from "$RESUME_FROM" \
  --batch_size 24 \
  --grad_accum_steps 1 \
  --max_steps 12000 \
  --eval_every 200 \
  --save_every 200 \
  --log_every 20 \
  --no-compile_model
