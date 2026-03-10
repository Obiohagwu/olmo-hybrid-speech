#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$ROOT"

python3 train.py \
  --arch olmo_hybrid \
  --preset speech_olmo_hybrid_a100_20m_ctx1024 \
  --device cuda \
  "$@"
