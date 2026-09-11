#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

if [[ -f .env.elder ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env.elder
  set +a
  export ELDER_ENV_LOADED=1
fi

: "${ELDER_SMOKE_OUTPUT_DIR:=/root/autodl-tmp/checkpoints/elder/stage1_full_smoke}"
export ELDER_OUTPUT_DIR="$ELDER_SMOKE_OUTPUT_DIR"
export ELDER_RUN_NAME="${ELDER_RUN_NAME:-elder-stage1-full-4h20-smoke}"
export ELDER_MAX_STEPS=1
export ELDER_PER_DEVICE_BATCH_SIZE=1
export ELDER_GRAD_ACCUM_STEPS=1
export ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE=1
export ELDER_GC_Q_CHUNK_SIZE=1
export ELDER_GC_P_CHUNK_SIZE=1
export ELDER_DATALOADER_NUM_WORKERS="${ELDER_DATALOADER_NUM_WORKERS:-0}"
export ELDER_SAVE_STEPS=1
export ELDER_SAVE_TOTAL_LIMIT=1
export ELDER_WARMUP_STEPS=0
export ELDER_RESUME_FROM=none

exec bash scripts/elder/train_stage1_full_4h20.sh
