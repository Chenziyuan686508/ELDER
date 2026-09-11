#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

if [[ ! -f .env.elder ]]; then
  printf 'Missing .env.elder. Copy experiments/elder/local_paths.autodl.env.example first.\n' >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a
source .env.elder
set +a
export ELDER_ENV_LOADED=1

# ELDER Stage 1 v2 defaults for 4x H20. Dedicated override names prevent the
# legacy values in .env.elder from silently restoring the old bs=4 recipe.
export ELDER_OUTPUT_DIR="${ELDER_STAGE1_V2_OUTPUT_DIR:-/root/autodl-tmp/checkpoints/elder/stage1_v2_4h20_bs256_fullres}"
export ELDER_CUDA_VISIBLE_DEVICES="${ELDER_STAGE1_V2_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ELDER_MAX_STEPS="${ELDER_STAGE1_V2_MAX_STEPS:-5000}"
export ELDER_PER_DEVICE_BATCH_SIZE="${ELDER_STAGE1_V2_PER_DEVICE_BATCH_SIZE:-64}"
export ELDER_GRAD_ACCUM_STEPS="${ELDER_STAGE1_V2_GRAD_ACCUM_STEPS:-1}"
export ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE="${ELDER_STAGE1_V2_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE:-8}"
export ELDER_GC_Q_CHUNK_SIZE="${ELDER_STAGE1_V2_GC_Q_CHUNK_SIZE:-4}"
export ELDER_GC_P_CHUNK_SIZE="${ELDER_STAGE1_V2_GC_P_CHUNK_SIZE:-4}"
export ELDER_DATALOADER_NUM_WORKERS="${ELDER_STAGE1_V2_DATALOADER_NUM_WORKERS:-1}"
export ELDER_RESIZE_MAX_PIXELS="${ELDER_STAGE1_V2_RESIZE_MAX_PIXELS:-1003520}"
export ELDER_SAVE_STEPS="${ELDER_STAGE1_V2_SAVE_STEPS:-500}"
export ELDER_SAVE_TOTAL_LIMIT="${ELDER_STAGE1_V2_SAVE_TOTAL_LIMIT:-3}"
export ELDER_WARMUP_STEPS="${ELDER_STAGE1_V2_WARMUP_STEPS:-100}"
export ELDER_LEARNING_RATE="${ELDER_STAGE1_V2_LEARNING_RATE:-5e-5}"
export ELDER_LOGGING_STEPS="${ELDER_STAGE1_V2_LOGGING_STEPS:-10}"
export ELDER_LORA_R="${ELDER_STAGE1_V2_LORA_R:-16}"
export ELDER_LORA_ALPHA="${ELDER_STAGE1_V2_LORA_ALPHA:-32}"
export ELDER_DDP_FIND_UNUSED_PARAMETERS="${ELDER_STAGE1_V2_DDP_FIND_UNUSED_PARAMETERS:-false}"
export ELDER_RUN_NAME="${ELDER_STAGE1_V2_RUN_NAME:-elder-stage1-v2-4h20-bs256-fullres}"
export ELDER_RESUME_FROM="${ELDER_STAGE1_V2_RESUME_FROM:-auto}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash scripts/elder/train_stage1_full_4h20.sh "$@"
