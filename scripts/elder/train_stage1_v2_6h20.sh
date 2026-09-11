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

# Six equal DDP shards cannot total exactly 1,024. A local batch of 171 gives
# 6 * 171 = 1,026 candidates, only two above the official global pool.
export ELDER_OUTPUT_DIR="${ELDER_STAGE1_V2_6H20_OUTPUT_DIR:-/root/autodl-tmp/checkpoints/elder/stage1_v2_fixed_6h20_bs1026_step2k}"
export ELDER_CUDA_VISIBLE_DEVICES="${ELDER_STAGE1_V2_6H20_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export ELDER_EXPECTED_GPU_COUNT=6
export ELDER_MAX_STEPS="${ELDER_STAGE1_V2_6H20_MAX_STEPS:-2000}"
export ELDER_PER_DEVICE_BATCH_SIZE="${ELDER_STAGE1_V2_6H20_PER_DEVICE_BATCH_SIZE:-171}"
export ELDER_GRAD_ACCUM_STEPS="${ELDER_STAGE1_V2_6H20_GRAD_ACCUM_STEPS:-1}"
export ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE="${ELDER_STAGE1_V2_6H20_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE:-11}"
# With the visual tower frozen and only the original DoRA targets trainable,
# match the released VLM2Vec-V2 GradCache micro-batch size.
export ELDER_GC_Q_CHUNK_SIZE="${ELDER_STAGE1_V2_6H20_GC_Q_CHUNK_SIZE:-8}"
export ELDER_GC_P_CHUNK_SIZE="${ELDER_STAGE1_V2_6H20_GC_P_CHUNK_SIZE:-8}"
# Two-worker testing was 4.8% slower than one worker over the same two-step run.
export ELDER_DATALOADER_NUM_WORKERS="${ELDER_STAGE1_V2_6H20_DATALOADER_NUM_WORKERS:-1}"
export ELDER_RESIZE_MAX_PIXELS="${ELDER_STAGE1_V2_6H20_RESIZE_MAX_PIXELS:-1003520}"
export ELDER_SAVE_STRATEGY="${ELDER_STAGE1_V2_6H20_SAVE_STRATEGY:-steps}"
export ELDER_SAVE_STEPS="${ELDER_STAGE1_V2_6H20_SAVE_STEPS:-250}"
export ELDER_SAVE_TOTAL_LIMIT="${ELDER_STAGE1_V2_6H20_SAVE_TOTAL_LIMIT:-5}"
export ELDER_WARMUP_STEPS="${ELDER_STAGE1_V2_6H20_WARMUP_STEPS:-100}"
export ELDER_LEARNING_RATE="${ELDER_STAGE1_V2_6H20_LEARNING_RATE:-5e-5}"
export ELDER_LOGGING_STEPS="${ELDER_STAGE1_V2_6H20_LOGGING_STEPS:-10}"
export ELDER_LOG_BATCH_COMPOSITION="${ELDER_STAGE1_V2_6H20_LOG_BATCH_COMPOSITION:-0}"
export ELDER_LORA_R="${ELDER_STAGE1_V2_6H20_LORA_R:-16}"
export ELDER_LORA_ALPHA="${ELDER_STAGE1_V2_6H20_LORA_ALPHA:-64}"
export ELDER_LORA_TARGET_MODULES="${ELDER_STAGE1_V2_6H20_LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"
export ELDER_LORA_ADAPTER_SCOPE="${ELDER_STAGE1_V2_6H20_LORA_ADAPTER_SCOPE:-full_model}"
export ELDER_STRICT_STAGE1_DORA="${ELDER_STAGE1_V2_6H20_STRICT_STAGE1_DORA:-true}"
export ELDER_DDP_FIND_UNUSED_PARAMETERS="${ELDER_STAGE1_V2_6H20_DDP_FIND_UNUSED_PARAMETERS:-false}"
export ELDER_RUN_NAME="${ELDER_STAGE1_V2_6H20_RUN_NAME:-elder-stage1-v2-fixed-6h20-bs1026-step2k}"
# Start this corrected run from the untouched Qwen2-VL base. Set this to auto
# or an explicit checkpoint step only after this new run has saved a checkpoint.
export ELDER_RESUME_FROM="${ELDER_STAGE1_V2_6H20_RESUME_FROM:-none}"
# Preserve the exact seeded data-source sequence and per-source cursor when
# resuming, even though replaying skipped image/video batches is expensive.
export ELDER_IGNORE_DATA_SKIP="${ELDER_STAGE1_V2_6H20_IGNORE_DATA_SKIP:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec bash scripts/elder/train_stage1_full_4h20.sh "$@"
