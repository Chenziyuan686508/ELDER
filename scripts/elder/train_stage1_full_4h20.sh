#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ $# -ne 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

if [[ -f .env.elder ]]; then
  if [[ "${ELDER_ENV_LOADED:-0}" != "1" ]]; then
    # shellcheck disable=SC1091
    set -a
    source .env.elder
    set +a
    export ELDER_ENV_LOADED=1
  fi
else
  printf 'Missing .env.elder. Copy experiments/elder/local_paths.autodl.env.example first.\n' >&2
  exit 1
fi

required_vars=(
  ELDER_MODEL_PATH
  ELDER_MMEB_DATA
  ELDER_LLAVAHOUND_DATA
  ELDER_COLPALI_DATA
  ELDER_VISRAG_DATA
  ELDER_OUTPUT_DIR
)
for variable_name in "${required_vars[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    printf 'Required environment variable is unset: %s\n' "$variable_name" >&2
    exit 1
  fi
done

: "${ELDER_CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${ELDER_EXPECTED_GPU_COUNT:=4}"
: "${ELDER_MAX_STEPS:=5000}"
: "${ELDER_PER_DEVICE_BATCH_SIZE:=4}"
: "${ELDER_GRAD_ACCUM_STEPS:=16}"
: "${ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE:=$ELDER_PER_DEVICE_BATCH_SIZE}"
: "${ELDER_GC_Q_CHUNK_SIZE:=2}"
: "${ELDER_GC_P_CHUNK_SIZE:=2}"
: "${ELDER_DATALOADER_NUM_WORKERS:=2}"
: "${ELDER_RESIZE_MAX_PIXELS:=200704}"
: "${ELDER_SAVE_STEPS:=500}"
: "${ELDER_SAVE_TOTAL_LIMIT:=3}"
: "${ELDER_SAVE_STRATEGY:=steps}"
: "${ELDER_WARMUP_STEPS:=100}"
: "${ELDER_LEARNING_RATE:=5e-5}"
: "${ELDER_LOGGING_STEPS:=1}"
: "${ELDER_RUN_NAME:=elder-stage1-full-4h20}"
: "${ELDER_RESUME_FROM:=auto}"
: "${ELDER_IGNORE_DATA_SKIP:=false}"
: "${ELDER_LORA_R:=16}"
: "${ELDER_LORA_ALPHA:=64}"
: "${ELDER_LORA_TARGET_MODULES:=qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj,gate_proj,up_proj}"
: "${ELDER_LORA_ADAPTER_SCOPE:=auto}"
: "${ELDER_STRICT_STAGE1_DORA:=false}"
: "${ELDER_DDP_FIND_UNUSED_PARAMETERS:=true}"

IFS=',' read -r -a elder_gpus <<< "$ELDER_CUDA_VISIBLE_DEVICES"
gpu_count="${#elder_gpus[@]}"
if [[ "$gpu_count" -ne "$ELDER_EXPECTED_GPU_COUNT" ]]; then
  printf 'Stage 1 launcher expected %s visible GPUs but got %s: %s.\n' \
    "$ELDER_EXPECTED_GPU_COUNT" "$gpu_count" "$ELDER_CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi
export CUDA_VISIBLE_DEVICES="$ELDER_CUDA_VISIBLE_DEVICES"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1

mkdir -p "$ELDER_OUTPUT_DIR" "${HF_HOME:-/tmp/hf_cache}" "${TMPDIR:-/tmp}"

python scripts/elder/stage1_model_contract.py \
  --model-path "$ELDER_MODEL_PATH"
python scripts/elder/stage1_full_local_preflight.py \
  --model-path "$ELDER_MODEL_PATH" \
  --dataset-config experiments/elder/stage1_full_local.yaml

if (( ELDER_GC_Q_CHUNK_SIZE > ELDER_PER_DEVICE_BATCH_SIZE )); then
  ELDER_GC_Q_CHUNK_SIZE="$ELDER_PER_DEVICE_BATCH_SIZE"
fi
if (( ELDER_GC_P_CHUNK_SIZE > ELDER_PER_DEVICE_BATCH_SIZE )); then
  ELDER_GC_P_CHUNK_SIZE="$ELDER_PER_DEVICE_BATCH_SIZE"
fi

infonce_global_batch="$((gpu_count * ELDER_PER_DEVICE_BATCH_SIZE))"
optimizer_global_batch="$((infonce_global_batch * ELDER_GRAD_ACCUM_STEPS))"
interleave_global_batch="$((gpu_count * ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE))"

printf '%s\n' \
  'Stage 1 launch configuration:' \
  "  GPUs: $ELDER_CUDA_VISIBLE_DEVICES" \
  "  world size: $gpu_count" \
  "  output: $ELDER_OUTPUT_DIR" \
  "  batch/device: $ELDER_PER_DEVICE_BATCH_SIZE" \
  "  true InfoNCE global batch: $infonce_global_batch" \
  "  gradient accumulation: $ELDER_GRAD_ACCUM_STEPS" \
  "  optimizer global batch: $optimizer_global_batch" \
  "  homogeneous batch/device: $ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE" \
  "  interleave global batch: $interleave_global_batch" \
  "  GradCache chunk size (q/p): $ELDER_GC_Q_CHUNK_SIZE/$ELDER_GC_P_CHUNK_SIZE" \
  "  dataloader workers: $ELDER_DATALOADER_NUM_WORKERS" \
  "  resize max pixels: $ELDER_RESIZE_MAX_PIXELS" \
  "  LoRA r/alpha: $ELDER_LORA_R/$ELDER_LORA_ALPHA" \
  "  LoRA target modules: $ELDER_LORA_TARGET_MODULES" \
  "  LoRA adapter scope: $ELDER_LORA_ADAPTER_SCOPE" \
  "  strict Stage 1 V2 DoRA audit: $ELDER_STRICT_STAGE1_DORA" \
  "  ignore data skip on resume: $ELDER_IGNORE_DATA_SKIP" \
  "  DDP find unused parameters: $ELDER_DDP_FIND_UNUSED_PARAMETERS"

if [[ "$dry_run" == "1" ]]; then
  printf '%s\n' 'Dry run complete; training was not started.'
  exit 0
fi

torchrun \
  --standalone \
  --nproc_per_node="$gpu_count" \
  train.py \
  --model_name "$ELDER_MODEL_PATH" \
  --dataset_config experiments/elder/stage1_full_local.yaml \
  --output_dir "$ELDER_OUTPUT_DIR" \
  --run_name "$ELDER_RUN_NAME" \
  --lora \
  --lora_r "$ELDER_LORA_R" \
  --lora_alpha "$ELDER_LORA_ALPHA" \
  --lora_target_modules "$ELDER_LORA_TARGET_MODULES" \
  --lora_adapter_scope "$ELDER_LORA_ADAPTER_SCOPE" \
  --strict_stage1_dora "$ELDER_STRICT_STAGE1_DORA" \
  --bf16 \
  --pooling eos \
  --normalize true \
  --temperature 0.02 \
  --resize_max_pixels "$ELDER_RESIZE_MAX_PIXELS" \
  --grad_cache true \
  --gc_q_chunk_size "$ELDER_GC_Q_CHUNK_SIZE" \
  --gc_p_chunk_size "$ELDER_GC_P_CHUNK_SIZE" \
  --per_device_train_batch_size "$ELDER_PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$ELDER_GRAD_ACCUM_STEPS" \
  --homogeneous_batch_size_per_device "$ELDER_HOMOGENEOUS_BATCH_SIZE_PER_DEVICE" \
  --dataloader_num_workers "$ELDER_DATALOADER_NUM_WORKERS" \
  --max_steps "$ELDER_MAX_STEPS" \
  --warmup_steps "$ELDER_WARMUP_STEPS" \
  --learning_rate "$ELDER_LEARNING_RATE" \
  --lr_scheduler_type linear \
  --logging_steps "$ELDER_LOGGING_STEPS" \
  --save_strategy "$ELDER_SAVE_STRATEGY" \
  --save_steps "$ELDER_SAVE_STEPS" \
  --save_total_limit "$ELDER_SAVE_TOTAL_LIMIT" \
  --save_safetensors true \
  --remove_unused_columns false \
  --ddp_find_unused_parameters "$ELDER_DDP_FIND_UNUSED_PARAMETERS" \
  --ignore_data_skip "$ELDER_IGNORE_DATA_SKIP" \
  --resume_from "$ELDER_RESUME_FROM" \
  --report_to none
