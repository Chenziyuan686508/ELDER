#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

: "${ELDER_MODEL_PATH:=/data/chenziyuan/models/Qwen2-VL-2B-Instruct}"
: "${ELDER_COLPALI_DATA:=/data/chenziyuan/datasets/colpali_train_set}"
: "${ELDER_OUTPUT_DIR:=/data/chenziyuan/checkpoints/elder/stage1_debug}"
: "${ELDER_CUDA_VISIBLE_DEVICES:=0}"
: "${ELDER_MAX_STEPS:=10}"
: "${ELDER_BATCH_SIZE:=2}"
: "${ELDER_GRAD_ACCUM_STEPS:=1}"
: "${ELDER_RESIZE_MAX_PIXELS:=200704}"
: "${ELDER_RUN_NAME:=elder-stage1-debug}"
: "${ELDER_RESUME_FROM:=none}"
: "${ELDER_SAVE_STEPS:=$ELDER_MAX_STEPS}"

export ELDER_MODEL_PATH
export ELDER_COLPALI_DATA
export CUDA_VISIBLE_DEVICES="$ELDER_CUDA_VISIBLE_DEVICES"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python scripts/elder/stage1_model_contract.py \
  --model-path "$ELDER_MODEL_PATH"

python train.py \
  --model_name "$ELDER_MODEL_PATH" \
  --dataset_config experiments/elder/stage1_debug_vidore.yaml \
  --output_dir "$ELDER_OUTPUT_DIR" \
  --run_name "$ELDER_RUN_NAME" \
  --lora \
  --lora_r 16 \
  --lora_alpha 64 \
  --bf16 \
  --pooling eos \
  --normalize true \
  --temperature 0.02 \
  --resize_max_pixels "$ELDER_RESIZE_MAX_PIXELS" \
  --grad_cache true \
  --gc_q_chunk_size 1 \
  --gc_p_chunk_size 1 \
  --per_device_train_batch_size "$ELDER_BATCH_SIZE" \
  --gradient_accumulation_steps "$ELDER_GRAD_ACCUM_STEPS" \
  --homogeneous_batch_size_per_device "$ELDER_BATCH_SIZE" \
  --dataloader_num_workers 0 \
  --max_steps "$ELDER_MAX_STEPS" \
  --warmup_steps 1 \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --logging_steps 1 \
  --save_steps "$ELDER_SAVE_STEPS" \
  --save_total_limit 2 \
  --save_safetensors true \
  --remove_unused_columns false \
  --resume_from "$ELDER_RESUME_FROM" \
  --report_to none
