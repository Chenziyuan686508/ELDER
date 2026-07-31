#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

if [[ -f .env.elder ]]; then
  # shellcheck disable=SC1091
  source .env.elder
fi

: "${ELDER_MODEL_PATH:=/data/chenziyuan/models/Qwen2-VL-2B-Instruct}"
: "${ELDER_COLPALI_DATA:=/data/chenziyuan/datasets/colpali_train_set}"
: "${ELDER_ACCEPTANCE_RUN_ID:=$(date +%Y%m%d-%H%M%S)}"
: "${ELDER_ACCEPTANCE_ROOT:=/data/chenziyuan/checkpoints/elder/stage1_acceptance/${ELDER_ACCEPTANCE_RUN_ID}}"
: "${ELDER_EVAL_NUM_SAMPLES:=128}"
: "${ELDER_EVAL_BATCH_SIZE:=2}"
: "${ELDER_TRAIN_STEPS:=100}"
: "${ELDER_BATCH_SIZE:=2}"
: "${ELDER_GRAD_ACCUM_STEPS:=1}"
: "${ELDER_RESIZE_MAX_PIXELS:=200704}"

train_dir="$ELDER_ACCEPTANCE_ROOT/train"
metrics_dir="$ELDER_ACCEPTANCE_ROOT/metrics"
status_file="$ELDER_ACCEPTANCE_ROOT/status.txt"
dataset_file="$ELDER_COLPALI_DATA/data/train-00000-of-00082.parquet"

if [[ -e "$status_file" || -e "$train_dir" || -e "$metrics_dir" ]]; then
  printf 'Refusing to reuse an existing acceptance run: %s\n' \
    "$ELDER_ACCEPTANCE_ROOT" >&2
  exit 1
fi
mkdir -p "$metrics_dir"
printf 'running\n' > "$status_file"
on_exit() {
  exit_code=$?
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'completed\n' > "$status_file"
  else
    printf 'failed (exit=%s)\n' "$exit_code" > "$status_file"
  fi
}
trap on_exit EXIT

export ELDER_MODEL_PATH
export ELDER_COLPALI_DATA
export ELDER_OUTPUT_DIR="$train_dir"
export ELDER_MAX_STEPS="$ELDER_TRAIN_STEPS"
export ELDER_BATCH_SIZE
export ELDER_GRAD_ACCUM_STEPS
export ELDER_RESIZE_MAX_PIXELS
export ELDER_RUN_NAME="elder-stage1-acceptance-${ELDER_ACCEPTANCE_RUN_ID}"
export ELDER_RESUME_FROM=none
export ELDER_SAVE_STEPS="$ELDER_TRAIN_STEPS"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

printf 'ELDER Stage 1 acceptance run: %s\n' "$ELDER_ACCEPTANCE_RUN_ID"
printf 'Output root: %s\n' "$ELDER_ACCEPTANCE_ROOT"
printf 'Visible GPU: %s\n' "${CUDA_VISIBLE_DEVICES:-not set}"

python scripts/elder/stage1_model_contract.py \
  --model-path "$ELDER_MODEL_PATH"

python scripts/elder/stage1_retrieval_eval.py \
  --base-model "$ELDER_MODEL_PATH" \
  --dataset-file "$dataset_file" \
  --num-samples "$ELDER_EVAL_NUM_SAMPLES" \
  --batch-size "$ELDER_EVAL_BATCH_SIZE" \
  --resize-max-pixels "$ELDER_RESIZE_MAX_PIXELS" \
  --output "$metrics_dir/before.json"

bash scripts/elder/train_stage1_debug.sh

python scripts/elder/stage1_retrieval_eval.py \
  --base-model "$ELDER_MODEL_PATH" \
  --checkpoint "$train_dir" \
  --dataset-file "$dataset_file" \
  --num-samples "$ELDER_EVAL_NUM_SAMPLES" \
  --batch-size "$ELDER_EVAL_BATCH_SIZE" \
  --resize-max-pixels "$ELDER_RESIZE_MAX_PIXELS" \
  --output "$metrics_dir/after.json"

python scripts/elder/compare_stage1_metrics.py \
  --before "$metrics_dir/before.json" \
  --after "$metrics_dir/after.json" \
  --output "$metrics_dir/acceptance.json"

printf 'Stage 1 acceptance passed. Report: %s\n' \
  "$metrics_dir/acceptance.json"
