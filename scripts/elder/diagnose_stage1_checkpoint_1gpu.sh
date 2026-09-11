#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

if [[ -f .env.elder && "${ELDER_ENV_LOADED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env.elder
  set +a
  export ELDER_ENV_LOADED=1
fi

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if [[ $# -ne 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

: "${ELDER_DIAG_BASE_MODEL:=${ELDER_MODEL_PATH:-/root/autodl-tmp/models/Qwen2-VL-2B-Instruct}}"
: "${ELDER_DIAG_CHECKPOINT:=/root/autodl-tmp/checkpoints/elder/stage1_v2_6h20_bs1026_step2k/checkpoint-2000}"
: "${ELDER_DIAG_DATA_ROOT:=/root/autodl-tmp/datasets/MMEB-V2}"
: "${ELDER_DIAG_DATASET:=ImageNet-R}"
: "${ELDER_DIAG_SAMPLE_COUNT:=64}"
: "${ELDER_DIAG_BATCH_SIZE:=8}"
: "${ELDER_DIAG_WORKERS:=0}"
: "${ELDER_DIAG_RESIZE_MAX_PIXELS:=200704}"
: "${ELDER_DIAG_CUDA_VISIBLE_DEVICES:=0}"
: "${ELDER_DIAG_MODES:=full_unmerged,full_merged,base,visual_only,dora_only}"
: "${ELDER_DIAG_INCLUDE_TRAJECTORY:=1}"
: "${ELDER_DIAG_OUTPUT_ROOT:=/root/autodl-tmp/eval/elder/checkpoint_diagnostics}"

required_checkpoint_files=(
  adapter_config.json
  adapter_model.safetensors
  model.safetensors
  vlm2vec_hybrid_checkpoint.json
)
for filename in "${required_checkpoint_files[@]}"; do
  if [[ ! -s "$ELDER_DIAG_CHECKPOINT/$filename" ]]; then
    printf 'Missing checkpoint file: %s\n' "$ELDER_DIAG_CHECKPOINT/$filename" >&2
    exit 1
  fi
done
if [[ ! -s "$ELDER_DIAG_BASE_MODEL/config.json" ]]; then
  printf 'Missing base model config: %s\n' "$ELDER_DIAG_BASE_MODEL/config.json" >&2
  exit 1
fi
if [[ ! -d "$ELDER_DIAG_DATA_ROOT" ]]; then
  printf 'Missing MMEB-V2 data root: %s\n' "$ELDER_DIAG_DATA_ROOT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$ELDER_DIAG_CUDA_VISIBLE_DEVICES"
export MMEB_V2_DATA_DIR="$ELDER_DIAG_DATA_ROOT"
export MMEB_V3_DATA_DIR="$ELDER_DIAG_DATA_ROOT"
export MMEB_V2_IMAGE_QUERY_DIR="$ELDER_DIAG_DATA_ROOT/image-query"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$ELDER_DIAG_OUTPUT_ROOT/$timestamp"
report_path="$run_dir/report.json"

command=(
  python scripts/elder/diagnose_stage1_checkpoint.py
  --base-model "$ELDER_DIAG_BASE_MODEL"
  --checkpoint "$ELDER_DIAG_CHECKPOINT"
  --data-root "$ELDER_DIAG_DATA_ROOT"
  --dataset "$ELDER_DIAG_DATASET"
  --sample-count "$ELDER_DIAG_SAMPLE_COUNT"
  --batch-size "$ELDER_DIAG_BATCH_SIZE"
  --workers "$ELDER_DIAG_WORKERS"
  --resize-max-pixels "$ELDER_DIAG_RESIZE_MAX_PIXELS"
  --device cuda:0
  --modes "$ELDER_DIAG_MODES"
  --output "$report_path"
)

if [[ "$ELDER_DIAG_INCLUDE_TRAJECTORY" == "1" ]]; then
  checkpoint_parent="$(dirname "$ELDER_DIAG_CHECKPOINT")"
  for step in 500 1000 1500 2000; do
    trajectory_checkpoint="$checkpoint_parent/checkpoint-$step"
    if [[ -d "$trajectory_checkpoint" ]]; then
      command+=(--trajectory-checkpoint "$trajectory_checkpoint")
    fi
  done
fi

printf '%s\n' \
  "Checkpoint diagnostic configuration:" \
  "  base model: $ELDER_DIAG_BASE_MODEL" \
  "  checkpoint: $ELDER_DIAG_CHECKPOINT" \
  "  dataset: $ELDER_DIAG_DATASET ($ELDER_DIAG_SAMPLE_COUNT queries)" \
  "  GPU: $ELDER_DIAG_CUDA_VISIBLE_DEVICES" \
  "  modes: $ELDER_DIAG_MODES" \
  "  trajectory: $ELDER_DIAG_INCLUDE_TRAJECTORY" \
  "  output: $report_path"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "$dry_run" == "1" ]]; then
  printf '%s\n' 'Dry run complete; no model was loaded.'
  exit 0
fi

mkdir -p "$run_dir"
"${command[@]}" 2>&1 | tee "$run_dir/diagnostic.log"

