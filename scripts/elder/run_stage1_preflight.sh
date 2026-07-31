#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

model_path="${ELDER_MODEL_PATH:-/data/chenziyuan/models/Qwen2-VL-2B-Instruct}"
dataset_file="${ELDER_DATASET_FILE:-/data/chenziyuan/datasets/colpali_train_set/data/train-00000-of-00082.parquet}"

python scripts/elder/stage1_preflight.py \
  --model-path "$model_path" \
  --dataset-file "$dataset_file"
