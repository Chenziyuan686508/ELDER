#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

# A representative one-step run without Trainer checkpoints or the final
# full-model export. This keeps GPU validation from consuming ~12 GiB/run.
export ELDER_STAGE1_V2_5H20_OUTPUT_DIR="${ELDER_STAGE1_V2_5H20_SMOKE_OUTPUT_DIR:-/tmp/elder_stage1_v2_5h20_smoke}"
export ELDER_STAGE1_V2_5H20_RUN_NAME="${ELDER_STAGE1_V2_5H20_SMOKE_RUN_NAME:-elder-stage1-v2-5h20-smoke}"
export ELDER_STAGE1_V2_5H20_MAX_STEPS="${ELDER_STAGE1_V2_5H20_SMOKE_MAX_STEPS:-1}"
export ELDER_STAGE1_V2_5H20_SAVE_STRATEGY=no
export ELDER_STAGE1_V2_5H20_WARMUP_STEPS=0
export ELDER_STAGE1_V2_5H20_RESUME_FROM=none
export ELDER_SKIP_FINAL_SAVE=1

exec bash scripts/elder/train_stage1_v2_5h20.sh "$@"
