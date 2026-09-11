#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

# Keep the full v2 batch and resolution in the smoke test so its peak memory is
# representative. Only the number of optimizer steps and checkpoint cadence
# are reduced.
export ELDER_STAGE1_V2_OUTPUT_DIR="${ELDER_STAGE1_V2_SMOKE_OUTPUT_DIR:-/root/autodl-tmp/checkpoints/elder/stage1_v2_4h20_bs256_fullres_smoke}"
export ELDER_STAGE1_V2_RUN_NAME="${ELDER_STAGE1_V2_SMOKE_RUN_NAME:-elder-stage1-v2-4h20-bs256-fullres-smoke}"
export ELDER_STAGE1_V2_MAX_STEPS="${ELDER_STAGE1_V2_SMOKE_MAX_STEPS:-1}"
export ELDER_STAGE1_V2_SAVE_STEPS="${ELDER_STAGE1_V2_SMOKE_SAVE_STEPS:-100}"
export ELDER_STAGE1_V2_SAVE_TOTAL_LIMIT=1
export ELDER_STAGE1_V2_WARMUP_STEPS=0
export ELDER_STAGE1_V2_RESUME_FROM=none

exec bash scripts/elder/train_stage1_v2_4h20.sh "$@"
