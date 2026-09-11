#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
conda_bin="${ELDER_CONDA_BIN:-/root/miniconda3/bin/conda}"
source_env="${ELDER_STAGE1_ENV_NAME:-elder}"
target_env="${ELDER_COT_ENV_NAME:-elder-cot}"
dry_run=0
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
elif [[ $# -gt 0 ]]; then
  printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
  exit 2
fi

if [[ ! -x "$conda_bin" ]]; then
  printf 'Conda executable not found: %s\n' "$conda_bin" >&2
  exit 1
fi

target_python="/root/miniconda3/envs/$target_env/bin/python"
printf '%s\n' \
  "Source environment: $source_env" \
  "CoT environment: $target_env" \
  "Requirements: $repo_dir/requirements/elder-cot.txt"

if [[ "$dry_run" == "1" ]]; then
  printf 'Would clone the Stage 1 environment if needed, then upgrade only the clone to transformers 4.57.1.\n'
  exit 0
fi

if [[ ! -x "$target_python" ]]; then
  "$conda_bin" create --yes --name "$target_env" --clone "$source_env"
fi

# Local Stage 1 settings deliberately enable HF offline mode.  They are not
# relevant to installing Python packages into this isolated environment.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
"$conda_bin" run --no-capture-output --name "$target_env" \
  python -m pip install --upgrade -r "$repo_dir/requirements/elder-cot.txt"

"$conda_bin" run --no-capture-output --name "$target_env" python - <<'PY'
import torch
import transformers
from transformers import Glm4vForConditionalGeneration

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("GLM class:", Glm4vForConditionalGeneration.__name__)
PY

printf 'CoT environment ready. Activate it with: conda activate %s\n' "$target_env"
