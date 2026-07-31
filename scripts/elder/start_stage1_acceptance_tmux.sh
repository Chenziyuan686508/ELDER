#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s GPU_INDEX [SESSION_NAME]\n' "$0" >&2
  exit 2
fi

gpu_index="$1"
session_name="${2:-elder-stage1-$(date +%Y%m%d-%H%M%S)}"
if [[ ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  printf 'GPU_INDEX must be a non-negative integer, got: %s\n' "$gpu_index" >&2
  exit 2
fi
if [[ ! "$session_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'SESSION_NAME may only contain letters, digits, _ and -.\n' >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  printf 'tmux is not installed.\n' >&2
  exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session_name" >&2
  exit 1
fi

gpu_state="$(
  nvidia-smi \
    --id="$gpu_index" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)"
IFS=',' read -r memory_used utilization <<< "$gpu_state"
memory_used="${memory_used//[[:space:]]/}"
utilization="${utilization//[[:space:]]/}"
if [[ "${ELDER_ALLOW_BUSY_GPU:-0}" != "1" ]] \
  && (( memory_used > 4096 || utilization > 10 )); then
  printf 'Refusing to start on busy GPU %s: %s MiB used, %s%% utilization.\n' \
    "$gpu_index" "$memory_used" "$utilization" >&2
  printf 'Wait for an idle card, or explicitly set ELDER_ALLOW_BUSY_GPU=1.\n' >&2
  exit 1
fi

run_id="${ELDER_ACCEPTANCE_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
run_root="${ELDER_ACCEPTANCE_ROOT:-/data/chenziyuan/checkpoints/elder/stage1_acceptance/${run_id}}"
log_file="$run_root/stage1_acceptance.log"
mkdir -p "$run_root"

tmux new-session -d -s "$session_name" -c "$repo_dir"
printf -v pipe_command 'cat >> %q' "$log_file"
tmux pipe-pane -o -t "$session_name" "$pipe_command"
printf -v launch_command \
  'export CUDA_VISIBLE_DEVICES=%q ELDER_CUDA_VISIBLE_DEVICES=%q ELDER_ACCEPTANCE_RUN_ID=%q ELDER_ACCEPTANCE_ROOT=%q; conda run -n elder --no-capture-output bash scripts/elder/run_stage1_acceptance.sh; elder_exit=$?; printf "\\nELDER pipeline exit code: %%s\\n" "$elder_exit"' \
  "$gpu_index" "$gpu_index" "$run_id" "$run_root"
tmux send-keys -t "$session_name" "$launch_command" C-m

printf 'Started ELDER Stage 1 acceptance in tmux.\n'
printf 'Session: %s\n' "$session_name"
printf 'GPU: %s\n' "$gpu_index"
printf 'Run root: %s\n' "$run_root"
printf 'Log: %s\n' "$log_file"
printf 'Attach: tmux attach -t %s\n' "$session_name"
printf 'Detach: Ctrl-b, then d\n'
