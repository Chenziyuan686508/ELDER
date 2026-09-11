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

build_manifest=0
force_manifest=0
dry_run=0
max_samples=""
tasks=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-manifest)
      build_manifest=1
      shift
      ;;
    --force-manifest)
      build_manifest=1
      force_manifest=1
      shift
      ;;
    --task)
      [[ $# -ge 2 ]] || { printf '%s\n' '--task requires a task ID.' >&2; exit 2; }
      tasks+=("$2")
      shift 2
      ;;
    --max-samples)
      [[ $# -ge 2 ]] || { printf '%s\n' '--max-samples requires an integer.' >&2; exit 2; }
      max_samples="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      printf '%s\n' \
        'Usage: bash scripts/elder/run_cot_generation_4h20.sh [options]' \
        '' \
        'Options:' \
        '  --build-manifest       Build manifests before generation' \
        '  --force-manifest       Rebuild existing manifests' \
        '  --task TASK_ID         Restrict generation; may be repeated' \
        '  --max-samples N        Process at most N new rows per GPU (smoke test)' \
        '  --dry-run              Print the four commands without loading GLM' \
        '' \
        'Environment overrides:' \
        '  ELDER_COT_PYTHON, ELDER_COT_MODEL, ELDER_COT_OUTPUT_DIR,' \
        '  ELDER_COT_CONFIG, ELDER_CUDA_VISIBLE_DEVICES'
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

: "${ELDER_COT_PYTHON:=/root/miniconda3/envs/elder-cot/bin/python}"
: "${ELDER_COT_MODEL:=/root/autodl-tmp/models/GLM-4.1V-9B-Thinking}"
: "${ELDER_COT_OUTPUT_DIR:=/root/autodl-tmp/datasets/ELDER-CoT}"
: "${ELDER_COT_CONFIG:=$repo_dir/experiments/elder/cot_generation_local.yaml}"
: "${ELDER_CUDA_VISIBLE_DEVICES:=0,1,2,3}"

ELDER_CUDA_VISIBLE_DEVICES="${ELDER_CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a gpus <<< "$ELDER_CUDA_VISIBLE_DEVICES"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  printf 'The launcher requires exactly four GPU IDs; got %s.\n' "$ELDER_CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi
if [[ "$dry_run" != "1" ]]; then
  mapfile -t available_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  for requested_gpu in "${gpus[@]}"; do
    gpu_found=0
    for available_gpu in "${available_gpus[@]}"; do
      if [[ "$requested_gpu" == "$available_gpu" ]]; then
        gpu_found=1
        break
      fi
    done
    if [[ "$gpu_found" != "1" ]]; then
      printf 'Requested GPU %s is not currently visible; nvidia-smi reports: %s\n' \
        "$requested_gpu" "${available_gpus[*]:-none}" >&2
      exit 1
    fi
  done
fi
if [[ "$dry_run" != "1" && ! -x "$ELDER_COT_PYTHON" ]]; then
  printf 'CoT Python not found: %s\nRun: bash scripts/elder/create_cot_env.sh\n' "$ELDER_COT_PYTHON" >&2
  exit 1
fi
if [[ ! -s "$ELDER_COT_MODEL/config.json" ]]; then
  printf 'GLM model config is missing: %s/config.json\n' "$ELDER_COT_MODEL" >&2
  exit 1
fi

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

manifest_dir="$ELDER_COT_OUTPUT_DIR/manifests"
manifest_index="$manifest_dir/manifest_index.json"
manifest_args=(
  scripts/elder/build_cot_manifest.py
  --config "$ELDER_COT_CONFIG"
  --output-dir "$ELDER_COT_OUTPUT_DIR"
)
for task in "${tasks[@]}"; do
  manifest_args+=(--task "$task")
done
if [[ "$force_manifest" == "1" ]]; then
  manifest_args+=(--force)
fi

if [[ "$dry_run" == "1" ]]; then
  printf '%s\n' \
    "CoT model: $ELDER_COT_MODEL" \
    "Output root: $ELDER_COT_OUTPUT_DIR" \
    "GPUs: ${gpus[*]}"
  if [[ "$build_manifest" == "1" || ! -s "$manifest_index" ]]; then
    printf 'Manifest command:'
    printf ' %q' "$ELDER_COT_PYTHON" "${manifest_args[@]}"
    printf '\n'
  fi
elif [[ "$build_manifest" == "1" || ! -s "$manifest_index" ]]; then
  "$ELDER_COT_PYTHON" "${manifest_args[@]}"
fi

if [[ "$dry_run" != "1" && ! -s "$manifest_index" ]]; then
  printf 'Manifest index is missing after preparation: %s\n' "$manifest_index" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="$ELDER_COT_OUTPUT_DIR/logs/$timestamp"
if [[ "$dry_run" != "1" ]]; then
  mkdir -p "$log_dir"
fi

pids=()
cleanup() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for shard_index in 0 1 2 3; do
  command=(
    "$ELDER_COT_PYTHON"
    scripts/elder/generate_cot.py
    --config "$ELDER_COT_CONFIG"
    --manifest-dir "$manifest_dir"
    --output-dir "$ELDER_COT_OUTPUT_DIR"
    --model-path "$ELDER_COT_MODEL"
    --num-shards 4
    --shard-index "$shard_index"
    --resume
  )
  for task in "${tasks[@]}"; do
    command+=(--task "$task")
  done
  if [[ -n "$max_samples" ]]; then
    command+=(--max-samples "$max_samples")
  fi
  if [[ "$dry_run" == "1" ]]; then
    command+=(--dry-run)
    printf 'GPU %s:' "${gpus[$shard_index]}"
    printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpus[$shard_index]}" "${command[@]}"
    printf '\n'
    continue
  fi

  log_path="$log_dir/gpu${gpus[$shard_index]}_shard${shard_index}.log"
  if [[ "$shard_index" == "0" ]]; then
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="${gpus[$shard_index]}" "${command[@]}" 2>&1 | tee "$log_path"
    ) &
  else
    (
      CUDA_VISIBLE_DEVICES="${gpus[$shard_index]}" "${command[@]}" >"$log_path" 2>&1
    ) &
  fi
  pids+=("$!")
  printf 'Started shard %s on GPU %s (PID %s, log %s)\n' \
    "$shard_index" "${gpus[$shard_index]}" "$!" "$log_path"
done

if [[ "$dry_run" == "1" ]]; then
  printf '%s\n' 'Dry run complete; no model was loaded and no files were written.'
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
trap - INT TERM
if [[ "$status" != "0" ]]; then
  printf 'At least one generation shard failed. Logs: %s\n' "$log_dir" >&2
  exit "$status"
fi

"$ELDER_COT_PYTHON" scripts/elder/validate_cot.py \
  --root "$ELDER_COT_OUTPUT_DIR"
printf 'Four-GPU CoT generation and validation complete: %s\n' "$ELDER_COT_OUTPUT_DIR"
