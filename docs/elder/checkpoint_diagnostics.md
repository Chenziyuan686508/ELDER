# Stage 1 checkpoint diagnostics

This read-only diagnostic separates checkpoint loading defects from training-time
representation collapse. It uses the same MMEB-V2 dataset parser, processor, and
collator as `eval.py`, but only materializes a fixed ImageNet-R subset.

## What it tests

- `full_unmerged`: full hybrid checkpoint with DoRA still attached.
- `full_merged`: the normal evaluation path after `merge_and_unload()`.
- `base`: untouched Qwen2-VL base model.
- `visual_only`: trained `visual.*` tensors with the original language model.
- `dora_only`: trained language-side DoRA with the original visual tower.
- optional trajectory: checkpoint-500/1000/1500/2000 in merged mode.

For every mode the report includes Hit@1/5/10, MRR, mean rank, query/candidate
embedding dispersion, and same-image prompt sensitivity. The comparison section
measures merged/unmerged row-wise cosine, absolute difference, and Top-1
agreement.

## Run

Run this after the regular MMEB-V2 evaluation has released at least one GPU:

```bash
cd /root/code/ELDER
conda activate elder
bash scripts/elder/diagnose_stage1_checkpoint_1gpu.sh
```

Default output:

```text
/root/autodl-tmp/eval/elder/checkpoint_diagnostics/<timestamp>/
  diagnostic.log
  report.json
```

Inspect the command without loading a model:

```bash
bash scripts/elder/diagnose_stage1_checkpoint_1gpu.sh --dry-run
```

Useful overrides:

```bash
ELDER_DIAG_CUDA_VISIBLE_DEVICES=1 \
ELDER_DIAG_SAMPLE_COUNT=128 \
ELDER_DIAG_INCLUDE_TRAJECTORY=1 \
bash scripts/elder/diagnose_stage1_checkpoint_1gpu.sh
```

For a faster merge-only check:

```bash
ELDER_DIAG_MODES=full_unmerged,full_merged \
ELDER_DIAG_INCLUDE_TRAJECTORY=0 \
bash scripts/elder/diagnose_stage1_checkpoint_1gpu.sh
```

The script writes `report.partial.json` after every completed mode and replaces it
with `report.json` only after all requested checks succeed.

## Interpretation

- unmerged healthy, merged collapsed: hybrid DoRA load/merge defect;
- unmerged and merged collapsed, DoRA-only collapsed: language DoRA or visual
  fusion failure;
- visual-only collapsed: trained visual tower failure;
- component modes healthy but full hybrid collapsed: interaction or full-state
  restoration problem.

The default near-collapse threshold is mean off-diagonal query cosine `>= 0.90`.
This threshold is diagnostic rather than a benchmark acceptance criterion.
