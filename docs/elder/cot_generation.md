# ELDER offline CoT generation

This pipeline creates fixed four-stage retrieval-oriented CoT supervision with
the local `GLM-4.1V-9B-Thinking` checkpoint. Query and candidate are generated
in separate model calls; neither call receives its paired side. Free-form
Thinking output is parsed but never written to the training JSONL.

## Task and sampling policy

The manifest builder uses deterministic, no-replacement reservoir sampling with
seed 42 and saves the original row index. It samples
`min(50,000, valid rows)` from each real task:

- 20 MMEB-train subsets;
- 3 LLaVA-Hound modes;
- 5 ColPali `source` groups;
- 6 VisRAG `source` groups.

That is 34 independently balanced tasks. Candidate CoT is enabled from actual
semantic richness, not only the dataset name: visual media and descriptive text
are enabled, while short labels, numbers, and short answers are disabled.

With the currently downloaded data, the full upper bound is 1,046,634 pairs
(661,879 MMEB + 150,000 LLaVA-Hound + 118,195 ColPali + 116,560
VisRAG). Hugging Face batch-1 generation can therefore take many weeks even on
four H20s, especially when video samples trigger retries. Benchmark a small
per-task pilot before committing to the full manifest.

## One-time environment setup

Stage 1 must remain on `transformers==4.52.3`. Create the isolated GLM
environment instead:

```bash
cd /root/code/ELDER
bash scripts/elder/create_cot_env.sh
```

The resulting interpreter is
`/root/miniconda3/envs/elder-cot/bin/python` and uses
`transformers==4.57.1`.

## Preflight and smoke test

List all configured task IDs:

```bash
source .env.elder
/root/miniconda3/envs/elder-cot/bin/python \
  scripts/elder/build_cot_manifest.py --list-tasks
```

Use a separate output root for a smoke test so that its filtered manifest cannot
be mistaken for the full manifest:

```bash
ELDER_COT_OUTPUT_DIR=/root/autodl-tmp/datasets/ELDER-CoT-smoke \
  bash scripts/elder/run_cot_generation_4h20.sh \
  --force-manifest --task mmeb__ImageNet_1K --max-samples 1
```

`--max-samples` is per GPU. The launcher verifies that all four requested GPU
IDs exist before loading four model replicas.

## Full run in tmux

```bash
tmux new -s elder-cot
cd /root/code/ELDER
bash scripts/elder/run_cot_generation_4h20.sh
```

The first run automatically builds a full manifest when none exists. If an old
or filtered manifest exists and should be replaced, use
`--force-manifest`. Press `Ctrl+B`, then `D` to detach. Reattach with:

```bash
tmux attach -t elder-cot
```

Stopping with `Ctrl+C` is safe. Each shard appends one validated JSON object at
a time; run the same command again to resume from completed `sample_id` values.
Old failure events remain as audit history, but a later successful result
resolves that sample during validation.

## Output layout

The default root is `/root/autodl-tmp/datasets/ELDER-CoT`:

```text
manifests/
  manifest_index.json
  <task>.jsonl
generated/
  cot-shard-00000-of-00004.jsonl
  ...
  failures-shard-00000-of-00004.jsonl
logs/<timestamp>/
validation_report.json
```

The final validator checks manifest coverage, duplicates, fixed K=4 schema,
stage roles, length, adjacent duplication, paths/page IDs, answer-like output,
and exact CoT template reuse across samples. Merge only a complete valid run:

```bash
/root/miniconda3/envs/elder-cot/bin/python scripts/elder/validate_cot.py \
  --root /root/autodl-tmp/datasets/ELDER-CoT \
  --merge-output /root/autodl-tmp/datasets/ELDER-CoT/elder_cot_train.jsonl
```

Generation defaults and all local dataset paths are declared in
`experiments/elder/cot_generation_local.yaml`.
