# Dual-endpoint teacher pilot

Implements the independent specification in `教师模型训练执行说明.md`. The former
four-stage ELDER CoT schema is **not** used. Training CoT is privileged data;
normal held-out evaluation requires separately generated endpoint-only CoT.

Use `/root/miniconda3/envs/elder/bin/python` from the repository root. For
`generate_eval_cot`, use `/root/miniconda3/envs/elder-cot/bin/python` instead.
All commands accept `--help` and `--config configs/teacher_pilot.yaml`.

Execution order: `audit`, `prepare`, `preflight`, `smoke`, `train --mode no_cot`,
`train --mode both_cot`, `generate_eval_cot`, `evaluate`, `report`. Training takes
`--resume`; generation resumes successful cache entries automatically. Do not
rerun preparation after starting A/B: resume fingerprints reject changed data.

Artifacts are written under `/root/autodl-tmp/eval/elder/teacher_pilot`:

- `reference`: immutable released adapter and tokenization files, reference metadata.
- `common`: merged retrieval base, shared random LoRA initialization, exact parameter audit.
- `data`: common training manifest, fixed holdout queries/candidate corpora, filtering records and batch plan.
- `review`: random samples and textual review notes (not a human visual-grounding score).
- `smoke`: real 10-step runs with an explicitly smaller 4-query pool.
- `runs`: final A/B runs; each optimizer step uses up to 64 queries and deduplicated candidates.
- `eval_cot`: independent GLM generation cache, with per-endpoint prompts and provenance.
- `evaluation`: full held-out corpus metrics, paired group bootstrap.
- `teacher_report.md`: evidence-only generated report, including incomplete checks.

Native Qwen2-VL last-content-token pooling adds no new special tokens. The local
VLM2Vec backbone uses per-sample visual lists but requires flattened grids to
compute position IDs; `model.py` explicitly bridges these conventions. The
vocabulary projection is bypassed during embedding forwards while preserving
its tied state-dict parameter. Vision and merger stay frozen; a new rank-32
language-only LoRA is attached after merging the released DoRA adapter.

This pilot uses a seeded 2048-pair sample and 200 held-out queries per task,
not an official benchmark run. The full filtered held-out candidate corpus is
encoded independently. Answer labels are canonicalized by case/terminal
punctuation within Visual7W; all known positives are masked in both directions.
Source media use decoded-pixel hashes to prevent cross-path split leakage.
Canonical candidate CoT is selected by a conservative lexical risk gate and
stable SHA256 ordering; provenance remains unknown. Source JSONs are untouched.

Optional process-state export is exposed by `process_export --enable` (at most 10 diagnostic samples); it includes a suffix-change causality probe. Student training is outside this implementation.

## Local video and visual-document readiness

`media_readiness` is a separate diagnostic, not part of the historical pilot's
training/validation preparation. It does not download data or screen CoT quality.
It scans the 15 LLaVA-Hound and 4 ViDoRe/VisRAG annotation files, checks video
frame existence, and matches document filename row indices against exact local
source queries (and ColPali answers). Missing `train_600k` is reported and excluded.
Document samples are extracted from local Arrow/Parquet as lossless RGB PNG;
this establishes source-row correspondence, not byte identity with the author's
unavailable JPEG export. All source annotations and CoT bodies are preserved.

```bash
/root/miniconda3/envs/elder/bin/python -m teacher_pipeline.media_readiness audit
/root/miniconda3/envs/elder/bin/python -m teacher_pipeline.media_readiness smoke
```

Outputs: `/root/autodl-tmp/eval/elder/media_readiness/`. The scan checks all
referenced video paths, but image decoding and GPU tests are sample-based. Four
seeded reservoir samples per task are decoded; two distinct candidates per task
are used for a real contrastive forward/backward/update in each input mode.
The smoke loads the shared initial adapter and does not save trained weights.
Replacing actual media tensors with zeros must change the embedding in CoT mode.
Successful smoke results establish input/training compatibility, not retrieval
quality or complete per-record media decoding. Production preparation of all
records still needs document extraction or indexed loading and unavailable-media
exclusion; the original two-task pilot config is not automatically expanded.
