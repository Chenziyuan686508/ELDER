# Teacher pilot handoff (2026-09-08)

Current authority: `教师模型训练执行说明.md`, superseding the earlier fixed-four-stage recurrent ELDER design for this task. The planned later student uses one-pass latent tokens; this implementation only trains the teacher.

Code: `teacher_pipeline/`; config: `configs/teacher_pilot.yaml`.
Artifacts: `/root/autodl-tmp/eval/elder/teacher_pilot/`.
Use `/root/miniconda3/envs/elder/bin/python`; generation uses `elder-cot/bin/python`.

Completed so far:

- Official released VLM2Vec-V2.0 DoRA downloaded at revision `e39ff079b8275ef876d3656da8c0bddbff3c4dde`; local Qwen2-VL-2B base fingerprinted.
- Released DoRA merged into shared base; fresh rank-32 language LoRA, 36,929,536 trainable parameters, visual tower/merger frozen. Native last valid content token pooling, no new special tokens.
- Full scans of MSCOCO_i2t (46,596) and Visual7W (41,677); source SHA256 matches specified Embed-RL revision. Real media decoded and verified. This is not a scan of all 39 JSONs.
- Fixed 2048-pair balanced pilot; 200 held-out queries/task, full filtered held-out candidate corpora of 1887 and 481. Grouped media split; source pretrained checkpoint may have seen these training sources.
- CoT provenance remains unknown. Candidate CoT conservative lexical filtering + stable canonical selection; see data_audit.json and review notes for limits.
- Ten tests passing; real-image CoT visibility, checkpoint reload and mixed inputs checked. A/B smoke each 10 steps, 4-query pools explicitly labeled.
- A/B pilot each 32 steps, real 64-query GradCache pool, deduplicated candidates. B paused at step 10 and resumed. Both final checkpoint reload errors 0.
- A/B compute-only times: 849.49 / 883.07 seconds; token counts 512335 / 1053880; peak allocated GPU memory 5.604 / 5.838 GiB.
- Base Hit@1: MSCOCO 68.5%, Visual7W 50.0%; A: 70.0%, 63.5%; B-off: 70.0%, 49.0%.
- Adapter base references repaired with `python -m teacher_pipeline.package`; deploy/copy shared `common/merged_base` together with the chosen adapter. Do not load the teacher adapter on raw Qwen.

Active work at last update:

- `python -m teacher_pipeline.run_remaining` is running, logs in `run_remaining.log`.
- Its child is generating independent endpoint-only GLM CoT for 2768 endpoints (2368 complete-corpus candidates + 400 queries). Text batch 64, image batch 8, greedy, max_new_tokens 512, max_pixels 1003520. First batches passed, no failures; outputs include completed natural reasoning and answer text. Check `generate_eval_cot.log` and `eval_cot/<active-version>/progress.json`.
- After generation, the supervisor automatically runs B normal evaluation, paired bootstrap, optional two-sample process export with causality probe, and report. It stops on any failure.
- Do not launch another GPU job concurrently or rerun prepare/preflight; A/B manifests are frozen and guarded.
- `teacher_report.md` under artifact root is an evidence-only progress report; it must be refreshed after completion.

Remaining verification/finalization:

1. Monitor generation completion and handle actual errors; never replace normal CoT with training/privileged CoT.
2. Confirm B evaluation and process export succeed; inspect B-A paired task intervals and errors. No benefit is a valid outcome.
3. Finish generation provenance/cap-rate/cost audit and source-code fingerprints; update report and this handoff with final facts.
4. Run final focused tests if needed after fixes, compile and diff check. Do not commit/reset or overwrite preexisting dirty worktree changes.
5. Report concise Chinese outcome with actual metrics, artifacts, and any incomplete checks.

Optional export lives in `teacher_pipeline/process_export.py`; disabled by default and uses sentence-near token boundaries, last-layer cumulative states, masks rather than copied padding states. It is not student training and does not establish interpretable reasoning.
