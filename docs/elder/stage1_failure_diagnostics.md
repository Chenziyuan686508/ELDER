# Stage 1 failure-localization diagnostics

These tools are read-only. They do not modify checkpoints, adapter files, model
weights, or training datasets.

## 1. DoRA module and layer ablation

[diagnose_dora_ablation.py](/root/code/ELDER/scripts/elder/diagnose_dora_ablation.py)
loads the saved adapter in memory and disables every adapter module outside a
chosen profile. The profiles are:

- base: no adapter.
- all: every saved DoRA module.
- attention: q_proj, k_proj, v_proj, o_proj.
- mlp: gate_proj, up_proj, down_proj.
- lower, middle, upper: layers 0-9, 10-18, and 19-27 respectively.

It measures retrieval, positive-vs-hard-negative margin, Top-1 concentration,
and embedding dispersion. With --trace-hidden-states, it additionally records
the same EOS-pooled representation after every decoder layer for base and all.

~~~bash
cd /root/code/ELDER
conda activate elder

CUDA_VISIBLE_DEVICES=0 python scripts/elder/diagnose_dora_ablation.py \
  --base-model /root/autodl-tmp/models/Qwen2-VL-2B-Instruct \
  --checkpoint /root/autodl-tmp/checkpoints/elder/stage1_v2_6h20_bs1026_step2k/checkpoint-2000 \
  --data-root /root/autodl-tmp/datasets/MMEB-V2 \
  --dataset ImageNet-R \
  --sample-count 64 \
  --batch-size 8 \
  --profiles base,all,attention,mlp,lower,middle,upper \
  --trace-hidden-states \
  --output /root/autodl-tmp/eval/elder/dora_diagnostics/manual_run/report.json
~~~

Interpretation:

- attention near-collapse: DoRA disrupts image-token attention fusion.
- mlp near-collapse: DoRA disrupts the language representation transform.
- only upper near-collapse: the final language layers and EOS representation
  are the first place to restrict or freeze.
- neither component alone collapses but all does: the joint adapter update is
  too strong.

## 2. Real Stage 1 batch integrity audit

[audit_stage1_batch_integrity.py](/root/code/ELDER/scripts/elder/audit_stage1_batch_integrity.py)
uses the same Stage 1 YAML mixture, parser registration, processor, collator,
and GradCache splitter as train.py. It reads a few real batches and checks:

- every row has a non-empty query and positive;
- expected image/video inputs survive preprocessing;
- query/positive metadata remain in the same order after GradCache chunking;
- local batch dimensions match on both retrieval sides;
- equal-size DDP gather uses diagonal global InfoNCE labels.

~~~bash
python scripts/elder/audit_stage1_batch_integrity.py \
  --base-model /root/autodl-tmp/models/Qwen2-VL-2B-Instruct \
  --batches 3 \
  --batch-size 171 \
  --world-size 6 \
  --interleave-global-chunk 66 \
  --gc-q-chunk-size 3 \
  --gc-p-chunk-size 3 \
  --output /root/autodl-tmp/eval/elder/batch_audits/manual_run/report.json
~~~

It may spend several minutes loading metadata and images, but uses no GPU. The
audit reads .env.elder by default to resolve the local dataset paths.

Inspect either command first without loading data or a model:

~~~bash
python scripts/elder/audit_stage1_batch_integrity.py \
  --base-model /root/autodl-tmp/models/Qwen2-VL-2B-Instruct \
  --output /tmp/elder_batch_audit.json \
  --dry-run
~~~

