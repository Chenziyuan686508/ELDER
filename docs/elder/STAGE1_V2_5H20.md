# Stage 1 v2: 5 x H20 candidate-pool configuration

## Recommended production recipe

```bash
bash scripts/elder/train_stage1_v2_5h20.sh
```

| Setting | Value |
| --- | ---: |
| GPUs | 5 x NVIDIA H20 (95 GiB) |
| Per-device batch | 205 |
| DDP InfoNCE candidate pool | 1,025 |
| Negatives per query | 1,024 |
| Gradient accumulation | 1 |
| Optimizer global batch | 1,025 |
| Homogeneous batch per device | 13 |
| Global interleave batch | 65 |
| GradCache query/target chunk size | 4 / 4 |
| Resize max pixels | 1,003,520 |
| LoRA/DoRA r / alpha | 16 / 32 |
| DDP find unused parameters | true |
| Dataloader workers | 1 |
| Learning rate | 5e-5 |
| Warmup / max steps | 100 / 2,000 |
| Per-rank batch composition logs | disabled |
| Checkpoint interval / retained | 500 / 3 |

The official 8-GPU recipe uses a candidate pool of 1,024. Equal local batch
sizes are required by the current fixed-shape all-gather implementation, so
`5 * 205 = 1,025` differs from the official pool by only one candidate.

`find_unused_parameters=true` is required for correctness. The mixed-modality
batch is split into GradCache chunks, and some chunks conditionally bypass
parameters. A real run with the flag disabled failed during DDP reduction even
though other chunks can emit PyTorch's "no unused parameters" warning.

## Local validation

The five GPUs are fully connected to one another through NV18 links. A
representative full-resolution batch contained image, video, visual-document,
and MMEB sources.

Results for the final 1,025-candidate one-step run:

- training time: 274.9 seconds;
- throughput: 3.728 samples/second;
- peak memory: approximately 85,643--85,991 MiB (83.64--83.98 GiB) per GPU.

Results for the 1,020-candidate two-step stability run with two different mixed batches:

- total training time: 467.2 seconds;
- average time: 233.6 seconds/step;
- throughput: 4.367 samples/second;
- peak memory: approximately 88,343--88,425 MiB (86.27--86.35 GiB) per GPU;
- both optimizer updates completed without OOM or NCCL failure.

The smoke launcher disables Trainer checkpointing and final model export, so it
does not create the approximately 12 GiB artifact produced by the older smoke
test:

```bash
bash scripts/elder/smoke_stage1_v2_5h20.sh
```

## Expected duration and storage

At the measured steady-state range, 2,000 steps are expected to take roughly
4.5--6.5 days. The run processes 2.05 million query/candidate pairs, closely
matching the approximately 2.048 million pair exposures in the paper's
1,024-candidate, 2,000-step setup. Actual time depends
on the sampled ratio of videos and high-resolution documents.

The production output directory is:

```text
/root/autodl-tmp/checkpoints/elder/stage1_v2_5h20_bs1025_step2k
```

Checkpoints are written every 500 steps and the latest three are retained. The
launcher uses `resume_from=auto`, so rerunning the same command resumes from the
latest valid checkpoint.

## Preflight and overrides

Run all checks without starting workers:

```bash
bash scripts/elder/train_stage1_v2_5h20.sh --dry-run
```

Dedicated `ELDER_STAGE1_V2_5H20_*` environment variables can override settings.
Set `ELDER_STAGE1_V2_5H20_LOG_BATCH_COMPOSITION=1` only when the per-rank
composition of the first five local batches is needed for debugging.
For example, a 5,000-step extended run can be launched with:

```bash
ELDER_STAGE1_V2_5H20_MAX_STEPS=5000 \
bash scripts/elder/train_stage1_v2_5h20.sh
```
