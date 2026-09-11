# Stage 1 v2: 4 x H20 configuration

This launcher is the spec-aligned Stage 1 recipe for the local 4 x H20 host:

```bash
bash scripts/elder/train_stage1_v2_4h20.sh
```

Run all model/data checks without starting training:

```bash
bash scripts/elder/train_stage1_v2_4h20.sh --dry-run
```

Run the representative one-step GPU smoke test:

```bash
bash scripts/elder/smoke_stage1_v2_4h20.sh
```

## Final defaults

| Setting | Value |
| --- | ---: |
| GPUs | 4 |
| Per-device batch | 64 |
| True distributed InfoNCE batch | 256 |
| Gradient accumulation | 1 |
| Optimizer global batch | 256 |
| Homogeneous batch per device | 8 |
| Global interleave batch | 32 |
| GradCache query/target chunk size | 4 / 4 |
| Resize max pixels | 1,003,520 |
| LoRA r / alpha | 16 / 32 |
| DDP find unused parameters | false |
| Dataloader workers | 1 |
| Learning rate | 5e-5 |
| Warmup / max steps | 100 / 5,000 |
| Checkpoint interval / retained | 500 / 3 |

The true contrastive negative pool is the current distributed microbatch. Gradient
accumulation does not enlarge it, so the previous `4 x 4` microbatch only supplied
16 examples to InfoNCE even though its optimizer batch was 256.

## Local validation

The full-resolution one-step test completed on four 95 GiB H20 GPUs with an
observed peak of about 85.3--85.9 GiB per GPU. The training step took 90.7 seconds
and the whole training call, including checkpoint work, took 97.8 seconds. A chunk
size of 8 reached about 95 GiB and OOMed, so 4 is the safe default.

The saved smoke checkpoint was loaded twice and produced identical embeddings
(`max_reload_difference = 0.0`).

## Overrides

All v2 settings have dedicated environment overrides, for example:

```bash
ELDER_STAGE1_V2_OUTPUT_DIR=/root/autodl-tmp/checkpoints/elder/my_run \
ELDER_STAGE1_V2_MAX_STEPS=100 \
bash scripts/elder/train_stage1_v2_4h20.sh
```

The default output directory is:

```text
/root/autodl-tmp/checkpoints/elder/stage1_v2_4h20_bs256_fullres
```

`ELDER_STAGE1_V2_RESUME_FROM=auto` resumes from the latest checkpoint in that
directory. Set it to `none` for an intentional fresh run in a new output directory.
