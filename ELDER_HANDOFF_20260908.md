# ELDER 项目交接文档（2026-09-08）

> 面向下一次 Codex/开发会话。本文记录的是磁盘与代码库在 2026-09-08 的实际状态，而不是理想设计状态。
>
> 首要结论：旧 Stage 1 已训练完成但出现 image embedding near-collapse，不能作为 Milestone 1 的合格 checkpoint；修复代码已经完成并通过短测，但修复后的正式 2000-step 训练尚未启动。

## 1. 项目目标与权威说明

ELDER 固定使用 `Qwen2-VL-2B-Instruct`，目标是按三个训练阶段实现：

1. Stage 1：普通 discriminative retrieval warm-up（VLM2Vec-V2 baseline）。
2. Stage 2：读取显式四阶段检索 CoT，训练 explicit teacher states。
3. Stage 3：EMA explicit-to-latent process distillation，推理时只输入原始多模态内容并执行 recurrent latent rollout。

当前权威设计文档是：

- `/root/code/ELDER/ELDER_IMPLEMENTATION_UPDATE.md`

原来的 `ELDER_IMPLEMENTATION_SPEC.md` 在工作区中已被重命名为上述文件。新会话应先完整阅读更新版。

重要不一致：更新版说明书 Stage 1 示例仍写 `lora_alpha: 32`、`global_batch_size: 256`、`max_steps: 5000`；当前为复现 released VLM2Vec-V2 行为而修复的六卡入口采用 `alpha=64`、全局候选池 1026、2000 steps。不要在未核实的情况下把说明书中的旧示例覆盖回训练脚本。

## 2. 当前阶段状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 本地 Stage 1 数据与模型 | 已准备 | MMEB-train、LLaVA-Hound train_300k、ColPali、VisRAG、本地 Qwen2-VL 均存在 |
| MMEB-V2 78 任务 | 已准备并跑通过 | 36 Image + 18 Video + 24 VisDoc |
| 旧 Stage 1 5000-step | 已完成但不是最终基线 | 输出 `stage1_full`；MMEB-V2 好于后续错误 checkpoint，但训练配置不是当前修复配置 |
| 旧六卡 Stage 1 2000-step | 已完成但判定无效 | Image 表征 near-collapse；不能用于 Stage 2 |
| Stage 1 根因诊断 | 已完成 | merge 基本等价；问题来自旧 language-side DoRA/visual fusion 训练路径 |
| Stage 1 修复代码 | 已完成 | full-model PEFT、冻结视觉塔、精确 DoRA scope、严格审计、标准 PEFT checkpoint |
| 修复后 2-step 六卡满配置 smoke | 已通过 | 显存约 54–55 GiB/卡，无 OOM |
| 修复后 20-step stress | 未完成 | 曾运行约 16 分钟后按用户要求停止，留给用户自行运行 |
| 修复后正式 2000-step | 未开始 | 正式输出目录当前为空 |
| Stage 2/3 核心训练 | 未实现 | 目前只有 CoT 生成/校验基础设施，没有 ELDERModel、explicit state trainer、recurrent student、EMA losses |
| Embed-RL 标注接入 | 仅完成盘点 | JSON/CoT 已下载，但媒体路径与现有原始数据尚未统一，当前 parser 不能直接训练 |

## 3. 服务器、环境和当前 GPU 可见性

仓库：

```text
/root/code/ELDER
```

Conda：

```text
/root/miniconda3/envs/elder      # Stage 1 / evaluation
/root/miniconda3/envs/elder-cot  # CoT generation
```

激活：

```bash
cd /root/code/ELDER
source /root/miniconda3/etc/profile.d/conda.sh
conda activate elder
```

历史上修复与 smoke 是在 6 张 NVIDIA H20（每张约 95.6 GiB）上完成的。但在 2026-09-08 盘点时，`nvidia-smi` 只显示 **1 张 H20（GPU 0）**。六卡脚本会检查 GPU 数并拒绝在一张卡上启动。正式训练前先运行：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

必须先确认当前任务实例是否重新分配了 6 张卡，不能根据历史状态假设六卡仍可见。

`.env.elder` 已存在，主要保存本地数据/模型路径。脚本读取它后会设置 `ELDER_ENV_LOADED=1`。手动传环境变量给评测脚本时，如不希望 `.env.elder` 覆盖它们，应同时显式设置 `ELDER_ENV_LOADED=1`。

## 4. 本地模型与数据

### 4.1 模型

```text
/root/autodl-tmp/models/Qwen2-VL-2B-Instruct
/root/autodl-tmp/models/GLM-4.1V-9B-Thinking
```

- Qwen2-VL-2B 是 Stage 1/2/3 固定 backbone。
- GLM-4.1V-9B-Thinking 用于离线 CoT 生成，不是 ELDER student backbone。

### 4.2 Stage 1 原始训练数据

```text
/root/autodl-tmp/datasets/MMEB-train
/root/autodl-tmp/datasets/train_video_and_instruction
/root/autodl-tmp/datasets/colpali_train_set
/root/autodl-tmp/datasets/VisRAG-Ret-Train-In-domain-data
```

Stage 1 混合配置：

```text
/root/code/ELDER/experiments/elder/stage1_full_local.yaml
```

该 YAML 当前包含 20 个 MMEB 子任务、ColPali、VisRAG、LLaVA-Hound caption/video/QA。数据加载采用 interleave：按权重选择数据源，每个数据源内部顺序读取并在耗尽后循环。

### 4.3 MMEB-V2

```text
/root/autodl-tmp/datasets/MMEB-V2
```

盘点大小约 138 GiB，已经补齐 78 任务需要的 image-query、video metadata 和 VisDoc 数据，完整评测曾成功跑完。

评测配置：

```text
/root/code/ELDER/experiments/elder/mmeb_v2/image.yaml
/root/code/ELDER/experiments/elder/mmeb_v2/video.yaml
/root/code/ELDER/experiments/elder/mmeb_v2/visdoc.yaml
```

### 4.4 Embed-RL-Train（新下载）

```text
/root/autodl-tmp/datasets/Embed-RL-Train/MMEB-train-contrastive-learning
```

状态：

- 总大小约 4.8 GiB。
- 39 个已解压 JSON。
- 共约 1,684,211 条记录。
- 全部记录的 `error` 字段为空字符串。
- 标注含 `query_cot`、`pos_cot`；Embed-RL CoT 是 `<thinking> / <rethink> / <answer>` 风格，不等同于 ELDER 要求的四阶段 schema。

来源统计：

| 来源 | 样本数 |
|---|---:|
| MMEB | 720,844 |
| LLaVA-Hound caption retrieval | 284,721 |
| LLaVA-Hound video retrieval | 260,410 |
| LLaVA-Hound QA | 274,006 |
| ViDoRe | 83,964 |
| VisRAG | 60,266 |
| 合计 | 1,684,211 |

LLaVA-Hound 媒体引用：

| 类型 | train_300k | train_600k |
|---|---:|---:|
| caption retrieval | 95,170 | 189,551 |
| video retrieval | 87,194 | 173,216 |
| video QA | 274,006 | 0 |
| 合计 | 456,370 | 362,767 |

当前服务器只确认有 `train_300k` 帧；要完整使用 Embed-RL 的 LLaVA-Hound 标注，还缺 `train_600k` 对应媒体（影响 362,767 条）。

此外 Embed-RL JSON 不包含媒体本体：

- MMEB 的 `images/...` 可映射到 `/root/autodl-tmp/datasets/MMEB-train/images/...`。
- ViDoRe JSON 引用抽取后的 `images/Vidore/*.jpeg`，而当前原始 ColPali 数据是 Arrow shards。
- VisRAG JSON 引用抽取后的 `images/VisRAG/*.jpeg`，而当前原始数据是 Parquet shards。
- 当前仓库没有 Embed-RL 专用 dataset parser。

因此下一步不能直接把该目录加入 Stage 1/2 YAML；应先实现只读检查/转换 manifest、媒体映射或媒体抽取，并把三段式 CoT 转成 ELDER 四阶段格式后再用于 Stage 2/3。

## 5. 历史 Stage 1 训练与评测

### 5.1 `stage1_full`（5000 steps）

路径：

```text
/root/autodl-tmp/checkpoints/elder/stage1_full
```

训练日志最终值：

```text
train_runtime = 190765.4576 s = 52:59:25
train_samples_per_second = 6.71
train_steps_per_second = 0.026
train_loss = 12.247168016245961
epoch = 1.0
```

MMEB-V2 完整评测：

```text
/root/autodl-tmp/eval/elder/stage1_mmeb_v2/20260808_174148
```

按官方类别指标做宏平均（Image/Video 用 Hit@1，VisDoc 用 `ndcg_linear@5`）：

| 部分 | 任务数 | 平均分 |
|---|---:|---:|
| Image | 36 | 44.35 |
| Video | 18 | 27.12 |
| VisDoc | 24 | 63.96 |
| All | 78 | 46.40 |

该结果比后续错误 checkpoint-2000 更好，但它不是当前精确修复后的 VLM2Vec-V2 baseline，不能替代重新训练和验收。

### 5.2 旧六卡 `checkpoint-2000`（判定无效）

```text
/root/autodl-tmp/checkpoints/elder/stage1_v2_6h20_bs1026_step2k/checkpoint-2000
```

旧训练使用了错误的 hybrid 结构：仅给 `encoder.model` 包 PEFT，外层 Qwen2-VL visual tower 没有被 PEFT 冻结；训练时约有 684,381,184 个可训练参数。DoRA 还采用 alpha 32，并包含过宽的 MLP targets（包括 gate/up）。checkpoint 含约 4.49 GB `model.safetensors`、约 76.5 MB adapter 和约 2.81 GB optimizer。

完整 MMEB-V2 评测：

```text
/root/autodl-tmp/eval/elder/stage1_mmeb_v2/20260815_085328
```

| 部分 | 任务数 | 平均分 |
|---|---:|---:|
| Image | 36 | 20.58 |
| Video | 18 | 30.61 |
| VisDoc | 24 | 58.20 |
| All | 78 | 34.47 |

Image 明显退化，部分分类任务接近随机；虽然 RefCOCO-Matching Hit@1 曾达到 0.789，但不能据此否定整体 collapse。

诊断报告：

```text
/root/autodl-tmp/eval/elder/checkpoint_diagnostics/20260815_111727/report.json
/root/autodl-tmp/eval/elder/dora_diagnostics/checkpoint2000/report.json
```

关键结论：

- full merged 与 unmerged 的 query row cosine 均值约 0.99966，Top-1 agreement 1.0：merge 不是主要故障。
- base 和 visual-only 没有 near-collapse。
- dora-only、full merged、full unmerged 均 near-collapse。
- ImageNet-R 64-query 样本中，旧 DoRA-only query mean off-diagonal cosine 约 0.9458。
- DoRA 消融中 all 约 0.9460，MLP-only 约 0.9241；结论指向旧 language DoRA/visual fusion 训练配置，而不是评测加载或 merge。

**不要从这个旧 checkpoint-2000 恢复修复后的正式训练，也不要用它初始化 Stage 2。**

## 6. 已完成的 Stage 1 修复

主要改动仍在 dirty worktree 中，未提交：

1. `src/arguments.py`
   - 新增 `lora_adapter_scope={auto,full_model,language_model}`。
   - 新增 `strict_stage1_dora`。
2. `src/model/model.py`
   - `full_model` 模式对完整 Qwen2-VL conditional-generation model 调用 `get_peft_model()`。
   - PEFT 自动冻结所有非 target base 参数，包括 visual tower。
3. `src/elder/stage1.py`
   - 新增严格 DoRA 参数审计。
   - 实际 Qwen2-VL targets 必须为 `q_proj,k_proj,v_proj,o_proj,down_proj`。
   - 禁止 `gate_proj,up_proj` 可训练。
   - 预期 trainable parameters 精确为 9,203,712。
4. `train.py`
   - strict 模式训练前 fail-fast，并在 output dir 写 `stage1_trainable_parameters.json`。
5. `src/trainer.py`
   - checkpoint resume 加载后再次严格审计。
   - 标准 full-model PEFT 不再误判为 legacy hybrid，不再额外保存 4.45 GB full state。
   - 保留旧 hybrid checkpoint 的只读兼容加载。
6. `scripts/elder/eval_stage1_mmeb_v2_4h20.sh`
   - 支持新的标准 PEFT adapter checkpoint，同时兼容旧 hybrid。
7. `scripts/elder/audit_stage1_v2_dora.py`
   - 可独立构建本地 Qwen2-VL 并核对精确 trainable scope。

当前修复后的实际配置：

```text
DoRA rank                     16
DoRA alpha                    64
DoRA dropout                  0.1
实际 target modules           q_proj,k_proj,v_proj,o_proj,down_proj
adapter scope                 full_model
visual tower                  frozen
gate_proj/up_proj             frozen
trainable parameters          9,203,712
per-device batch              171
world size                    6
global candidate pool         1,026
homogeneous batch/device      11
global homogeneous chunk      66
GradCache q/p chunk           8 / 8
resize_max_pixels             1,003,520
max steps                     2,000
learning rate                 5e-5
warmup steps                  100
save interval                 250
save_total_limit              5
ddp_find_unused_parameters    false
ignore_data_skip              false
fresh-run default             resume_from=none
```

注意：脚本 `lora_target_modules` 字符串保留了跨架构 released target 名称（如 `qkv_proj`、`gate_up_proj`），但 Qwen2-VL-2B 实际匹配且允许训练的只有上述五类；strict audit 会检查实际参数并拒绝 gate/up。

## 7. 已完成验证与未完成验证

已通过：

- 独立真实模型 DoRA 审计：9,203,712 trainable，visual/gate/up frozen。
- 六卡 2-step 满配置 smoke：
  - step loss 约 22.4631、19.8471；
  - 平均约 126.71 s/step；
  - 六卡 GPU 利用率接近 100%；
  - 显存约 53.6–54.7 GiB/卡，距离 95 GiB 上限有较大余量；
  - 无 OOM。
- 小 batch checkpoint save：新格式只保存约 36.9 MB adapter，不再保存 4.45 GB full model/hybrid metadata。
- 标准 checkpoint 确定性加载：两次加载 embedding 最大差异 0。
- 六卡 checkpoint-1 -> step 2 resume：正确识别 global step 1 并只训练下一步。
- MMEB-V2 评测入口对标准 PEFT checkpoint 的 dry-run。
- `tests/elder/test_stage1_baseline.py`：13 passed。
- shell syntax、Python compile 和 `git diff --check` 曾通过。

未完成：

- 20-step 六卡满配置 stress 没有跑完。曾稳定运行约 16 分钟，显存未爬升，但之后按用户要求安全终止。
- 修复后的 2000-step 正式训练未开始。
- 修复 checkpoint 的 MMEB-V2 分数尚不存在。
- batch integrity audit 报告当前为 failed：

```text
/root/autodl-tmp/eval/elder/batch_audits/stage1_6h20/report.json
```

该报告 3 个 batch 中 2 pass、1 fail；失败 batch 的 66 行被标记为 `empty positive`，同时数据源恰含 66 个 VisRAG 和 66 个 DocVQA homogeneous rows。必须先核对审计脚本是否把“空文本但有 positive image”的合法 text-to-image 样本误判为空，还是 parser/collator 真丢失了正样本。不要忽略该报告，也不要未经确认就认定训练数据坏了。

## 8. 下一步建议（严格顺序）

### P0：完成修复后 Stage 1 的最终前置验证

1. 确认 6 张 GPU 实际可见且空闲。
2. 修复或解释 batch integrity audit 的 66 个 `empty positive`。
3. 重新运行 20-step 满配置 stress，确认全部 loss/grad norm 有限且无显存累积。
4. 保留一份 stress 日志供交接，不需要保存大 checkpoint。

20-step 命令：

```bash
cd /root/code/ELDER
source /root/miniconda3/etc/profile.d/conda.sh
conda activate elder

ELDER_STAGE1_V2_6H20_SMOKE_OUTPUT_DIR=/tmp/elder_stage1_v2_6h20_stress \
ELDER_STAGE1_V2_6H20_SMOKE_MAX_STEPS=20 \
bash scripts/elder/smoke_stage1_v2_6h20.sh 2>&1 | \
  tee /tmp/elder_stage1_v2_6h20_stress.log
```

### P1：从 base 全新训练修复后的 Stage 1

不要 resume 旧目录。正式入口默认 `resume_from=none`，输出到新目录：

```text
/root/autodl-tmp/checkpoints/elder/stage1_v2_fixed_6h20_bs1026_step2k
```

tmux：

```bash
tmux new -s elder-stage1-fixed
cd /root/code/ELDER
source /root/miniconda3/etc/profile.d/conda.sh
conda activate elder

bash scripts/elder/train_stage1_v2_6h20.sh 2>&1 | \
  tee /root/autodl-tmp/checkpoints/elder/stage1_v2_fixed_6h20_bs1026_step2k/train.log
```

启动日志必须出现：

```text
Strict Stage 1 V2 DoRA audit passed: trainable=9,203,712
Number of trainable parameters = 9,203,712
```

如果不是这个数字，立即停止。

正确新目录产生 checkpoint 后，如需恢复：

```bash
ELDER_STAGE1_V2_6H20_RESUME_FROM=auto \
bash scripts/elder/train_stage1_v2_6h20.sh 2>&1 | \
  tee -a /root/autodl-tmp/checkpoints/elder/stage1_v2_fixed_6h20_bs1026_step2k/train.log
```

`ignore_data_skip=false` 会为保持 seeded data-source sequence 而重放并跳过已训练 batches；恢复 500/1000 步时数据解析可能很慢，但序列可复现。不要再临时改成 `true`，除非明确接受样本序列变化。

### P2：分阶段诊断并跑 MMEB-V2

建议在 checkpoint-250 首先跑 ImageNet-R/小型 image diagnostics，避免训练 2000 步后才发现 collapse。至少在 250、500、1000、1500、2000 检查 embedding dispersion；最终 checkpoint 再跑完整 78 任务。

评测结果默认写入：

```text
/root/autodl-tmp/eval/elder/stage1_mmeb_v2/<timestamp>/
```

评测入口默认是 4 卡。若只有 1 卡：

```bash
ELDER_ENV_LOADED=1 \
ELDER_CUDA_VISIBLE_DEVICES=0 \
ELDER_MMEV2_EXPECTED_GPU_COUNT=1 \
ELDER_STAGE1_CHECKPOINT=/path/to/corrected/checkpoint \
bash scripts/elder/eval_stage1_mmeb_v2_4h20.sh --modality image
```

完整 78 任务应使用：

```bash
ELDER_STAGE1_CHECKPOINT=/path/to/corrected/checkpoint \
bash scripts/elder/eval_stage1_mmeb_v2_4h20.sh --modality all
```

正式 Milestone 1 验收至少比较 base、旧 `stage1_full` 与 corrected Stage 1 的：Image Hit@1、Video Hit@1、VisDoc NDCG@5、All macro average、embedding off-diagonal cosine/dispersion、InfoNCE 或正负 margin。

### P3：整理 Embed-RL 数据，而不是直接训练

建议新增独立 preflight/converter，完成：

1. 流式读取 39 个大 JSON，禁止一次性加载全部 4.8 GiB。
2. 标准化 MMEB、LLaVA、ViDoRe、VisRAG 的 schema。
3. 对每条样本解析 query/candidate modality，并验证媒体存在。
4. 对 ViDoRe Arrow、VisRAG Parquet 建立行 ID 到抽取 JPEG 的可复现映射。
5. 决定下载 LLaVA `train_600k`，或明确过滤 362,767 条缺媒体记录。
6. 把 Embed-RL 三段式 CoT 转换/重新生成成 ELDER 四阶段 schema；执行 target leakage 与 `cot_step_mask` 校验。
7. 输出 manifest，不修改原始 Embed-RL JSON。

### P4：实现 Stage 2，再实现 Stage 3

只有 corrected Stage 1 通过 MMEB-V2 验收后再推进：

- Stage 2：special `<STEP_1...4>` tokens、structured CoT collator、真实 token positions/masks、explicit teacher states、final/intermediate retrieval losses、query-only candidate fallback、独立 checkpoint/resume。
- Stage 3：student recurrent rollout、frozen teacher debug、state/transition losses、rank distillation、EMA update、cached-graph recurrent gradient 测试、只导出 student 推理 checkpoint。

目前 `src/elder/cot.py` 和 `scripts/elder/generate_cot.py` 等只覆盖 CoT 构建/校验，不代表 Stage 2/3 训练已经实现。

## 9. 关键脚本与诊断入口

```text
scripts/elder/train_stage1_v2_6h20.sh          # 修复后的六卡正式训练
scripts/elder/smoke_stage1_v2_6h20.sh          # 可覆盖步数的 smoke
scripts/elder/audit_stage1_v2_dora.py          # 精确 trainable scope 审计
scripts/elder/audit_stage1_batch_integrity.py  # 真实 batch 完整性审计
scripts/elder/diagnose_stage1_checkpoint.py    # merge/base/visual/DoRA 诊断
scripts/elder/diagnose_dora_ablation.py        # DoRA 模块/层级消融
scripts/elder/eval_stage1_mmeb_v2_4h20.sh      # MMEB-V2 评测
scripts/elder/check_mmeb_v2.py                 # 78 任务 readiness
scripts/elder/generate_cot.py                  # GLM CoT generation
scripts/elder/validate_cot.py                  # ELDER CoT schema 校验
```

快速测试：

```bash
cd /root/code/ELDER
conda run -n elder pytest -q tests/elder/test_stage1_baseline.py
bash -n scripts/elder/train_stage1_v2_6h20.sh \
  scripts/elder/smoke_stage1_v2_6h20.sh \
  scripts/elder/eval_stage1_mmeb_v2_4h20.sh
git diff --check
```

独立 DoRA 审计：

```bash
conda run -n elder python scripts/elder/audit_stage1_v2_dora.py \
  --model-path /root/autodl-tmp/models/Qwen2-VL-2B-Instruct \
  --output /tmp/elder_stage1_v2_dora_audit.json
```

## 10. Git/worktree 注意事项

工作区很脏，包含用户此前的大量 staged、unstaged 和 untracked 修改。没有做 commit，也不能通过 reset/checkout 清理。开始下一项工作前应先执行：

```bash
git status --short
git diff --check
```

尤其注意：

- `train.py`、`src/trainer.py` 同时有 staged/unstaged 变化（状态 `MM`）。
- 若干 `scripts/elder/*` 是 `AM`。
- `scripts/elder/eval_stage1_mmeb_v2_4h20.sh.orig` 和 `scripts/elder/train_stage1_v2_6h20.sh.orig` 是未跟踪备份文件，删除前先确认用户是否需要。
- `/tmp/elder_stage1_v2_*` 下仍有若干小型 smoke audit JSON；没有正式模型 checkpoint。
- 修复后的正式目录 `/root/autodl-tmp/checkpoints/elder/stage1_v2_fixed_6h20_bs1026_step2k` 当前大小为 0。

不要删除旧 checkpoints/评测结果，除非用户明确授权；它们仍是故障复现与 ablation 对照。

## 11. 给下一会话的最短行动清单

1. 阅读本文与 `ELDER_IMPLEMENTATION_UPDATE.md`。
2. `git status --short`，不要覆盖 dirty worktree。
3. `nvidia-smi`，确认到底有几张 GPU。
4. 排查 batch integrity audit 的 66 条 `empty positive` 是否为合法 image-only positive 误报。
5. 六卡可用时跑完 20-step full stress。
6. 从 Qwen2-VL base 启动新的 corrected 2000-step Stage 1，绝不 resume 旧 checkpoint-2000。
7. 在 checkpoint-250 提前做 image-collapse 诊断；最终跑完整 MMEB-V2。
8. corrected Stage 1 通过后，先做 Embed-RL manifest/media/CoT 转换，再开始 Stage 2 实现。

