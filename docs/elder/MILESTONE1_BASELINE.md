# ELDER Milestone 1：VLM2Vec-V2 Baseline

本阶段只建立判别式多模态检索基线，不包含 CoT、`<STEP>`、recurrent
latent rollout 或 EMA teacher。

## 上游版本

- Repository: `https://github.com/TIGER-AI-Lab/VLM2Vec`
- Branch: `main`
- Commit: `2638a8413fda4b98668a29ea763b4898814bfea7`
- Imported: 2026-07-31

主分支同时包含 MMEB-V3 扩展，但本阶段只使用其中的 VLM2Vec-V2
Qwen2-VL、ViDoRe 数据加载、InfoNCE、GradCache 和训练入口。

## 本地资源

默认配置使用：

```text
Model:
/data/chenziyuan/models/Qwen2-VL-2B-Instruct

Debug dataset shard:
/data/chenziyuan/datasets/colpali_train_set/data/train-00000-of-00082.parquet
```

模型和数据路径不会提交到 Git。可以复制路径模板：

```bash
cp experiments/elder/local_paths.env.example .env.elder
source .env.elder
```

## 环境安装

```bash
conda activate elder

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements/elder-stage1.txt
```

Qwen2-VL 的官方训练路径使用 FlashAttention 2。其 wheel 必须和 Python、
PyTorch 及 CUDA ABI 匹配，不在通用 requirements 中自动编译。本机当前使用：

```text
flash-attn==2.7.3
Python 3.10
PyTorch 2.6.0+cu124
```

## 预检

预检不会加载 2B 参数到 GPU，也不会访问网络。它验证：

- 权重索引、分片和 safetensors 元数据；
- Qwen2-VL config、tokenizer 和 processor；
- text-only collator；
- ViDoRe/ColPali image-text 数据变换与 collator；
- InfoNCE 的反向传播。

```bash
conda activate elder
bash scripts/elder/run_stage1_preflight.sh
```

## Debug 训练

默认运行 10 steps、batch size 2、LoRA rank 16 和 GradCache：

```bash
conda activate elder
source .env.elder
bash scripts/elder/train_stage1_debug.sh
```

可以临时覆盖：

```bash
ELDER_CUDA_VISIBLE_DEVICES=5 \
ELDER_MAX_STEPS=1 \
ELDER_BATCH_SIZE=2 \
bash scripts/elder/train_stage1_debug.sh
```

训练输出默认写到 `/data/chenziyuan/checkpoints/elder/stage1_debug`。

checkpoint 保存完成后可验证两次加载的一致性：

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/elder/stage1_checkpoint_smoke.py \
  --checkpoint /data/chenziyuan/checkpoints/elder/stage1_smoke_v2
```

## 128-pair 检索验收

Stage 1 的下一道门槛是在固定的前 128 个 ViDoRe/ColPali pairs 上执行：

1. 评测未训练的 Qwen2-VL-2B；
2. 训练 100 steps；
3. 加载训练 checkpoint 并评测同一批 pairs；
4. 比较双向 Recall@1/5/10、InfoNCE、embedding norm 和正负样本间隔。

这是 correctness-first 的小规模过拟合测试，不代表测试集泛化性能。验收要求
query-to-candidate InfoNCE 下降、Recall@1 不下降且高于随机水平的 3 倍，同时
归一化 embedding 的平均范数维持在 `1±0.05`。

训练统一放在 `tmux` 中。启动脚本会拒绝显存使用超过 4096 MiB 或利用率超过
10% 的 GPU，防止误占用其他任务：

```bash
cd /home/chenziyuan/code/ELDER
bash scripts/elder/start_stage1_acceptance_tmux.sh GPU编号
```

启动后脚本会打印 session、日志和输出目录。查看实时输出：

```bash
tmux attach -t SESSION名称
```

按 `Ctrl-b`，再按 `d` 可退出界面但保留训练。最终文件位于：

```text
/data/chenziyuan/checkpoints/elder/stage1_acceptance/<run-id>/
├── status.txt
├── stage1_acceptance.log
├── train/
└── metrics/
    ├── before.json
    ├── after.json
    └── acceptance.json
```

只有明确确认可以共享一张繁忙 GPU 时，才允许使用
`ELDER_ALLOW_BUSY_GPU=1` 跳过保护；默认不使用该选项。

## 验收顺序

1. `python -m pytest -q tests/elder`
2. Stage 1 preflight 返回 `"status": "ok"`
3. 单 GPU 1-step 训练完成并写出 checkpoint
4. `stage1_checkpoint_smoke.py` 两次加载结果一致
5. 128–1024 pairs 上 loss 可下降
6. 再扩展到完整 VLM2Vec-V2 配方与 8×L40S

## 当前验证记录（2026-07-31）

- 单元测试：9 passed；
- 只读 preflight：通过；
- Qwen2-VL-2B + ViDoRe/ColPali 单 GPU 1-step：通过；
- step 1 loss：3.4853，gradient norm：2204.61；
- checkpoint 两次加载：embedding shape `[2, 1536]`，最大差异 `0.0`；
- 从 `checkpoint-1` 自动恢复并训练至 step 2：通过；
- 有效 smoke checkpoint：
  `/data/chenziyuan/checkpoints/elder/stage1_smoke_v2`。

为使官方 V2 代码在当前单 GPU debug 配方中真正可运行，本阶段修复了：

- YAML 中 `dataset_path` 等本地路径的环境变量展开；
- ViDoRe `neg_image` 的嵌套 feature schema；
- collator 对未返回 hard negatives 的无效解码；
- 单 GPU 下 `--grad_cache true` 未生效；
- 混合视觉全量权重与语言 LoRA checkpoint 缺少 resume 元数据。
