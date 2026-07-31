# ELDER 项目实现说明书

> **项目名称**：ELDER — Explicit-to-Latent Process Distillation for Efficient Multimodal Retrieval  
> **文档用途**：交给 Codex 作为实现依据。本文档优先描述工程接口、训练流程和验收标准，而不是论文写作。  
> **推荐基础代码库**：在官方 `TIGER-AI-Lab/VLM2Vec` V2 代码上增量开发，不建议从零搭建数据和评测框架。  
> **固定主干模型**：`Qwen/Qwen2-VL-2B-Instruct`。Stage 1、Stage 2、Stage 3 以及最终推理均使用同一主干模型；当前项目不进行跨 backbone 或跨模型规模切换。

---

## 1. 项目目标

ELDER 的目标是训练一个通用多模态 embedding 模型，使其在不生成显式自然语言 CoT 的情况下，通过固定的 \(K\) 次 recurrent hidden-state feedback 完成 latent reasoning，并输出用于检索的单向量 embedding。

训练阶段使用显式 CoT 作为 **privileged process supervision**：

```text
强 MLLM 离线生成分阶段 CoT
             ↓
EMA Teacher：输入 + 显式 CoT → 阶段状态 t1 ... tK
             ↓ 逐阶段蒸馏
Student：输入 → z1 → z2 → ... → zK
             ↓
         Retrieval Embedding
```

推理阶段只保留 Student：

```text
多模态输入
   ↓ prefix prefill
latent seed → z1 → z2 → ... → zK
   ↓
L2-normalized embedding
```

Query 和 Candidate 默认采用**对称编码结构**。Candidate 的 latent reasoning 在建库阶段离线执行，在线检索时只需编码 Query。

---

## 2. 核心设计原则

### 2.1 训练和推理解耦

以下组件只在训练期间存在：

- 强 MLLM；
- 预生成显式 CoT；
- `<STEP>` 锚点；
- Explicit Teacher 分支；
- EMA 参数副本；
- state、transition、rank distillation losses。

部署模型只包含：

- 多模态 backbone；
- latent seed；
- feedback adapter；
- recurrent latent rollout；
- retrieval head。

### 2.2 显式 CoT 不是推理时输入

显式 CoT 仅用于构造教师状态。Student 在训练和测试时都不能读取显式 CoT，否则会产生训练—推理不一致。

### 2.3 Latent state 是动态的

模型保存的是共享的 `latent_seed` 和模型参数，而不是固定的推理内容。对不同输入 \(x\)，得到的 \(z_i(x)\) 必须不同。

### 2.4 不生成词表 token

Latent rollout 不经过 LM head，不做 vocabulary softmax、sampling、beam search 或文本解码。每一步直接把连续 hidden state 映射为下一步输入 embedding。

### 2.5 默认不切断 recurrent 梯度

主方法中，`z_i` 反馈到下一步时不能 `detach()`。必须允许最终检索损失和后续阶段损失反向传播到早期 latent states。`detach_feedback=true` 只作为节省显存的消融选项。

---

## 3. 总体架构

### 3.1 统一编码接口

ELDER 对任意文本、图片、视频、视觉文档或交错多模态输入提供统一接口：

```python
embedding = model.encode_latent(batch, num_latent_steps=K)
```

输出：

```python
{
    "embedding": Tensor[B, D],
    "latent_states": Tensor[B, K, H],   # 可选，用于训练和分析
}
```

其中：

- `H` 为 backbone hidden size；
- `D` 为 embedding dimension；
- 默认 `retrieval_head=identity`，即 `D=H`，以保持与 VLM2Vec 基线一致；
- 可选线性 projection 到 1024/2048 维。

### 3.2 Student recurrent latent encoder

对输入 \(x\)：

1. 使用正常的 processor 和 MLLM forward 完成多模态 prefix prefill；
2. 加入可学习连续向量 `latent_seed`；
3. 得到当前 latent hidden state；
4. 使用 `feedback_adapter` 将当前 state 转为下一步的连续输入；
5. 重复 \(K\) 次；
6. 使用最终状态 `z_K` 生成 embedding。

推荐默认反馈模块：

```python
feedback_adapter = LayerNorm(H) + Linear(H, H)
```

线性层初始化为接近单位映射。必须支持以下配置：

- `identity`
- `layernorm`
- `linear`
- `ln_linear`
- `mlp`

主实验默认 `ln_linear`。

伪代码：

```python
past = multimodal_prefix_prefill(batch, use_cache=True)
u = latent_seed.expand(batch_size, 1, hidden_size)
states = []

for step in range(K):
    output = decoder_step(
        inputs_embeds=u,
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
    )
    z = output.last_hidden_state[:, -1, :]
    states.append(z)
    past = output.past_key_values
    u = feedback_adapter(z).unsqueeze(1)

embedding = l2_normalize(retrieval_head(states[-1]))
```

### 3.3 Explicit Teacher encoder

教师输入包含原始多模态输入和结构化 CoT：

```text
[INPUT]
<COT_1> ... </COT_1> <STEP>
<COT_2> ... </COT_2> <STEP>
...
<COT_K> ... </COT_K> <STEP>
```

Teacher 只做一次完整 causal forward，并提取所有 `<STEP>` 位置最后一层 hidden state：

```python
teacher_states: Tensor[B, K, H]
```

由于 causal mask，第 \(i\) 个 `<STEP>` 只能读取 `INPUT + COT_1 ... COT_i`，因此对应累计显式推理状态。

### 3.4 EMA Teacher

Teacher 与 Student 同构。参数更新规则：

\[
\bar{\theta}\leftarrow\mu\bar{\theta}+(1-\mu)\theta
\]

要求：

- Teacher 参数 `requires_grad=False`；
- Teacher forward 位于 `torch.no_grad()`；
- Teacher 默认 `eval()`，避免 dropout 产生不稳定目标；
- EMA 更新在 optimizer step 后执行；
- LoRA 模式下至少 EMA 更新所有可训练 LoRA 参数、latent seed、feedback adapter 和 retrieval head；
- 最简单可靠的实现是维护完整 teacher module，后续再优化为只维护可训练参数。

---

## 4. 数据集方案

ELDER 沿用 VLM2Vec-V2 式通用多模态训练语料，覆盖 Image、Video 和 Visual Document。

### 4.1 Image-centric：MMEB-train

数据源：

```text
TIGER-Lab/MMEB-train
```

覆盖约 20 个训练子任务，主要包括：

- 分类；
- VQA；
- 图文检索；
- composed image retrieval；
- visual grounding；
- 文档、图表和信息图理解。

初始复现使用 `original` split；后续可将 `diverse_instruction` 作为增强或消融。

### 4.2 Video：LLaVA-Hound

建议沿用 VLM2Vec-V2 配方：

- 300k video-caption pairs；
- caption → video retrieval；
- video → caption retrieval；
- 240k video QA pairs；
- 每个视频均匀采样 8 帧。

### 4.3 Visual Document：ViDoRe + VisRAG

建议包含：

- `vidore/colpali_train_set`，约 118k；
- VisRAG synthetic，约 239k；
- VisRAG in-domain，约 123k。

实现时数据集名称和本地路径全部放入 YAML，不要硬编码。

### 4.4 训练数据规模策略

#### Stage 1

尽量使用完整 VLM2Vec-V2 语料，建立通用 embedding space。

#### Stage 2/3

CoT 生成成本较高，第一版采用平衡抽样：

```text
MVP：100k–300k pairs
中等规模：500k–800k pairs
完整论文版：尽可能覆盖全部训练 pairs
```

必须保证 Image、Video、VisDoc 都有样本，不能只在图片任务上蒸馏。

### 4.5 Interleaved sub-batching

沿用 VLM2Vec-V2 的 interleaved sub-batching：

- 同一 sub-batch 内保持数据结构相近；
- 一个 global batch 内混合多个任务/模态；
- 默认 `interleave_batch_size=64`；
- L40S 初始实验可先用 16 或 32；
- 保留 GradCache 和跨 GPU negatives。

---

## 5. CoT 离线生成

### 5.1 CoT 生成目标

CoT 不是用于回答问题，而是用于形成可蒸馏的检索过程。每条 CoT 必须有固定 \(K\) 个宏观阶段。

默认 `K=4`：

1. **Content Understanding**：识别当前输入中的核心语义、实体和视觉内容；
2. **Evidence Abstraction**：抽象与匹配有关的属性、证据类型或关键线索；
3. **Relation and Constraint Reasoning**：分析关系、条件、时间、布局或跨模态约束；
4. **Retrieval Semantics Synthesis**：形成适合检索匹配的最终语义表示。

Query 与 Candidate 可以使用不同提示词，但阶段编号必须保持相近的语义粒度。

### 5.2 禁止 target leakage

生成 Query CoT 时只能看到 Query。  
生成 Candidate CoT 时只能看到 Candidate。

禁止使用：

```text
Query + Positive Candidate → Query CoT
```

否则 Teacher 会包含 Student 在真实推理时不可获得的信息。

### 5.3 数据格式

推荐 JSONL：

```json
{
  "sample_id": "dataset/subset/123",
  "dataset": "DocVQA",
  "query": {
    "text": "...",
    "media": ["path/to/query_image.png"],
    "cot_steps": ["...", "...", "...", "..."]
  },
  "candidate": {
    "text": "...",
    "media": ["path/to/candidate_image.png"],
    "cot_steps": ["...", "...", "...", "..."]
  },
  "metadata": {
    "generator": "strong-mllm-name",
    "prompt_version": "v1",
    "num_steps": 4
  }
}
```

允许 Candidate 没有 CoT，以支持 query-only ablation，但主方法默认两侧都有 CoT。

### 5.4 CoT 质量过滤

离线脚本至少执行：

- 步数是否严格等于 \(K\)；
- 单步是否为空；
- 是否包含答案泄露或正样本 ID；
- 每步 token 数上下限；
- 重复率检查；
- 拒绝模板化、无意义或完全相同的多步内容；
- 保存失败原因和重试次数。

推荐每步 32–96 tokens，总 CoT 不超过 384 tokens。

---

## 6. 三阶段训练

## Stage 1：Discriminative Retrieval Warm-up

### 目的

先让 backbone 具备基础检索能力，建立稳定的跨模态 embedding space。

### 输入

普通 VLM2Vec query/candidate pairs，不使用 CoT，不执行 latent rollout。

### 模型路径

尽量复用 VLM2Vec 的：

- processor；
- dataset parser；
- interleaved sampler；
- `MMEBModel`；
- GradCache trainer；
- MMEB-V2 evaluation。

### Loss

使用双向或单向 InfoNCE。主实现应支持双向，默认以 VLM2Vec 官方设置为准。

### 固定默认配置

```yaml
model: Qwen/Qwen2-VL-2B-Instruct
tuning: LoRA
lora_r: 16
lora_alpha: 32
normalize: true
temperature: 0.02
global_batch_size: 256
interleave_batch_size: 32
max_steps: 5000
learning_rate: 5e-5
```

### 输出 checkpoint

```text
checkpoints/elder_stage1_discriminative/
```

### 验收

- 能在小规模 MMEB/ViDoRe debug set 上稳定下降；
- embedding norm 正常；
- Recall/Hit@1 明显高于随机；
- 与未修改 VLM2Vec baseline 基本一致。

---

## Stage 2：Explicit Process Warm-up

### 目的

让 `<STEP>` hidden states 从普通文本状态转化为具有检索语义的阶段性教师状态。

### 输入

Query 和 Candidate 分别拼接自己的分阶段 CoT，并在每一阶段后加入 `<STEP>`。

### 初始化

从 Stage 1 checkpoint 初始化。

### Forward

单个模型执行 explicit forward：

```python
q_states = encode_explicit(query, query_cot_steps)      # [B, K, H]
c_states = encode_explicit(candidate, candidate_steps)  # [B, K, H]
```

### Loss

主损失：

- 最终阶段 query/candidate states 的 InfoNCE。

辅助损失：

- 中间阶段的 weighted InfoNCE；
- 权重随阶段递增，例如 `[0.25, 0.50, 0.75, 1.00]`。

可写为：

\[
\mathcal L_{\text{explicit}}
=
\mathcal L_{\text{final-ret}}
+
\lambda_{\text{step}}\sum_i\alpha_i\mathcal L_{\text{ret}}^{(i)}
\]

### 注意

- Stage 2 不训练 recurrent student；
- Stage 2 不需要 EMA；
- `<STEP>` 位置必须从 tokenizer 后的真实 token index 提取，不能用字符串长度推断；
- 如果同一个 `<STEP>` token 重复出现，collator 要返回 `step_positions[B, K]`；
- Stage 2 结束后用该 checkpoint 同时初始化 Student 和 EMA Teacher。

### 输出 checkpoint

```text
checkpoints/elder_stage2_explicit/
```

### 验收

- 最终 `<STEP>` retrieval accuracy 高于 Stage 1 或至少不下降；
- 中间阶段 retrieval accuracy 随步骤总体上升；
- 相邻 teacher states 不应全部相同；
- `cos(t_i, t_{i+1})` 不能长期接近 1.0；
- 不同样本的 teacher trajectory 有明显差异。

---

## Stage 3：EMA Explicit-to-Latent Process Distillation

### 初始化

```python
student.load(stage2_checkpoint)
teacher.load(stage2_checkpoint)
teacher.requires_grad_(False)
```

### Teacher view

Teacher 输入原始 input + CoT，输出：

```python
t_q: [B, K, H]
t_c: [B, K, H]
```

### Student view

Student 只输入原始多模态内容，执行 recurrent feedback：

```python
z_q: [B, K, H]
z_c: [B, K, H]
e_q: [B, D]
e_c: [B, D]
```

### 关键 Loss

#### 1. Retrieval loss

最终 latent states 生成 embedding，使用 InfoNCE：

\[
\mathcal L_{\mathrm{ret}}=\operatorname{InfoNCE}(e_q,e_c)
\]

#### 2. State distillation

每一步 latent state 对齐对应显式 CoT prefix state：

\[
\mathcal L_{\mathrm{state}}
=
\sum_{x\in\{q,c\}}\sum_i
\left[1-\cos(\operatorname{LN}(z_i^x),\operatorname{sg}(\operatorname{LN}(t_i^x)))\right]
\]

#### 3. Transition distillation

对齐相邻阶段的变化：

\[
\mathcal L_{\mathrm{trans}}
=
\sum_{x\in\{q,c\}}\sum_{i=2}^{K}
\left[1-\cos(z_i^x-z_{i-1}^x,\operatorname{sg}(t_i^x-t_{i-1}^x))\right]
\]

#### 4. Rank distillation（可选，默认实现但可关闭）

Teacher 和 Student 在同一 in-batch candidate set 上形成相似度分布，使用 KL：

\[
\mathcal L_{\mathrm{rank}}
=
\sum_i D_{\mathrm{KL}}(P_i^T\Vert P_i^S)
\]

### 总 Loss

\[
\mathcal L_{\mathrm{ELDER}}
=
\mathcal L_{\mathrm{ret}}
+
\lambda_s\mathcal L_{\mathrm{state}}
+
\lambda_t\mathcal L_{\mathrm{trans}}
+
\lambda_r\mathcal L_{\mathrm{rank}}
\]

默认初值：

```yaml
lambda_state: 1.0
lambda_transition: 0.5
lambda_rank: 0.2
ema_decay: 0.999
```

前 5%–10% steps 对蒸馏权重线性 warm-up。

### EMA 更新顺序

```python
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
update_ema(student, teacher, decay)
```

### 输出 checkpoint

只需保存 Student 供推理：

```text
checkpoints/elder_stage3_student/
```

为了恢复训练，也可以额外保存 EMA teacher 和 optimizer state。

---

## 7. Recurrent feedback 的训练实现要求

这是项目中最关键、最容易实现错误的部分。

### 7.1 正确语义

第 \(i\) 步输出 `z_i` 必须成为第 \(i+1\) 步的连续输入，而不是提前插入 \(K\) 个固定 learnable tokens。

正确：

```text
latent_seed → z1 → feedback(z1) → z2 → feedback(z2) → ...
```

错误：

```text
<L1><L2><L3><L4> 一次前向
```

后者属于 one-pass latent slots，不是主方法。

### 7.2 梯度要求

主实现不得在反馈时切断梯度：

```python
u = feedback_adapter(z)          # 正确
u = feedback_adapter(z.detach()) # 仅消融
```

### 7.3 KV cache 与训练图

推理阶段必须使用 KV cache。

训练阶段建议实现两种后端：

#### A. `cached_graph`（主目标）

- prefix prefill 使用 `use_cache=True`；
- recurrent steps 使用 `past_key_values`；
- cache 不 detach；
- 保留完整 recurrent computation graph；
- Stage 3 禁用与 cache 冲突的 gradient checkpointing。

#### B. `cached_detached`（显存消融）

- cache 正常复用；
- feedback state detach；
- 失去跨 step 的 BPTT；
- 仍可依靠每一步 state loss 训练；
- 只用于对比或显存不足时验证。

如果 Hugging Face 当前 backbone 的 cache 无法稳定反向传播，先实现一个 correctness-first reference path，再做优化，不要默认静默 detach。

### 7.4 Qwen-VL 适配

建议新增 `QwenVLRecurrentAdapter`，统一封装：

- multimodal prefix forward；
- text decoder/backbone 获取；
- `past_key_values`；
- `position_ids`；
- `cache_position`；
- 单步 `inputs_embeds`；
- hidden state extraction。

不要把 Qwen2-VL 的内部字段散落在 ELDER 主模型中。

---

## 8. 推荐代码结构

基于 VLM2Vec repo 增加：

```text
src/
  elder/
    __init__.py
    configuration_elder.py
    model_elder.py
    recurrent_rollout.py
    qwen_recurrent_adapter.py
    ema.py
    losses.py
    outputs.py

    data/
      cot_dataset.py
      cot_collator.py
      cot_schema.py
      prompts.py
      validators.py

    trainers/
      stage1_trainer.py
      stage2_trainer.py
      stage3_trainer.py

scripts/
  elder/
    generate_cot.py
    validate_cot.py
    train_stage1.sh
    train_stage2.sh
    train_stage3.sh
    evaluate_elder.sh
    benchmark_latency.py

experiments/
  elder/
    data_vlm2vec_v2.yaml
    stage1.yaml
    stage2.yaml
    stage3.yaml

tests/
  elder/
    test_step_positions.py
    test_recurrent_shapes.py
    test_recurrent_gradients.py
    test_teacher_no_grad.py
    test_ema_update.py
    test_losses.py
    test_checkpoint_roundtrip.py
```

---

## 9. 建议类与接口

### 9.1 `ELDERModel`

```python
class ELDERModel(nn.Module):
    def encode_discriminative(self, batch) -> Tensor:
        ...

    def encode_explicit(
        self,
        batch,
        step_positions,
        return_states=True,
    ) -> ElderExplicitOutput:
        ...

    def encode_latent(
        self,
        batch,
        num_steps,
        return_states=True,
        detach_feedback=False,
    ) -> ElderLatentOutput:
        ...

    def forward_stage1(self, qry, tgt):
        ...

    def forward_stage2(self, qry, tgt, qry_step_positions, tgt_step_positions):
        ...

    def forward_stage3(
        self,
        teacher,
        qry,
        tgt,
        qry_explicit,
        tgt_explicit,
        qry_step_positions,
        tgt_step_positions,
    ):
        ...
```

### 9.2 输出类型

```python
@dataclass
class ElderLatentOutput:
    embedding: torch.Tensor
    latent_states: torch.Tensor
    past_key_values: Optional[Any] = None

@dataclass
class ElderExplicitOutput:
    embedding: torch.Tensor
    step_states: torch.Tensor
```

### 9.3 Loss 返回

Trainer 日志必须分别记录：

```text
loss_total
loss_retrieval
loss_state
loss_transition
loss_rank
teacher_student_state_cos_step_1 ... step_K
student_transition_norm_step_2 ... step_K
teacher_transition_norm_step_2 ... step_K
embedding_norm
```

不能只记录总 loss，否则无法诊断 collapse。

---

## 10. 数据 Collator 要求

Stage 2/3 的 collator 需要同时返回：

```python
{
  "qry_student": processor_output,
  "tgt_student": processor_output,
  "qry_teacher": processor_output,
  "tgt_teacher": processor_output,
  "qry_step_positions": LongTensor[B, K],
  "tgt_step_positions": LongTensor[B, K],
  "sample_ids": list[str],
}
```

要求：

- Student batch 中不得出现 CoT；
- Teacher batch 才包含 CoT；
- padding side 与 backbone 保持一致；
- `<STEP>` 注册为 special token；
- 若修改 tokenizer vocab，模型 embedding matrix 必须 resize；
- step position 在 padding 后仍正确；
- 每条样本必须恰好有 \(K\) 个有效 `<STEP>`；
- 不符合格式的样本在数据预处理阶段过滤，而不是训练时随机报错。

---

## 11. 训练资源和建议配置

实验室资源：8×L40S。

### MVP 固定配置

```yaml
backbone: Qwen/Qwen2-VL-2B-Instruct
precision: bf16
tuning: LoRA
lora_r: 32
lora_alpha: 64
num_latent_steps: 4
max_cot_tokens: 384
num_video_frames: 8
per_device_batch_size: 1-4
gradient_accumulation_steps: 按 global batch 调整
deepspeed: ZeRO-2
```

Stage 3 同时存在 Student 和 Teacher，因此 batch 应从小规模开始。Teacher 无梯度，但仍需要参数和 forward 显存。

### 扩展实验

- `K ∈ {2, 4, 8}`；
- 所有主实验和消融实验均固定使用 `Qwen/Qwen2-VL-2B-Instruct`；
- Stage 2 必须从同 backbone 的 Stage 1 checkpoint 初始化，Stage 3 必须从同 backbone 的 Stage 2 checkpoint 初始化；
- checkpoint 加载时应校验 backbone 标识，禁止静默加载其他模型规模或架构的权重；
- LoRA 与 full fine-tuning 做资源允许下的对比。

---

## 12. 推理与索引

### Candidate indexing

```python
for candidate in corpus:
    embedding = model.encode_latent(candidate, K)
    index.add(embedding)
```

Candidate 的 \(K\) 次 recurrent reasoning 只在离线建库或索引更新时发生。

### Online query

```python
query_embedding = model.encode_latent(query, K)
scores = vector_index.search(query_embedding, top_k)
```

线上不加载 EMA teacher，不读取 CoT。

### 推理 checkpoint

推理目录应包含：

```text
config.json
model / LoRA weights
processor files
tokenizer files
elder_config.json
latent_seed
feedback_adapter
retrieval_head
```

加载后必须可以直接执行：

```python
model = ELDERModel.from_pretrained(path)
embedding = model.encode(...)
```

---

## 13. 评测计划

### 13.1 主评测

使用 MMEB-V2 的 78 tasks，分别报告：

- Image；
- Video；
- Visual Document；
- Overall。

保持 VLM2Vec-V2 的任务 prompt 和评测脚本不变。

### 13.2 效率评测

必须额外报告：

- query encoding latency；
- throughput；
- peak GPU memory；
- 显式 CoT baseline 的平均生成 token 数；
- ELDER 的 latent step 数；
- candidate indexing time；
- 不同 \(K\) 的性能—速度曲线。

### 13.3 必要消融

至少包含：

1. Stage 1 only：普通 discriminative embedding；
2. Recurrent latent without distillation；
3. State distillation only；
4. State + transition；
5. State + transition + rank；
6. Frozen teacher vs EMA teacher；
7. Query-only distillation vs symmetric distillation；
8. one-pass latent slots vs recurrent feedback；
9. `detach_feedback` vs full recurrent gradient；
10. \(K=2/4/8\)；
11. 无 Stage 2 warm-up；
12. 不同 CoT 质量或不同 CoT 数据量。

---

## 14. 单元测试和验收标准

### 14.1 单元测试

- `step_positions` 在不同 padding 下提取正确；
- Teacher 一次 forward 得到恰好 `[B,K,H]`；
- Student recurrent rollout 得到 `[B,K,H]`；
- 第 `i+1` 步输入确实依赖第 `i` 步输出；
- 不 detach 时，最终 loss 对 `z_1` 和 latent seed 有非零梯度；
- Teacher 参数没有梯度；
- EMA 更新数值正确；
- checkpoint 保存/加载后 embedding 一致；
- query/candidate 的 text-only、image-text、video、visdoc batch 都可运行。

### 14.2 训练 sanity check

先在 128–1024 pairs 上过拟合：

- retrieval loss 明显下降；
- state cosine 明显上升；
- transition loss 下降；
- latent states 不出现 NaN；
- 不同步骤 state variance 不为零；
- 同一条样本的连续 states 不完全重合；
- 训练后正样本得分高于负样本。

### 14.3 最小可交付版本

MVP 完成条件：

- Stage 1/2/3 均可独立启动和 resume；
- 1 个 image 数据集 + ViDoRe debug subset 可端到端训练；
- Query 和 Candidate 都可 recurrent 编码；
- Stage 3 能同时输出四项 loss；
- 推理模型不依赖 CoT 或 Teacher；
- 延迟脚本能与显式 CoT baseline 比较。

---

## 15. 推荐实现顺序

Codex 请严格按以下顺序推进，不要一开始同时实现所有功能：

### Milestone 1：复现 VLM2Vec baseline

- 跑通官方 Stage 1；
- 保存和评测 checkpoint；
- 不修改原始 baseline 行为。

### Milestone 2：Explicit teacher states

- 添加 `<STEP>`；
- 实现 structured CoT collator；
- 单次 forward 提取 K 个 states；
- 跑通 Stage 2。

### Milestone 3：Recurrent latent encoder

- 先在 text-only 输入上实现；
- 再适配 image-text；
- 最后适配 video 和 visual document；
- 验证 recurrent gradient。

### Milestone 4：Frozen teacher distillation debug

- 暂时冻结 Stage 2 teacher；
- 跑通 state loss 和 retrieval loss；
- 这是调试步骤，不是最终论文设置。

### Milestone 5：EMA teacher

- 加入 EMA 更新；
- 加入 transition loss；
- 加入 rank loss；
- 完成 Stage 3 正式训练。

### Milestone 6：完整数据和评测

- 扩展到 VLM2Vec-V2 全语料；
- 完成 MMEB-V2 评测；
- 完成效率和消融实验。

---

## 16. 明确的非目标

第一版不要实现：

- 强化学习或 GRPO；
- 在线 CoT generation；
- 多向量 late interaction；
- 动态 latent step 数；
- MoE latent router；
- 文本 CoT reconstruction decoder；
- 跨页 RAG；
- reranker。

先验证核心命题：

> 显式 CoT prefix states 能否通过 EMA process distillation，迁移为可用于通用多模态检索的 recurrent latent reasoning trajectory。

---

## 17. 主要风险与处理

### 风险 1：CoT 阶段只是机械切分

处理：强制固定阶段语义，检查相邻 teacher transition norm 和 state cosine。

### 风险 2：latent collapse

处理：

- transition loss；
- 每步监控 state variance；
- 不同阶段权重；
- latent seed 和 feedback adapter 合理初始化；
- 必要时增加 decorrelation regularization，但不要默认加入主方法。

### 风险 3：Teacher 是 moving target

处理：

- Stage 2 预热；
- EMA decay 0.999；
- 蒸馏权重 warm-up；
- Teacher eval mode；
- 先 frozen teacher 调试。

### 风险 4：Stage 3 显存过高

处理：

- 固定使用 2B + LoRA；
- 减小 CoT 长度；
- 减小 K；
- Teacher no-grad；
- ZeRO-2/3；
- 使用 balanced sampling；
- `cached_detached` 只能作为备用或消融，不能悄悄替代主实现。

### 风险 5：Qwen-VL cache API 不兼容连续 embedding

处理：把所有 backbone-specific 逻辑封装到 adapter，写独立测试；可参考 PLUME 官方代码中的 prefix/latent/suffix 和 `past_key_values` 处理方式，但不要直接复制其 curriculum 或 Latent MoE。

---

## 18. 推荐配置文件字段

见随本文档提供的 `elder_config_template.yaml`。实现时所有关键参数必须可配置，不允许写死：

- backbone；
- K；
- feedback adapter；
- loss 权重；
- EMA decay；
- query/candidate 是否都蒸馏；
- CoT 最大长度；
- dataset weights；
- cache/gradient mode；
- LoRA 和 distributed training；
- checkpoint 路径；
- debug subset size。

---

## 19. 参考实现与事实依据

Codex 开发时优先参考以下公开资源：

1. `TIGER-AI-Lab/VLM2Vec`：数据加载、interleaved sub-batching、GradCache、InfoNCE、MMEB-V2 评测和 Qwen-VL processor。
2. VLM2Vec-V2（arXiv:2507.04590）：训练数据由 MMEB-train、LLaVA-Hound、ViDoRe 和 VisRAG 构成；官方 2B 配方使用 temperature 0.02、LoRA 和大 global batch。
3. `haoxiangzhao12138/PLUME`：Qwen2-VL 中连续 latent rollout、KV cache、position IDs 和 multimodal prefix 的工程处理可作为参考。
4. UME-R1（arXiv:2511.00405）：使用同类 MMEB-V2 训练语料，并为 query 和 target 构造推理数据，可参考其 CoT 数据组织方式，但 ELDER 不使用 RL。
5. ELDER 的核心差异：不采用 PLUME 的 progressive replacement curriculum，也不在推理时生成文本；采用 EMA teacher 对每个 CoT prefix state 与 recurrent latent state 做阶段对齐和 transition distillation。

---

## 20. 一句话任务描述

> 在 VLM2Vec-V2 代码库上实现 ELDER：先进行判别式检索预热，再用分阶段显式 CoT 预热 `<STEP>` states，最后通过 EMA teacher 将每个显式 CoT prefix 的 hidden state 和状态转移蒸馏到 Student 的 \(K\) 次 recurrent hidden-state feedback 中；推理时只保留 Student，并对 Query 和 Candidate 输出可直接用于向量检索的 embedding。
