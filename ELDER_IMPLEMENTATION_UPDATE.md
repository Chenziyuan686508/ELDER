# ELDER 项目实现说明书

> **项目名称**：ELDER — Explicit-to-Latent Process Distillation for Efficient Multimodal Retrieval  
> **文档用途**：交给 Codex 作为实现依据。本文档优先描述工程接口、训练流程和验收标准，而不是论文写作。  
> **推荐基础代码库**：在官方 `TIGER-AI-Lab/VLM2Vec` V2 代码上增量开发，不建议从零搭建数据和评测框架。  
> **固定主干模型**：`Qwen/Qwen2-VL-2B-Instruct`。Stage 1、Stage 2、Stage 3 以及最终推理均使用同一主干模型；当前项目不进行跨 backbone 或跨模型规模切换。

> **当前版本关键约定**：默认采用固定 `K=4` 的宏观语义检查点。固定的是过程蒸馏的表示分辨率和循环计算预算，而不是假设所有样本天然具有相同数量的真实推理步骤。CoT 采用检索导向、输入侧独立生成的四阶段结构；对于类别标签和极短答案等语义贫乏目标，不强制构造完整 Candidate CoT。

---

## 1. 项目目标

ELDER 的目标是训练一个通用多模态 embedding 模型，使其在不生成显式自然语言 CoT 的情况下，通过固定的 \(K\) 次 recurrent hidden-state feedback 完成 latent reasoning，并输出用于检索的单向量 embedding。这里的 \(K\) 表示统一的过程蒸馏分辨率与最大循环计算预算：不同输入的具体推理内容和信息增量可以不同，但均被压缩到同一组宏观语义检查点中，以建立稳定的教师—学生阶段对应关系。

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

### 2.6 固定 `K` 的含义

固定 `K` 不表示所有样本天然具有相同数量的真实推理步骤，也不表示简单样本必须人为制造复杂推理。ELDER 将长度、粒度和措辞各异的显式 CoT 规范化为 `K` 个跨模态通用的宏观功能检查点，并将其作为：

- 教师状态与学生状态的一一对齐接口；
- 状态转移蒸馏的统一时间轴；
- 推理成本可预测的最大循环预算；
- 控制过程表示粒度的超参数。

对于简单输入，后续阶段可以承担证据确认、噪声过滤和表示稳定化；对于复杂输入，后续阶段持续融合关系、比较、时间、空间和跨模态约束。默认 `K=4` 仅作为方法主配置，必须通过 `K ∈ {1, 2, 4, 6, 8}` 的性能—效率消融验证其合理性。自适应步数属于正交扩展，不纳入第一版主方法。

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

教师输入包含原始多模态输入和结构化 CoT。默认四阶段格式如下：

```text
[INPUT]
<STAGE_1 type="modality_perception"> ... </STAGE_1> <STEP_1>
<STAGE_2 type="evidence_selection"> ... </STAGE_2> <STEP_2>
<STAGE_3 type="relation_reasoning"> ... </STAGE_3> <STEP_3>
<STAGE_4 type="retrieval_synthesis"> ... </STAGE_4> <STEP_4>
<RET>
```

Teacher 只做一次完整 causal forward，并提取所有 `<STEP_i>` 位置最后一层 hidden state：

```python
teacher_states: Tensor[B, K, H]
```

由于 causal mask，第 `i` 个阶段锚点只能访问 `INPUT + STAGE_1 ... STAGE_i`，因此对应截至该宏观检查点的累计显式推理状态。`<RET>` 状态用于可选的最终教师检索表示或 rank distillation，不替代 `<STEP_i>` 的过程监督。

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

### 4.4 CoT 处理范围与任务路由

ELDER 可沿用 UME-R1、Embed-RL 所使用的同类通用多模态数据池，即 MMEB-train、LLaVA-Hound、ViDoRe 和 VisRAG，但不应机械地为所有 Query 和 Target 构造完整四阶段 CoT。是否启用过程蒸馏由输入的语义丰富度和任务形式决定。

#### A. 优先进行双侧完整 CoT

适用于 Query 与 Candidate 两侧都包含充分实体、属性、关系或视觉内容的任务，例如：

- composed image retrieval；
- text–image / image–text retrieval；
- image–image retrieval；
- video caption/retrieval；
- ViDoRe、VisRAG 等文档页面检索；
- 内容丰富的图像、视频或文档 Candidate。

这些样本默认生成 Query CoT 和 Candidate CoT，并在 Stage 2/3 对两侧执行 state/transition distillation。

#### B. 默认仅进行 Query 侧完整 CoT

适用于 Candidate 为类别名称、单词、短数字或极短答案的任务，例如部分：

- ImageNet-1K、SUN397、VOC2007 等分类任务；
- A-OKVQA、OK-VQA、ChartQA、DocVQA、InfographicsVQA、Visual7W 等短答案任务。

此时 Query 侧仍可包含图像、问题和复杂约束，适合生成四阶段 CoT；Candidate 侧若仅为 `dog`、`yes`、`12` 等短文本，则关闭过程蒸馏，仅参与最终 InfoNCE。不要为了保持形式对称而生成空泛的四阶段 Target CoT。

#### C. 运行时语义丰富度门控

不能只依赖数据集名称。预处理脚本应根据实际 Candidate 内容决定 `cot_enabled`：

- Candidate 含图像、视频、文档页面：默认启用；
- Candidate 为较长描述性文本：默认启用；
- Candidate 为短标签/答案：默认关闭；
- 边界样本可由规则与生成模型复核共同决定。

建议保存：

```json
{
  "query_cot_enabled": true,
  "candidate_cot_enabled": false,
  "candidate_disable_reason": "short_label"
}
```

### 4.5 训练数据规模策略

#### Stage 1

尽量使用完整 VLM2Vec-V2 语料，建立通用 embedding space。

#### Stage 2/3

CoT 生成成本较高，第一版采用分任务平衡抽样：

```text
MVP：100k–300k pairs
中等规模：500k–800k pairs
完整论文版：尽可能覆盖全部训练 pairs
```

要求：

- Image、Video、VisDoc 均有过程蒸馏样本；
- 保留分类、VQA 等无 Candidate CoT 的样本，以维持通用检索能力；
- 记录每个数据集的 Query/Candidate CoT 启用比例；
- 避免数据量大的单一任务主导 Stage 2/3。

### 4.6 Interleaved sub-batching

沿用 VLM2Vec-V2 的 interleaved sub-batching：

- 同一 sub-batch 内保持数据结构相近；
- 一个 global batch 内混合多个任务/模态；
- 默认 `interleave_batch_size=64`；
- L40S 初始实验可先用 16 或 32；
- 保留 GradCache 和跨 GPU negatives；
- 可将 `query_only_distill` 与 `symmetric_distill` 样本分到不同 sub-batch，简化 padding 与 mask 处理。

---

## 5. CoT 离线生成

### 5.1 CoT 的角色与固定 `K` 动机

CoT 不用于产生最终答案，也不在推理阶段输入 Student。它只作为训练期的 **privileged process supervision**，用于构造具有明确检索语义的教师阶段状态。

原始自由 CoT 的句子数、token 数和展开粒度可以不同。ELDER 不逐 token 模仿自由文本，而是要求强 MLLM 在完成内部分析后，将结果整理为固定 `K` 个宏观语义检查点：

\[
C(x)=\{c_1(x),c_2(x),\ldots,c_K(x)\}.
\]

固定 `K` 的目的不是声称所有输入真实地按照完全相同的步数思考，而是：

1. 将可变长度显式过程压缩为统一分辨率；
2. 使 `t_k ↔ z_k` 和 `Δt_k ↔ Δz_k` 具有稳定对应关系；
3. 避免逐 token 蒸馏迫使 latent reasoning 复现语言措辞；
4. 提供固定、可预测的在线循环计算预算；
5. 将研究重点限定为“显式过程如何迁移到 recurrent latent trajectory”，而非同时解决动态计算深度。

默认 `K=4`。四阶段固定的是宏观功能，具体内容、信息量和状态变化幅度由输入决定。

### 5.2 四阶段检索导向 CoT Schema

#### Stage 1 — Modality-grounded Content Perception

**目标：客观识别当前输入中存在什么。**

- 文本：实体、主题、属性、关键术语、否定表达；
- 图像：主体、场景、颜色、动作、空间位置；
- 图文组合：图像现状、文本指令及两者对应关系；
- 视频：主体、动作、关键事件和关键时间片段；
- 视觉文档：页面类型、OCR 字段、表格、图表、区域和布局结构。

此阶段不得提前猜测正样本，只描述当前输入自身可验证的内容。

#### Stage 2 — Retrieval Evidence Selection

**目标：从全部内容中筛选真正影响相关/不相关候选区分的证据。**

需要明确：

- 哪些实体、属性、区域、字段或关键帧最具判别性；
- 哪些背景内容与检索无关，应被忽略；
- 哪些证据需要候选侧出现或得到支持。

该阶段可以借鉴 Embed-RL 中检索导向 T-CoT 的关键词、图像区域、视频关键帧和 rethink 式证据过滤思想，但输出必须转换为 ELDER 的统一阶段结构。

#### Stage 3 — Relation and Constraint Reasoning

**目标：建模证据之间的关系与组合约束。**

包括但不限于：

- 实体—属性绑定；
- 主体—动作—客体关系；
- 空间方位与布局对应；
- 时间顺序和状态变化；
- 数值大小、趋势与比较关系；
- 图像条件与文本修改要求；
- 多个检索条件的逻辑组合。

该阶段是区分“关键词堆叠”与“复杂检索意图理解”的关键。

#### Stage 4 — Retrieval Semantics Synthesis

**目标：将前述信息压缩为直接服务于向量匹配的检索语义。**

Query 侧描述：需要检索什么样的候选、候选必须满足哪些核心条件。  
Candidate 侧描述：当前候选包含哪些核心证据、能够支持什么类型的检索需求。

禁止输出正样本 ID、页面编号、文件路径或“正确答案是……”等标签信息。Stage 4 是检索语义综合，不是任务答案生成。

### 5.3 简单样本如何填充四阶段

简单 Query 仍可使用四个宏观检查点，但后续阶段不需要人为制造复杂关系。例如“检索红色汽车图片”：

```json
{
  "stage_1": "输入主体为汽车，关键可见属性为红色。",
  "stage_2": "最具判别性的检索证据是汽车类别与红色外观，背景信息可忽略。",
  "stage_3": "候选需同时满足对象类别和颜色属性，没有额外时间或空间约束。",
  "stage_4": "目标候选是以红色汽车为核心内容的图像。"
}
```

此时后续阶段承担约束确认、去噪和语义压缩。固定四阶段不等于四次同等复杂的推理。

### 5.4 Query 与 Candidate 的差异化模板

#### Query 侧

最后一阶段必须以需求形式表达：

```text
需要检索包含……、满足……关系/条件的候选。
```

#### Candidate 侧

最后一阶段必须以证据能力形式表达：

```text
该候选包含……证据，可支持……类型的检索需求。
```

两侧使用相同的四阶段宏观功能，但措辞角色不同。不得把配对对象的内容提供给生成器。

### 5.5 CoT 生成模型与生成流程

第一版 MVP 推荐使用：

```text
zai-org/GLM-4.1V-9B-Thinking
```

原因是模型规模适中，支持文本、图像、视频/多帧和视觉文档理解，适合在本地进行批量离线生成。正式扩大数据前，应在多模态开发集上与更强的 Qwen-VL Thinking 模型比较格式遵循、视觉忠实度、阶段差异和生成吞吐量，再决定是否更换主生成器或仅对困难样本复核。

Thinking 模型的自由分析内容不直接写入训练数据。推荐流程：

```text
原始输入
  ↓
Thinking MLLM 内部充分分析
  ↓
仅输出固定四阶段 JSON
  ↓
规则校验 + 泄漏检查 + 语义复核
  ↓
写入 ELDER CoT 数据
```

生成参数建议：

- 低到中等 temperature，优先稳定而非多样性；
- 强制 JSON schema；
- 每阶段只承担一个宏观功能；
- 每阶段建议 24–80 tokens；
- 总 CoT 建议不超过 320 tokens，`384` 作为硬上限；
- 失败样本最多重试 2–3 次。

### 5.6 与 UME-R1 / Embed-RL 数据使用方式的关系

可复用两者所覆盖的 MMEB-train、LLaVA-Hound、ViDoRe 和 VisRAG 数据池及已有 CoT 组织思路，但 ELDER 的监督目标不同：

- 不使用 RL；
- 不直接保存自由 `<think>` 作为阶段标签；
- 不要求所有短 Target 都生成完整 CoT；
- 将检索导向信息重新规范化为四个宏观检查点；
- Query 与 Candidate 独立生成，禁止 pair-conditioned Query CoT；
- 需要显式保存阶段级状态掩码，供 Stage 2/3 使用。

### 5.7 禁止 target leakage

生成 Query CoT 时只能看到 Query。  
生成 Candidate CoT 时只能看到 Candidate。

禁止使用：

```text
Query + Positive Candidate → Query CoT
```

禁止在提示词、元数据或图像文件名中暴露：

- 正样本 ID；
- 配对候选内容；
- 数据集答案字段；
- 页面编号或检索排名；
- 人工构造的负样本差异描述。

### 5.8 数据格式

推荐 JSONL：

```json
{
  "sample_id": "dataset/subset/123",
  "dataset": "DocVQA",
  "query": {
    "text": "...",
    "media": ["path/to/query_image.png"],
    "cot_enabled": true,
    "cot_steps": ["...", "...", "...", "..."],
    "cot_step_mask": [1, 1, 1, 1]
  },
  "candidate": {
    "text": "12",
    "media": [],
    "cot_enabled": false,
    "cot_steps": [],
    "cot_step_mask": [0, 0, 0, 0]
  },
  "metadata": {
    "generator": "zai-org/GLM-4.1V-9B-Thinking",
    "prompt_version": "elder_cot_v2",
    "stage_schema": "perception-evidence-relation-synthesis",
    "num_steps": 4,
    "candidate_disable_reason": "short_answer"
  }
}
```

`cot_step_mask` 必须进入 Stage 2/3 collator 和 loss。关闭 CoT 的一侧仍参与 retrieval loss，但不参与 state/transition distillation。

### 5.9 CoT 质量过滤

离线脚本至少检查：

- JSON 与字段格式是否合法；
- 启用 CoT 时是否严格包含 `K` 个阶段；
- 单步是否为空或超长；
- 是否描述输入中不存在的实体、数值、区域或事件；
- 是否出现答案、正样本 ID 或配对信息泄漏；
- 相邻阶段是否只是同义改写；
- Stage 2 是否真正筛选证据；
- Stage 3 是否明确关系或说明无额外关系；
- Stage 4 是否为检索语义而非直接答案；
- 是否出现高度模板化、复制粘贴或多步完全相同；
- 保存失败原因、重试次数、生成模型和 prompt 版本。

建议额外计算：

- 相邻阶段文本重复率；
- 每阶段 token 长度分布；
- 各数据集 CoT 启用率；
- Query/Candidate 泄漏关键词命中率；
- 抽样人工检查通过率。

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

让 `<STEP_i>` hidden states 从普通文本位置状态转化为具有检索语义的阶段性教师状态，使四个宏观检查点分别承载模态感知、证据筛选、关系建模和检索语义综合功能。

### 输入

Query 和 Candidate 分别拼接自己的结构化 CoT。只有 `cot_enabled=true` 的一侧才构造 explicit view；短标签或短答案 Candidate 不强制拼接 CoT。

### 初始化

从 Stage 1 checkpoint 初始化。

### Forward

```python
q_out = encode_explicit(
    query_explicit_batch,
    step_positions=query_step_positions,
    step_mask=query_step_mask,
)

c_out = encode_explicit(
    candidate_explicit_batch,
    step_positions=candidate_step_positions,
    step_mask=candidate_step_mask,
)
```

输出：

```python
q_states: Tensor[B, K, H]
q_step_mask: BoolTensor[B, K]
c_states: Tensor[B, K, H]
c_step_mask: BoolTensor[B, K]
```

关闭 Candidate CoT 的样本，其 `candidate_step_mask` 全为 0，不计算 Candidate 过程损失。

### Loss

#### 1. 最终阶段检索损失

对存在完整 explicit view 的两侧，使用最终有效 `<STEP>` 状态构造检索 embedding。对于 Candidate 无 CoT 的样本，Candidate 继续使用 Stage 1 的普通 discriminative encoding，与 Query explicit embedding 进行 InfoNCE。

#### 2. 中间阶段辅助检索损失

对有效阶段使用递增权重：

```text
alpha = [0.25, 0.50, 0.75, 1.00]
```

\[
\mathcal L_{\mathrm{explicit}}
=
\mathcal L_{\mathrm{final-ret}}
+
\lambda_{\mathrm{step}}
\sum_{k=1}^{K}\alpha_k
\mathcal L_{\mathrm{ret}}^{(k)}.
\]

所有阶段损失必须按 `step_mask` 归一化。中间阶段只要求总体具有检索判别性，不强制每条简单样本的相似度按固定 margin 单调上升。

### 注意

- Stage 2 不训练 recurrent student；
- Stage 2 不需要 EMA；
- `<STEP_i>` 位置必须从 tokenizer 后的真实 token index 提取；
- 建议注册独立的 `<STEP_1>` 至 `<STEP_K>`，避免重复 token 位置解析歧义；
- 若使用统一 `<STEP>`，collator 必须返回按出现顺序排列的 `step_positions[B, K]`；
- `step_positions=-1` 的位置必须由 `step_mask=0` 屏蔽；
- Stage 2 结束后用该 checkpoint 同时初始化 Student 和 Teacher；
- 需要分别统计 symmetric-distill 与 query-only-distill 样本的损失。

### 输出 checkpoint

```text
checkpoints/elder_stage2_explicit/
```

### 验收

- 最终阶段检索性能高于 Stage 1 或至少不明显下降；
- 中间阶段平均检索性能随阶段总体改善或趋于稳定；
- 不要求所有简单样本严格单调提升；
- 相邻 teacher states 不应全部相同；
- `cos(t_k, t_{k+1})` 与 `||t_{k+1}-t_k||` 分布合理；
- 四个阶段的文本和 hidden-state 统计具有可区分性；
- Query-only 样本能正常训练，Candidate 无 CoT 时不产生非法 step loss。

---

## Stage 3：EMA Explicit-to-Latent Process Distillation

### 初始化

```python
student.load(stage2_checkpoint)
teacher.load(stage2_checkpoint)
teacher.requires_grad_(False)
```

### Teacher view

Teacher 只对 `cot_enabled=true` 的输入读取原始内容 + CoT，输出：

```python
t_q: [B, K, H]
t_c: [B, K, H]
mask_q: [B, K]
mask_c: [B, K]
```

短标签/短答案 Candidate 的 `mask_c` 全为 0。它们不产生 Candidate state/transition loss，但仍参与最终检索对比学习。

### Student view

Student 始终只输入原始多模态内容，并对 Query 和 Candidate 执行固定 `K` 步 recurrent feedback：

```python
z_q: [B, K, H]
z_c: [B, K, H]
e_q: [B, D]
e_c: [B, D]
```

固定 `K` 是学生的统一 latent process scaffold 和最大计算预算。简单样本的后续状态可以执行确认、去噪和稳定化，不要求每一步都增加新的复杂关系。

### 关键 Loss

#### 1. Retrieval loss

最终 latent states 生成 embedding，使用 InfoNCE：

\[
\mathcal L_{\mathrm{ret}}
=
\operatorname{InfoNCE}(e_q,e_c).
\]

该损失对所有样本生效，与是否具有双侧 CoT 无关。

#### 2. Masked state distillation

设 `m_k^x ∈ {0,1}` 表示输入侧 `x` 的第 `k` 个教师状态是否有效：

\[
\mathcal L_{\mathrm{state}}
=
\frac{
\sum_{x\in\{q,c\}}
\sum_{k=1}^{K}
m_k^x
\left[
1-\cos\left(
\operatorname{LN}(z_k^x),
\operatorname{sg}(\operatorname{LN}(t_k^x))
\right)
\right]
}{
\sum_{x\in\{q,c\}}\sum_{k=1}^{K}m_k^x+\epsilon
}.
\]
#### 3. Masked transition distillation

只有相邻两个教师阶段均有效时才计算转移监督：

\[
\tilde m_k^x=m_{k-1}^x m_k^x,
\]

\[
\mathcal L_{\mathrm{trans}}
=
\frac{
\sum_{x\in\{q,c\}}
\sum_{k=2}^{K}
\tilde m_k^x
\left[
1-\cos\left(
 z_k^x-z_{k-1}^x,
 \operatorname{sg}(t_k^x-t_{k-1}^x)
\right)
\right]
}{
\sum_{x\in\{q,c\}}\sum_{k=2}^{K}\tilde m_k^x+\epsilon
}.
\]
转移损失用于约束宏观检查点之间的信息演化方向，避免四个 latent states 退化为重复表示。它不要求每条样本的状态变化范数完全相同。

#### 4. Rank distillation（可选，默认实现但可关闭）

Teacher 和 Student 在同一 in-batch candidate set 上形成相似度分布，使用 KL：

\[
\mathcal L_{\mathrm{rank}}
=
D_{\mathrm{KL}}(P^T\Vert P^S).
\]

若 Candidate 没有 explicit CoT，rank teacher 可选择：

- 关闭该样本的 rank distillation；或
- 使用 Teacher 的普通 discriminative Candidate embedding。

主实现必须通过配置显式选择，禁止静默混用。

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
\lambda_r\mathcal L_{\mathrm{rank}}.
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

为了恢复训练，也可以额外保存 EMA teacher、optimizer、scheduler 和 CoT schema version。

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

Stage 2/3 的 collator 需要同时返回 Student 原始输入、Teacher explicit 输入以及阶段掩码：

```python
{
  "qry_student": processor_output,
  "tgt_student": processor_output,
  "qry_teacher": Optional[processor_output],
  "tgt_teacher": Optional[processor_output],
  "qry_step_positions": LongTensor[B, K],
  "tgt_step_positions": LongTensor[B, K],
  "qry_step_mask": BoolTensor[B, K],
  "tgt_step_mask": BoolTensor[B, K],
  "qry_cot_enabled": BoolTensor[B],
  "tgt_cot_enabled": BoolTensor[B],
  "sample_ids": list[str],
}
```

要求：

- Student batch 中不得出现 CoT；
- Teacher batch 只为 `cot_enabled=true` 的一侧包含 CoT；
- 无 CoT 的位置统一令 `step_positions=-1`、`step_mask=0`；
- padding side 与 backbone 保持一致；
- `<STEP_i>` 注册为 special token；
- 若修改 tokenizer vocab，模型 embedding matrix 必须 resize；
- step position 在 padding 后仍正确；
- 有效 CoT 必须恰好有 `K` 个阶段锚点；
- loss 必须按有效 mask 数量归一化，避免 query-only batch 改变损失尺度；
- 不符合 schema 的样本在预处理阶段过滤，不能在训练时随机报错；
- collator 日志应统计每个 batch 的 Query/Candidate CoT 启用率。

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
cot_stage_schema: perception-evidence-relation-synthesis
cot_generator: zai-org/GLM-4.1V-9B-Thinking
max_cot_tokens: 384
query_cot_policy: always_for_stage23
candidate_cot_policy: semantic_richness_gate
short_target_max_tokens: 4
num_video_frames: 8
per_device_batch_size: 1-4
gradient_accumulation_steps: 按 global batch 调整
deepspeed: ZeRO-2
```

Stage 3 同时存在 Student 和 Teacher，因此 batch 应从小规模开始。Teacher 无梯度，但仍需要参数和 forward 显存。

### 扩展实验

- `K ∈ {1, 2, 4, 6, 8}`，主表至少报告 `K=1/2/4/8`；
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
12. 不同 CoT 质量或不同 CoT 数据量；
13. GLM-4.1V-Thinking vs 其他强 MLLM 生成器；
14. 四阶段检索导向 CoT vs 自由 CoT / Embed-RL 三段式直接迁移；
15. 短 Target 强制生成 CoT vs semantic-richness gate；
16. `K=1/2/4/6/8` 的性能—延迟—显存曲线。

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

### 风险 1：CoT 阶段只是机械切分或相互改写

处理：

- 使用固定的四类宏观语义功能，而不是按句子数量切分自由 CoT；
- 强制 Stage 2 进行证据筛选、Stage 3 建模关系或明确“无额外关系”；
- 监控相邻阶段文本重复率、teacher transition norm 和 state cosine；
- 对简单样本允许后期阶段承担确认、去噪和压缩，不要求虚构复杂推理；
- 对短标签/答案 Candidate 关闭过程蒸馏；
- 抽样人工复核不同模态的阶段语义一致性。

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
- `K` 与 `cot_stage_schema`；
- CoT 生成模型、prompt version、temperature 和重试次数；
- Query/Candidate CoT 启用策略；
- short-target 判定阈值与 semantic-richness gate；
- feedback adapter；
- state/transition/rank loss 权重；
- EMA decay；
- CoT 最大长度和单阶段长度；
- dataset weights；
- Query/Candidate 是否都蒸馏；
- `step_positions` 与 `step_mask` 字段；
- cache/gradient mode；
- LoRA 和 distributed training；
- checkpoint 路径；
- debug subset size；
- CoT schema version 与数据生成版本。

---

## 19. 参考实现与事实依据

Codex 开发时优先参考以下公开资源：

1. `TIGER-AI-Lab/VLM2Vec`：数据加载、interleaved sub-batching、GradCache、InfoNCE、MMEB-V2 评测和 Qwen-VL processor。
2. VLM2Vec-V2：训练数据由 MMEB-train、LLaVA-Hound、ViDoRe 和 VisRAG 构成；ELDER 沿用其通用多模态数据组织方式。
3. `haoxiangzhao12138/PLUME`：Qwen2-VL 中连续 latent rollout、KV cache、position IDs 和 multimodal prefix 的工程处理可作为参考，但 ELDER 不采用其 progressive replacement curriculum。
4. UME-R1：可参考其在 MMEB-train、LLaVA-Hound、ViDoRe 和 VisRAG 上构造 Query/Target 推理数据的组织方式；ELDER 不使用 RL，也不强制为所有短 Target 生成完整过程。
5. Embed-RL：可参考其检索导向 T-CoT 中的关键词、图像区域、视频关键帧和 evidence rethinking 思路；ELDER 将这些信息重新规范化为“内容感知—证据筛选—关系建模—检索综合”四个宏观检查点。
6. GLM-4.1V-9B-Thinking：作为第一版离线 CoT 生成器候选；正式大规模生成前必须在 ELDER 自建多模态开发集上验证格式遵循、视觉忠实度、阶段差异性和吞吐量。
7. ELDER 的核心差异：通过固定分辨率的结构化显式过程，为 recurrent latent states 提供阶段状态与状态转移监督；固定 `K` 是过程对齐接口和计算预算，不是对真实推理步数的强假设。

---

## 20. 一句话任务描述

> 在 VLM2Vec-V2 代码库上实现 ELDER：先进行判别式检索预热，再用输入侧独立生成的四阶段检索导向 CoT 预热 `<STEP_i>` states，最后通过 EMA teacher 将固定分辨率的显式阶段状态和状态转移蒸馏到 Student 的 \(K\) 次 recurrent hidden-state feedback 中；短标签/答案 Candidate 仅参与检索损失，推理时只保留 Student，并输出可直接用于向量检索的 embedding。
