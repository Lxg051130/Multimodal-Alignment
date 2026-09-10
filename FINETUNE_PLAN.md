# GRAM/CLIP 下游微调路线评估（RTX 4090 视角）

> 生成日期：2026-09-09  
> 依据：本仓库代码 + codegraph 索引 + 仓库自带 annotation 统计 + 官方公开权重说明

## 1. codegraph 扫描结论

已在本仓库初始化并索引 codegraph（`.codegraph/`）：

- 79 个 Python 文件，1,529 个节点，3,213 条边；
- 主要调用链：
  `run.py -> utils/build_model.py::build_model -> model/gram.py::GRAM -> general_module（EVA-CLIP/BEATs/BERT 编码器）`
  `run.py -> utils/pipeline.py::train -> model/gram.py::forward_ret -> utils/volume.py::volume_computation3/4/5`
  `evaluation/evaluation_mm.py::evaluate_ret` 负责 retrieval 指标。

架构要点：

- 默认视觉编码器是 `evaclip01_giant`（EVA01-CLIP-g-14，约 1.1B 参数，fp16 checkpoint 约 2.2 GB）；
- 音频编码器默认 BEATs（约 90M 参数）；
- 文本/多模态编码器是同一个 BERT-base（约 109M），同时负责文本编码和带 cross-attention 的下游生成；
- 对比学习用 Gramian volume loss，而不是 cosine InfoNCE；最终特征统一投影到 `contra_dim=512`；
- 默认 `default_model_cfg.json` 中 `frozen_vision=false`、`frozen_audio=false`，即官方下游微调是全参训练；
- `run.py` 只支持 `torch.distributed.launch` 启动，即使单卡也需要 `--nproc_per_node 1`；
- 仓库只有 `fp16` AMP 路径，没有 DeepSpeed/FSDP/8-bit Adam，也没有 LoRA 注入点。

## 2. 本地数据和权重现状

### 数据集

`datasets/annotations/` 下只有 annotation JSON，没有视频/音频/图片原始文件。各 annotation 内的 media 路径是作者服务器的 `/mnt/...`、`/leonardo_...` 或仓库假设的 `datasets/srcdata/...`，本机不存在，需要按官方来源下载原始数据并改路径。

本地现有 annotation 统计（JSON 行数，不是去重后的 clip 数）：

| 数据集 | 可用 annotation | 大致规模 | 官方配置里的任务 |
|---|---:|---|---|
| MSRVTT | ret / cap / QA | unique train 9,000 条；ret_test 1,000；cap_train 130k；QA_trainval 170k | `ret%tv%ta`、cap、qa |
| VATEX | ret / cap | ret_train 259,910；ret_test 1,500 | `ret%tvas`、cap |
| ActivityNet | ret / QA | ret_train 10,009；ret_test 4,917；QA_train 32,000；QA_test 800 | `ret%tva`、qa |
| DiDeMo | ret | train 8,363；test 1,003 | `ret%tva` |
| AudioCaps | ret / cap | ret_trainval 44,102；ret_test 729 | 官方配置里同时写了 video+audio，若只用 audio 需要改代码 |
| YouCook2 | cap / ret | cap_train 9,283；cap_test 3,216 | 用 cap annotation 做检索 |

建议按这个顺序做数据“冒烟测试”：

1. MSRVTT：`unique_descs_ret_train.json` + `descs_ret_test.json`；
2. DiDeMo：规模最小，适合验证训练流程；
3. VATEX / ActivityNet / MSRVTT-QA：留给更大资源的任务。

### 权重

以下权重本仓库目录中都没有，需要先准备：

```text
pretrained_weights/
  clip/EVA01_CLIP_g_14_psz14_s11B.pt
  beats/BEATs_iter3_plus_AS2M.pt
  bert/bert-base-uncased/            # 完整 HF BERT 目录

gram_ckpt/                           # 任意 GRAM_pretrained_* 或 finetuned_* 
  log/hps.json
  ckpt/model_step_*.pt
```

官方模型 zoo 里的 `GRAM_pretrained_4modalities` / `GRAM_pretrained_5modalities` 都可以作为下游任务起点；`build_model` 会读取 `log/hps.json` 中的 encoder 类型并把 checkpoint 加载进模型。

## 3. 三个从易到难的下游任务

> 目标不是“预训练 GRAM”，而是在公开视频/文本/音频数据上做自己的下游微调，并逐步增加训练参数比例和输入规模。

### T1：冻结 GRAM 骨干 + 轻量头微调 / 线性探测

- **目标**：先在 MSRVTT 或 DiDeMo 上打通 GRAM 推理/特征链路，并验证“只用 GRAM 512 维 embedding + 轻量头”能否追到一定检索指标。
- **数据**：MSRVTT unique train（9,000 clip）→ test 1,000；或 DiDeMo 8,363/1,003。
- **权重**：`GRAM_pretrained_4modalities`（内含 EVA-CLIP-g-14 + BEATs + BERT）。
- **训练范围**：只训练 `contra_head_*`、`hidden_trans_*`、`itm_head`、`vision/audio_frame_embedding`；EVA-CLIP、BEATs、BERT 全部冻结。
- **实现路径**：最省事的做法是先用 `--mode testing` 把 video/audio/text embedding 离线抽出来存盘，再写一个轻量线性/MLP head 做跨模态检索；若要在 run.py 里做，需要补一段“除 head 外全部 `requires_grad=False`”的逻辑。
- **显存预估**：权重即使保持 fp32 也只有约 5 GB，冻结下无反向激活，批大小 2–4、4 帧时约 **6–9 GB**。
- **4090 结论**：可以，而且很轻松。
- **预计时间**：数据下载和抽特征通常比训练本身更久；训练/验证数小时内可完成。

### T2：GRAM 参数高效微调（PEFT / 部分层解冻 + volume loss）

- **目标**：开始真正调整 GRAM 的视觉/音频/文本特征，观察 volume contrastive 微调能否超过冻结基线。推荐用 LoRA 或只解冻每棵编码器最后几层。
- **数据**：DiDeMo 或 ActivityNet；想加难度再用 VATEX 子集。
- **权重**：从 `GRAM_pretrained_4modalities` 继续微调。
- **训练范围**：
  - 首选：EVA-CLIP visual q/k/v/mlp、BEATs、BERT 的 attention/FFN 加 LoRA；
  - 退而求其次：只解冻各编码器最后 2–4 层 + 全部投影头。
- **关键配置**：`--checkpointing true`（激活重计算）、`--fp16 true`、每步 micro-batch 1–4 + 梯度累积到 16–32、train frame 4、audio sample 1、`vision_resolution 224`。
- **显存预估**：冻结骨干 fp32 约 5 GB + LoRA/小规模可训参数 + 跨模态激活，约 **12–18 GB**。
- **4090 结论**：可以，但仓库目前没有 LoRA 实现，需要做一次性 PEFT 改造；不改代码的话，只做“部分层解冻 + 头训练”也可以落在 24 GB 内。
- **预计时间**：单个数据集数天到一周（取决于步骤数和是否逐帧解码）。

### T3：GRAM 全参下游微调 / 视频问答与视频字幕（生成式）

- **目标**：复现官方风格的 retrieval 微调，或者做更重的下游：MSRVTT-QA、ActivityNet-QA、VATEX caption。
- **数据**：MSRVTT-QA（约 170k 条）、VATEX cap（约 290k 条）或官方 `retrieval-vatex.json` 配置。
- **权重**：从 `GRAM_pretrained_4modalities` 或对应官方 finetuned checkpoint 继续全参微调。
- **训练范围**：EVA-CLIP visual/text tower、BEATs、BERT、全部头一起训练。
- **显存预估**：
  - 模型约 1.3B+ 参数；fp32 权重约 5 GB，Adam 双 moment 约 10.6 GB，梯度约 5 GB，仅“参数+优化器+梯度”就约 21–22 GB，超过 24 GB 卡可安全使用的净显存；
  - 加上 8 帧 224×224 的 EVA-CLIP 激活、BEATs 激活、BERT 对约 2,000 个 condition token 的 cross-attention，官方 4×A100/H100 的配置是单卡跑不下来的；
  - `use_ddp=true` 只同步梯度，不切分模型，双 4090 直接跑 DDP 每张卡仍是全模型副本，不能解决问题；需要 FSDP/ZeRO + 8-bit Adam 才有机会。
- **4090 结论**：官方形态**不行**。若一定要在本地做，只能把它“降级”成：冻结 EVA-CLIP/BEATs、只全参微调 BERT 融合/生成头，或切到更小的 CLIP（如 OpenAI ViT-B/16）重新设计轻量 GRAM 头；否则建议租 1×A100-80GB 或 4×A100/H100。

## 4. 显存测算备忘

| 阶段 | 可训练参数 | 输入 | 预估峰值显存 | 1×RTX 4090 |
|---|---:|---:|---:|---|
| T1 冻结骨干 + head | ~10M | 4 帧视频 + 1 音频 + 文本，batch 2–4 | 6–9 GB | 可以 |
| T2 PEFT / 部分解冻 | ~30–80M | 4 帧视频 + 1 音频 + 文本，batch 1–4，grad ckpt | 12–18 GB | 可以（需 LoRA/冻结改造） |
| T3 全参 retrieval/QA/caption | ~1.3B | 8 帧视频 + 2 音频，batch 8–16 | >30 GB | 不可以（官方配置） |

几个容易误判的点：

1. 仓库默认从头构造整个 EVA-CLIP `CustomCLIP`（含 text tower），即使 GRAM 文本路径主要走 BERT；动手前建议确认并裁掉真正不用的参数，能省约 100–200 MB 级别内存。
2. 仓库的 `batch_size` 在 `build_dataloader` 中还会除以 world size。官方脚本里写 32/64 通常对应 4–8 卡，单卡跑时要显式用 `--train_batch_size` 压小。
3. `--checkpointing true` 会关闭 `use_ddp`，不影响单卡，但会明显降低吞吐；对 4090 的 24 GB 来说通常是“保命开关”。
4. 评估时的 ITM rerank 会为每个 candidate 跑 BERT，`--itm_rerank_num` 和 val batch 要调小，否则显存峰值可能出现在 evaluation 而不是 training。

## 5. 建议执行顺序

1. 下载 EVA-CLIP/BEATs/BERT/GRAM 权重并核对路径；
2. 下载 MSRVTT 或 DiDeMo 视频 + 音频，按 annotation 里的 `video_id` 改名/放到配置路径下；
3. 先用 `--mode testing --zero_shot` 跑通 T1 的 embedding/指标，确认环境与数据；
4. 做一次“head-only 微调”作为基线（T1）；
5. 给 GRAM 加 LoRA/冻结逻辑（T2），在 DiDeMo 上验证；
6. 如果 T2 效果稳定，再评估是否有预算上 A100/H100 做 T3；否则把 T3 改成“冻结大骨干 + 微调生成头”的本地可行版本。
