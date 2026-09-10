# GRAM 单卡(4090/24GB)下游微调：三个任务的数据、权重、代码与跑法

本文档把根目录 `FINETUNE_PLAN.md` 的三个任务落地成可直接运行的代码，
并回答三件事：数据集去哪下、权重去哪下、每个任务怎么跑。

代码库核实结论（codegraph + 源码）：
GRAM 的 retrieval 训练/评测是 **text+video+audio 三模态**的
（`forward_ret` 的 volume loss 与 `evaluate_ret` 无条件使用
`feat_t/feat_v/feat_a`），所以三个任务都要准备“视频 + 从视频抽出的音频 +
文本”，不能只给视频。

---

## 0. 目录结构

```text
GRAM/
├── datasets/annotations/...   # 仓库自带，无需下载（只缺原始视频/音频）
├── data/vision_mapper.py      # 已加目录列表缓存（本方案唯一的仓库改动，纯提速）
├── pretrained_weights/
│   ├── clip/EVA01_CLIP_g_14_psz14_s11B.pt     # EVA-CLIP-g（GRAM 的 vision tower）
│   ├── beats/BEATs_iter3_plus_AS2M.pt          # BEATs（audio tower）
│   └── bert/bert-base-uncased/                 # HF BERT（text + fusion tower）
├── gram_ckpt/GRAM_pretrained_4modalities/      # GRAM 预训练模型（log/ + ckpt/）
└── finetune_4090/          # 本方案
    ├── README_TASKS.md
    ├── make_cfgs.py        # 生成任务配置 json（替换本地路径）
    ├── prepare_data.py     # 抽音频 + annotation∩本地视频 交集过滤
    ├── peft_utils.py       # 冻结 / LoRA / 解冻尾部 的实现
    ├── run_tasks.py        # 单卡训练器（无需 DDP/wandb）
    ├── run_t1.sh / run_t2.sh / run_t3.sh
    └── cfgs/               # make_cfgs.py 生成的配置
```

---

## 1. 权重下载（三个任务共用）

### 1.1 三个基础 encoder（EVA-CLIP、BEATs、BERT 都从 Hugging Face 下载）

| 权重 | Hugging Face 下载地址 | 本机目标位置（相对仓库根目录） |
|---|---|---|
| EVA-CLIP-g-14（约 2.2 GB） | `https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt` | `pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt` |
| BEATs_iter3_plus_AS2M（约 345 MB） | `https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt` | `pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt` |
| BERT-base-uncased | <https://huggingface.co/bert-base-uncased> | `pretrained_weights/bert/bert-base-uncased/`（完整 HF 目录：config.json + pytorch_model.bin + vocab.txt 等） |

```powershell
# 在仓库根目录 D:\PycharmProjects\GRAM 下执行
New-Item -ItemType Directory -Force pretrained_weights\clip, pretrained_weights\beats | Out-Null

curl.exe -L -o pretrained_weights\clip\EVA01_CLIP_g_14_psz14_s11B.pt `
  https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt

curl.exe -L -o pretrained_weights\beats\BEATs_iter3_plus_AS2M.pt `
  https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt
```

```python
from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('bert-base-uncased').save_pretrained(
    'pretrained_weights/bert/bert-base-uncased')
BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained(
    'pretrained_weights/bert/bert-base-uncased')
```

> 你当前这台机器的 `pretrained_weights/bert/bert-base-uncased/` 已经存在完整
> `pytorch_model.bin`，不需要重复下载；EVA-CLIP 和 BEATs 尚未看到，需要补齐。
> GRAM 源码用**相对路径硬编码**加载这三处：`./pretrained_weights/...`，
> 所以 encoder 权重必须放在仓库根目录下，不能随意放到 `D:/weights`。

> “GRAM + CLIP”：GRAM 官方仓库把 **EVA-CLIP-g-14** 作为视觉 tower，上面的
> EVA-CLIP 权重就是代码里 `vision_encoder_type=evaclip01_giant` 使用的那一个。

### 1.2 GRAM 官方权重

GRAM 自己的多模态预训练 checkpoint 目前**只发在 Google Drive**，没有官方
Hugging Face 仓库；这是本方案唯一没法“全放 HF”的一项。

| 需要的目录 | 地址 | 用途 |
|---|---|---|
| `GRAM_pretrained_4modalities` | <https://drive.google.com/drive/folders/1mD9PDvugLx3t1KtTCwJtYO8VZDFKfMgs> | 三个任务的共同起点（推荐） |
| Model Zoo 总目录 | <https://drive.google.com/drive/folders/15CGPSut2Bgcsuce1Fjaozfts0f9QK1Ya> | 想拿 `GRAM_finetuned_DIDEMO/_MSRVTT/_VATEX` 做对照时在这里找 |

下载后目录必须保持（本方案三个任务都会把这条路径作为 `--pretrain_dir`）：

```text
gram_ckpt/GRAM_pretrained_4modalities/
├── log/hps.json
└── ckpt/model_step_<step>.pt
```

本方案会读 `log/hps.json` 的 `model_cfg` 自动沿用 encoder 类型，所以不要混放。
Windows 下建议直接浏览器打开上面链接整目录下载；命令行可用：

```powershell
python -m pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1mD9PDvugLx3t1KtTCwJtYO8VZDFKfMgs" -O gram_ckpt
```

### 1.3 关于 EVA-CLIP 名称，以及“能不能换成别的 CLIP”

先纠正一个容易搜不到的点：**它叫 EVA-CLIP（EVA01-CLIP-g-14），不叫
“EVAL-CLIP”**。Hugging Face 的 `QuanSun/EVA-CLIP` 仓库里就有 GRAM 需要的那个文件
（`EVA01_CLIP_g_14_psz14_s11B.pt`），所以不存在“找不到要换一个”的问题。

那能不能换成 OpenAI CLIP（ViT-B/16 等）当替代？

- 只替换 encoder 权重文件：**不行**。GRAM 的多模态 checkpoint 是在 EVA-CLIP-g-14
  结构上训练的（40 层 Transformer、width 1408、patch 14），权重键名和形状都对不上
  OpenAI CLIP 的 ViT-B/16，`load_state_dict` 会大面积 missing/unexpected，
  等价于没加载上。
- 仓库源码确实支持 `vision_encoder_type=clip_vit_base_16 / clip_vit_large_14_336px`
  等 OpenAI CLIP 变体，但那是“从零搭一个 GRAM 结构、不加载官方 GRAM 多模态权重”
  的另一条路线，需要自备 OpenAI 的 `ViT-B-16.pt`（torch.jit 格式）并重新预训练/对齐，
  不是本方案里 T1–T3 的“GRAM 预训练权重下游微调”。

因此本方案默认：三个基础 encoder 全部从 Hugging Face 拿，GRAM 多模态权重从
官方 Google Drive 拿，二者不互换。

---

## 2. 数据集下载与本地化

仓库自带 GRAM 处理好的 annotation（train/test JSON），**不需要再找 annotation**；
你只需要补原始视频，再运行 `prepare_data.py`：给每个 `<video_id>` 抽
`<video_id>.mp3`，并把 annotation 与本地视频做交集，输出 `annos_ret_train.json` /
`annos_ret_test.json`。

三个任务的统一目录约定（T1/T2/T3 完全一样，只是根目录不同）：

```text
D:\datasets\didemo\videos\...      # T1：<video_id>.mp4
D:\datasets\msrvtt\videos\...      # T2：video0.mp4 ... video9999.mp4
D:\datasets\vatex\videos\...       # T3：<video_id>.mp4
```

视频文件**平铺**在 `videos\` 下，文件名就是 annotation 里的 `video_id`（不带目录层级）。

### T1：DiDeMo

- annotation：仓库 `datasets/annotations/didemo/`（train 8,363 描述 / test 1,003）
- 原始视频（Flickr 原片约 10.6k 个，约 20–25 GB）：
  - 官方入口 <https://github.com/LisaAnne/LocalizingMoments>
  - 社区整理 <https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo>
    （内含 train/test 清单与直链）
  - HF 镜像：<https://huggingface.co/datasets/friedrichor/DiDeMo>、
    <https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/tree/main/didemo>
- 文件命名：`<video_id>.mp4`（示例 `100233434@N08_9622921646_c39ac10ea3`）

### T2：MSRVTT（10k 视频 / 200k 描述）

**是，直接下载一个整包 zip（约 10 GB 级别，不同镜像约 6–20 GB 都有说法）。**
推荐用 Frozen-in-Time 备份的 `MSRVTT.zip`，一个包内就是 10,000 个短视频：

```powershell
# 1) 下载（也可以直接用浏览器打开下面的链接另存）
curl.exe -L -o D:\datasets\download\MSRVTT.zip `
  https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip

# 2) 解压（Windows 自带的 Expand-Archive 解大包容易慢/失败，优先用 7-Zip）
7z x D:\datasets\download\MSRVTT.zip -oD:\datasets\download\msrvtt_extract

# 3) 不管 zip 内部是 AllVideo/、TrainValVideo/ + TestVideo/ 还是视频直接在根目录，
#    一律把所有 video*.mp4 平铺到一个 videos 目录（prepare_data.py 只认这个平铺结构）
New-Item -ItemType Directory -Force D:\datasets\msrvtt\videos | Out-Null
Get-ChildItem D:\datasets\download\msrvtt_extract -Recurse -Filter *.mp4 |
  Copy-Item -Destination D:\datasets\msrvtt\videos -Force
(Get-ChildItem D:\datasets\msrvtt\videos).Count   # 应看到 10000
```

最后的数据根目录长这样（后面的命令都传 `D:/datasets/msrvtt` 这一个根目录）：

```text
D:\datasets\msrvtt\
├── videos\video0.mp4 ... video9999.mp4   ← 第 3 步放进来
├── audios\video0.mp3 ...                ← prepare_data.py --extract-audio 自动抽
├── annos_ret_train.json                 ← prepare_data.py 自动生成
└── annos_ret_test.json                  ← prepare_data.py 自动生成
```

- annotation：仓库 `datasets/annotations/msrvtt/`（train=9,000 唯一视频的 180k 描述；
  test=官方 1k-A 的 1,000 视频）
- 原始视频（zip 内 10k 个 mp4；不同镜像对压缩/解压体积口径不一，约 6–20 GB）：
  - Frozen-in-Time 备份直链：`https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip`
  - HF 镜像：<https://huggingface.co/datasets/friedrichor/MSR-VTT>、
    <https://huggingface.co/datasets/morpheushoc/msrvtt>
- 文件命名：`video0.mp4 ... video9999.mp4`
- zip 里若还带 json/csv/CLIP 特征等文件，忽略即可，**不要把它们当作 annotation 或
  vision 目录**；上面第 3 步的 `-Filter *.mp4` 只会搬视频。
- 若不想一开始就下载全量 10k，可先只从解压目录复制几十个 `video*.mp4` 到
  `videos\` 做冒烟；下面的 `--limit` / `--cap-per-video` 会自动按“本地实际存在的
  视频”来过滤，不会因为少了文件而报错。

### T3：VATEX（26k 视频 / 260k 描述）

- annotation：仓库 `datasets/annotations/vatex/`
- 原始视频：
  - 官方页面 <https://eric-xw.github.io/vatex-website/index.html>
  - HF：<https://huggingface.co/datasets/HuggingFaceM4/vatex>、
    <https://huggingface.co/datasets/qingy2024/VaTeX>
- 文件命名：`<video_id>.mp4`（示例 `Ptf_2VRj-V0_000122_000132`）

> 建议先 `--limit N` 下载小子集冒烟，再全量。

---

## 3. 三个任务总览（GPU 用量从少到多）

| 任务 | 数据（默认） | 训练范围 | 4090 显存（估） | 吞吐/时间 |
|---|---|---|---|---|
| T1 | DiDeMo 8.3k | 冻结 EVA-CLIP/BEATs；训练 BERT + 各投影头 | ~8–12 GB | 分钟级/千步；1 epoch 小时级 |
| T2 | MSRVTT（脚本先 2k 视频子集 × 每视频 1 caption） | T1 + BERT 的 LoRA(q/k/v/FFN) | ~10–15 GB | 千步/数小时级 |
| T3 | VATEX（脚本先 4k 视频子集 × 每视频 1 caption） | T2 + 解冻 EVA 尾部 N 块、BEATs 尾部 M 层 | ~18–23 GB | 4090 单卡半天级起步 |

官方那种“1.1B 全参数解冻”不进 4090 清单，原因见 FAQ 4。

---

## 4. 每个任务怎么跑

前置：装好第 1 节权重；把视频放到 `<root>/videos/`。命令都在仓库根目录执行。

### 4.1 T1：DiDeMo —— 冻结主干，训练 BERT + heads

```bash
bash finetune_4090/run_t1.sh \
  /path/to/didemo \
  /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  /path/to/outputs
```

手动分解（Windows 用 PowerShell 也是这几行）：

```bash
python finetune_4090/prepare_data.py --name didemo \
  --video-dir /path/to/didemo/videos \
  --audio-dir /path/to/didemo/audios \
  --data-root /path/to/didemo --extract-audio

python finetune_4090/make_cfgs.py --task t1 --data-root /path/to/didemo --epochs 5

python finetune_4090/run_tasks.py --task t1 \
  --config finetune_4090/cfgs/t1_didemo.json \
  --pretrain_dir /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  --output_dir /path/to/outputs/t1_didemo \
  --train_batch_size 2 --epochs 5 --lr 3e-5 \
  --eval_steps 1000 --save_steps 2000 --first_eval
```

要点：
- `make_cfgs.py` 生成的 json 里 `frozen_vision/frozen_audio=true`，视觉/音频主干只前向；
- `run_tasks.py` 训练时自动使用 volume-only 版 `forward_ret`（同款 volume loss，
  跳过官方训练里最贵的 BERT-ITM 分支），评测仍走仓库原生 `evaluate_ret`；
- 首次建议 `--first_eval` 看零样本基线；终端找 `ret_itc_tv.forward_r1` /
  `volume_ITM_T2D.forward_r1`。

### 4.2 T2：MSRVTT —— BERT-LoRA

```bash
bash finetune_4090/run_t2.sh \
  /path/to/msrvtt \
  /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  /path/to/outputs
```

手动分解：

```bash
python finetune_4090/prepare_data.py --name msrvtt \
  --video-dir /path/to/msrvtt/videos \
  --audio-dir /path/to/msrvtt/audios \
  --data-root /path/to/msrvtt --extract-audio \
  --limit 2000 --cap-per-video 1

python finetune_4090/make_cfgs.py --task t2 --data-root /path/to/msrvtt --epochs 3

python finetune_4090/run_tasks.py --task t2 \
  --config finetune_4090/cfgs/t2_msrvtt.json \
  --pretrain_dir /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  --output_dir /path/to/outputs/t2_msrvtt_lora \
  --train_batch_size 2 --epochs 3 --lr 1e-4 \
  --lora_r 16 --lora_alpha 32 --eval_steps 1000
```

要点：
- `--limit 2000` = 只保留约 2,000 个视频；
- `--cap-per-video 1` = 每个视频只保留 1 条描述（MSRVTT 默认一个视频 20 条，
  不限制的话 2,000 个视频会膨胀成 ~32k 行，明显拖慢）；
- 想跑官方 9k×20=180k 描述的完整规模时，把这两个参数都去掉（那属于 4×A100
  级别的全量训练，不在 4090 默认清单里）；

`run_tasks.py` 会给 BERT 的 `attention.self.query/key/value`、
`intermediate.dense`、`output.dense` 注入 LoRA（基座冻结，A/B 训练），
启动时打印可训练参数量。

**先跑冒烟（PowerShell，可直接照抄）**：只放几十~几百个视频进
`D:\datasets\msrvtt\videos\`，或用 `--limit` 从全量中截断；再加
`--cap-per-video 1` 和 `--max_train_steps`，把训练量压到几十步内：

```powershell
python finetune_4090/prepare_data.py --name msrvtt `
  --video-dir D:\datasets\msrvtt\videos `
  --audio-dir D:\datasets\msrvtt\audios `
  --data-root D:\datasets\msrvtt `
  --extract-audio --limit 200 --cap-per-video 1

python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt `
  --data-root D:\datasets\msrvtt --epochs 1

python finetune_4090/run_tasks.py --task t2 `
  --config finetune_4090/cfgs/t2_msrvtt.json `
  --pretrain_dir D:\PycharmProjects\GRAM\gram_ckpt\GRAM_pretrained_4modalities `
  --output_dir D:\runs\t2_msrvtt_smoke `
  --train_batch_size 2 --epochs 1 --max_train_steps 20 --lr 1e-4 `
  --lora_r 16 --lora_alpha 32 --eval_steps 20 --save_steps 10 --log_steps 1
```

终端先打印 `[data] ... kept N/M rows`，再打印 `[1/20]...` 之类的步数；能看到 loss
下降、第 20 步完成一次 eval 并在 `D:\runs\t2_msrvtt_smoke\ckpt\final.pt` 保存，
就说明整条链路通了。若本地只有几十个视频，`--limit 200` 不会报错，它只会按实际
存在的视频继续。

### 4.3 T3：VATEX —— BERT-LoRA + 视觉/音频尾部解冻

```bash
bash finetune_4090/run_t3.sh \
  /path/to/vatex \
  /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  /path/to/outputs
```

手动分解：

```bash
python finetune_4090/prepare_data.py --name vatex \
  --video-dir /path/to/vatex/videos \
  --audio-dir /path/to/vatex/audios \
  --data-root /path/to/vatex --extract-audio \
  --limit 4000 --cap-per-video 1

python finetune_4090/make_cfgs.py --task t3 --data-root /path/to/vatex --epochs 1

python finetune_4090/run_tasks.py --task t3 \
  --config finetune_4090/cfgs/t3_vatex.json \
  --pretrain_dir /path/to/gram_ckpt/GRAM_pretrained_4modalities \
  --output_dir /path/to/outputs/t3_vatex_tail \
  --train_batch_size 2 --epochs 1 --lr 2e-5 \
  --lora_r 16 --lora_alpha 32 \
  --unfreeze_vision_blocks 4 --unfreeze_audio_layers 2 \
  --eval_steps 2000
```

要点：EVA-CLIP-g 共 40 个 block、BEATs 约 12 层；上面只解冻尾部 4 块/2 层并开
梯度检查点，4090 峰值约 18–23 GB。默认脚本同样是子集版
（`--limit 4000 --cap-per-video 1`）；完整 VATEX 的 26k×10 描述全量请去掉这两个
参数并放多卡/租 A100 跑。

---

## 5. 常见问题

1. **为什么不能只用 text-video 两模态？**
   仓库 retrieval loss/eval 强制三模态 volume；音频缺失时 AudioMapper 返回全零谱图
   （能跑但伤指标），所以统一用 `prepare_data.py --extract-audio`。
2. **4090 OOM？** 依次：确认 `--fp16` 已开（默认开）、把 `--train_batch_size`
   从 2 调到最小可用值 2（**volume 对比损失必须有 ≥2 的 batch**，否则没有负样本、
   梯度趋近于 0）、
   json 中 `vision_sample_num` 4→2、把 `--eval_steps` 调大（评测 ITM-rerank 最占
   显存）、用 `prepare_data.py --limit 200 --cap-per-video 1` + `--max_train_steps 20`
   冒烟。
3. **EVA-CLIP 搜不到，能不能换成别的 CLIP？** 拼写是 **EVA-CLIP**，不是
   EVAL-CLIP；`QuanSun/EVA-CLIP` 里就有 GRAM 需要的 `EVA01_CLIP_g_14_psz14_s11B.pt`，
   不需要“换一个”。直接换 OpenAI CLIP 与官方 GRAM checkpoint 结构不匹配，
   会大面积加载失败；只有从零构造、不加载 GRAM 多模态权重时才可把
   `vision_encoder_type` 改成 `clip_vit_base_16` 等（见第 1.3 节）。
4. **全参数微调 4090 行不行？** 1.1B(EVA)+109M(BERT)+BEATs 全解冻时，参数+梯度+
   Adam 状态就约 21 GB 起步，加逐帧激活必超 24 GB。T3 的“尾部解冻+前段检查点”
   是把 24 GB 用满的折中。
5. **与官方 run.py 的关系？** `run_tasks.py` 与官方共用模型、数据读取与评测；
   差异仅在不跑 ITM 训练分支、不要求 DDP/wandb。多卡/A100 用户可直接切官方
   `torch.distributed.launch run.py --config config/gram/finetune_cfg/retrieval-*.json`。

代码结构：
- `peft_utils.py`：`freeze_module` / `add_lora_to_module` / `unfreeze_vision_tail`
  / `unfreeze_audio_tail`；
- `run_tasks.py`：`RetVideoAudioTxtDataset`（读原始视频+音频）、volume-only
  `forward_ret`、AMP+AdamW 训练循环、原生 `evaluate_ret` 评测与 checkpoint 保存；
- `make_cfgs.py`/`prepare_data.py`：路径替换、mp3 抽取、train/test 交集过滤。
