# T2 微调手册：GRAM 预训练权重 → MSRVTT 检索（单卡 4090 / BERT-LoRA）

> 本文只讲 **finetune_4090 里的 t2 任务**（`run_t2.sh`、`run_tasks.py --task t2`、`make_cfgs.py --task t2`、`prepare_data.py --name msrvtt`、`peft_utils.apply_peft(task="t2")`）。
> 内容按“源码逐行读 + 本机实测”整理，整理日期 2026-09-10，仓库根目录 `D:\PycharmProjects\Multimodal-Alignment`。
> T1（DiDeMo，冻结主干只训头和 BERT）与 T3（VATEX，再解冻视觉/音频尾部）见 `finetune_4090/README_TASKS.md`。

---

## 0. 最短路径（5 步）与本机现状

T2 做的事：**冻结 EVA-CLIP 视觉塔和 BEATs 音频塔，在 BERT 文本/融合塔里注入 LoRA（q/k/v/out/intermediate），用 Gramian volume 对比损失在 MSRVTT 上微调**。

最短路径（在仓库根目录执行 PowerShell）：

```powershell
# 1) 权重就位（详见第 2 节）
#    pretrained_weights\clip\EVA01_CLIP_g_14_psz14_s11B.pt   <- 目前缺，必须下
#    gram_ckpt\ckpt\model_step_150000.pt                      <- 由 gram.pt 改名而来

# 2) 数据就位（详见第 3 节）
#    D:\datasets\msrvtt\videos\video0.mp4 ...
python finetune_4090/prepare_data.py --name msrvtt `
  --video-dir D:\datasets\msrvtt\videos --audio-dir D:\datasets\msrvtt\audios `
  --data-root D:\datasets\msrvtt --extract-audio --limit 2000 --cap-per-video 1

# 3) 生成 t2 配置（详见第 4 节，生成后要补两个字段）
python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt --data-root D:\datasets\msrvtt --epochs 3

# 4) 开始微调（详见第 5 节）
python finetune_4090/run_tasks.py --task t2 `
  --config finetune_4090/cfgs/t2_msrvtt.json `
  --pretrain_dir D:\PycharmProjects\Multimodal-Alignment\gram_ckpt `
  --output_dir D:\runs\t2_msrvtt_lora `
  --train_batch_size 2 --epochs 3 --lr 1e-4 `
  --lora_r 16 --lora_alpha 32 --eval_steps 1000 --save_steps 2000
```

### 0.1 本机实测现状（2026-09-10）

| 依赖项 | 要求 | 本机实测 | 结论 |
|---|---|---|---|
| Python 环境 | 装有 CUDA 版 PyTorch 的环境 | 默认 `python` = `D:\Anaconda\python.exe` 3.13.5，**torch 2.13.0+cpu，`torch.cuda.is_available() == False`** | ❌ 需换/建 CUDA 环境 |
| 运行时依赖 | `easydict` `librosa` `torchaudio` `timm` `transformers` | base 环境**缺 `easydict`/`librosa`/`torchaudio`/`timm`**（`transformers 5.8.0` 有） | ❌ 需 `pip install` |
| `pretrained_weights/bert/bert-base-uncased/` | 完整 HF BERT 目录 | ✅ 已有（config.json + pytorch_model.bin + vocab.txt 等） | 可用 |
| `pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt` | BEATs 权重 | ✅ 已有（约 345 MB） | 可用 |
| `pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt` | EVA-CLIP-g-14 | ❌ **缺失**（连 `pretrained_weights/clip/` 目录都没有） | 必须下载 |
| GRAM 预训练权重 | `<dir>/log/hps.json` + `<dir>/ckpt/model_step_<N>.pt` | ⚠️ `gram_ckpt/log/hps.json` ✅；checkpoint 目前叫 `gram_ckpt/ckpt/gram.pt`，**文件名不符合代码的 `model_step_*.pt` 约定** | 需改名/复制 |
| MSRVTT 标注 | `datasets/annotations/msrvtt/*.json` | ✅ 自带：train 180,000 行（9,000 视频 × 20 描述）、test 1,000 行 | 可用 |
| MSRVTT 视频 | `<data-root>/videos/video*.mp4` | ⚠️ `datasets/annotations/msrvtt/MSRVTT_Videos.zip` 在（2.19 GB，内含 `video/video0.mp4 ... video9999.mp4`），但 `videos/` 目录为空、未解压 | 需解压 |
| ffmpeg | 抽 `.mp3` 用 | ✅ `D:\ffmpeg-8.1.2-essentials_build\...\ffmpeg.exe` | 可用 |

---

## 1. 目录总览：数据、权重、代码、输出到底放哪

```text
D:\PycharmProjects\Multimodal-Alignment\        <- 仓库根：所有命令都在这里执行
├── finetune_4090\
│   ├── run_t2.sh              # t2 一键脚本（bash/WSL 用）
│   ├── prepare_data.py        # 解压后的视频/音频 + 标注过滤 -> annos_ret_*.json
│   ├── make_cfgs.py           # 生成本地路径的任务配置
│   ├── run_tasks.py           # 单卡训练器（t2 的真正入口）
│   ├── peft_utils.py          # 冻结 / LoRA 注入 / 解冻尾部
│   └── cfgs\t2_msrvtt.json    # make_cfgs.py 生成的配置
│
├── pretrained_weights\        # 三个基础 encoder 权重（源码里写死为 ./pretrained_weights/...）
│   ├── clip\EVA01_CLIP_g_14_psz14_s11B.pt   # 视觉塔（必须）
│   ├── beats\BEATs_iter3_plus_AS2M.pt       # 音频塔（必须）
│   └── bert\bert-base-uncased\              # 文本/融合塔（必须，整个目录）
│
├── gram_ckpt\                 # GRAM 多模态预训练权重 = 传给 --pretrain_dir 的目录
│   ├── log\hps.json           # 必须：trainer 从里面继承 encoder 类型等
│   └── ckpt\model_step_<数字>.pt   # 必须：文件名以 model_step 开头、以 .pt 结尾
│
└── datasets\annotations\msrvtt\   # 仓库自带标注与视频 zip（原始 annotation 不用自己找）
    ├── descs_ret_train.json       # 180,000 行
    ├── descs_ret_test.json        # 1,000 行
    └── MSRVTT_Videos.zip          # 10,000 个 mp4（video/video0.mp4 ...）

D:\datasets\msrvtt\              <- 数据根（--data-root，可以放仓库外，推荐）
├── videos\video0.mp4 ... video9999.mp4    # 解压后平铺，文件名 = 标注里的 video_id
├── audios\video0.mp3 ...                  # prepare_data.py --extract-audio 自动抽
├── annos_ret_train.json                   # prepare_data.py 自动生成（只保留本地有视频的行）
└── annos_ret_test.json                    # 同上

D:\runs\t2_msrvtt_lora\          <- --output_dir（训练产物）
├── ckpt\final.pt、model_step_<step>.pt、best_ret.pt（后者仅在评估成功时出现）
└── log\                                    # 目录会被创建，但当前实现不往里写文件
```

两条硬约束：

1. **三个 encoder 权重必须放在仓库根的 `pretrained_weights/` 下**，因为 `model/general_module.py` 与 `model/gram.py` 里写死了 `./pretrained_weights/clip/...`、`./pretrained_weights/beats/...`、`./pretrained_weights/bert/bert-base-uncased`。该目录已被 `.gitignore` 忽略。
2. **`--pretrain_dir` 必须同时含 `log/hps.json` 和 `ckpt/model_step_<N>.pt`**，见 2.2。

---

## 2. 第 1 步：权重放哪、怎么下

### 2.1 三个基础 encoder（路径固定，不能挪）

| 权重 | 下载地址 | 放到（相对仓库根） | 大小 |
|---|---|---|---|
| EVA-CLIP-g-14（视觉塔，`evaclip01_giant`） | `https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt` | `pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt` | ~2.2 GB |
| BEATs（音频塔，`beats`） | `https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt` | `pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt` | ~345 MB |
| BERT-base-uncased | <https://huggingface.co/bert-base-uncased> | `pretrained_weights/bert/bert-base-uncased/`（整个 HF 目录：`config.json`+`pytorch_model.bin`+`vocab.txt`…） | ~440 MB |

```powershell
# 在仓库根执行
New-Item -ItemType Directory -Force pretrained_weights\clip | Out-Null

curl.exe -L -o pretrained_weights\clip\EVA01_CLIP_g_14_psz14_s11B.pt `
  https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA01_CLIP_g_14_psz14_s11B.pt

# BEATs 与 BERT 本机已有，仅换机器时才需要：
# curl.exe -L -o pretrained_weights\beats\BEATs_iter3_plus_AS2M.pt `
#   https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt
# python -c "from transformers import BertModel, BertTokenizer; BertModel.from_pretrained('bert-base-uncased').save_pretrained('pretrained_weights/bert/bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('pretrained_weights/bert/bert-base-uncased')"
```

> 注意是 **EVA-CLIP**（EVA01-CLIP-g-14），不是 “EVAL-CLIP”。`vision_encoder_type=evaclip01_giant` 时源码只认这一个文件，换成 OpenAI CLIP 的 `ViT-B-16.pt` 会大面积 key 不匹配。

### 2.2 GRAM 预训练 checkpoint（`--pretrain_dir` 的真实要求）

`finetune_4090/run_tasks.py` 对 `--pretrain_dir` 只有两处要求：

```python
# 1) 读超参（_load_model_cfg）：从 hps.json 继承 vision_encoder_type / audio_encoder_type /
#    audio_melbins / audio_target_length（+ pool_video，如果存在）
json.load(open(os.path.join(pretrain_dir, "log", "hps.json")))

# 2) 选 checkpoint（_latest_ckpt）：只认 ckpt/ 里 “model_step 开头、.pt 结尾”的文件，取编号最大的
files = [f for f in os.listdir(os.path.join(pretrain_dir, "ckpt"))
         if f.startswith("model_step") and f.endswith(".pt")]
files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))
return os.path.join(ckpt_dir, files[-1])
```

所以目录必须是：

```text
gram_ckpt\                          <- --pretrain_dir 传这一层
├── log\hps.json
└── ckpt\model_step_150000.pt       <- 文件名里的数字随便，但前缀/后缀必须对
```

**本机当前状态**：`gram_ckpt\log\hps.json` ✅，checkpoint 是 `gram_ckpt\ckpt\gram.pt`（5.59 GB，已确认是标准 GRAM 4 模态 state_dict：`vision_encoder.visual.*` 1408 维、`audio_encoder.*`、`multimodal_encoder.bert.*`、`contra_head_*`、`hidden_trans_*`，共 1288 个键），只差改名：

```powershell
# 方案 A：直接改名（省 5.6 GB 空间）
Rename-Item gram_ckpt\ckpt\gram.pt model_step_150000.pt

# 方案 B：复制一份，保留原文件
# Copy-Item gram_ckpt\ckpt\gram.pt gram_ckpt\ckpt\model_step_150000.pt
```

> 训练启动时会打印 `[ckpt] <文件路径>` 和 `[ckpt] missing X unexpected Y`；`X/Y` 应该接近 0，若 missing 上千，说明 checkpoint 与 `vision_encoder_type` 不匹配。

### 2.3 一条命令自检权重

```powershell
Get-ChildItem pretrained_weights -Recurse -Depth 2 -File | Select-Object FullName, Length
Get-ChildItem gram_ckpt -Recurse -File | Select-Object FullName, Length
```

---

## 3. 第 2 步：数据放哪、怎么准备

### 3.1 标注：仓库自带，不用自己造

| 文件 | 行数 | 说明 |
|---|---:|---|
| `datasets/annotations/msrvtt/descs_ret_train.json` | 180,000 | 9,000 个视频 × 20 条描述，字段 `video_id` / `desc` / `subtitle` |
| `datasets/annotations/msrvtt/descs_ret_test.json` | 1,000 | 官方 1k-A 测试集，字段同上 |
| `datasets/annotations/msrvtt/unique_descs_ret_train.json` | — | 每视频只留 1 条描述的训练集版本（想跑“单描述全量训练”时可用） |

### 3.2 视频：解压到 `<data-root>/videos/`，文件名就是 `video_id`

```powershell
# 建议把数据放仓库外，避免 10 GB 视频进 git
New-Item -ItemType Directory -Force D:\datasets\msrvtt\videos | Out-Null

# zip 内层是 video\video*.mp4，先解压到临时目录再平铺
Expand-Archive -LiteralPath datasets\annotations\msrvtt\MSRVTT_Videos.zip `
  -DestinationPath D:\datasets\msrvtt\_extract -Force
Get-ChildItem D:\datasets\msrvtt\_extract -Recurse -Filter *.mp4 |
  Move-Item -Destination D:\datasets\msrvtt\videos -Force
(Get-ChildItem D:\datasets\msrvtt\videos -Filter *.mp4).Count   # 期望 10000
```

> 只想先冒烟，也可以只复制前 50 个 mp4 到 `videos\`：`prepare_data.py` 是“按本地实际存在的视频”过滤的，不会因为文件少而报错。

### 3.3 抽音频 + 生成过滤后的标注（`prepare_data.py`）

```powershell
python finetune_4090/prepare_data.py --name msrvtt `
  --video-dir D:\datasets\msrvtt\videos `
  --audio-dir D:\datasets\msrvtt\audios `
  --data-root D:\datasets\msrvtt `
  --extract-audio --limit 2000 --cap-per-video 1
```

`prepare_data.py` 全部参数：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--name` | 必填 | 数据集名，t2 用 `msrvtt`（可选 `didemo` / `msrvtt` / `vatex` / `activitynet`） |
| `--video-dir` | 必填 | 视频目录，文件名 `<video_id>.mp4` |
| `--audio-dir` | `""` | 音频输出目录；配 `--extract-audio` 时必填 |
| `--data-root` | 视频目录的父目录 | `annos_ret_train.json` / `annos_ret_test.json` 写到哪 |
| `--extract-audio` | 关 | 用 ffmpeg 从每个视频抽单声道 16 kHz `<id>.mp3`（已存在则跳过） |
| `--ffmpeg` | `ffmpeg` | ffmpeg 可执行文件路径 |
| `--limit` | `0`（不限） | 只保留前 N 个“本地存在”的唯一视频（train 取 0.8N、test 取 N，两者共用一个池子） |
| `--cap-per-video` | `0`（不限） | 每个视频最多保留 N 条描述；`1` = 一视频一描述 |

它做三件事：(1) 把仓库标注与本地实际存在的视频求交集；(2) 按 `--limit/--cap-per-video` 裁剪；(3) 写出 `<data-root>/annos_ret_train.json`、`annos_ret_test.json`，并打印形如 `train: 2000 rows, 2000 videos -> ...`。

**为什么必须抽音频**：GRAM 的检索损失/评估是 text+video+audio 三模态的 Gramian volume。音频缺失时 `AudioMapper` 会打印 `not have audios <id>` 并返回全零频谱，能跑但指标会坏。

### 3.4 数据自检

```powershell
(Get-ChildItem D:\datasets\msrvtt\videos -Filter *.mp4).Count
(Get-ChildItem D:\datasets\msrvtt\audios -Filter *.mp3).Count
Get-ChildItem D:\datasets\msrvtt\annos_ret_*.json | Select-Object Name, Length
```

---

## 4. 第 3 步：生成 t2 配置（`make_cfgs.py`）

```powershell
python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt `
  --data-root D:\datasets\msrvtt --epochs 3
```

输出覆盖写 `finetune_4090/cfgs/t2_msrvtt.json`（仓库里现存那份是旧机器生成的，`default` 还指向 `D:\PycharmProjects\GRAM\...`，**必须重新生成**）。

`make_cfgs.py` 全部参数：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--task` | 必填 | `t1` / `t2` / `t3`；本文流程只选 `t2` |
| `--dataset` | 按 task 推导：t2→`msrvtt` | `didemo` / `msrvtt` / `vatex` |
| `--data-root` | 必填 | 含 `videos/` `audios/` `annos_ret_*.json` 的目录 |
| `--epochs` | t2→`3` | 写进 train 的 `epoch`（t1→5、t3→1） |
| `--vision-sample-num` | `4` | 每个视频采样帧数（显存与时间的主开关） |
| `--audio-sample-num` | `1` | 每段音频采样数（BEATs 输入片段数） |
| `--train-bs` | `2` | 写进 train 的 `batch_size`（仅记录，实际以 `run_tasks.py --train_batch_size` 为准） |
| `--val-bs` | `1` | 同上，验证集 |
| `--max-caption-len` | `40` | BERT 文本最大 token 数 |

生成的 `t2_msrvtt.json` 关键内容：

```jsonc
{
  "run_cfg":   { "learning_rate": 1e-4, "gradient_accumulation_steps": 1, "checkpointing": true,
                 "fp16": true, "use_ddp": false, "mode": "training", "valid_freq": 10 },
  "model_cfg": { "default": "<仓库根>/config/gram/default_model_cfg.json",
                 "frozen_vision": true, "frozen_audio": true, "checkpointing": true,
                 "max_caption_len": 40, "itm_ratio": 0.0, "itm_rerank_num": 10,
                 "ret_bidirection_evaluation": false },
  "data_cfg": {
    "train": [{ "name": "msrvtt_ret", "txt": "<data-root>/annos_ret_train.json",
                "vision": "<data-root>/videos", "audio": "<data-root>/audios",
                "task": "ret%tv%ta", "epoch": 3, "batch_size": 2,
                "vision_sample_num": 4, "audio_sample_num": 1, "n_workers": 4, "training": true }],
    "val":   [{ "txt": "<data-root>/annos_ret_test.json", "task": "ret%tvas",
                "epoch": 1, "batch_size": 1, "training": false, "...": "..." }]
  }
}
```

> `make_cfgs.py` 优先用 `<data-root>/annos_ret_train.json`（prepare_data 产物），不存在时回退到仓库自带标注。所以**先跑 prepare_data 再跑 make_cfgs**。

### 4.1 ⚠️ 生成后必须补两个字段（否则模型构建就会崩）

仓库官方的 `utils/args.py` 会从 data_cfg 算出两个模型字段再塞进 `model_cfg`：

```python
model_cfg.max_vision_sample_num = compute_max_vision_sample_num_for_position_embeddings(data_cfg)
model_cfg.max_audio_sample_num  = compute_max_audio_sample_num_for_position_embeddings(data_cfg)
```

而 `finetune_4090/run_tasks.py::_load_model_cfg` 只复刻了「文件合并 + 从 hps.json 继承」两条规则，**没有复刻这两行**；但 `model/gram.py` 的 `GRAM.__init__` 直接用了这两个字段：

```python
self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, ...))
self.audio_frame_embedding  = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num,  ...))
```

实测按 `_load_model_cfg` 的合并规则跑一遍，生成的配置里这两个键**确实缺失**，模型构建时会抛 `AttributeError: 'EasyDict' object has no attribute 'max_vision_sample_num'`。两种补法：

**补法 1（不改代码，改 json）**：在 `finetune_4090/cfgs/t2_msrvtt.json` 的 `model_cfg` 里加两行（值 = train 的 `vision_sample_num` / `audio_sample_num`，t2 默认即 4 和 1）。用 Python 一行搞定：

```powershell
python -c "import json; p='finetune_4090/cfgs/t2_msrvtt.json'; c=json.load(open(p,encoding='utf-8')); t=c['data_cfg']['train'][0]; c['model_cfg']['max_vision_sample_num']=t['vision_sample_num']; c['model_cfg']['max_audio_sample_num']=t['audio_sample_num']; json.dump(c, open(p,'w',encoding='utf-8'), indent=2, ensure_ascii=False); print('patched', p)"
```

等价的手工做法：直接编辑 json，在 `"model_cfg"` 里加 `"max_vision_sample_num": 4,` 和 `"max_audio_sample_num": 1,`。
注意 `make_cfgs.py` 每次都会覆盖这个文件，改完 json 后不要再重跑 `make_cfgs.py`。

**补法 2（改代码，一劳永逸）**：在 `finetune_4090/make_cfgs.py` 的 `model_extra` 字典里补上

```python
"max_vision_sample_num": vision_sample_num,
"max_audio_sample_num": audio_sample_num,
```

这样以后每次 `make_cfgs.py` 生成的配置就是完整的。

---

## 5. 第 4 步：跑 t2 微调

### 5.1 一键脚本（bash / WSL / Git Bash）

```bash
bash finetune_4090/run_t2.sh \
  /path/to/msrvtt \
  /path/to/gram_ckpt \
  /path/to/outputs
```

三个位置参数分别是：`$1 = --data-root`、`$2 = --pretrain_dir`、`$3 = --output_dir 的父目录`。
脚本内部等价于下面三条命令，并固定了这些值：`--limit 2000 --cap-per-video 1`、`--epochs 3`、`--train_batch_size 2`、`--lr 1e-4`、`--lora_r 16`、`--lora_alpha 32`、`--eval_steps 1000`、`--save_steps 2000`，输出目录是 `$3/t2_msrvtt_lora`。

### 5.2 手动三条命令（Windows PowerShell，推荐）

```powershell
cd D:\PycharmProjects\Multimodal-Alignment

# (1) 数据：抽音频 + 生成过滤后的标注
python finetune_4090/prepare_data.py --name msrvtt `
  --video-dir D:\datasets\msrvtt\videos `
  --audio-dir D:\datasets\msrvtt\audios `
  --data-root D:\datasets\msrvtt `
  --extract-audio --limit 2000 --cap-per-video 1

# (2) 生成 t2 配置（记得按 4.1 补两个字段）
python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt `
  --data-root D:\datasets\msrvtt --epochs 3

# (3) 微调
python finetune_4090/run_tasks.py --task t2 `
  --config finetune_4090/cfgs/t2_msrvtt.json `
  --pretrain_dir D:\PycharmProjects\Multimodal-Alignment\gram_ckpt `
  --output_dir D:\runs\t2_msrvtt_lora `
  --train_batch_size 2 --epochs 3 --lr 1e-4 `
  --lora_r 16 --lora_alpha 32 `
  --eval_steps 1000 --save_steps 2000
```

### 5.3 `run_tasks.py` 全部参数

| 参数 | 默认值 | t2 建议 | 作用 |
|---|---|---|---|
| `--task` | 必填 | `t2` | 决定 PEFT 策略：t2 = 冻结双塔 + BERT LoRA |
| `--config` | 必填 | `finetune_4090/cfgs/t2_msrvtt.json` | 任务配置（data_cfg / model_cfg 覆盖） |
| `--pretrain_dir` | `""` | `...\gram_ckpt` | GRAM 预训练权重目录；**留空 = 随机初始化从头练**，必须填 |
| `--output_dir` | 必填 | `D:\runs\t2_msrvtt_lora` | 产物目录，自动建 `ckpt/` 与 `log/` |
| `--train_batch_size` | `2` | `2`（不能降到 1） | 每优化步 batch；volume 对比损失需要 ≥2 个样本才有负样本，脚本会 `SystemExit` 拒绝 1 |
| `--val_batch_size` | `1` | `1` | 验证 batch |
| `--epochs` | `1.0` | `3` | 训练轮数，支持小数（最后一轮按比例截断步数） |
| `--max_train_steps` | `0` | `0`（冒烟用 20） | 硬性限制梯度步数；>0 时取 `min(自动步数, 本值)` |
| `--lr` | `3e-5` | `1e-4` | 峰值学习率（AdamW），配 warmup + 线性衰减 |
| `--weight_decay` | `0.01` | 保持 | 只作用于非 bias / 非 LayerNorm 参数 |
| `--warmup_ratio` | `0.05` | 保持 | 前 5% 步数线性升温 |
| `--seed` | `42` | 保持 | torch / random 随机种子 |
| `--fp16` / `--no_fp16` | `--fp16`（开） | 保持开 | CUDA AMP 混合精度；CPU 上自动失效 |
| `--first_eval` | 关 | 可选 | 训练前先评估一次（当前实现下只会打印 warning，见 8.2） |
| `--eval_steps` | `2000` | `1000` | 每 N 步评估一次 |
| `--save_steps` | `2000` | `2000` | 每 N 步存 `ckpt/model_step_<step>.pt` |
| `--log_steps` | `20` | `20` | 每 N 步打印 loss（第一步必打） |
| `--lora_r` | `16` | `16` | LoRA 秩；越大容量越高、显存/时间越多 |
| `--lora_alpha` | `32.0` | `32` | LoRA 缩放系数，实际缩放 = `alpha / r`（默认 2.0） |
| `--unfreeze_vision_blocks` | `4` | t2 忽略 | 仅 t3 生效：解冻 EVA-CLIP 末尾 N 个 block |
| `--unfreeze_audio_layers` | `2` | t2 忽略 | 仅 t3 生效：解冻 BEATs 末尾 N 层 |

### 5.4 内建超参（命令行改不了，要改只能动 `run_tasks.py`）

| 项 | 取值 | 位置 |
|---|---|---|
| 优化器 | AdamW，`betas=(0.9, 0.98)` | 优化器构造 |
| 参数分组 | bias / LayerNorm 不做权重衰减，其余 `weight_decay=0.01` | 同上 |
| 梯度裁剪 | `clip_grad_norm_(..., 5.0)` | 反向之后 |
| 学习率调度 | 线性 warmup + 线性衰减到 0（按全局步数比例） | 训练循环 |
| 训练损失 | `volume_computation3(feat_t, feat_v_all, feat_a_all) / contra_temp` 的双向 Gramian volume + `label_smoothing=0.1` 交叉熵；`contra_temp` 可学习（初值 0.07） | `_forward_ret_volume_only` |
| 训练前向 | **只算 volume 损失，跳过 ITM 分支**（4090 显存换来的） | 同上 |
| 评估前向 | 调仓库原生 `forward_ret(compute_loss=False)` + `evaluate_ret` | `run_eval` |
| DataLoader | 训练 `drop_last=True`；`num_workers = data_cfg.n_workers`（默认 4，Windows 卡就改小） | 数据集构造 |
| DDP | 不开；仅 `init_process_group(world_size=1)` 让 `all_gather` 类函数可用（NCCL 失败自动退 gloo） | `main()` 开头 |

### 5.5 训练时该看到什么

```text
[data] msrvtt_ret: kept 2000/2000 rows                          <- 数据过滤结果（分母是 annos_ret_train.json 的行数）
[peft_utils] injected 72 LoRA adapters into multimodal_encoder  <- 12 层 × 6 个位置
[peft_utils] trainable 30.xxM / 1xxx.xxM params                 <- 可训练 / 总参数
[ckpt] D:\...\gram_ckpt\ckpt\model_step_150000.pt
[ckpt] missing 0 unexpected 0
[1/3000] ep0 {'loss_area': 1.23, 'loss_itc': 0.0, 'loss_itm': 0.0} lr=2.00e-05 0.3min
...
done: D:\runs\t2_msrvtt_lora\ckpt\final.pt
```

步数估算：`steps/epoch = 训练行数 / train_batch_size`（2000 行 / bs2 = 1000 步），`总步数 = steps/epoch × epochs`（3 epoch ≈ 3000 步）。

### 5.6 产物

| 文件 | 何时产生 | 内容 |
|---|---|---|
| `<output_dir>\ckpt\final.pt` | 训练结束必产生 | 完整 `state_dict`（LoRA 包装后的键名结构） |
| `<output_dir>\ckpt\model_step_<step>.pt` | 每 `--save_steps` 步 | 同上 |
| `<output_dir>\ckpt\best_ret.pt` | **仅当评估成功且指标刷新** | 最佳检索指标权重；当前评估实现有问题（8.2），默认跑不出 |

加载微调结果做推理/二次微调：`torch.load('<output_dir>/ckpt/final.pt')` → `{k.replace('module.',''):v}` → `model.modify_checkpoint(...)` → `load_state_dict(strict=False)`，与 `run_tasks.py` 加载 `--pretrain_dir` 的写法完全一致。

### 5.7 冒烟测试（先确认整条链路通，再上全量）

```powershell
python finetune_4090/prepare_data.py --name msrvtt `
  --video-dir D:\datasets\msrvtt\videos --audio-dir D:\datasets\msrvtt\audios `
  --data-root D:\datasets\msrvtt --extract-audio --limit 200 --cap-per-video 1

python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt --data-root D:\datasets\msrvtt --epochs 1

python finetune_4090/run_tasks.py --task t2 `
  --config finetune_4090/cfgs/t2_msrvtt.json `
  --pretrain_dir D:\PycharmProjects\Multimodal-Alignment\gram_ckpt `
  --output_dir D:\runs\t2_msrvtt_smoke `
  --train_batch_size 2 --epochs 1 --max_train_steps 20 --lr 1e-4 `
  --lora_r 16 --lora_alpha 32 --eval_steps 20 --save_steps 10 --log_steps 1
```

能打印出 `[1/20] ... loss_area`、并在最后写出 `D:\runs\t2_msrvtt_smoke\ckpt\final.pt`，就说明权重装载、数据读取、LoRA 注入、前向反向、保存全部通了。

---

## 6. 参数总表（一页速查）

```text
数据层   prepare_data.py  --name msrvtt --video-dir <videos> --audio-dir <audios>
                          --data-root <root> --extract-audio [--limit N] [--cap-per-video N]
配置层   make_cfgs.py     --task t2 --dataset msrvtt --data-root <root> --epochs 3
                          [--vision-sample-num 4] [--audio-sample-num 1]
                          [--train-bs 2] [--val-bs 1] [--max-caption-len 40]
训练层   run_tasks.py     --task t2 --config finetune_4090/cfgs/t2_msrvtt.json
                          --pretrain_dir <gram_ckpt> --output_dir <out>
                          --train_batch_size 2 --val_batch_size 1 --epochs 3
                          --max_train_steps 0 --lr 1e-4 --weight_decay 0.01
                          --warmup_ratio 0.05 --seed 42 [--no_fp16] [--first_eval]
                          --eval_steps 1000 --save_steps 2000 --log_steps 20
                          --lora_r 16 --lora_alpha 32
                          [--unfreeze_vision_blocks 4 --unfreeze_audio_layers 2]  # 仅 t3
```

T2 默认有效组合（本文推荐值）：

```text
数据        MSRVTT，2,000 视频 × 1 描述 = 2,000 训练行；测试 1,000 行
帧/音频     每视频 4 帧（vision_sample_num=4）+ 1 段音频（audio_sample_num=1），分辨率 224
训练范围    冻结 EVA-CLIP / BEATs / contra_head_* / hidden_trans_*；
            BERT 基座可训 + 在 attention.self(q,k,v) / intermediate.dense / output.dense 注入 LoRA(r=16, alpha=32)
优化        AdamW(lr=1e-4, betas=0.9/0.98, wd=0.01 除 bias/LayerNorm), warmup 5% + 线性衰减,
            fp16 AMP, grad clip 5.0, batch 2, 3 epoch(≈3000 步), seed 42
损失        三模态 Gramian volume + label_smoothing 0.1，训练时不含 ITM 分支
评估        每 1000 步（当前实现会因字段名不匹配打 warning，见 8.2）
```

显存参考（4090 24 GB）：T2 约 **10–15 GB**。OOM 时优先降 `--vision-sample-num`（4→2）、加大 `--eval_steps`（评估里的 ITM rerank 峰值最高），**不要**把 `--train_batch_size` 降到 1。

---

## 7. T2 到底改了什么（`peft_utils.apply_peft(task="t2")`）

1. `model.vision_encoder`（EVA-CLIP）与 `model.audio_encoder`（BEATs）：全部 `requires_grad=False`，`train` 被替换为 `disabled_train`，永远保持 eval 行为。
2. 冻结 `contra_head_v / contra_head_a / contra_head_va / contra_head_d / contra_head_vas / contra_head_vs` 与 `hidden_trans_vision_multimodal` / `hidden_trans_audio_multimodal`，切断梯度回流到主干，省激活显存。
3. 对 `model.multimodal_encoder`（BERT）中名字含 `attention.self` / `intermediate` / `output.dense` 的每个 `nn.Linear`，用 `_LoRALinear` 就地替换：原权重冻结，只训 `lora_a (r×in)` 与 `lora_b (out×r)`（`lora_b` 零初始化，起点等价于原模型）。BERT-base 12 层 × 6 个位置 = **72 个 LoRA 适配器**。
4. BERT 的 embedding、LayerNorm、MLM 头等原参数**保持可训练**。
5. t1 在第 2 步后返回；t2 在第 3 步后返回；t3 才继续解冻视觉/音频尾部并重新打开那些投影头。

想换成 t1 策略验证数据链路：把 `--task t2` 换成 `--task t1 --config finetune_4090/cfgs/t1_didemo.json`（数据换 DiDeMo）。

---

## 8. 已知问题与修法（按代码路径核对过）

### 8.1 `max_vision_sample_num` / `max_audio_sample_num` 缺失 → 启动即崩

见 4.1。现象：`AttributeError: 'EasyDict' object has no attribute 'max_vision_sample_num'`，发生在模型构造阶段（还没读数据）。修法二选一（补 json 或补 `make_cfgs.py`）。

### 8.2 内置评估跑不通 → 只有 `[warn]`，没有指标、没有 `best_ret.pt`

训练 batch 是 `run_tasks.py` 自己拼的（键名 `raw_captions` / `vision_pixels` / `audio_spectrograms` / `ids` / `ids_txt`），但验证集 DataLoader 用的是 `RetVideoAudioTxtDataset` 返回的键名（`id` / `caption` / `video` / `audio`）。而仓库原生 `evaluate_ret` 与 `GRAM.forward_ret(compute_loss=False)` 读的是：

```python
batch.raw_captions          # forward_ret 第一行
batch.ids / batch.ids_txt   # evaluate_ret 统计用
```

于是评估抛 `AttributeError`，被 `run_eval` 的 `try/except` 吞掉，只打印：

```text
[warn] evaluate_ret failed at step 1000: 'EasyDict' object has no attribute 'raw_captions'
```

后果：训练照跑、`final.pt` 照存，但**看不到 R@1、也拿不到 `best_ret.pt`**。想让评估生效，最小改动三处（都在 `finetune_4090/run_tasks.py`）：

```python
# (1) _read_row 的返回键改成仓库原生命名
row = {"ids": vid, "ids_txt": vid, "raw_captions": desc,
       "vision_pixels": video, "audio_spectrograms": audio}
if "subtitle" in anno and self.use_subtitles:
    row["raw_subtitles"] = anno.get("subtitle", "")   # 验证任务 ret%tvas 需要它

# (2) _collate 里的“字符串字段”名单同步改名
if key in ("ids", "ids_txt", "raw_captions", "raw_subtitles"):

# (3) 训练循环里构造模型输入时直接用这些键
model_batch = {"raw_captions": batch["raw_captions"], "vision_pixels": batch["vision_pixels"],
               "audio_spectrograms": batch["audio_spectrograms"],
               "ids": batch["ids"], "ids_txt": batch["ids_txt"]}
```

改完评估会走原生路径，日志里会出现 `volume_ITM_T2D.forward_r1`、`volume_D2T...`、`gramian_value` 等指标，`best_ret.pt` 也会在刷新指标时保存。
想拿“官方数字”则用仓库原生 `run.py + config/gram/finetune_cfg/retrieval-msrvtt.json`（需 DDP 启动 + wandb）。

### 8.3 `FileNotFoundError: ...\ckpt` / `IndexError: list index out of range`

`--pretrain_dir` 下没有 `ckpt/model_step_*.pt`。注意：`log/hps.json` 存在但 ckpt 文件名不对时，报的是 **IndexError**（列表为空仍取 `[-1]`）。按 2.2 改名即可。

### 8.4 `0 rows with videos under ...`

`--video-dir` 里的文件名与标注 `video_id` 对不上。MSRVTT 必须是 `video0.mp4 … video9999.mp4`（自带 `video` 前缀）。检查：`(Get-ChildItem D:\datasets\msrvtt\videos).Name | Select-Object -First 5`。

### 8.5 `not have audios <id>` 刷屏

没抽音频或 `.mp3` 名字不对。跑 `prepare_data.py --extract-audio`（输出 `<audio-dir>/<id>.mp3`）；`AudioMapper` 依次找 `<id>`、`<id>.wav`、`<id>.mp3`、`<id>.mkv`。

### 8.6 CUDA OOM

处理顺序：确认 `--fp16`（默认开）→ `make_cfgs.py --vision-sample-num 2`（改完要重跑 make_cfgs，并重补 4.1 的两个字段）→ 调大 `--eval_steps` → 先用 `--limit` 小数据 + `--max_train_steps` 验证流程。**不要把 `--train_batch_size` 降到 1**（脚本直接拒绝）。

### 8.7 Windows 下 DataLoader 卡住 / spawn 报错

json 里 `"n_workers": 4` 会起 4 个进程重复导入 torch。改小到 0–2（`finetune_4090/cfgs/t2_msrvtt.json` 的 `data_cfg.train/val` 两处），或在 `make_cfgs.py` 里加 `"n_workers": 2`。

### 8.8 `torch.cuda.amp.GradScaler` 弃用告警

`run_tasks.py` 用的是 `torch.cuda.amp.*`。在 torch 2.x 上仍可用（本机 torch 2.13 里 `torch.cuda.amp.GradScaler` 存在），只是提示改用 `torch.amp.*`，可忽略。

---

## 9. 环境准备（当前机器必须先做）

```powershell
# 建独立环境（示例）
conda create -n gram4090 python=3.10 -y
conda activate gram4090

# CUDA 版 PyTorch（cu121 / cu124 按显卡驱动选）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 训练所需其余依赖
pip install easydict librosa timm transformers einops numpy scipy pandas soundfile audioread

# 自检：必须打印 True 和显卡名
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import easydict, librosa, torchaudio, timm, transformers; print('deps ok')"
```

若仓库提供了 `requirements.txt`，也可以先 `pip install -r requirements.txt`；但其中的 torch 可能是 CPU 版，装完再按上面的 index-url 覆盖安装 torch / torchvision / torchaudio。

---

## 10. 相关文件速查

| 文件 | 作用 |
|---|---|
| `finetune_4090/run_t2.sh` | t2 一键脚本（三条命令 + 固定超参） |
| `finetune_4090/prepare_data.py` | 抽音频、标注与本地视频求交集、写 `annos_ret_*.json` |
| `finetune_4090/make_cfgs.py` | 生成本地路径的 `cfgs/t2_msrvtt.json` |
| `finetune_4090/run_tasks.py` | 单卡训练器：`RetVideoAudioTxtDataset`、volume-only `forward_ret`、AdamW+AMP 训练循环、原生 `evaluate_ret` 评估、checkpoint 保存 |
| `finetune_4090/peft_utils.py` | `freeze_module` / `add_lora_to_module` / `unfreeze_vision_tail` / `unfreeze_audio_tail` / `apply_peft` |
| `finetune_4090/README_TASKS.md` | T1/T2/T3 三任务总体说明（含 T3 细节） |
| `FINETUNE_PLAN.md` | 更早的可行性评估（显存测算、权重来源、任务分级） |
| `config/gram/default_model_cfg.json` | 模型默认配置（`contra_dim=512`、`vision_resolution=224`、`itm_rerank_num=50` 等） |
| `config/gram/finetune_cfg/retrieval-msrvtt.json` | 官方（多卡）MSRVTT 检索配置，可对照 |
