#!/usr/bin/env bash
# Task 1 - Frozen EVA-CLIP/BEATs backbones, train BERT text tower + heads only.
#
# Usage:
#   bash finetune_4090/run_t1.sh /path/to/didemo_videos /path/to/GRAM_pretrained_4modalities /path/to/outputs
set -euo pipefail

DATA_ROOT="$1"        # dir that contains videos/ (and audios/ after prepare_data)
PRETRAIN_DIR="$2"
OUT_ROOT="$3"

python finetune_4090/prepare_data.py \
  --name didemo \
  --video-dir "$DATA_ROOT/videos" \
  --audio-dir "$DATA_ROOT/audios" \
  --data-root "$DATA_ROOT" \
  --extract-audio

python finetune_4090/make_cfgs.py --task t1 --dataset didemo \
  --data-root "$DATA_ROOT" --epochs 5

python finetune_4090/run_tasks.py --task t1 \
  --config finetune_4090/cfgs/t1_didemo.json \
  --pretrain_dir "$PRETRAIN_DIR" \
  --output_dir "$OUT_ROOT/t1_didemo" \
  --train_batch_size 2 \
  --epochs 5 \
  --lr 3e-5 \
  --eval_steps 1000 \
  --save_steps 2000 \
  --first_eval
