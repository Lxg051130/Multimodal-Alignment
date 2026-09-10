#!/usr/bin/env bash
# Task 2 - Frozen backbones + LoRA adapters inside BERT + heads on MSRVTT.
#
# Usage:
#   bash finetune_4090/run_t2.sh /path/to/msrvtt_videos /path/to/GRAM_pretrained_4modalities /path/to/outputs
set -euo pipefail

DATA_ROOT="$1"
PRETRAIN_DIR="$2"
OUT_ROOT="$3"

python finetune_4090/prepare_data.py \
  --name msrvtt \
  --video-dir "$DATA_ROOT/videos" \
  --audio-dir "$DATA_ROOT/audios" \
  --data-root "$DATA_ROOT" \
  --extract-audio \
  --limit 2000 \
  --cap-per-video 1

python finetune_4090/make_cfgs.py --task t2 --dataset msrvtt \
  --data-root "$DATA_ROOT" --epochs 3

python finetune_4090/run_tasks.py --task t2 \
  --config finetune_4090/cfgs/t2_msrvtt.json \
  --pretrain_dir "$PRETRAIN_DIR" \
  --output_dir "$OUT_ROOT/t2_msrvtt_lora" \
  --train_batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --eval_steps 1000 \
  --save_steps 2000
