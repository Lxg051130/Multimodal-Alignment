#!/usr/bin/env bash
# Task 3 - Frozen-early/fine-late backbones (last EVA blocks + last BEATs
# layers unfrozen) + BERT LoRA on VATEX. Heaviest of the three.
#
# Usage:
#   bash finetune_4090/run_t3.sh /path/to/vatex_videos /path/to/GRAM_pretrained_4modalities /path/to/outputs
set -euo pipefail

DATA_ROOT="$1"
PRETRAIN_DIR="$2"
OUT_ROOT="$3"

python finetune_4090/prepare_data.py \
  --name vatex \
  --video-dir "$DATA_ROOT/videos" \
  --audio-dir "$DATA_ROOT/audios" \
  --data-root "$DATA_ROOT" \
  --extract-audio \
  --limit 4000 \
  --cap-per-video 1

python finetune_4090/make_cfgs.py --task t3 --dataset vatex \
  --data-root "$DATA_ROOT" --epochs 1

python finetune_4090/run_tasks.py --task t3 \
  --config finetune_4090/cfgs/t3_vatex.json \
  --pretrain_dir "$PRETRAIN_DIR" \
  --output_dir "$OUT_ROOT/t3_vatex_tail" \
  --train_batch_size 2 \
  --epochs 1 \
  --lr 2e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --unfreeze_vision_blocks 4 \
  --unfreeze_audio_layers 2 \
  --eval_steps 2000 \
  --save_steps 2000
