#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
accelerate launch --num_processes 8 st_train_bge-m3.py \
    --model "" \
    --max_seq_length 512 \
    --dataset "" \
    --hardneg_number 1 \
    --query_instruction ""  \
    --passage_instruction "" \
    --split_dataset_ratio 0.05 \
    --gather_across_devices true \
    --output_dir "" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --save_steps 2000 \
    --eval_steps 500 \
    --run_name ""