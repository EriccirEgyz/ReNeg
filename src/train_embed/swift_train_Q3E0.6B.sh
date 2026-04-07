nproc_per_node=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
INFONCE_HARD_NEGATIVES=1 \
INFONCE_MASK_FAKE_NEGATIVE=False \
MASTER_PORT=29600 \
swift sft \
    --model "" \
    --template qwen3_emb \
    --task_type embedding \
    --train_type full \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --dataset "" \
    --dataset_num_proc 32 \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir "" \
    --save_steps 2000 \
    --eval_steps 250 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero2 \
    --max_length 512 \
    --save_only_model true