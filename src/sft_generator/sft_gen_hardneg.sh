nproc_per_node=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model '' \
    --model_type qwen3_nothinking \
    --train_type full \
    --dataset '' \
    --val_dataset '' \
    --dataset_num_proc 32 \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 250 \
    --save_steps 500 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir '' \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --packing True \
    --packing_length 8192 \
    --deepspeed zero2 \
    --save_only_model True