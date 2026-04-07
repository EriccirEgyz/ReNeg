export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python ./analyze_hardneg_diversity.py \
    --device_count 8 \
    --model_path "" \
    --model_name "Qwen3-Embedding-4B" \
    --data_path "" \
    --data_name "" \
    --batch_size 32 \
    --n_clusters 50 \
    --random_seed 42 \
    --save_encodings 1