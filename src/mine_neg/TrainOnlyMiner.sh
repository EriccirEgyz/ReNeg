export CUDA_VISIBLE_DEVICES=0,1,2,3

python ./TrainOnlyMiner.py \
    --device_count 4 \
    --model_path "" \
    --model_name "Qwen3-Embedding-0.6B" \
    --traindata_path "" \
    --traindata_name "" \
    --output_traindata_path "" \
    --mode Naive \
    --perc_pos 0.95 \
    --initial_search_ratio 0.01 \
    --sample_upper_bound 10 \
    --sample_lower_bound 1 \
    --neg_number 4 \
    --batch_size 64 \
    --faiss_gpu 1 \
    --save_encoding 1 \
    --save_querypos_minsim 1