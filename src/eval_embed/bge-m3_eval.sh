export CUDA_VISIBLE_DEVICES=0,1,2,3
python ./eval.py \
    --device_count 4 \
    --model_path "" \
    --model_name "" \
    --query_instruction ""  \
    --document_instruction "" \
    --dataset_path "" \
    --dataset_name "msmarco-passagedev" \
    --special_encode_corpus_path "" \
    --batch_size 64 \
    --stream_batch_size 5000000 \
    --faiss_gpu 1 \
    --search_k 1000 \
    --trec_eval_m -1 \
    --trec_eval_path "" \
    --config_name "no_instruction" \
    --save_encoding 0