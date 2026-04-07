python3 -m sglang.launch_server \
    --model-path /mnt/ali-sh-1/usr/lihaitao/model/Qwen3/Qwen3-32B \
    --port 30000 \
    --host 0.0.0.0 \
    --tp 4 \
    --data-parallel-size 2