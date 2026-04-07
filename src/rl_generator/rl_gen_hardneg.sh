export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=""

project_name=""
experiment_name=""

log_dir=""
mkdir -p "${log_dir}"
log_file="${log_dir}/${experiment_name}.log"

train_data=""

export HYDRA_FULL_ERROR=1
export SWANLAB_API_KEY=
export SWANLAB_LOG_DIR=""
export SWANLAB_MODE=local

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.model.path="" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    data.train_files="${train_data}" \
    data.val_files="${train_data}" \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.train_batch_size=128 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.return_raw_chat=True \
    data.truncation='error' \
    reward_model.enable=true \
    reward_model.n_gpus_per_node=8 \
    reward_model.nnodes=1 \
    reward_model.model.path= \
    reward_model.micro_batch_size_per_gpu=64 \
    reward_model.use_dynamic_bsz=False \
    reward_model.reward_manager=prime \
    reward_model.use_reward_loop=False \
    reward_model.rollout.name=vllm \
    reward_model.rollout.gpu_memory_utilization=0.8 \
    reward_model.rollout.data_parallel_size=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=1 \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger=['console','swanlab'] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=50 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir= \
    2>&1 | tee "${log_file}"