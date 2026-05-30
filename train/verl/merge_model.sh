steps=(2)

for step in "${steps[@]}"; do
    local_dir=/mnt/work/yunxiao_4/CuRLRank/train/verl/model/CuRLRank-4B-grpo/global_step_${step}/actor
    target_dir=/mnt/work/yunxiao_4/CuRLRank/train/verl/model/CuRLRank-4B-merged/global_step_${step}
    mkdir -p "$target_dir"

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --target_dir "$target_dir"
done