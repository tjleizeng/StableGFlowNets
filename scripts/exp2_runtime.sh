#!/usr/bin/env bash
# exp2: runtime profiling (40 runs).
# Edit the loops and sbatch arguments below, then run with bash on the login node.
set -euo pipefail

if (( $# )); then
    echo "This script takes no arguments. Edit its loops, launcher, output_root, and sbatch flags directly." >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$repo_root/outputs"
launcher="$repo_root/scripts/standby.sh"
cd -- "$repo_root"
mkdir -p -- "$output_root/slurm/runtime"

for environment in hypergrid-mild hypergrid-hard
do
    for seed in 470 3825 4444 8888 9528
    do
        name="${environment}-TB-seed${seed}"
        echo "queueing runtime-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/runtime/${name}-%j.out" \
            --job-name="runtime-${name}" \
            "$launcher" \
            --sim "${environment}" --method TB --seed "${seed}" --total_iterations 10000 --learning_rate 0.0001 \
            --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
            --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --confidence_level 0.95 \
            --runtime-profile --cuda-profile --checkpoint-interval 0 --no-final-evaluation --no-save-final-model \
            --no-save-memory --no-progress \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag runtime --output_dir "$output_root/runtime/${name}"

        name="${environment}-Stable-seed${seed}"
        echo "queueing runtime-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/runtime/${name}-%j.out" \
            --job-name="runtime-${name}" \
            "$launcher" \
            --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 10000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --confidence_level 0.95 --target_TV 0.01 --moving_average_factor 0.05 --patience 10 \
            --max_state_size 10000 --stabilizing_factor max --forward_traj 16 --backward_traj 16 \
            --runtime-profile --cuda-profile --checkpoint-interval 0 --no-final-evaluation --no-save-final-model \
            --no-save-memory --no-progress \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag runtime --output_dir "$output_root/runtime/${name}"
    done
done

environment="rna14"
for seed in 470 3825 4444 8888 9528
do
    name="${environment}-TB-seed${seed}"
    echo "queueing runtime-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/runtime/${name}-%j.out" \
        --job-name="runtime-${name}" \
        "$launcher" \
        --sim "${environment}" --method TB --seed "${seed}" --total_iterations 10000 --learning_rate 0.0001 \
        --batch_size 64 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --confidence_level 0.95 \
        --off_policy --sample_method 1 --runtime-profile --cuda-profile --checkpoint-interval 0 \
        --no-final-evaluation --no-save-final-model --no-save-memory --no-progress \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag runtime --output_dir "$output_root/runtime/${name}"

    name="${environment}-Stable-seed${seed}"
    echo "queueing runtime-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/runtime/${name}-%j.out" \
        --job-name="runtime-${name}" \
        "$launcher" \
        --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 10000 --learning_rate 0.0001 \
        --batch_size 64 --train_freq 64 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --confidence_level 0.95 \
        --off_policy --sample_method 1 --target_TV 0.01 --moving_average_factor 0.05 --patience 10 \
        --max_state_size 10000 --stabilizing_factor max --forward_traj 32 --backward_traj 32 --runtime-profile \
        --cuda-profile --checkpoint-interval 0 --no-final-evaluation --no-save-final-model --no-save-memory \
        --no-progress \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag runtime --output_dir "$output_root/runtime/${name}"
done

environment="seh_large"
for seed in 470 3825 4444 8888 9528
do
    name="${environment}-TB-seed${seed}"
    echo "queueing runtime-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/runtime/${name}-%j.out" \
        --job-name="runtime-${name}" \
        "$launcher" \
        --sim "${environment}" --method TB --seed "${seed}" --total_iterations 10000 --learning_rate 0.0005 \
        --batch_size 32 --train_freq 32 --hidden_sizes 256 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 1 --deterministic --confidence_level 0.95 \
        --seh_large_proxy_device cpu --seh_large_max_blocks 8 --seh_large_min_blocks 2 --seh_large_reward_t 8 \
        --seh_large_reward_exp 10 --seh_large_reward_min 0.01 --seh_large_mode_r_threshold 7.5 --runtime-profile \
        --cuda-profile --checkpoint-interval 0 --no-final-evaluation --no-save-final-model --no-save-memory \
        --no-progress \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag runtime --output_dir "$output_root/runtime/${name}"

    name="${environment}-Stable-seed${seed}"
    echo "queueing runtime-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/runtime/${name}-%j.out" \
        --job-name="runtime-${name}" \
        "$launcher" \
        --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 10000 --learning_rate 0.0005 \
        --batch_size 32 --train_freq 32 --hidden_sizes 256 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 1 --deterministic --confidence_level 0.95 \
        --seh_large_proxy_device cpu --seh_large_max_blocks 8 --seh_large_min_blocks 2 --seh_large_reward_t 8 \
        --seh_large_reward_exp 10 --seh_large_reward_min 0.01 --seh_large_mode_r_threshold 7.5 --target_TV 0.1 \
        --moving_average_factor 0.05 --patience 10 --max_state_size 10000 --stabilizing_factor max \
        --forward_traj 16 --backward_traj 16 --runtime-profile --cuda-profile --checkpoint-interval 0 \
        --no-final-evaluation --no-save-final-model --no-save-memory --no-progress \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag runtime --output_dir "$output_root/runtime/${name}"
done
