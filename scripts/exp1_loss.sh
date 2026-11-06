#!/usr/bin/env bash
# exp1: loss comparisons (24 runs).
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
mkdir -p -- "$output_root/slurm/loss"

environment="regular-tree"
for seed in 470
do
    for depth in 5 10 15
    do
        for method in TB DB FM
        do
            name="${environment}-h${depth}-${method}-seed${seed}"
            echo "queueing loss-${name}"
            sbatch --chdir="$repo_root" \
                --output="$output_root/slurm/loss/${name}-%j.out" \
                --job-name="loss-${name}" \
                "$launcher" \
                --sim "${environment}" --method "${method}" --seed "${seed}" --total_iterations 10000 \
                --learning_rate 0.001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 \
                --activation leakyrelu --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 \
                --deterministic --tree_g 3 --tree_h "${depth}" \
                --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
                --run-tag loss --output_dir "$output_root/loss/${name}"
        done
    done
done

for environment in hypergrid-simple hypergrid-mild hypergrid-hard
do
    for seed in 470
    do
        for method in TB DB FM
        do
            name="${environment}-${method}-seed${seed}"
            echo "queueing loss-${name}"
            sbatch --chdir="$repo_root" \
                --output="$output_root/slurm/loss/${name}-%j.out" \
                --job-name="loss-${name}" \
                "$launcher" \
                --sim "${environment}" --method "${method}" --seed "${seed}" --total_iterations 10000 \
                --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 \
                --activation leakyrelu --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 \
                --deterministic \
                --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
                --run-tag loss --output_dir "$output_root/loss/${name}"
        done
    done
done

environment="rna14"
for seed in 470
do
    for method in TB DB FM
    do
        name="${environment}-${method}-seed${seed}"
        echo "queueing loss-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/loss/${name}-%j.out" \
            --job-name="loss-${name}" \
            "$launcher" \
            --sim "${environment}" --method "${method}" --seed "${seed}" --total_iterations 10000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 128 128 --activation leakyrelu \
            --initial_z 0 --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy \
            --sample_method 1 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag loss --output_dir "$output_root/loss/${name}"
    done
done

environment="seh_large"
for seed in 470
do
    for method in TB DB FM
    do
        name="${environment}-${method}-seed${seed}"
        echo "queueing loss-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/loss/${name}-%j.out" \
            --job-name="loss-${name}" \
            "$launcher" \
            --sim "${environment}" --method "${method}" --seed "${seed}" --total_iterations 10000 \
            --learning_rate 0.0005 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 256 \
            --activation leakyrelu --initial_z 0 --epsilon 0.05 --memory_size 1000 --gradient_steps 1 \
            --deterministic --seh_large_proxy_device cpu --seh_large_max_blocks 8 --seh_large_min_blocks 2 \
            --seh_large_reward_t 8 --seh_large_reward_exp 10 --seh_large_reward_min 0.01 \
            --seh_large_mode_r_threshold 7.5 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag loss --output_dir "$output_root/loss/${name}"
    done
done
