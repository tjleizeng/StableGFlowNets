#!/usr/bin/env bash
# exp3: TV/FCS monitoring (7 runs).
# Edit the loops and sbatch arguments below, then run with bash on the login node.
set -euo pipefail

if (( $# )); then
    echo "This script takes no arguments. Edit its loops, launcher, output_root, and sbatch flags directly." >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$repo_root/outputs"
launcher="$repo_root/scripts/dedicated.sh"
cd -- "$repo_root"
mkdir -p -- "$output_root/slurm/monitoring"

environment="regular-tree"
for seed in 0
do
    for depth in 5 10 15
    do
        name="${environment}-h${depth}-Stable-seed${seed}"
        echo "queueing monitoring-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/monitoring/${name}-%j.out" \
            --job-name="monitoring-${name}" \
            "$launcher" \
            --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 20000 \
            --learning_rate 0.001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --tree_g 3 \
            --tree_h "${depth}" --target_TV 0.01 --confidence_level 0.9 --moving_average_factor 0.05 \
            --patience 10 --max_state_size 10000 --stabilizing_factor max --forward_traj 16 --backward_traj 16 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag monitoring --output_dir "$output_root/monitoring/${name}"
    done
done

for environment in hypergrid-simple hypergrid-mild hypergrid-hard
do
    for seed in 0
    do
        name="${environment}-Stable-seed${seed}"
        echo "queueing monitoring-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/monitoring/${name}-%j.out" \
            --job-name="monitoring-${name}" \
            "$launcher" \
            --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 50000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --target_TV 0.01 \
            --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 --max_state_size 10000 \
            --stabilizing_factor max --forward_traj 16 --backward_traj 16 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag monitoring --output_dir "$output_root/monitoring/${name}"
    done
done

environment="rna14"
for seed in 0
do
    name="${environment}-Stable-seed${seed}"
    echo "queueing monitoring-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/monitoring/${name}-%j.out" \
        --job-name="monitoring-${name}" \
        "$launcher" \
        --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 300000 --learning_rate 0.0001 \
        --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy --sample_method 1 \
        --target_TV 0.01 --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 \
        --max_state_size 10000 --stabilizing_factor max --forward_traj 16 --backward_traj 16 \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag monitoring --output_dir "$output_root/monitoring/${name}"
done
