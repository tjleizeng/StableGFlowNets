#!/usr/bin/env bash
# exp2: long RNA comparisons (4 runs).
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
mkdir -p -- "$output_root/slurm/rna-long"

environment="rna14"
for seed in 0
do
    name="${environment}-TB-seed${seed}"
    echo "queueing rna-long-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/rna-long/${name}-%j.out" \
        --job-name="rna-long-${name}" \
        "$launcher" \
        --sim "${environment}" --method TB --seed "${seed}" --total_iterations 300000 --learning_rate 0.0001 \
        --batch_size 48 --train_freq 48 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy --sample_method 1 \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag rna-long --output_dir "$output_root/rna-long/${name}"

    name="${environment}-Teacher-seed${seed}"
    echo "queueing rna-long-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/rna-long/${name}-%j.out" \
        --job-name="rna-long-${name}" \
        "$launcher" \
        --sim "${environment}" --method Teacher --seed "${seed}" --total_iterations 300000 \
        --learning_rate 0.0001 --batch_size 48 --train_freq 48 --hidden_sizes 256 256 --activation leakyrelu \
        --initial_z 0 --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy \
        --sample_method 1 --alpha 0.0 \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag rna-long --output_dir "$output_root/rna-long/${name}"

    name="${environment}-Stable-seed${seed}"
    echo "queueing rna-long-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/rna-long/${name}-%j.out" \
        --job-name="rna-long-${name}" \
        "$launcher" \
        --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 300000 --learning_rate 0.0001 \
        --batch_size 48 --train_freq 48 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
        --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy --sample_method 1 \
        --target_TV 0.01 --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 \
        --max_state_size 10000 --stabilizing_factor max --forward_traj 24 --backward_traj 24 \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag rna-long --output_dir "$output_root/rna-long/${name}"

    name="${environment}-StableTeacher-seed${seed}"
    echo "queueing rna-long-${name}"
    sbatch --chdir="$repo_root" \
        --output="$output_root/slurm/rna-long/${name}-%j.out" \
        --job-name="rna-long-${name}" \
        "$launcher" \
        --sim "${environment}" --method StableTeacher --seed "${seed}" --total_iterations 300000 \
        --learning_rate 0.0001 --batch_size 48 --train_freq 48 --hidden_sizes 256 256 --activation leakyrelu \
        --initial_z 0 --epsilon 0.05 --memory_size 1000 --gradient_steps 2 --deterministic --off_policy \
        --sample_method 1 --target_TV 0.01 --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 \
        --max_state_size 10000 --stabilizing_factor max --forward_traj 16 --backward_traj 16 --teacher_traj 16 \
        --alpha 0.0 \
        --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
        --run-tag rna-long --output_dir "$output_root/rna-long/${name}"
done
