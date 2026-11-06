#!/usr/bin/env bash
# exp2: long Hypergrid comparisons (120 runs).
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
mkdir -p -- "$output_root/slurm/grid-long"

for environment in hypergrid-simple hypergrid-mild hypergrid-hard
do
    for seed in 470 3825 4444 8888 9528
    do
        name="${environment}-TB-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method TB --seed "${seed}" --total_iterations 30000 --learning_rate 0.0001 \
            --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
            --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-DB-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method DB --seed "${seed}" --total_iterations 30000 --learning_rate 0.0001 \
            --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
            --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-FM-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method FM --seed "${seed}" --total_iterations 30000 --learning_rate 0.0001 \
            --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
            --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-SubTB-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method SubTB --seed "${seed}" --total_iterations 30000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-WDB-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method WDB --seed "${seed}" --total_iterations 30000 --learning_rate 0.0001 \
            --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu --initial_z 0 \
            --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-Teacher-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method Teacher --seed "${seed}" --total_iterations 30000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --alpha 0.0 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-Stable-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method Stable --seed "${seed}" --total_iterations 30000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --target_TV 0.01 \
            --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 --max_state_size 10000 \
            --stabilizing_factor max --forward_traj 16 --backward_traj 16 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"

        name="${environment}-StableTeacher-seed${seed}"
        echo "queueing grid-long-${name}"
        sbatch --chdir="$repo_root" \
            --output="$output_root/slurm/grid-long/${name}-%j.out" \
            --job-name="grid-long-${name}" \
            "$launcher" \
            --sim "${environment}" --method StableTeacher --seed "${seed}" --total_iterations 30000 \
            --learning_rate 0.0001 --batch_size 32 --train_freq 32 --hidden_sizes 256 256 --activation leakyrelu \
            --initial_z 0 --epsilon 0 --memory_size 1000 --gradient_steps 1 --deterministic --target_TV 0.01 \
            --confidence_level 0.9 --moving_average_factor 0.05 --patience 10 --max_state_size 10000 \
            --stabilizing_factor max --forward_traj 16 --backward_traj 8 --teacher_traj 8 --alpha 0.0 \
            --metrics-jsonl --wandb-mode "${WANDB_MODE:-disabled}" \
            --run-tag grid-long --output_dir "$output_root/grid-long/${name}"
    done
done
