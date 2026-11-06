#!/usr/bin/env bash
#SBATCH --ntasks=1 --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G

# Select account, partition, and QoS via sbatch flags or SBATCH_* environment
# variables before submission; #SBATCH directives do not expand shell variables.

set -euo pipefail

# Slurm copies this script; resolve its checkout from the submission directory.
if [[ -n "${GFN_PROJECT_ROOT:-}" ]]; then
    repo_root="$GFN_PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    repo_root="$SLURM_SUBMIT_DIR"
    case "$(basename -- "$repo_root")" in
        scripts|code) repo_root="$(dirname -- "$repo_root")" ;;
    esac
else
    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -f "$repo_root/code/train.py" ]]; then
    printf 'Cannot find code/train.py under %s. Set GFN_PROJECT_ROOT to the StableGFlowNets directory.\n' "$repo_root" >&2
    exit 1
fi
repo_root="$(cd -- "$repo_root" && pwd)"

# Set reproducibility controls before starting the training interpreter.
seed=42
expect_seed=0
for argument in "$@"; do
    if (( expect_seed )); then
        seed="$argument"
        expect_seed=0
        continue
    fi
    case "$argument" in
        --seed|-se) expect_seed=1 ;;
        --seed=*|-se=*) seed="${argument#*=}" ;;
    esac
done
export PYTHONHASHSEED="$seed" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# Use the active Python environment by default. Conda activation is opt-in;
# all site-specific names/paths come from the submitting user's environment.
if [[ "${GFN_SKIP_CONDA:-0}" != 1 && -n "${GFN_CONDA_ENV:-}" ]]; then
    if [[ -n "${GFN_CONDA_MODULE:-}" ]]; then
        if ! type module >/dev/null 2>&1; then
            printf 'GFN_CONDA_MODULE is set, but the module command is unavailable.\n' >&2
            exit 1
        fi
        module load "$GFN_CONDA_MODULE"
    fi
    if ! command -v conda >/dev/null 2>&1; then
        printf 'Conda is unavailable. Activate an environment before submission, or configure GFN_CONDA_MODULE.\n' >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        printf 'Cannot initialize Conda. Activate the environment first and set GFN_SKIP_CONDA=1.\n' >&2
        exit 1
    fi
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$GFN_CONDA_ENV"
fi
if ! command -v "${PYTHON:-python}" >/dev/null 2>&1; then
    printf 'Python is unavailable. Activate your environment or set PYTHON to an interpreter.\n' >&2
    exit 1
fi

if [[ -z "${WANDB_BASE:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        WANDB_BASE="$SCRATCH/StableGFN/wandb"
    elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
        WANDB_BASE="$SLURM_TMPDIR/wandb"
    else
        WANDB_BASE="${TMPDIR:-/tmp}/wandb_${USER:-unknown}"
    fi
fi
export WANDB_BASE WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_BASE/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_BASE/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_BASE/config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_BASE/data}"
mkdir -p -- "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
unset WANDB_X_REQUIRE_LEGACY_SERVICE

cd -- "$repo_root/code"
# Record source provenance where the analysis notebooks discover run metadata.
"${PYTHON:-python}" - "$repo_root" "$repo_root/outputs/manual" "$@" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

repo_root = Path(sys.argv[1])
default_output = sys.argv[2]
arguments = sys.argv[3:]
if "-h" in arguments or "--help" in arguments:
    raise SystemExit(0)
parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--output_dir", default=default_output)
options, _ = parser.parse_known_args(arguments)
output_dir = Path(options.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
fingerprint_path = output_dir / "fingerprint.json"
source_hashes = {
    path.relative_to(repo_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((repo_root / "code").rglob("*.py"))
}
fingerprint = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "files_sha256": source_hashes,
}
if fingerprint_path.exists():
    previous = json.loads(fingerprint_path.read_text(encoding="utf-8-sig"))
    if previous.get("files_sha256") != source_hashes:
        raise SystemExit(
            f"Source files differ from {fingerprint_path}; use a new --output_dir "
            "to keep existing experiment provenance intact."
        )
else:
    if any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} is nonempty but has no fingerprint.json; use a new "
            "--output_dir to avoid attributing existing results to the current source."
        )
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")
command = shlex.join([sys.executable, "train.py", "--output_dir", default_output, *arguments])
command_path = output_dir / "command.txt"
previous_commands = command_path.read_text(encoding="utf-8").splitlines() if command_path.exists() else []
if command not in previous_commands:
    with command_path.open("a", encoding="utf-8") as command_file:
        if command_path.stat().st_size and previous_commands:
            command_file.write("\n")
        command_file.write(command + "\n")
PY
exec "${PYTHON:-python}" train.py --output_dir "$repo_root/outputs/manual" "$@"
