"""Strict runtime configuration and provenance for reproducible experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


REPRODUCIBILITY_FILENAME = "reproducibility.json"


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    """Hash Python source by relative path and content in stable order."""
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(source_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _seh_large_artifact_hashes(source_root: Path) -> Dict[str, Optional[str]]:
    paths = {
        "proxy_parameters": (
            source_root
            / "envs"
            / "sEH_large"
            / "pretrained_proxy"
            / "best_params.pkl.gz"
        ),
        "proxy_metadata": (
            source_root / "envs" / "sEH_large" / "pretrained_proxy" / "info.pkl.gz"
        ),
        "building_blocks": (
            source_root
            / "envs"
            / "sEH_large"
            / "data"
            / "blocks_PDB_105.json"
        ),
    }
    return {
        name: sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }


def _nvidia_driver_version(cuda_available: bool) -> Optional[str]:
    if not cuda_available:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    return ",".join(sorted(versions)) or None


def collect_reproducibility_record(
    *,
    seed: int,
    deterministic: bool,
    source_root: str,
    proxy_device: Optional[str],
) -> Dict[str, Any]:
    """Collect the execution contract without recording secrets."""
    root = Path(source_root).resolve()
    cuda_available = torch.cuda.is_available()

    gpu = None
    if cuda_available:
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        gpu = {
            "index": device_index,
            "name": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "total_memory_bytes": properties.total_memory,
        }

    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    warnings = []
    if deterministic and python_hash_seed is None:
        warnings.append(
            "PYTHONHASHSEED was not set before Python started. The supplied "
            "Slurm launchers set it; direct invocations should do the same."
        )

    return {
        "schema_version": 1,
        "seed": int(seed),
        "deterministic_requested": bool(deterministic),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "proxy_device": proxy_device,
        "environment": {
            "python_hash_seed": python_hash_seed,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "argv": list(sys.argv),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_geometric": _distribution_version("torch-geometric"),
            "torch_scatter": _distribution_version("torch-scatter"),
            "rdkit": _distribution_version("rdkit"),
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "driver": _nvidia_driver_version(cuda_available),
            "cudnn": (
                torch.backends.cudnn.version() if cuda_available else None
            ),
            "gpu": gpu,
            "matmul_allow_tf32": (
                torch.backends.cuda.matmul.allow_tf32
                if hasattr(torch.backends, "cuda")
                else None
            ),
            "cudnn_allow_tf32": (
                torch.backends.cudnn.allow_tf32
                if hasattr(torch.backends, "cudnn")
                else None
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "threads": {
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
        },
        "source": {
            "root": str(root),
            "python_tree_sha256": source_tree_sha256(root),
            "seh_large_artifacts_sha256": _seh_large_artifact_hashes(root),
        },
        "warnings": warnings,
    }


def write_reproducibility_record(
    model_dir: str, record: Dict[str, Any]
) -> Path:
    output_path = Path(model_dir) / REPRODUCIBILITY_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
