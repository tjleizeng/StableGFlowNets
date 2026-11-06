"""Shared helpers for the four analysis notebooks.

Edit calculations and plot styles here; the notebooks contain configuration and
the ordered workflow calls. All GF_* environment controls remain available.

Notebook entry points:
    AnalysisSession     Run discovery, configuration, figure export, provenance.
    TrainingAnalysis    Training logs, loss/coverage plots, runtime summaries.
    CheckpointAnalysis  Fresh forward-policy samples, metrics, projection plots.
    OraclePreparation   Bounded RNA-cache preview and full support preparation.
    TVAnalysis          Exact/estimated TV, calibration, comparisons, support sweeps.

The numbered ``# %%`` sections below provide editor navigation. TV formulas,
exact probabilities, and trajectory samplers are ordinary functions in this
file. Importing the helpers does not load experiments or initialize RNA/sEH data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import heapq
import itertools
import json
import math
import os
import pickle
import pickletools
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Configure deterministic CUDA before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

# Also support direct imports when only code/notebooks is on the Python path.
_CODE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(_CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_CODE_DIRECTORY))

from envs.hypergrid import HyperGridEnv
from envs.regular_tree import RegularTreeEnv
from nns.discrete_nn import DiscreteBackwardPolicy, DiscreteForwardPolicy

__all__ = [
    "AnalysisSession",
    "TrainingAnalysis",
    "CheckpointAnalysis",
    "OraclePreparation",
    "TVAnalysis",
]


# %% 1. Shared configuration, run discovery, and exports

PROFILES = (
    "smoke",
    "smoke-tv",
    "main",
    "ablation",
    "loss",
    "grid-long",
    "rna-long",
    "monitoring",
    "runtime",
)
SUPPORTED_METHODS = {
    "tb",
    "db",
    "fm",
    "subtb",
    "wdb",
    "teacher",
    "stable",
    "stableteacher",
}
NON_SCIENCE = {
    "seed",
    "output_dir",
    "run_name",
    "wandb_name",
    "wandb_project",
    "wandb_mode",
    "wandb_entity",
    "verbose",
    "local_metrics",
    "local_metrics_interval",
    "checkpoint_interval",
    "runtime_profile",
    "save_memory",
    "save_model",
    "evaluate_modes",
    "run_tag",
    "metrics_jsonl",
    "progress",
    "cuda_profile",
    "final_evaluation",
    "save_final_model",
    "skip_existing",
}


def find_repository_root(start=None):
    """Find the nearest source checkout; no local manuscript is required."""
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "code/train.py").is_file() and (
            candidate / "requirements.txt"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Run from the repository root or one of its subdirectories."
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(config):
    scientific = {key: value for key, value in config.items() if key not in NON_SCIENCE}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def run_label(config):
    method = config.get("method", "unknown")
    if str(method).lower() == "stable":
        factor = config.get("stabilizing_factor", "max")
        return (
            "TB + backward sampling"
            if factor in (None, "null", "none")
            else "Stable (" + str(factor) + ")"
        )
    return str(method)


@dataclass
class AnalysisSession:
    """One explicit analysis configuration and its discovered input runs.

    ``None`` overrides in :meth:`from_environment` retain the documented GF_*
    environment controls, so both interactive and headless execution work.
    """

    root: Path
    profile: str
    run_root: Path
    figure_root: Path
    strict: bool = False
    expected_seeds: list = field(default_factory=lambda: [470, 3825, 4444, 8888, 9528])
    sd_ddof: int = 1
    issues: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)

    sha = staticmethod(sha256_file)
    signature = staticmethod(signature)
    run_label = staticmethod(run_label)

    @classmethod
    def from_environment(
        cls,
        notebook_name,
        *,
        root=None,
        profile=None,
        run_root=None,
        figure_root=None,
        strict=None,
        warn_h16_target=True,
    ):
        root = find_repository_root(root)
        profile = (
            os.environ.get("GF_ANALYSIS_PROFILE", "smoke")
            if profile is None
            else profile
        )
        if profile not in PROFILES:
            raise ValueError("Unknown analysis profile: " + str(profile))

        def resolve(value):
            path = Path(value)
            return (path if path.is_absolute() else root / path).resolve()

        run_root = resolve(
            os.environ.get("GF_RUN_ROOT", "outputs/" + profile)
            if run_root is None
            else run_root
        )
        figure_root = resolve(
            os.environ.get("GF_FIGURE_ROOT", str(root / "figures" / run_root.name))
            if figure_root is None
            else figure_root
        )
        if root not in figure_root.parents:
            raise ValueError("All generated files must stay inside the repository.")
        figure_root.mkdir(parents=True, exist_ok=True)
        strict = (
            os.environ.get("GF_STRICT", "0") == "1" if strict is None else bool(strict)
        )
        session = cls(root, profile, run_root, figure_root, strict)
        if str(root / "code") not in sys.path:
            sys.path.insert(0, str(root / "code"))
        session._discover_runs()
        session.record(root / "code/notebooks" / notebook_name)
        for path in (root / "code").rglob("*.py"):
            if "__pycache__" not in path.parts:
                session.record(path)
        session._check_sources(warn_h16_target)
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
            }
        )
        print(f"Found {len(session.runs)} run(s) under {run_root}")
        return session

    def repository_path(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.root / path).resolve()

    def record(self, path):
        path = Path(path)
        key = (
            str(path.relative_to(self.root)) if self.root in path.parents else str(path)
        )
        self.sources[key] = self.sha(path)

    def read_json(self, path):
        self.record(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def save_figure(self, figure, name):
        """Export the same PNG/PDF pair, embedding TrueType PDF fonts."""
        figure.savefig(self.figure_root / (name + ".png"), dpi=180, bbox_inches="tight")
        with plt.rc_context(
            {"pdf.fonttype": 42, "ps.fonttype": 42, "pdf.use14corefonts": False}
        ):
            figure.savefig(self.figure_root / (name + ".pdf"), bbox_inches="tight")
        plt.show()
        plt.close(figure)

    def _profile_accepts(self, config):
        total, sim = int(config.get("total_iterations", 0)), config.get("sim", "")
        if self.profile == "grid-long":
            return sim.startswith("hypergrid") and total == 30000
        if self.profile == "rna-long":
            return sim == "rna14" and total == 300000
        if self.profile == "monitoring":
            return total == {"regular-tree": 20000, "rna14": 300000}.get(sim, 50000)
        return total == 10000 if self.profile in ("main", "ablation", "loss") else True

    def _training_provenance(self, path):
        for directory in (path, *path.parents):
            if self.root not in (directory, *directory.parents):
                break
            fingerprint = directory / "fingerprint.json"
            if fingerprint.is_file():
                self.record(fingerprint)
                data = json.loads(fingerprint.read_text(encoding="utf-8-sig"))
                hashes = data.get("files_sha256", {})
                scientific = {
                    key: value
                    for key, value in hashes.items()
                    if key.endswith(".py") and not key.startswith("code/notebooks/")
                }
                digest = hashlib.sha256(
                    json.dumps(scientific, sort_keys=True).encode()
                ).hexdigest()
                return digest, hashes
        return "unrecorded", {}

    def _discover_runs(self):
        paths = (
            sorted(self.run_root.rglob("hyperparameters.json"))
            if self.run_root.exists()
            else []
        )
        for path in paths:
            try:
                config = self.read_json(path)
                if str(config.get("method", "")).lower() not in SUPPORTED_METHODS:
                    self.issues.append(
                        {
                            "path": str(path),
                            "issue": "Run excluded: method is not part of the supported GFlowNet workflows.",
                        }
                    )
                    continue
                if not self._profile_accepts(config):
                    self.issues.append(
                        {
                            "path": str(path),
                            "issue": "Run excluded: training task/budget does not match analysis profile "
                            + self.profile,
                        }
                    )
                    continue
                source_signature, source_hashes = self._training_provenance(path.parent)
                key = hashlib.sha256(
                    str(path.parent.relative_to(self.run_root)).encode()
                ).hexdigest()[:12]
                self.runs.append(
                    {
                        "run_key": key,
                        "path": path.parent,
                        "config": config,
                        "cohort": hashlib.sha256(
                            (signature(config) + source_signature).encode()
                        ).hexdigest()[:12],
                        "source_signature": source_signature,
                        "source_hashes": source_hashes,
                        "method": run_label(config),
                        "sim": config.get("sim", "unknown"),
                        "seed": config.get("seed"),
                    }
                )
            except Exception as exc:
                self.issues.append({"path": str(path), "issue": str(exc)})
        if not self.runs:
            self.issues.append(
                {
                    "path": str(self.run_root),
                    "issue": "No runs found. Train the README quickstart, then rerun the analysis.",
                }
            )

    def hypergrid_training_target(self, run):
        recorded = run["source_hashes"].get("code/envs/hypergrid.py")
        if recorded == self.sha(self.root / "code/envs/hypergrid.py"):
            return "float64_consistent_current_source"
        if (
            recorded
            == "752f13d9c4dafc7829c3ea7fa4aa5572aadf39c47a344ccbecf33c67d4cee337"
        ):
            return "legacy_float32_operational_rewards"
        return "training_source_not_established"

    def _check_sources(self, warn_h16_target):
        for run in self.runs:
            if (
                warn_h16_target
                and run["sim"] == "hypergrid-mild"
                and self.hypergrid_training_target(run)
                != "float64_consistent_current_source"
            ):
                self.issues.append(
                    {
                        "path": str(run["path"]),
                        "issue": "Historical/unknown H16 training provenance: "
                        + self.hypergrid_training_target(run)
                        + ". Evaluation uses the current float64 target; rerun affected old checkpoints.",
                    }
                )
            method = str(run["config"].get("method", "")).lower()
            if method not in ("fm", "wdb"):
                continue
            source = (
                "code/learning/gfn_FM.py"
                if method == "fm"
                else "code/learning/gfn_DB_WDB.py"
            )
            historical = run["source_hashes"].get(source) != self.sha(
                self.root / source
            )
            warm_start = bool(run["config"].get("load_model"))
            if historical:
                objective = (
                    "local objective" if method == "fm" else "per-trajectory weighting"
                )
                message = f"{method.upper()} checkpoint predates the corrected {objective} or has unverified source; train a fresh run before paper comparison."
                self.issues.append({"path": str(run["path"]), "issue": message})
                if self.strict:
                    raise RuntimeError(message + " " + str(run["path"]))
            if warm_start:
                message = f"Warm-started {method.upper()} is a separate fine-tuning protocol, not the from-scratch paper baseline."
                self.issues.append({"path": str(run["path"]), "issue": message})
                if self.strict:
                    raise RuntimeError(message + " " + str(run["path"]))
            if method == "fm":
                if historical:
                    run["method"] = "FM (historical/unverified source)"
                if warm_start:
                    run["method"] = "FM (warm start; separate protocol)"
            elif historical or warm_start:
                qualifiers = (
                    ["historical/unverified source"] if historical else []
                ) + (["warm start; separate protocol"] if warm_start else [])
                run["method"] = "WDB (" + "; ".join(qualifiers) + ")"

    def finish(self, kind, extra=None, complete=True):
        """Write input hashes and diagnostics, then enforce completeness in strict mode."""
        from IPython.display import display

        issues = pd.DataFrame(self.issues, columns=["path", "issue"])
        issues.to_csv(self.figure_root / f"REVIEW_{kind}.csv", index=False)
        manifest = {
            "input_sha256": self.sources,
            "run_root": str(self.run_root),
            "analysis_profile": self.profile,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "std_ddof": self.sd_ddof,
            "issues": self.issues,
        }
        manifest.update(extra or {})
        (self.figure_root / f"{kind}_analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        display(issues)
        print("Exports:", self.figure_root)
        if self.strict and (not self.runs or not complete):
            raise RuntimeError(
                f"Strict analysis failed: no inputs or incomplete outputs; inspect REVIEW_{kind}.csv"
            )
        return manifest


# %% 2. Training diagnostics and plots — TrainingAnalysis

RUN_COLUMNS = ("run_key", "cohort", "method", "sim", "seed")
SINGLE_UPDATE_RATIO_METHODS = {
    "TB",
    "FM",
    "DB",
    "WDB",
    "STABLE",
    "STABLETEACHER",
    "TEACHER",
}


class TrainingAnalysis:
    """Read local training logs while keeping cohorts and timing protocols separate."""

    def __init__(self, session):
        self.session = session
        self.metrics = pd.DataFrame()
        self.training = pd.DataFrame()
        self.runtime = pd.DataFrame()
        self.modes = pd.DataFrame()
        self.runtime_summary = pd.DataFrame()

    def load(self):
        """Load JSONL metrics and export per-run training, mode, and timing tables."""
        session = self.session
        frames, train_rows, runtime_rows, mode_rows = [], [], [], []
        for run in session.runs:
            path = run["path"] / "training_metrics.jsonl"
            if not path.is_file():
                session.issues.append(
                    {
                        "path": str(path),
                        "issue": "Missing optional local metrics; rerun with --metrics-jsonl.",
                    }
                )
                continue
            session.record(path)
            rows = []
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                try:
                    row = json.loads(line)
                    if row.get("schema_version") != 1:
                        raise ValueError("Unsupported metrics schema")
                    rows.append(row)
                except Exception as exc:
                    session.issues.append(
                        {"path": str(path), "issue": f"Line {line_no}: {exc}"}
                    )
            frame = pd.DataFrame(rows)
            if frame.empty or "iteration" not in frame:
                continue
            if frame["iteration"].duplicated().any():
                session.issues.append(
                    {
                        "path": str(path),
                        "issue": "Duplicate iterations: resumed or repeated runs require manual separation; excluded.",
                    }
                )
                continue
            frame = frame.sort_values("iteration")
            if (np.diff(frame.iteration) != 1).any():
                session.issues.append(
                    {
                        "path": str(path),
                        "issue": "Non-contiguous logged rounds: smoothing is time-indexed but peak ratio may miss unlogged rounds.",
                    }
                )
            for key in RUN_COLUMNS:
                frame[key] = run[key]
            frame["loss"] = frame.get(
                "train/loss_mean_unrounded", frame.get("train/loss_mean", np.nan)
            )
            self._loss_ratios(frame, run, path)
            frame["tree_h"] = run["config"].get("tree_h")
            frame["training_budget"] = run["config"].get("total_iterations")
            frames.append(frame)
            train_rows.append(
                {
                    **{key: run[key] for key in RUN_COLUMNS},
                    "last_round": int(frame.iteration.max()),
                    "requested_rounds": run["config"].get("total_iterations"),
                    "train_modes": frame.get(
                        "train/total_modes", pd.Series([np.nan])
                    ).iloc[-1],
                    "peak_max_rest_ratio": frame.max_rest_ratio.max(),
                }
            )
            mode_path = run["path"] / "mode_metrics.json"
            if mode_path.exists():
                payload = session.read_json(mode_path)
                if payload.get("status") == "ok":
                    mode_rows.append({**train_rows[-1], **payload.get("metrics", {})})
                else:
                    session.issues.append(
                        {
                            "path": str(mode_path),
                            "issue": "End-of-run mode evaluation failed.",
                        }
                    )
            runtime_path = run["path"] / "runtime_profile.json"
            if runtime_path.exists():
                runtime_rows.extend(
                    self._runtime_records(run, session.read_json(runtime_path))
                )

        self.metrics = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        self.training = pd.DataFrame(train_rows)
        self.runtime = pd.DataFrame(runtime_rows)
        self.modes = pd.DataFrame(mode_rows)
        for name, frame in [
            ("training_summary", self.training),
            ("runtime_per_seed", self.runtime),
            ("molecular_training_metrics", self.modes),
        ]:
            frame.to_csv(session.figure_root / (name + ".csv"), index=False)
        return self

    def _loss_ratios(self, frame, run, path):
        """Prefer exact per-update peaks; never infer multi-update peaks from averages."""
        frame["max_rest_ratio"] = frame.get("train/loss_max_to_rest_ratio_max", np.nan)
        method = str(run["config"].get("method", "")).upper()
        batch_size = int(run["config"].get("batch_size", 32))
        if (
            "train/loss_means_per_batch" in frame
            and "train/loss_max_per_batch" in frame
            and method != "SUBTB"
        ):

            def round_peak(row):
                means = np.asarray(row["train/loss_means_per_batch"], dtype=float)
                maxima = np.asarray(row["train/loss_max_per_batch"], dtype=float)
                if means.size == 0 or means.shape != maxima.shape:
                    return np.nan
                rest = batch_size * means - maxima
                ratios = np.divide(
                    maxima, rest, out=np.full_like(maxima, np.nan), where=rest > 0
                )
                return np.nanmax(ratios) if np.any(np.isfinite(ratios)) else np.nan

            frame["max_rest_ratio"] = frame.apply(round_peak, axis=1)
        if frame.max_rest_ratio.isna().all():
            if (
                int(run["config"].get("gradient_steps", 1)) == 1
                and method in SINGLE_UPDATE_RATIO_METHODS
            ):
                maximum = frame.get(
                    "train/loss_max_unrounded", frame.get("train/loss_max", np.nan)
                )
                rest = batch_size * frame.loss - maximum
                frame["max_rest_ratio"] = np.where(rest > 0, maximum / rest, np.nan)
            else:
                self.session.issues.append(
                    {
                        "path": str(path),
                        "issue": "Exact max/rest ratio unavailable for this method/update aggregation; not reconstructed from averaged losses.",
                    }
                )

    @staticmethod
    def _runtime_records(run, payload):
        records = (
            payload.get("runs", [payload]) if isinstance(payload, dict) else payload
        )
        rows = []
        for profile in records:
            result = {
                **{key: run[key] for key in RUN_COLUMNS},
                "status": profile.get("status"),
            }
            for family, label in [
                ("wall_timings_seconds", "wall"),
                ("timings_seconds", "cpu_gpu"),
            ]:
                values = profile.get(family, {}) or {}
                numerator, denominator = values.get("reference_flow"), values.get(
                    "training"
                )
                result[label + "_reference_flow_percent"] = (
                    100 * numerator / denominator
                    if numerator is not None
                    and denominator is not None
                    and denominator > 0
                    else np.nan
                )
            runtime_config = {
                "learning_cohort": run["cohort"],
                "runtime_profile": run["config"].get("runtime_profile"),
                "cuda_profile": run["config"].get("cuda_profile"),
                "device": run["config"].get("device"),
                "torch_num_threads": run["config"].get("torch_num_threads"),
                "single_threaded": run["config"].get("single_threaded"),
                "timing_semantics": profile.get("timing_semantics"),
                "kernel_available": profile.get("gpu_kernel_profiling", {}).get(
                    "available"
                ),
                "kernel_sampling_interval": profile.get("gpu_kernel_profiling", {}).get(
                    "sampling_interval_top_level_calls"
                ),
            }
            result["runtime_cohort"] = hashlib.sha256(
                json.dumps(runtime_config, sort_keys=True).encode()
            ).hexdigest()[:12]
            rows.append(result)
        return rows

    def plot_traces(self):
        """Plot each cohort, smoothing losses over 1,000 rounds but retaining ratio spikes."""
        if self.metrics.empty:
            return
        plot_metrics = [
            ("loss", "Mean training loss"),
            ("max_rest_ratio", "Peak update max/rest ratio"),
            ("train/total_modes", "Online discovered modes"),
            ("train/current_mass", "Retained reward mass"),
        ]
        for cohort, group in self.metrics.groupby("cohort"):
            present = [
                (key, label)
                for key, label in plot_metrics
                if key in group and group[key].notna().any()
            ]
            if not present:
                continue
            fig, axes = plt.subplots(
                1, len(present), figsize=(4.5 * len(present), 3.2), squeeze=False
            )
            for ax, (key, label) in zip(axes[0], present):
                for _, trace in group.groupby("run_key"):
                    trace = trace.sort_values("iteration")
                    values = pd.to_numeric(trace[key], errors="coerce")
                    if key == "loss":
                        time_index = pd.to_timedelta(
                            trace.iteration.to_numpy(), unit="s"
                        )
                        values = (
                            pd.Series(values.to_numpy(), index=time_index)
                            .rolling("1000s", min_periods=1)
                            .mean()
                            .to_numpy()
                        )
                    ax.plot(
                        trace.iteration,
                        values,
                        label=f"seed {trace.seed.iloc[0]}",
                        lw=1,
                    )
                ax.set(xlabel="Training round", ylabel=label)
                ax.legend(fontsize=8)
            fig.suptitle(f"{group.sim.iloc[0]} / {group.method.iloc[0]} / {cohort}")
            fig.tight_layout()
            self.session.save_figure(fig, "training_" + cohort)

    def summarize_runtime(self):
        """Export runtime means/SD and report incomplete or incompatible cohorts."""
        session = self.session
        if not self.runtime.empty:
            self.runtime_summary = (
                self.runtime.groupby(
                    ["sim", "method", "cohort", "runtime_cohort"], dropna=False
                )
                .agg(
                    n_seeds=("seed", "nunique"),
                    wall_mean=("wall_reference_flow_percent", "mean"),
                    wall_std=("wall_reference_flow_percent", "std"),
                    cpu_gpu_mean=("cpu_gpu_reference_flow_percent", "mean"),
                    cpu_gpu_std=("cpu_gpu_reference_flow_percent", "std"),
                )
                .reset_index()
            )
            self.runtime_summary.to_csv(
                session.figure_root / "table10_runtime_summary.csv", index=False
            )
        if not self.runtime.empty:
            for cohort, group in self.runtime.groupby("cohort"):
                if group.runtime_cohort.nunique() > 1:
                    session.issues.append(
                        {
                            "path": cohort,
                            "issue": "Mixed profiler flags or timing semantics: runtime results are kept in separate runtime_cohort groups.",
                        }
                    )
        if not self.training.empty:
            for cohort, group in self.training.groupby("cohort"):
                observed = sorted(set(group.seed.dropna().astype(int)))
                if observed != session.expected_seeds:
                    session.issues.append(
                        {
                            "path": cohort,
                            "issue": f"Observed seeds {observed}; five-seed RQ2 requires {session.expected_seeds}. RQ1/RQ3 use their separate protocol.",
                        }
                    )
        return self.runtime_summary

    def plot_normalized_traces(self):
        """Export the existing eight-panel loss/ratio layout for seed 470."""
        if self.metrics.empty or os.environ.get("GF_PAPER_LAYOUTS", "1") != "1":
            return
        panels = [
            ("regular-tree", 5, 2, 0.01),
            ("regular-tree", 10, 2, 0.01),
            ("regular-tree", 15, 2, 0.01),
            ("rna14", None, 10, 10),
            ("hypergrid-simple", None, 2, 1),
            ("hypergrid-mild", None, 50, 2),
            ("hypergrid-hard", None, 50, 3),
            ("seh_large", None, 100, 300),
        ]
        colors = {"TB": "#4C78A8", "FM": "#F58518", "DB": "#54A24B"}
        fig, axes = plt.subplots(2, 4, figsize=(15, 6), squeeze=False)
        for ax, (sim, depth, ratio_scale, loss_scale) in zip(axes.flat, panels):
            subset = self.metrics[
                (self.metrics.sim == sim)
                & (self.metrics.seed == 470)
                & (self.metrics.method.isin(colors))
            ]
            if depth is not None:
                subset = subset[subset.tree_h == depth]
            title = (
                f"Tree D={depth}" if depth else sim
            ) + f"\nratio/{ratio_scale:g}, loss/{loss_scale:g}"
            for method, group in subset.groupby("method"):
                if group.run_key.nunique() != 1:
                    self.session.issues.append(
                        {
                            "path": "figure3/" + sim + "/" + method,
                            "issue": "Multiple configurations match this panel; select a single suite/configuration. Trace omitted.",
                        }
                    )
                    continue
                group = group.sort_values("iteration")
                index = pd.to_timedelta(group.iteration.to_numpy(), unit="s")
                smooth = (
                    pd.Series(group.loss.to_numpy(), index=index)
                    .rolling("1000s", min_periods=1)
                    .mean()
                )
                ax.plot(
                    group.iteration / 100,
                    group.max_rest_ratio / ratio_scale,
                    color=colors[method],
                    label=method,
                    lw=0.8,
                )
                ax.plot(
                    group.iteration / 100,
                    smooth.to_numpy() / loss_scale,
                    color=colors[method],
                    ls=":",
                    lw=1.4,
                )
            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No matching runs",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set(
                title=title,
                xlabel="Training rounds (x100)",
                ylabel="Normalized ratio / loss",
            )
            if len(ax.lines):
                ax.legend(fontsize=8)
        fig.tight_layout()
        self.session.save_figure(fig, "figure3_loss_concentration")

    def finish(self):
        """Write provenance and diagnostics, enforcing strict completeness."""
        complete = bool(self.session.runs) and len(self.training) == len(
            self.session.runs
        )
        return self.session.finish("training", complete=complete)


# %% 3. Fresh policy evaluation — CheckpointAnalysis

def _reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_policy(env, policy, count):
    """Sample complete trajectories with epsilon zero and temperature one."""
    states = []
    rewards = []
    with torch.inference_mode():
        for _ in range(count):
            observation, _ = env.reset()
            finished = False
            for _ in range(10000):
                previous = np.asarray(observation).copy()
                mask = env.get_forward_action_masks(previous)
                action = int(
                    policy(
                        previous, action_masks=mask, epsilon=0.0, temperature=1.0
                    ).item()
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise RuntimeError(
                        "Evaluation trajectory was truncated; no silent resampling"
                    )
                if terminated:
                    states.append(env.get_state(previous))
                    rewards.append(float(reward))
                    finished = True
                    break
            if not finished:
                raise RuntimeError("Evaluation exceeded 10,000 steps")
    return states, np.asarray(rewards)


class CheckpointAnalysis:
    """Evaluate saved policies without changing their training state.

    Explicit constructor arguments override the corresponding ``GF_*``
    environment variables. Results remain separated by training cohort,
    checkpoint, sample count, and evaluation seed.
    """

    def __init__(self, session, samples=None, seed=None, checkpoints=None, device=None):
        self.session = session
        self.samples = int(
            os.environ.get("GF_EVAL_SAMPLES", "256") if samples is None else samples
        )
        self.seed = int(os.environ.get("GF_EVAL_SEED", "42") if seed is None else seed)
        default_checkpoints = (
            "2000,4000,6000,8000,10000" if session.profile == "grid-long" else "final"
        )
        if checkpoints is None:
            checkpoints = os.environ.get("GF_CHECKPOINTS", default_checkpoints)
        if isinstance(checkpoints, str):
            checkpoints = checkpoints.split(",")
        self.checkpoints = [str(checkpoint) for checkpoint in checkpoints]
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            os.environ.get("GF_DEVICE", default_device) if device is None else device
        )
        if self.samples < 1:
            raise ValueError("GF_EVAL_SAMPLES must be positive")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.evaluation = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.snapshot_panels = {}

    def evaluate(self):
        """Draw fresh samples, save terminal states, and return per-run metrics."""
        session = self.session
        sys.path.insert(0, str(session.root / "code"))
        from train import create_environment, get_activation_fn, get_arguments
        from nns.discrete_nn import DiscreteForwardPolicy

        results = []
        self.snapshot_panels = {}
        for run in session.runs:
            for checkpoint in self.checkpoints:
                folder = (
                    run["path"]
                    if checkpoint == "final"
                    else run["path"] / ("model_" + checkpoint)
                )
                policy_path = folder / "forward_policy.pth"
                try:
                    if not policy_path.exists():
                        raise FileNotFoundError(policy_path)
                    session.record(policy_path)
                    args = get_arguments([])
                    for key, value in run["config"].items():
                        setattr(args, key, value)
                    args.wandb_project = None

                    # Preserve the original reset order: environment and policy
                    # construction precede the reset used for actual sampling.
                    _reset_seed(self.seed)
                    env = create_environment(args)
                    env.reset(seed=self.seed)
                    policy = DiscreteForwardPolicy(
                        int(env.observation_space.shape[0]),
                        int(env.action_space.n),
                        hidden_sizes=args.hidden_sizes,
                        activation_fn=get_activation_fn(args.activation),
                        device=str(self.device),
                    ).to(self.device)
                    policy.load_state_dict(
                        torch.load(
                            policy_path, map_location=self.device, weights_only=True
                        )
                    )
                    policy.eval()
                    _reset_seed(self.seed)
                    states, rewards = _sample_policy(env, policy, self.samples)
                    row = {
                        **{
                            key: run[key]
                            for key in ("run_key", "cohort", "method", "sim", "seed")
                        },
                        "checkpoint": checkpoint,
                        "n_samples": self.samples,
                        "evaluation_seed": self.seed,
                        "mean_reward": float(rewards.mean()),
                        "protocol": (
                            "paper_sample_count"
                            if self.samples == 100000
                            else "smoke_sample_count"
                        ),
                    }
                    sample_stem = f"samples_{run['run_key']}_{checkpoint}"
                    if run["sim"].startswith("hypergrid"):
                        row["empirical_total_l1"] = float(env.get_error(states))
                        row["empirical_tv"] = row["empirical_total_l1"] / 2
                        row["target_definition"] = "stored_float64_truth_dist"
                        row["operational_target_mismatch"] = (
                            run["sim"] == "hypergrid-mild"
                            and session.hypergrid_training_target(run)
                            == "legacy_float32_operational_rewards"
                        )
                        row["training_target_provenance"] = (
                            session.hypergrid_training_target(run)
                        )
                        array = np.asarray(states, dtype=np.int16)
                        self._hypergrid_marginals(run, checkpoint, env, array, row)
                        np.savez_compressed(
                            session.figure_root / (sample_stem + ".npz"),
                            states=array,
                            rewards=rewards,
                        )
                    elif run["sim"] == "rna14":
                        row["test_modes"] = int(env.get_error(states))
                        self._save_sequence_samples(sample_stem, states, rewards)
                    elif run["sim"] == "seh_large":
                        values = env.get_error(states, rewards=rewards)
                        for name, value in zip(env.MODE_METRIC_NAMES, values):
                            row["test_" + name] = int(value)
                        self._save_sequence_samples(sample_stem, states, rewards)
                    else:
                        # The tree metric expects a batch array with .shape.
                        evaluation_states = (
                            np.asarray(states) if run["sim"] == "regular-tree" else states
                        )
                        row["environment_error"] = float(env.get_error(evaluation_states))
                        np.savez_compressed(
                            session.figure_root / (sample_stem + ".npz"),
                            states=np.asarray(states),
                            rewards=rewards,
                        )
                    self._training_discovery_metrics(run, row)
                    results.append(row)
                    env.close()
                    print(run["run_key"], checkpoint, "evaluated", self.samples)
                except Exception as exc:
                    session.issues.append(
                        {"path": str(folder), "issue": f"{type(exc).__name__}: {exc}"}
                    )
        self.evaluation = pd.DataFrame(results)
        self.evaluation.to_csv(
            session.figure_root / "checkpoint_metrics.csv", index=False
        )
        return self.evaluation

    def _save_sequence_samples(self, stem, states, rewards):
        (self.session.figure_root / (stem + ".json")).write_text(
            json.dumps({"states": states, "rewards": rewards.tolist()}),
            encoding="utf-8",
        )

    def _hypergrid_marginals(self, run, checkpoint, env, array, row):
        pairs = [(0, 1)] + ([(2, 3)] if env.ndim >= 4 else [])
        pair_panels = []
        for first, second in pairs:
            histogram = np.zeros((env.height, env.height))
            np.add.at(histogram, (array[:, first], array[:, second]), 1)
            histogram /= len(array)
            reduced = tuple(
                axis for axis in range(env.ndim) if axis not in (first, second)
            )
            target = env.truth_dist.sum(axis=reduced) if reduced else env.truth_dist
            pair_panels.append(
                {"pair": (first, second), "sample": histogram, "target": target}
            )
        record = self.snapshot_panels.setdefault(
            run["run_key"], {"run": run, "checkpoints": []}
        )
        record["checkpoints"].append({"checkpoint": checkpoint, "panels": pair_panels})

        fig, axes = plt.subplots(
            len(pairs),
            2,
            figsize=(7, 3.1 * len(pairs)),
            squeeze=False,
            layout="constrained",
        )
        for index, panel in enumerate(pair_panels):
            first, second = panel["pair"]
            vmax = max(panel["sample"].max(), panel["target"].max())
            for ax, data, label in zip(
                axes[index],
                [panel["target"], panel["sample"]],
                ["Stored target marginal", "Sample marginal"],
            ):
                image = ax.imshow(
                    data.T, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
                )
                ax.set(
                    title=label,
                    xlabel=f"Coordinate {first + 1}",
                    ylabel=f"Coordinate {second + 1}",
                )
            fig.colorbar(image, ax=list(axes[index]), shrink=0.8)
        affected = (
            " | Historical H16 training source: rerun required"
            if row["operational_target_mismatch"]
            else ""
        )
        fig.suptitle(
            f"{run['sim']} / {run['method']} / seed {run['seed']} / {checkpoint}"
            + affected
        )
        self.session.save_figure(fig, f"hypergrid_{run['run_key']}_{checkpoint}")

    def _training_discovery_metrics(self, run, row):
        session = self.session
        metrics_path = run["path"] / "training_metrics.jsonl"
        if metrics_path.exists():
            session.record(metrics_path)
            logs = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if logs:
                row["last_training_round"] = max(item["iteration"] for item in logs)
                row["train_modes"] = logs[-1].get("train/total_modes")
        mode_path = run["path"] / "mode_metrics.json"
        if mode_path.exists():
            payload = session.read_json(mode_path)
            if payload.get("status") == "ok":
                for name, value in payload.get("metrics", {}).items():
                    row["train_" + name] = value
        if run["sim"] == "seh_large" and not mode_path.exists():
            session.issues.append(
                {
                    "path": str(mode_path),
                    "issue": "Missing training-discovered sEH canonical/scaffold/diversity metrics; test-policy counts cannot fill Table 2/6 training columns.",
                }
            )
        if run["sim"] == "rna14" and row.get("train_modes") is None:
            session.issues.append(
                {
                    "path": str(metrics_path),
                    "issue": "Missing online RNA training mode count; no train-mode value is inferred from final-policy samples.",
                }
            )

    def plot_snapshots(self):
        """Export projections with a shared color scale across each row."""
        session = self.session
        composite_rows = []
        for runkey, panel_record in self.snapshot_panels.items():
            run = panel_record["run"]
            snapshots = panel_record["checkpoints"]
            if not snapshots:
                continue
            order = {
                checkpoint: index for index, checkpoint in enumerate(self.checkpoints)
            }
            snapshots = sorted(snapshots, key=lambda item: order[item["checkpoint"]])
            pairs = [panel["pair"] for panel in snapshots[0]["panels"]]
            fig, axes = plt.subplots(
                len(pairs),
                len(self.checkpoints) + 1,
                figsize=(2.6 * (len(self.checkpoints) + 1), 2.8 * len(pairs)),
                squeeze=False,
                layout="constrained",
            )
            lookup = {item["checkpoint"]: item for item in snapshots}
            for row_index, (first, second) in enumerate(pairs):
                target = snapshots[0]["panels"][row_index]["target"]
                vmax = max(
                    [target.max()]
                    + [item["panels"][row_index]["sample"].max() for item in snapshots]
                )
                image = axes[row_index, 0].imshow(
                    target.T, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
                )
                axes[row_index, 0].set(
                    title="Stored target",
                    xlabel=f"Coordinate {first + 1}",
                    ylabel=f"Coordinate {second + 1}",
                )
                for column, checkpoint in enumerate(self.checkpoints, 1):
                    ax = axes[row_index, column]
                    if checkpoint in lookup:
                        data = lookup[checkpoint]["panels"][row_index]["sample"]
                        image = ax.imshow(
                            data.T, origin="lower", vmin=0, vmax=vmax, cmap="viridis"
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "Missing checkpoint",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    ax.set(
                        title=(
                            "Final policy"
                            if checkpoint == "final"
                            else f"Round {int(checkpoint):,}"
                        ),
                        xlabel=f"Coordinate {first + 1}",
                        ylabel=f"Coordinate {second + 1}",
                    )
                fig.colorbar(
                    image,
                    ax=list(axes[row_index]),
                    shrink=0.85,
                    label="Marginal probability",
                )
            affected = (
                " | Historical H16 training source: rerun required"
                if run["sim"] == "hypergrid-mild"
                and session.hypergrid_training_target(run)
                != "float64_consistent_current_source"
                else ""
            )
            fig.suptitle(
                f"{run['sim']} / {run['method']} / seed {run['seed']} | {self.samples:,} samples per checkpoint"
                + affected
            )
            session.save_figure(fig, "hypergrid_checkpoints_" + runkey)
            composite_rows.append(
                {
                    "run_key": runkey,
                    "environment": run["sim"],
                    "method": run["method"],
                    "training_seed": run["seed"],
                    "requested_checkpoints": ",".join(self.checkpoints),
                    "available_checkpoints": ",".join(
                        item["checkpoint"] for item in snapshots
                    ),
                    "coordinate_pairs": ";".join(f"{a + 1},{b + 1}" for a, b in pairs),
                    "n_samples": self.samples,
                    "target_definition": "stored_float64_truth_dist",
                    "operational_target_mismatch": run["sim"] == "hypergrid-mild"
                    and session.hypergrid_training_target(run)
                    == "legacy_float32_operational_rewards",
                    "training_target_provenance": session.hypergrid_training_target(
                        run
                    ),
                }
            )
        pd.DataFrame(composite_rows).to_csv(
            session.figure_root / "hypergrid_checkpoint_panels.csv", index=False
        )

    def summarize(self):
        """Summarize independent training seeds without pooling sample budgets."""
        session = self.session
        evaluation = self.evaluation
        if not evaluation.empty:
            keys = [
                "cohort",
                "sim",
                "method",
                "checkpoint",
                "n_samples",
                "evaluation_seed",
            ]
            numeric = [
                column
                for column in evaluation.select_dtypes(include=np.number).columns
                if column not in keys + ["seed", "last_training_round"]
            ]
            rows = []
            for groupkey, group in evaluation.groupby(keys, dropna=False):
                row = dict(zip(keys, groupkey))
                row["n_training_seeds"] = group.seed.nunique()
                if group.seed.duplicated().any():
                    session.issues.append(
                        {
                            "path": row["cohort"],
                            "issue": "Duplicate training seed within cohort; excluded from aggregate.",
                        }
                    )
                    continue
                for metric in numeric:
                    values = group[metric].dropna()
                    row[metric + "_mean"] = values.mean()
                    row[metric + "_std"] = values.std(ddof=session.sd_ddof)
                    row[metric + "_n"] = len(values)
                rows.append(row)
                observed = sorted(group.seed.astype(int).unique())
                if observed != session.expected_seeds:
                    session.issues.append(
                        {
                            "path": row["cohort"],
                            "issue": f"Incomplete RQ2 seed set: {observed}",
                        }
                    )
            self.summary = pd.DataFrame(rows)
            self.summary.to_csv(
                session.figure_root / "performance_summary.csv", index=False
            )
        if self.samples != 100000:
            session.issues.append(
                {
                    "path": str(session.run_root),
                    "issue": f"{self.samples} evaluation samples used; the paper's Hypergrid final-policy tables/snapshots require 100000. RNA and sEH final-policy metrics are auxiliary diagnostics.",
                }
            )
        return self.summary

    def finish(self):
        """Write provenance and diagnostics, enforcing strict completeness."""
        extra = {
            "analysis_scope": "Hypergrid final-policy sample metrics; RNA test_modes and final-policy sEH metrics are auxiliary diagnostics, not the training-discovery counts in final Tables 2/5.",
            "evaluation_samples": self.samples,
            "evaluation_seed": self.seed,
            "checkpoints": self.checkpoints,
            "device": str(self.device),
            "epsilon": 0,
            "temperature": 1,
        }
        complete = bool(self.session.runs) and len(self.evaluation) == len(
            self.session.runs
        ) * len(self.checkpoints)
        return self.session.finish("checkpoint", extra=extra, complete=complete)


# %% 4. RNA oracle preparation — OraclePreparation

class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, n=-1):
        data = self.stream.read(n)
        self.digest.update(data)
        return data

    def readline(self):
        data = self.stream.readline()
        self.digest.update(data)
        return data

    def tell(self):
        return self.stream.tell()


def iter_prediction_dictionary(reader):
    """Primitive PROTO4/5 dict[str,float] only; no pickle.load or memo allocation."""
    pending = None
    dictionary_seen = False
    stopped = False
    allowed = {"PROTO", "FRAME", "MEMOIZE", "MARK", "SETITEMS", "SETITEM"}
    for opcode, value, position in pickletools.genops(reader):
        name = opcode.name
        if name == "EMPTY_DICT":
            if dictionary_seen:
                raise ValueError("Nested dictionaries are unsupported")
            dictionary_seen = True
        elif name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"):
            if not dictionary_seen or pending is not None:
                raise ValueError("Unexpected string in prediction cache")
            if len(value) != 14 or set(value) - set("UCGA"):
                raise ValueError("Invalid RNA14 key")
            pending = value
        elif name == "BINFLOAT":
            if pending is None or not math.isfinite(value):
                raise ValueError("Invalid score in prediction cache")
            yield pending, float(value)
            pending = None
        elif name == "STOP":
            if pending is not None or not dictionary_seen:
                raise ValueError("Incomplete prediction dictionary")
            stopped = True
            break
        elif name not in allowed:
            raise ValueError(
                f"Unsupported pickle opcode {name} at {position}; no objects were unpickled"
            )
    if not stopped:
        raise ValueError("Truncated prediction dictionary")
    if reader.read(1):
        raise ValueError("Trailing content after prediction dictionary")


def select_oracle(
    entries,
    top_k,
    expected_states,
    scale,
    beta=40,
    reward_min=0.001,
    chunk_size=262144,
    reward_epsilon=1e-12,
):
    """Select ranked states and normalize the environment's full reward formula.

    The unscaled moment remains available for comparison with the bundled
    statistics; rewards and their normalization include the additive epsilon.
    """
    if not math.isfinite(reward_epsilon) or reward_epsilon < 0:
        raise ValueError("Reward epsilon must be finite and nonnegative")
    heap = []
    logs = []
    global_log_z = -np.inf
    previous = None
    count = 0
    translation = bytes.maketrans(b"UCGA", b"0123")
    for sequence, score in entries:
        order = sequence.encode("ascii").translate(translation)
        if previous is not None and order <= previous:
            raise ValueError("Cache keys must be unique and ordered in the source UCGA enumeration")
        previous = order
        candidate = (max(score, reward_min), -count, sequence)
        if len(heap) < top_k:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
        logs.append(beta * math.log(max(score, reward_min)))
        count += 1
        if len(logs) >= chunk_size:
            global_log_z = float(
                np.logaddexp(global_log_z, logsumexp(np.asarray(logs, dtype=np.float64)))
            )
            logs.clear()
    if logs:
        global_log_z = float(
            np.logaddexp(global_log_z, logsumexp(np.asarray(logs, dtype=np.float64)))
        )
    if count != expected_states:
        raise ValueError(
            f"Expected exhaustive {expected_states} states, found {count}; no oracle artifact written"
        )
    selected = sorted(heap, key=lambda item: (-item[0], -item[1]))
    sequences = np.asarray([item[2] for item in selected], dtype="U14")
    clipped_scores = np.asarray([item[0] for item in selected], dtype=np.float64)
    log_epsilon = math.log(reward_epsilon) if reward_epsilon > 0 else -np.inf
    log_rewards = np.logaddexp(beta * np.log(clipped_scores) + math.log(scale), log_epsilon)
    rewards = np.exp(log_rewards)
    full_log_reward_sum = float(
        np.logaddexp(global_log_z + math.log(scale), math.log(count) + log_epsilon)
    )
    return (
        sequences,
        rewards,
        {
            "state_count": count,
            "global_log_reward_sum": full_log_reward_sum,
            "global_log_reward_sum_unscaled": global_log_z,
            "oracle_log_reward_sum": float(logsumexp(log_rewards)),
            "oracle_reward_mass": float(np.exp(logsumexp(log_rewards) - full_log_reward_sum)),
        },
    )


class OraclePreparation:
    """Preview by default; full exhaustive preparation requires explicit opt-in."""

    def __init__(self, root=None, cache=None, output=None, prepare=None):
        self.root = find_repository_root(root)

        def resolve(value):
            path = Path(value)
            return (path if path.is_absolute() else self.root / path).resolve()

        self.cache = resolve(
            os.environ.get("GF_RNA_PREDICTION_CACHE", "code/envs/rna/L14_RNA1_allpreds.pkl")
            if cache is None
            else cache
        )
        self.output = resolve(
            os.environ.get("GF_RNA_ORACLE_OUTPUT", "outputs/rna-oracle")
            if output is None
            else output
        )
        if self.root not in self.output.parents:
            raise ValueError("Oracle outputs must stay inside this repository")
        self.prepare = (
            os.environ.get("GF_PREPARE_ORACLE", "0") == "1" if prepare is None else bool(prepare)
        )
        self.top_k = 200000
        self.expected_states = 4**14
        self.stats_path = self.root / "code/envs/rna/L14_RNA1_info.csv"
        self.stats = pd.read_csv(self.stats_path).set_index("reward_exp").loc[40]
        self.scale = float(self.stats["scale"])
        self.beta = 40
        self.reward_min = 0.001
        self.reward_epsilon = 1e-12

    def describe(self):
        return {
            "cache_bytes": self.cache.stat().st_size,
            "full_state_count": self.expected_states,
            "top_k": self.top_k,
            "prepare_enabled": self.prepare,
            "output": str(self.output),
        }

    def preview(self, count=20):
        """Read only a bounded prefix without unpickling arbitrary objects."""
        with self.cache.open("rb") as stream:
            values = list(itertools.islice(iter_prediction_dictionary(DigestReader(stream)), count))
        return pd.DataFrame(values, columns=["sequence", "raw_score"])

    def run(self):
        """Return a preview table or build the exhaustive support and return metadata."""
        if not self.prepare:
            print("Schema preview only; no full cache loaded and no oracle artifact written.")
            return self.preview()
        return self.build()

    def build(self):
        """Stream the complete cache, verifying exhaustive enumeration before export."""
        CACHE, OUTPUT, STATS = self.cache, self.output, self.stats_path
        TOP_K, EXPECTED_STATES = self.top_k, self.expected_states
        SCALE, BETA, REWARD_MIN, stats = (
            self.scale,
            self.beta,
            self.reward_min,
            self.stats,
        )
        artifact = OUTPUT / "oracle_top200000.npz"
        metadata_path = artifact.with_suffix(".metadata.json")
        if artifact.exists() or metadata_path.exists():
            raise FileExistsError(
                "Oracle output already exists; select a fresh GF_RNA_ORACLE_OUTPUT"
            )
        with CACHE.open("rb") as stream:
            reader = DigestReader(stream)
            sequences, rewards, normalization = select_oracle(
                iter_prediction_dictionary(reader),
                TOP_K,
                EXPECTED_STATES,
                SCALE,
                BETA,
                REWARD_MIN,
                reward_epsilon=self.reward_epsilon,
            )
            cache_sha = reader.digest.hexdigest()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, sequences=sequences, rewards=rewards)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        metadata = {
            "schema_version": 2,
            "definition": "ground_truth_top_200000_reward_states",
            "source_cache": str(CACHE),
            "source_cache_sha256": cache_sha,
            "source_cache_bytes": CACHE.stat().st_size,
            "stats_sha256": hashlib.sha256(STATS.read_bytes()).hexdigest(),
            "artifact_sha256": digest,
            "top_k": TOP_K,
            "reward_exponent": BETA,
            "reward_min": REWARD_MIN,
            "reward_scale": SCALE,
            "reward_epsilon": self.reward_epsilon,
            "reward_formula": "scale * max(raw_score, reward_min)**reward_exponent + reward_epsilon",
            "selection_order": "descending clipped raw score; equal-score ties in UCGA enumeration order",
            "full_state_space_verified": "strictly increasing valid UCGA RNA14 keys and exactly 4^14 entries",
            "normalization": "all_4^14_states_from_same_cache",
            **normalization,
            "stored_csv_logz_unscaled": float(stats["logz"]),
            "stored_csv_logz_difference": normalization["global_log_reward_sum_unscaled"]
            - float(stats["logz"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata


# %% 5. TV data structures, environments, and policy loading

@dataclass(frozen=True)
class SeriesSpec:
    """A single environment/method checkpoint series."""

    series_id: str
    environment: str
    method: str
    kind: str
    run_dir: Path
    checkpoints: tuple[int, ...]
    preset: str | None = None
    tree_g: int | None = None
    tree_h: int | None = None

    @property
    def ground_truth_kind(self) -> str:
        if self.series_id == "rna14_stable":
            return "conditional_tv_on_stable_retained_states"
        if self.series_id == "rna14_stable_teacher_new":
            return "conditional_tv_on_stable_teacher_retained_states"
        return "global_tv"


@dataclass
class Trajectory:
    """A trajectory stored in forward order, including the terminal exit."""

    observations: list[np.ndarray]
    actions: list[int]
    next_observations: list[np.ndarray]
    reward: float
    source: str


@dataclass
class EstimatorResult:
    value: float
    unclipped: float
    optimal_c: float


@dataclass
class BoundResult:
    value: float
    unclipped: float
    empirical: float
    confidence: float
    optimal_c: float


@dataclass
class FCSResult:
    value: float
    epoch_std: float
    mean_unique_states: float


@dataclass
class HyperGridTopology:
    coords: np.ndarray
    order: np.ndarray
    layer_bounds: np.ndarray
    strides: np.ndarray


@dataclass
class RNASupport:
    support_id: str
    source_file: str
    sequences: tuple[str, ...]
    sequence_to_index: dict[str, int]
    observations: np.ndarray
    rewards: np.ndarray
    target_probabilities: np.ndarray
    substring_states: tuple[str, ...]
    substring_to_index: dict[str, int]
    prepend_sources: np.ndarray
    append_sources: np.ndarray
    character_indices: np.ndarray
    file_hashes: dict[str, str]
    canonical_hash: str
    source_size: int

    @property
    def size(self) -> int:
        return len(self.sequences)

    @property
    def length(self) -> int:
        return int(self.character_indices.shape[1])


@dataclass
class RNAEvaluation:
    support: RNASupport
    substring_log_probs: np.ndarray
    window_log_flows: np.ndarray
    terminal_log_probabilities: np.ndarray
    learned_probabilities: np.ndarray
    log_pf_support_mass: float
    log_reward_support_mass: float
    effective_log_z: float
    true_tv: float


class RNAOps:
    """Minimal RNA14 state/mask operations without constructing the FLEX oracle."""

    alphabet = ("U", "C", "G", "A")
    length = 14
    state_dim = 56
    action_dim = 9

    @classmethod
    def encode(cls, sequence: str) -> np.ndarray:
        obs = np.zeros(cls.state_dim, dtype=np.float32)
        for position, character in enumerate(sequence):
            obs[position * 4 + cls.alphabet.index(character)] = 1.0
        return obs

    @classmethod
    def decode(cls, observation: np.ndarray) -> str:
        obs = np.asarray(observation).reshape(cls.length, 4)
        chars: list[str] = []
        for row in obs:
            if float(row.sum()) <= 0.5:
                break
            chars.append(cls.alphabet[int(np.argmax(row))])
        return "".join(chars)

    @classmethod
    def get_forward_action_masks(cls, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        original_shape = obs.shape[:-1]
        flat = obs.reshape(-1, cls.state_dim)
        lengths = flat.reshape(-1, cls.length, 4).sum(axis=(1, 2))
        masks = np.zeros((len(flat), cls.action_dim), dtype=bool)
        masks[lengths < cls.length, :8] = True
        masks[lengths >= cls.length, 8] = True
        return masks.reshape(original_shape + (cls.action_dim,))

    @classmethod
    def get_backward_action_masks(cls, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        original_shape = obs.shape[:-1]
        flat = obs.reshape(-1, cls.length, 4)
        masks = np.zeros((len(flat), cls.action_dim), dtype=bool)
        for index, rows in enumerate(flat):
            occupied = np.flatnonzero(rows.sum(axis=1) > 0.5)
            if len(occupied) == 0:
                continue
            first = int(np.argmax(rows[occupied[0]]))
            last = int(np.argmax(rows[occupied[-1]]))
            masks[index, first] = True
            masks[index, last + 4] = True
        return masks.reshape(original_shape + (cls.action_dim,))

    @classmethod
    def apply_backward_action(cls, observation: np.ndarray, action: int) -> np.ndarray:
        sequence = cls.decode(observation)
        if not sequence:
            return np.asarray(observation, dtype=np.float32).copy()
        previous = sequence[1:] if action < 4 else sequence[:-1]
        return cls.encode(previous)


class MemoryEfficientRegularTreeEnv(RegularTreeEnv):
    """RegularTree API without allocating the unused 3^h target array."""

    def _compute_truth_distribution(self) -> np.ndarray:
        return np.empty(0, dtype=np.float64)


def activation_class(name: str) -> type[torch.nn.Module]:
    activations: dict[str, type[torch.nn.Module]] = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "leakyrelu": torch.nn.LeakyReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported saved activation {name!r}") from exc


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{1_000.0 * seconds:.3f} ms"
    return f"{seconds:.1f} s"




def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def checkpoint_dir(spec: SeriesSpec, step: int) -> Path:
    return spec.run_dir / f"model_{step}"


def read_hyperparameters(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "hyperparameters.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_policies(
    state_dim: int,
    action_dim: int,
    hyperparameters: Mapping[str, Any],
    device: torch.device,
) -> tuple[DiscreteForwardPolicy, DiscreteBackwardPolicy]:
    hidden_sizes = [int(size) for size in hyperparameters.get("hidden_sizes", [256, 256])]
    activation = activation_class(str(hyperparameters.get("activation", "leakyrelu")))
    forward = DiscreteForwardPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        activation_fn=activation,
        device=str(device),
    ).to(device)
    backward = DiscreteBackwardPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        activation_fn=activation,
        device=str(device),
    ).to(device)
    return forward, backward


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(
    spec: SeriesSpec,
    step: int,
    forward_policy: DiscreteForwardPolicy,
    backward_policy: DiscreteBackwardPolicy,
    device: torch.device,
) -> float:
    model_dir = checkpoint_dir(spec, step)
    forward_policy.load_state_dict(_torch_load(model_dir / "forward_policy.pth", device))
    backward_policy.load_state_dict(_torch_load(model_dir / "backward_policy.pth", device))
    forward_policy.eval()
    backward_policy.eval()
    saved_log_z = _torch_load(model_dir / "logZ.pth", device)
    if isinstance(saved_log_z, torch.Tensor):
        if saved_log_z.numel() != 1:
            raise ValueError(f"Expected scalar logZ in {model_dir}, got shape {saved_log_z.shape}")
        return float(saved_log_z.detach().cpu().item())
    if isinstance(saved_log_z, Mapping):
        for key in ("logZ", "log_z", "value"):
            if key in saved_log_z:
                value = saved_log_z[key]
                return float(value.item() if isinstance(value, torch.Tensor) else value)
    return float(saved_log_z)


def create_environment(spec: SeriesSpec) -> HyperGridEnv | RegularTreeEnv | RNAOps:
    if spec.kind == "hypergrid":
        assert spec.preset is not None
        return HyperGridEnv(preset=spec.preset)
    if spec.kind == "tree":
        assert spec.tree_g is not None and spec.tree_h is not None
        # h=15 has 14,348,907 leaves. The target is the known scalar 1/3^h,
        # so skip the base constructor's otherwise unnecessary 115 MB array.
        return MemoryEfficientRegularTreeEnv(g=spec.tree_g, h=spec.tree_h)
    if spec.kind == "rna":
        return RNAOps()
    raise ValueError(f"Unknown series kind: {spec.kind}")


def environment_dimensions(
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
) -> tuple[int, int]:
    if isinstance(environment, RNAOps):
        return environment.state_dim, environment.action_dim
    return int(environment.observation_space.shape[0]), int(environment.action_space.n)


def build_hypergrid_topology(env: HyperGridEnv) -> HyperGridTopology:
    shape = (env.height,) * env.ndim
    flat_indices = np.arange(env.height**env.ndim, dtype=np.int64)
    coords = np.stack(np.unravel_index(flat_indices, shape), axis=1).astype(np.int16)
    layer_ids = coords.sum(axis=1, dtype=np.int32)
    order = np.argsort(layer_ids, kind="stable")
    counts = np.bincount(layer_ids, minlength=env.ndim * (env.height - 1) + 1)
    layer_bounds = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    strides = np.asarray(
        [env.height ** (env.ndim - dimension - 1) for dimension in range(env.ndim)],
        dtype=np.int64,
    )
    return HyperGridTopology(coords, order, layer_bounds, strides)


# %% 6. TV and flow-consistency estimators

def direct_masked_log_probs(
    policy: torch.nn.Module,
    observations: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Log probabilities of the actual categorical sampler (without -50 log-p clamp)."""

    logits = policy.action_net(observations).clamp(min=-50.0, max=50.0)
    masked_logits = torch.where(masks, logits, torch.full_like(logits, -1e8))
    return torch.log_softmax(masked_logits, dim=-1)


def _tail_values(log_ratios: np.ndarray, c: float) -> np.ndarray:
    """The per-sample m_c values shared by M_TV and B_TV."""

    values = np.zeros_like(log_ratios, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        positive = log_ratios > c
        if np.any(positive):
            values[positive] = np.expm1(log_ratios[positive] - c) / (-np.expm1(-c))
        negative = log_ratios < -c
        if np.any(negative):
            values[negative] = -np.expm1(c + log_ratios[negative]) / np.expm1(c)
    return values


def _robust_scalar_minimum(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Minimize a continuous piecewise objective with endpoint safeguards."""

    if not (np.isfinite(lower) and np.isfinite(upper)) or upper < lower:
        return upper, float("inf")
    if upper == lower:
        return upper, float(objective(upper))

    linear_grid = np.linspace(lower, upper, 257)
    positive_thresholds = np.asarray(thresholds, dtype=np.float64)
    positive_thresholds = positive_thresholds[
        np.isfinite(positive_thresholds)
        & (positive_thresholds >= lower)
        & (positive_thresholds <= upper)
    ]
    candidates = np.unique(np.concatenate(([lower, upper], linear_grid, positive_thresholds)))
    values = np.asarray([objective(float(c)) for c in candidates], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return upper, float("inf")

    finite_indices = np.flatnonzero(finite)
    grid_index = int(finite_indices[np.argmin(values[finite])])
    best_c = float(candidates[grid_index])
    best_value = float(values[grid_index])

    left_index = max(0, grid_index - 1)
    right_index = min(len(candidates) - 1, grid_index + 1)
    local_lower = float(candidates[left_index])
    local_upper = float(candidates[right_index])
    if local_upper > local_lower:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize_scalar(
                objective,
                bounds=(local_lower, local_upper),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 500},
            )
        if result.success and np.isfinite(result.fun) and float(result.fun) < best_value:
            best_c = float(result.x)
            best_value = float(result.fun)
    return best_c, best_value


def compute_m_tv(log_z_pf: np.ndarray, log_r_pb: np.ndarray) -> EstimatorResult:
    """Compute the Proposition-style empirical M_TV estimator."""

    ratios = np.asarray(log_z_pf, dtype=np.float64) - np.asarray(log_r_pb, dtype=np.float64)
    if len(ratios) == 0 or not np.all(np.isfinite(ratios)):
        return EstimatorResult(1.0, float("inf"), 0.0)
    upper = float(np.max(np.abs(ratios)))
    if upper <= 1e-14:
        return EstimatorResult(0.0, 0.0, 0.0)

    lower = min(1e-9, upper * 1e-6)

    def objective(c: float) -> float:
        if c <= 0:
            return float("inf")
        values = _tail_values(ratios, c)
        mean_value = float(np.mean(values))
        estimate = -np.expm1(-2.0 * c) * (1.0 + mean_value)
        if not np.isfinite(estimate) or estimate < 0:
            return float("inf")
        return float(estimate)

    optimal_c, raw = _robust_scalar_minimum(objective, lower, upper, np.abs(ratios))
    if not np.isfinite(raw):
        return EstimatorResult(1.0, raw, optimal_c)
    return EstimatorResult(float(np.clip(raw, 0.0, 1.0)), raw, optimal_c)


def corrected_confidence_term(alpha: float, num_forward: int, num_backward: int) -> float:
    """Manuscript Theorem 3.11: 2*log(2/alpha)/n_f + 2*log(2/alpha)/n_b."""

    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between zero and one")
    if num_forward <= 0 or num_backward <= 0:
        return float("inf")
    return 2.0 * math.log(2.0 / alpha) / num_forward + 2.0 * math.log(2.0 / alpha) / num_backward


def compute_b_tv(
    log_z_pf: np.ndarray,
    log_r_pb: np.ndarray,
    *,
    alpha: float,
    num_forward: int,
    num_backward: int,
) -> BoundResult:
    """Compute B_TV with direct constraint checking and the corrected confidence term."""

    ratios = np.asarray(log_z_pf, dtype=np.float64) - np.asarray(log_r_pb, dtype=np.float64)
    confidence = corrected_confidence_term(alpha, num_forward, num_backward)
    if len(ratios) == 0 or not np.all(np.isfinite(ratios)):
        return BoundResult(1.0, float("inf"), float("inf"), confidence, 0.0)
    upper = float(np.max(np.abs(ratios)))
    if upper <= 1e-14:
        raw = confidence
        return BoundResult(float(np.clip(raw, 0.0, 1.0)), raw, 0.0, confidence, 0.0)

    lower = min(1e-9, upper * 1e-6)

    def objective(c: float) -> float:
        if c <= 0:
            return float("inf")
        values = _tail_values(ratios, c)
        maximum = float(np.max(values))
        if not np.isfinite(maximum) or maximum < 0:
            return float("inf")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            k_value = np.expm1(c) * maximum
            exp_c = np.exp(c)
            if not np.isfinite(k_value) or k_value >= 1.0:
                return float("inf")
            bound = exp_c * (exp_c + k_value) / (1.0 - k_value) - 1.0
        if not np.isfinite(bound) or bound < 0:
            return float("inf")
        return float(bound)

    optimal_c, empirical = _robust_scalar_minimum(objective, lower, upper, np.abs(ratios))
    if not np.isfinite(empirical):
        return BoundResult(1.0, float("inf"), empirical, confidence, optimal_c)
    raw = empirical + confidence
    return BoundResult(float(np.clip(raw, 0.0, 1.0)), raw, empirical, confidence, optimal_c)


def compute_fcs_epoch(
    forward_trajectories: Sequence[Trajectory],
    terminal_log_weights: np.ndarray,
) -> tuple[float, int]:
    """Compute one FCS epoch as implemented in exp2_rebuttal2.ipynb.

    Forward samples determine the random set S. Terminal model weights and
    rewards are restricted and normalized on the unique states in S. Duplicate
    occurrences are averaged in probability space before normalization.
    """

    values = np.asarray(terminal_log_weights, dtype=np.float64)
    if len(forward_trajectories) == 0 or len(forward_trajectories) != len(values):
        raise ValueError("FCS requires one finite log p_T(x) estimate per forward trajectory")
    if not np.all(np.isfinite(values)):
        raise ValueError("FCS received a non-finite model terminal log-weight")

    state_model_log_weights: dict[bytes, list[float]] = {}
    state_rewards: dict[bytes, float] = {}
    for index, trajectory in enumerate(forward_trajectories):
        if trajectory.source != "forward":
            raise ValueError("FCS buckets may contain only forward trajectories")
        terminal = np.ascontiguousarray(trajectory.observations[-1], dtype=np.float32)
        key = terminal.tobytes()
        reward = float(trajectory.reward)
        if not np.isfinite(reward) or reward <= 0:
            raise ValueError(f"FCS received invalid terminal reward {reward}")
        if key in state_rewards and not math.isclose(
            state_rewards[key], reward, rel_tol=1e-12, abs_tol=1e-14
        ):
            raise ValueError("Duplicate FCS terminal state has inconsistent rewards")
        model_log_weight = float(values[index])
        state_rewards[key] = reward
        state_model_log_weights.setdefault(key, []).append(model_log_weight)

    keys = list(state_model_log_weights)
    log_marginal_weights = np.asarray(
        [
            float(
                logsumexp(state_model_log_weights[key])
                - math.log(len(state_model_log_weights[key]))
            )
            for key in keys
        ],
        dtype=np.float64,
    )
    log_target_weights = np.log(np.asarray([state_rewards[key] for key in keys], dtype=np.float64))
    learned = np.exp(log_marginal_weights - logsumexp(log_marginal_weights))
    target = np.exp(log_target_weights - logsumexp(log_target_weights))
    fcs = 0.5 * float(np.abs(learned - target).sum())
    return float(np.clip(fcs, 0.0, 1.0)), len(keys)


def compute_fcs(
    forward_trajectories: Sequence[Trajectory],
    terminal_log_weights: np.ndarray,
    *,
    epochs: int,
    bucket_size: int,
) -> FCSResult:
    """Average FCS over consecutive forward-sample buckets."""

    expected = epochs * bucket_size
    if len(forward_trajectories) != expected or len(terminal_log_weights) != expected:
        raise ValueError(
            f"FCS expected {epochs} x {bucket_size} = {expected} trajectories, "
            f"got {len(forward_trajectories)} trajectories and "
            f"{len(terminal_log_weights)} values"
        )
    epoch_values: list[float] = []
    unique_counts: list[int] = []
    for epoch in range(epochs):
        start = epoch * bucket_size
        end = start + bucket_size
        value, unique = compute_fcs_epoch(
            forward_trajectories[start:end],
            np.asarray(terminal_log_weights)[start:end],
        )
        epoch_values.append(value)
        unique_counts.append(unique)
    return FCSResult(
        value=float(np.mean(epoch_values)),
        epoch_std=float(np.std(epoch_values)),
        mean_unique_states=float(np.mean(unique_counts)),
    )


# %% 7. Exact terminal probabilities and RNA reference supports

def one_hot_hypergrid_observations(
    coords: np.ndarray,
    ndim: int,
    height: int,
) -> np.ndarray:
    observations = np.zeros((len(coords), ndim * height), dtype=np.float32)
    rows = np.arange(len(coords))
    for dimension in range(ndim):
        observations[rows, dimension * height + coords[:, dimension]] = 1.0
    return observations


def exact_hypergrid_tv(
    env: HyperGridEnv,
    policy: DiscreteForwardPolicy,
    topology: HyperGridTopology,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, np.ndarray]:
    """Exact terminal distribution by forward mass propagation on the DAG."""
    number_states = len(topology.coords)
    incoming = np.zeros(number_states, dtype=np.float64)
    terminal = np.zeros(number_states, dtype=np.float64)
    incoming[0] = 1.0

    with torch.inference_mode():
        for layer in range(len(topology.layer_bounds) - 1):
            start = int(topology.layer_bounds[layer])
            end = int(topology.layer_bounds[layer + 1])
            layer_indices = topology.order[start:end]
            for chunk_start in range(0, len(layer_indices), batch_size):
                indices = layer_indices[chunk_start : chunk_start + batch_size]
                coordinates = topology.coords[indices]
                observations = one_hot_hypergrid_observations(coordinates, env.ndim, env.height)
                masks = np.concatenate(
                    (
                        coordinates < env.height - 1,
                        np.ones((len(indices), 1), dtype=bool),
                    ),
                    axis=1,
                )
                obs_tensor = torch.as_tensor(observations, device=device)
                mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
                probabilities = (
                    direct_masked_log_probs(policy, obs_tensor, mask_tensor)
                    .exp()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                masses = incoming[indices]
                terminal[indices] = masses * probabilities[:, env.ndim]
                for dimension in range(env.ndim):
                    valid = coordinates[:, dimension] < env.height - 1
                    if np.any(valid):
                        child_indices = indices[valid] + topology.strides[dimension]
                        np.add.at(
                            incoming,
                            child_indices,
                            masses[valid] * probabilities[valid, dimension],
                        )

    learned_mass = float(terminal.sum())
    if not np.isclose(learned_mass, 1.0, atol=2e-7, rtol=2e-7):
        raise RuntimeError(f"HyperGrid forward mass sums to {learned_mass:.12g}, not one")
    target = np.asarray(env.truth_dist, dtype=np.float64).reshape(-1)
    tv = 0.5 * float(np.abs(terminal - target).sum())
    return tv, learned_mass, terminal


def regular_tree_observations(
    node_indices: np.ndarray,
    depth: int,
    g: int,
    h: int,
) -> np.ndarray:
    observations = np.zeros((len(node_indices), h * (g + 1)), dtype=np.float32)
    rows = np.arange(len(node_indices))
    unvisited_columns = np.arange(h) * (g + 1) + g
    observations[:, unvisited_columns] = 1.0
    remaining = node_indices.copy()
    for level in range(depth - 1, -1, -1):
        child = remaining % g
        remaining //= g
        observations[rows, level * (g + 1) + g] = 0.0
        observations[rows, level * (g + 1) + child] = 1.0
    return observations


def exact_regular_tree_tv(
    env: RegularTreeEnv,
    policy: DiscreteForwardPolicy,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Exact TV using a streaming frontier, including the 3^15 tree."""
    masses = np.ones(1, dtype=np.float64)
    target_leaf_mass = 1.0 / float(env.g**env.h)
    tv_l1 = 0.0
    learned_mass = 0.0

    with torch.inference_mode():
        for depth in range(env.h):
            number_nodes = env.g**depth
            if len(masses) != number_nodes:
                raise RuntimeError("RegularTree frontier size is inconsistent")
            next_masses = (
                None if depth == env.h - 1 else np.empty(number_nodes * env.g, dtype=np.float64)
            )
            for start in range(0, number_nodes, batch_size):
                end = min(number_nodes, start + batch_size)
                indices = np.arange(start, end, dtype=np.int64)
                observations = regular_tree_observations(indices, depth, env.g, env.h)
                masks = np.zeros((len(indices), env.g + 1), dtype=bool)
                masks[:, : env.g] = True
                obs_tensor = torch.as_tensor(observations, device=device)
                mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
                child_probabilities = (
                    direct_masked_log_probs(policy, obs_tensor, mask_tensor)[:, : env.g]
                    .exp()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                child_probabilities /= child_probabilities.sum(axis=1, keepdims=True)
                children = masses[start:end, None] * child_probabilities
                if next_masses is None:
                    tv_l1 += float(np.abs(children - target_leaf_mass).sum())
                    learned_mass += float(children.sum())
                else:
                    next_masses[start * env.g : end * env.g] = children.reshape(-1)
            if next_masses is not None:
                masses = next_masses

    if not np.isclose(learned_mass, 1.0, atol=2e-7, rtol=2e-7):
        raise RuntimeError(f"RegularTree leaf mass sums to {learned_mass:.12g}, not one")
    return 0.5 * tv_l1, learned_mass


def decode_terminal_state(value: Any) -> tuple[str, np.ndarray, float]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError("terminal_states.pkl values must contain (observation, reward)")
    observation = np.asarray(value[0], dtype=np.float32).reshape(-1)
    if observation.shape != (RNAOps.state_dim,):
        raise ValueError(
            f"Expected RNA observation shape {(RNAOps.state_dim,)}, got {observation.shape}"
        )
    blocks = observation.reshape(RNAOps.length, 4)
    if not np.all((blocks == 0.0) | (blocks == 1.0)) or not np.all(blocks.sum(axis=1) == 1.0):
        raise ValueError("RNA terminal observations must be binary one-hot at all 14 positions")
    reward = float(value[1])
    sequence = RNAOps.decode(observation)
    if len(sequence) != RNAOps.length:
        raise ValueError(f"Expected a terminal RNA14 observation, decoded {sequence!r}")
    if not np.isfinite(reward) or reward <= 0:
        raise ValueError(f"RNA support contains non-positive reward {reward}")
    if not np.array_equal(RNAOps.encode(sequence), observation):
        raise ValueError(f"RNA terminal observation is not canonical for {sequence}")
    return sequence, observation, reward


def load_terminal_state_file(path: Path) -> dict[str, tuple[np.ndarray, float]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a dictionary in {path}")
    decoded: dict[str, tuple[np.ndarray, float]] = {}
    for value in raw.values():
        sequence, observation, reward = decode_terminal_state(value)
        if sequence in decoded and not math.isclose(
            decoded[sequence][1], reward, rel_tol=1e-12, abs_tol=1e-14
        ):
            raise ValueError(f"Inconsistent duplicate reward for RNA sequence {sequence}")
        decoded[sequence] = (observation, reward)
    return decoded


def canonical_rna_support_hash(
    sequences: Sequence[str],
    rewards: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for sequence, reward in zip(sequences, rewards):
        digest.update(sequence.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.float64(reward).tobytes())
    return digest.hexdigest()


def build_rna_support(
    run_dir: Path,
    *,
    support_id: str,
    expected_file_hash: str,
    verify_known_hashes: bool = True,
) -> RNASupport:
    support_path = run_dir / "terminal_states.pkl"
    if not support_path.is_file():
        raise FileNotFoundError(
            f"RNA14 support {support_id!r} requires terminal_states.pkl in " f"{run_dir}"
        )
    file_hashes = {
        "terminal_states.pkl": sha256_file(support_path),
    }
    if verify_known_hashes:
        observed = file_hashes["terminal_states.pkl"]
        if observed.lower() != expected_file_hash.lower():
            raise ValueError(
                f"Unexpected {support_id} terminal-state file hash: {observed}; "
                f"expected {expected_file_hash}. Use "
                "--allow-rna-support-change only after intentionally reviewing "
                "the changed support."
            )

    retained_states = load_terminal_state_file(support_path)
    if len(retained_states) != 10_000:
        raise ValueError(
            f"RNA14 support {support_id!r} must contain exactly 10,000 unique "
            f"terminal states, but {support_path} contains {len(retained_states)}."
        )

    sequences = tuple(sorted(retained_states))
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
    observations = np.stack([retained_states[sequence][0] for sequence in sequences]).astype(
        np.float32, copy=False
    )
    rewards = np.asarray([retained_states[sequence][1] for sequence in sequences], dtype=np.float64)
    target_probabilities = rewards / rewards.sum()
    alphabet_index = {character: index for index, character in enumerate(RNAOps.alphabet)}
    character_indices = np.asarray(
        [[alphabet_index[character] for character in sequence] for sequence in sequences],
        dtype=np.int8,
    )

    substring_set = {""}
    for sequence in sequences:
        for start in range(RNAOps.length):
            for end in range(start + 1, RNAOps.length + 1):
                if start != 0 or end != RNAOps.length:
                    substring_set.add(sequence[start:end])
    substring_states = tuple(sorted(substring_set, key=lambda state: (len(state), state)))
    substring_to_index = {state: index for index, state in enumerate(substring_states)}

    number_sequences = len(sequences)
    prepend_sources = np.full((number_sequences, RNAOps.length, RNAOps.length), -1, dtype=np.int32)
    append_sources = np.full_like(prepend_sources, -1)
    for sequence_index, sequence in enumerate(sequences):
        for span in range(2, RNAOps.length + 1):
            for left in range(0, RNAOps.length - span + 1):
                right = left + span - 1
                prepend_sources[sequence_index, left, right] = substring_to_index[
                    sequence[left + 1 : right + 1]
                ]
                append_sources[sequence_index, left, right] = substring_to_index[
                    sequence[left:right]
                ]

    return RNASupport(
        support_id=support_id,
        source_file=str(support_path.resolve()),
        sequences=sequences,
        sequence_to_index=sequence_to_index,
        observations=observations,
        rewards=rewards,
        target_probabilities=target_probabilities,
        substring_states=substring_states,
        substring_to_index=substring_to_index,
        prepend_sources=prepend_sources,
        append_sources=append_sources,
        character_indices=character_indices,
        file_hashes=file_hashes,
        canonical_hash=canonical_rna_support_hash(sequences, rewards),
        source_size=len(retained_states),
    )


def rna_substring_observations(states: Sequence[str]) -> np.ndarray:
    observations = np.zeros((len(states), RNAOps.state_dim), dtype=np.float32)
    alphabet_index = {character: index for index, character in enumerate(RNAOps.alphabet)}
    for row, state in enumerate(states):
        for position, character in enumerate(state):
            observations[row, position * 4 + alphabet_index[character]] = 1.0
    return observations


def evaluate_rna_support(
    support: RNASupport,
    policy: DiscreteForwardPolicy,
    device: torch.device,
    batch_size: int,
) -> RNAEvaluation:
    """Evaluate all retained RNA terminals via a deduplicated window DP."""
    number_substrings = len(support.substring_states)
    substring_log_probs = np.empty((number_substrings, RNAOps.action_dim), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, number_substrings, batch_size):
            end = min(number_substrings, start + batch_size)
            observations = rna_substring_observations(support.substring_states[start:end])
            masks = np.zeros((end - start, RNAOps.action_dim), dtype=bool)
            masks[:, :8] = True
            obs_tensor = torch.as_tensor(observations, device=device)
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
            substring_log_probs[start:end] = (
                direct_masked_log_probs(policy, obs_tensor, mask_tensor)
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            substring_log_probs[start:end, :8] -= logsumexp(
                substring_log_probs[start:end, :8], axis=1, keepdims=True
            )

    number_sequences = support.size
    length = support.length
    flow = np.full((number_sequences, length, length), -np.inf, dtype=np.float64)
    empty_index = support.substring_to_index[""]
    for position in range(length):
        character = support.character_indices[:, position].astype(np.int64)
        flow[:, position, position] = np.logaddexp(
            substring_log_probs[empty_index, character],
            substring_log_probs[empty_index, character + 4],
        )

    for span in range(2, length + 1):
        for left in range(0, length - span + 1):
            right = left + span - 1
            first_character = support.character_indices[:, left].astype(np.int64)
            last_character = support.character_indices[:, right].astype(np.int64)
            prepend = (
                flow[:, left + 1, right]
                + substring_log_probs[support.prepend_sources[:, left, right], first_character]
            )
            append = (
                flow[:, left, right - 1]
                + substring_log_probs[support.append_sources[:, left, right], last_character + 4]
            )
            flow[:, left, right] = np.logaddexp(prepend, append)

    terminal_log_probabilities = flow[:, 0, length - 1]
    log_pf_support_mass = float(logsumexp(terminal_log_probabilities))
    learned_probabilities = np.exp(terminal_log_probabilities - log_pf_support_mass)
    learned_sum = float(learned_probabilities.sum())
    if not np.isclose(learned_sum, 1.0, atol=2e-10, rtol=2e-10):
        raise RuntimeError(f"Conditional RNA learned probabilities sum to {learned_sum}")
    log_reward_support_mass = float(np.log(support.rewards.sum()))
    effective_log_z = log_reward_support_mass - log_pf_support_mass
    true_tv = 0.5 * float(np.abs(learned_probabilities - support.target_probabilities).sum())
    return RNAEvaluation(
        support=support,
        substring_log_probs=substring_log_probs,
        window_log_flows=flow,
        terminal_log_probabilities=terminal_log_probabilities,
        learned_probabilities=learned_probabilities,
        log_pf_support_mass=log_pf_support_mass,
        log_reward_support_mass=log_reward_support_mass,
        effective_log_z=effective_log_z,
        true_tv=true_tv,
    )


def exact_regular_tree_terminal_log_probabilities(
    environment: RegularTreeEnv,
    forward_policy: DiscreteForwardPolicy,
    terminal_observations: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate exact p_T(x) for selected tree leaves along their unique paths."""

    terminal_observations = np.asarray(terminal_observations, dtype=np.float32)
    paths = np.asarray(environment.get_state(terminal_observations), dtype=np.int64)
    if paths.ndim != 2 or paths.shape[1] != environment.h or np.any(paths < 0):
        raise ValueError("FCS-Exact RegularTree observations must be complete leaves")
    rows = np.arange(len(paths))
    node_indices = np.zeros(len(paths), dtype=np.int64)
    terminal_log_probabilities = np.zeros(len(paths), dtype=np.float64)
    with torch.inference_mode():
        for depth in range(environment.h):
            observations = regular_tree_observations(
                node_indices, depth, environment.g, environment.h
            )
            masks = np.zeros((len(paths), environment.g + 1), dtype=bool)
            masks[:, : environment.g] = True
            log_probabilities = (
                direct_masked_log_probs(
                    forward_policy,
                    torch.as_tensor(observations, device=device),
                    torch.as_tensor(masks, dtype=torch.bool, device=device),
                )
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            actions = paths[:, depth]
            terminal_log_probabilities += log_probabilities[rows, actions]
            node_indices = node_indices * environment.g + actions
    return terminal_log_probabilities


def exact_fcs_terminal_log_weights(
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
    forward_trajectories: Sequence[Trajectory],
    forward_policy: DiscreteForwardPolicy,
    device: torch.device,
    *,
    hypergrid_terminal_probabilities: np.ndarray | None,
    rna_evaluation: RNAEvaluation | None,
) -> np.ndarray:
    """Return exact model terminal log probabilities aligned to forward samples."""

    if not forward_trajectories or not all(
        trajectory.source == "forward" for trajectory in forward_trajectories
    ):
        raise ValueError("FCS-Exact requires forward samples to define S")
    terminal_observations = np.asarray(
        [trajectory.observations[-1] for trajectory in forward_trajectories],
        dtype=np.float32,
    )
    if isinstance(environment, HyperGridEnv):
        if hypergrid_terminal_probabilities is None:
            raise RuntimeError("HyperGrid FCS-Exact requires the exact DP distribution")
        coordinates = np.asarray(environment.get_state(terminal_observations), dtype=np.int64)
        flat_indices = np.ravel_multi_index(coordinates.T, (environment.height,) * environment.ndim)
        probabilities = np.asarray(hypergrid_terminal_probabilities[flat_indices], dtype=np.float64)
        if np.any(probabilities <= 0) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("HyperGrid exact terminal probabilities must be positive")
        return np.log(probabilities)
    if isinstance(environment, RegularTreeEnv):
        return exact_regular_tree_terminal_log_probabilities(
            environment, forward_policy, terminal_observations, device
        )
    if rna_evaluation is None:
        raise RuntimeError("RNA FCS-Exact requires the exact support DP")
    indices = np.asarray(
        [
            rna_evaluation.support.sequence_to_index[RNAOps.decode(observation)]
            for observation in terminal_observations
        ],
        dtype=np.int64,
    )
    values = np.asarray(rna_evaluation.terminal_log_probabilities[indices], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("RNA exact terminal log probabilities must be finite")
    return values


def compute_exact_checkpoint(
    spec: SeriesSpec,
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
    forward_policy: DiscreteForwardPolicy,
    device: torch.device,
    exact_batch_size: int,
    hypergrid_topology: HyperGridTopology | None,
    rna_support: RNASupport | None,
) -> tuple[float, float, RNAEvaluation | None, np.ndarray | None, dict[str, float]]:
    started = time.perf_counter()
    diagnostics: dict[str, float] = {}
    if spec.kind == "hypergrid":
        assert isinstance(environment, HyperGridEnv) and hypergrid_topology is not None
        true_tv, learned_mass, hypergrid_terminal_probabilities = exact_hypergrid_tv(
            environment,
            forward_policy,
            hypergrid_topology,
            device,
            exact_batch_size,
        )
        effective_log_z = float("nan")
        rna_evaluation = None
        diagnostics["learned_terminal_mass"] = learned_mass
        diagnostics["exact_policy_states_evaluated"] = float(len(hypergrid_topology.coords))
        diagnostics["exact_terminal_states"] = float(environment.height**environment.ndim)
    elif spec.kind == "tree":
        assert isinstance(environment, RegularTreeEnv)
        true_tv, learned_mass = exact_regular_tree_tv(
            environment, forward_policy, device, exact_batch_size
        )
        effective_log_z = float("nan")
        rna_evaluation = None
        hypergrid_terminal_probabilities = None
        diagnostics["learned_terminal_mass"] = learned_mass
        diagnostics["exact_policy_states_evaluated"] = float(
            (environment.g**environment.h - 1) // (environment.g - 1)
        )
        diagnostics["exact_terminal_states"] = float(environment.g**environment.h)
    elif spec.kind == "rna":
        if rna_support is None:
            raise RuntimeError("RNA support was not initialized")
        rna_evaluation = evaluate_rna_support(rna_support, forward_policy, device, exact_batch_size)
        true_tv = rna_evaluation.true_tv
        effective_log_z = rna_evaluation.effective_log_z
        hypergrid_terminal_probabilities = None
        diagnostics.update(
            {
                "learned_terminal_mass": 1.0,
                "log_pf_support_mass": rna_evaluation.log_pf_support_mass,
                "log_reward_support_mass": rna_evaluation.log_reward_support_mass,
                "exact_policy_states_evaluated": float(len(rna_support.substring_states)),
                "exact_terminal_states": float(rna_support.size),
            }
        )
    else:
        raise ValueError(spec.kind)
    diagnostics["exact_seconds"] = time.perf_counter() - started
    if not (0.0 <= true_tv <= 1.0 + 1e-8):
        raise RuntimeError(f"Exact TV is outside [0,1]: {true_tv}")
    return (
        float(np.clip(true_tv, 0.0, 1.0)),
        effective_log_z,
        rna_evaluation,
        hypergrid_terminal_probabilities,
        diagnostics,
    )


# %% 8. Forward and backward trajectory sampling

def sample_policy_action(
    policy: torch.nn.Module,
    observation: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> int:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.inference_mode():
        action = policy.forward(obs_tensor, action_masks=mask_tensor)
    return int(torch.as_tensor(action).reshape(-1)[0].detach().cpu().item())


def collect_forward_trajectories(
    env: HyperGridEnv | RegularTreeEnv,
    forward_policy: DiscreteForwardPolicy,
    number: int,
    device: torch.device,
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for _ in range(number):
        observation, _ = env.reset()
        observations: list[np.ndarray] = []
        actions: list[int] = []
        next_observations: list[np.ndarray] = []
        reward = 0.0
        terminated = False
        while not terminated:
            mask = np.asarray(env.get_forward_action_masks(observation), dtype=bool)
            action = sample_policy_action(forward_policy, observation, mask, device)
            next_observation, reward, done, truncated, _ = env.step(action)
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            actions.append(action)
            next_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
            observation = next_observation
            terminated = bool(done or truncated)
        if reward <= 0:
            raise RuntimeError("A forward trajectory terminated with non-positive reward")
        trajectories.append(
            Trajectory(observations, actions, next_observations, float(reward), "forward")
        )
    return trajectories


def backward_pairs_to_trajectory(
    env: HyperGridEnv | RegularTreeEnv | RNAOps,
    terminal_observation: np.ndarray,
    reward: float,
    pairs: list[tuple[np.ndarray, int]],
    action_dim: int,
) -> Trajectory:
    if not pairs:
        # HyperGrid can terminate at its origin. That valid reference trajectory
        # has no construction transition and consists only of the exit action.
        terminal = np.asarray(terminal_observation, dtype=np.float32).copy()
        return Trajectory(
            [terminal],
            [action_dim - 1],
            [terminal.copy()],
            float(reward),
            "reference",
        )
    reversed_pairs = list(reversed(pairs))
    first_next, first_action = reversed_pairs[0]
    previous = env.apply_backward_action(first_next, first_action)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    for next_observation, action in reversed_pairs:
        observations.append(np.asarray(previous, dtype=np.float32).copy())
        actions.append(int(action))
        next_observations.append(np.asarray(next_observation, dtype=np.float32).copy())
        previous = next_observation
    observations.append(np.asarray(terminal_observation, dtype=np.float32).copy())
    actions.append(action_dim - 1)
    next_observations.append(np.asarray(terminal_observation, dtype=np.float32).copy())
    return Trajectory(observations, actions, next_observations, float(reward), "reference")


def sample_backward_trajectory_conditioned_on_terminal(
    env: HyperGridEnv | RegularTreeEnv | RNAOps,
    backward_policy: DiscreteBackwardPolicy,
    terminal_observation: np.ndarray,
    reward: float,
    device: torch.device,
) -> Trajectory:
    """Sample one trajectory from p_B(.|x) for a fixed terminal x."""

    current = np.asarray(terminal_observation, dtype=np.float32).copy()
    pairs: list[tuple[np.ndarray, int]] = []
    while True:
        mask = np.asarray(env.get_backward_action_masks(current), dtype=bool)
        if not np.any(mask):
            break
        action = sample_policy_action(backward_policy, current, mask, device)
        pairs.append((current.copy(), action))
        current = env.apply_backward_action(current, action)
    _, action_dim = environment_dimensions(env)
    return backward_pairs_to_trajectory(
        env,
        np.asarray(terminal_observation),
        float(reward),
        pairs,
        action_dim,
    )


def collect_reference_trajectories(
    env: HyperGridEnv | RegularTreeEnv,
    backward_policy: DiscreteBackwardPolicy,
    number: int,
    device: torch.device,
) -> list[Trajectory]:
    terminal_observations, rewards = env.sample_terminal_states(number)
    return [
        sample_backward_trajectory_conditioned_on_terminal(
            env,
            backward_policy,
            np.asarray(terminal_observation),
            float(reward),
            device,
        )
        for terminal_observation, reward in zip(terminal_observations, rewards)
    ]


def sample_rna_forward_trajectory(
    evaluation: RNAEvaluation,
    sequence_index: int,
    rng: np.random.Generator,
) -> Trajectory:
    support = evaluation.support
    sequence = support.sequences[sequence_index]
    left = 0
    right = support.length - 1
    reverse_steps: list[tuple[str, int, str]] = []

    while left < right:
        prepend_source = int(support.prepend_sources[sequence_index, left, right])
        first_character = int(support.character_indices[sequence_index, left])
        last_character = int(support.character_indices[sequence_index, right])
        prepend_log_flow = (
            evaluation.window_log_flows[sequence_index, left + 1, right]
            + evaluation.substring_log_probs[prepend_source, first_character]
        )
        total_log_flow = evaluation.window_log_flows[sequence_index, left, right]
        prepend_probability = float(np.exp(prepend_log_flow - total_log_flow))
        current = sequence[left : right + 1]
        if rng.random() < np.clip(prepend_probability, 0.0, 1.0):
            previous = sequence[left + 1 : right + 1]
            reverse_steps.append((previous, first_character, current))
            left += 1
        else:
            previous = sequence[left:right]
            reverse_steps.append((previous, last_character + 4, current))
            right -= 1

    character = int(support.character_indices[sequence_index, left])
    empty_index = support.substring_to_index[""]
    prepend_log = evaluation.substring_log_probs[empty_index, character]
    append_log = evaluation.substring_log_probs[empty_index, character + 4]
    prepend_probability = float(np.exp(prepend_log - np.logaddexp(prepend_log, append_log)))
    initial_action = character if rng.random() < prepend_probability else character + 4
    singleton = sequence[left : left + 1]
    reverse_steps.append(("", initial_action, singleton))

    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    for previous, action, current in reversed(reverse_steps):
        observations.append(RNAOps.encode(previous))
        actions.append(action)
        next_observations.append(RNAOps.encode(current))
    terminal = support.observations[sequence_index]
    observations.append(terminal.copy())
    actions.append(8)
    next_observations.append(terminal.copy())
    return Trajectory(
        observations,
        actions,
        next_observations,
        float(support.rewards[sequence_index]),
        "forward",
    )


def collect_rna_conditional_trajectories(
    evaluation: RNAEvaluation,
    backward_policy: DiscreteBackwardPolicy,
    num_forward: int,
    num_backward: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[Trajectory]:
    support = evaluation.support
    forward_indices = rng.choice(
        support.size,
        size=num_forward,
        replace=True,
        p=evaluation.learned_probabilities,
    )
    trajectories = [
        sample_rna_forward_trajectory(evaluation, int(index), rng) for index in forward_indices
    ]

    reference_indices = rng.choice(
        support.size,
        size=num_backward,
        replace=True,
        p=support.target_probabilities,
    )
    for index_value in reference_indices:
        index = int(index_value)
        terminal = support.observations[index]
        trajectories.append(
            sample_backward_trajectory_conditioned_on_terminal(
                RNAOps(),
                backward_policy,
                terminal,
                float(support.rewards[index]),
                device,
            )
        )
    return trajectories


def evaluate_trajectory_log_ratios(
    trajectories: Sequence[Trajectory],
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
    forward_policy: DiscreteForwardPolicy,
    backward_policy: DiscreteBackwardPolicy,
    log_z: float,
    device: torch.device,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("No trajectories were supplied")
    state_dim, action_dim = environment_dimensions(environment)
    batch_size = len(trajectories)
    maximum_length = max(len(trajectory.actions) for trajectory in trajectories)
    observations = np.zeros((batch_size, maximum_length, state_dim), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    actions = np.zeros((batch_size, maximum_length), dtype=np.int64)
    lengths = np.zeros(batch_size, dtype=np.int64)
    rewards = np.zeros(batch_size, dtype=np.float64)
    sources: list[str] = []
    for row, trajectory in enumerate(trajectories):
        length = len(trajectory.actions)
        if not (length == len(trajectory.observations) == len(trajectory.next_observations)):
            raise ValueError("Malformed trajectory with inconsistent step counts")
        observations[row, :length] = np.asarray(trajectory.observations)
        next_observations[row, :length] = np.asarray(trajectory.next_observations)
        actions[row, :length] = np.asarray(trajectory.actions)
        lengths[row] = length
        rewards[row] = trajectory.reward
        sources.append(trajectory.source)
    if np.any(actions < 0) or np.any(actions >= action_dim):
        raise ValueError("Trajectory contains an invalid action index")
    if np.any(rewards <= 0):
        raise ValueError("Trajectory rewards must be positive")

    forward_masks = np.asarray(environment.get_forward_action_masks(observations), dtype=bool)
    backward_masks = np.asarray(
        environment.get_backward_action_masks(next_observations), dtype=bool
    )
    obs_tensor = torch.as_tensor(observations, device=device)
    next_obs_tensor = torch.as_tensor(next_observations, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    length_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    forward_mask_tensor = torch.as_tensor(forward_masks, dtype=torch.bool, device=device)
    backward_mask_tensor = torch.as_tensor(backward_masks, dtype=torch.bool, device=device)
    positions = torch.arange(maximum_length, device=device).unsqueeze(0)
    forward_valid = positions < length_tensor.unsqueeze(1)
    backward_valid = positions < (length_tensor - 1).unsqueeze(1)
    selected_forward_valid = forward_mask_tensor.gather(-1, action_tensor.unsqueeze(-1)).squeeze(-1)
    selected_backward_valid = backward_mask_tensor.gather(-1, action_tensor.unsqueeze(-1)).squeeze(
        -1
    )
    if torch.any(forward_valid & ~selected_forward_valid):
        raise RuntimeError("A recorded forward action is invalid under its state mask")
    if torch.any(backward_valid & ~selected_backward_valid):
        raise RuntimeError("A recorded backward action is invalid under its next-state mask")

    with torch.inference_mode():
        forward_step_log_probs = forward_policy.evaluate_actions(
            obs_tensor,
            action_tensor,
            action_masks=forward_mask_tensor,
            lengths=length_tensor,
        )
        backward_step_log_probs = backward_policy.evaluate_actions(
            next_obs_tensor,
            action_tensor,
            action_masks=backward_mask_tensor,
            lengths=length_tensor,
        )
        raw_forward = (
            direct_masked_log_probs(forward_policy, obs_tensor, forward_mask_tensor)
            .gather(-1, action_tensor.unsqueeze(-1))
            .squeeze(-1)
        )
        raw_backward = (
            direct_masked_log_probs(backward_policy, next_obs_tensor, backward_mask_tensor)
            .gather(-1, action_tensor.unsqueeze(-1))
            .squeeze(-1)
        )

    # The theorem uses the actual categorical densities. evaluate_actions adds
    # a training-stability floor at log p=-50, so retain it only as a diagnostic.
    forward_sums = (raw_forward * forward_valid).sum(dim=1).cpu().numpy().astype(np.float64)
    backward_sums = (raw_backward * backward_valid).sum(dim=1).cpu().numpy().astype(np.float64)
    log_z_pf = float(log_z) + forward_sums
    log_r_pb = np.log(rewards) + backward_sums
    compatibility_log_z_pf = float(log_z) + forward_step_log_probs.sum(dim=1).cpu().numpy().astype(
        np.float64
    )
    compatibility_log_r_pb = np.log(rewards) + backward_step_log_probs.sum(
        dim=1
    ).cpu().numpy().astype(np.float64)
    compatibility_difference = (compatibility_log_z_pf - compatibility_log_r_pb) - (
        log_z_pf - log_r_pb
    )
    num_forward_clipped = int(((raw_forward < -50.0) & forward_valid).sum().cpu().item())
    num_backward_clipped = int(((raw_backward < -50.0) & backward_valid).sum().cpu().item())
    return {
        "log_pf": forward_sums,
        "log_pb": backward_sums,
        "log_z_pf": log_z_pf,
        "log_r_pb": log_r_pb,
        "log_ratios": log_z_pf - log_r_pb,
        "sources": np.asarray(sources),
        "num_forward_logprob_clipped": num_forward_clipped,
        "num_backward_logprob_clipped": num_backward_clipped,
        "max_abs_training_floor_log_ratio_difference": float(
            np.max(np.abs(compatibility_difference))
        ),
    }


def estimate_fcs_mc_terminal_log_weights(
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
    forward_trajectories: Sequence[Trajectory],
    forward_policy: DiscreteForwardPolicy,
    backward_policy: DiscreteBackwardPolicy,
    device: torch.device,
    *,
    epochs: int,
    bucket_size: int,
    budgets: Sequence[int],
) -> tuple[dict[int, np.ndarray], int]:
    """Estimate p_T(x) with fresh conditional-backward importance samples.

    For each unique forward-visited x and each budget K, estimate
    p_T(x)=E_{tau~p_B(.|x)}[p_F(tau)/p_B(tau|x)] with K samples. Budgets are
    nested: smaller K uses the first K paths from the largest-budget sample.
    Forward trajectories define S only and are never reused as backward paths.
    """

    expected = epochs * bucket_size
    if len(forward_trajectories) != expected:
        raise ValueError(f"FCS expected {expected} forward trajectories")
    normalized_budgets = tuple(sorted({int(budget) for budget in budgets}))
    if not normalized_budgets or normalized_budgets[0] <= 0:
        raise ValueError("FCS-MC budgets must be positive integers")
    if not all(trajectory.source == "forward" for trajectory in forward_trajectories):
        raise ValueError("FCS subsets may contain only forward trajectories")

    aligned_log_weights = {
        budget: np.empty(expected, dtype=np.float64) for budget in normalized_budgets
    }
    number_backward_trajectories = 0
    maximum_budget = normalized_budgets[-1]
    for epoch in range(epochs):
        start = epoch * bucket_size
        end = start + bucket_size
        bucket = forward_trajectories[start:end]
        unique: dict[bytes, Trajectory] = {}
        keys: list[bytes] = []
        for trajectory in bucket:
            terminal = np.ascontiguousarray(trajectory.observations[-1], dtype=np.float32)
            key = terminal.tobytes()
            keys.append(key)
            unique.setdefault(key, trajectory)

        conditional: list[Trajectory] = []
        path_slices: dict[bytes, slice] = {}
        unique_keys = list(unique)
        for key in unique_keys:
            terminal_trajectory = unique[key]
            path_start = len(conditional)
            for _ in range(maximum_budget):
                conditional.append(
                    sample_backward_trajectory_conditioned_on_terminal(
                        environment,
                        backward_policy,
                        terminal_trajectory.observations[-1],
                        terminal_trajectory.reward,
                        device,
                    )
                )
            path_slices[key] = slice(path_start, len(conditional))
        number_backward_trajectories += len(conditional)
        likelihoods = evaluate_trajectory_log_ratios(
            conditional,
            environment,
            forward_policy,
            backward_policy,
            0.0,
            device,
        )
        if not np.all(likelihoods["sources"] == "reference"):
            raise RuntimeError("FCS weights must use conditional-backward trajectories")
        log_importance_weights = np.asarray(
            likelihoods["log_pf"] - likelihoods["log_pb"], dtype=np.float64
        )
        if not np.all(np.isfinite(log_importance_weights)):
            raise RuntimeError("FCS-MC log p_F(tau)/p_B(tau|x) weights must be finite")
        for budget in normalized_budgets:
            by_key: dict[bytes, float] = {}
            for key in unique_keys:
                path_slice = path_slices[key]
                first = int(path_slice.start)
                values = log_importance_weights[first : first + budget]
                by_key[key] = float(logsumexp(values) - math.log(budget))
            aligned_log_weights[budget][start:end] = np.asarray(
                [by_key[key] for key in keys], dtype=np.float64
            )

    return aligned_log_weights, number_backward_trajectories


# %% 9. One seeded TV evaluation

def run_repetition(
    spec: SeriesSpec,
    checkpoint: int,
    repeat: int,
    environment: HyperGridEnv | RegularTreeEnv | RNAOps,
    forward_policy: DiscreteForwardPolicy,
    backward_policy: DiscreteBackwardPolicy,
    checkpoint_log_z: float,
    estimator_log_z: float,
    true_tv: float,
    exact_diagnostics: Mapping[str, float],
    rna_evaluation: RNAEvaluation | None,
    hypergrid_terminal_probabilities: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    repetition_seed = int(args.seeds[repeat])
    set_all_seeds(repetition_seed)
    rng = np.random.default_rng(repetition_seed)
    started = time.perf_counter()

    if spec.kind == "rna":
        if rna_evaluation is None:
            raise RuntimeError("RNA conditional sampler requires its exact DP table")
        trajectories = collect_rna_conditional_trajectories(
            rna_evaluation,
            backward_policy,
            args.num_forward,
            args.num_backward,
            device,
            rng,
        )
    else:
        assert isinstance(environment, (HyperGridEnv, RegularTreeEnv))
        trajectories = collect_forward_trajectories(
            environment, forward_policy, args.num_forward, device
        )
        trajectories.extend(
            collect_reference_trajectories(environment, backward_policy, args.num_backward, device)
        )

    likelihoods = evaluate_trajectory_log_ratios(
        trajectories,
        environment,
        forward_policy,
        backward_policy,
        estimator_log_z,
        device,
    )
    source_values, source_counts = np.unique(likelihoods["sources"], return_counts=True)
    counts = dict(zip(source_values.tolist(), source_counts.tolist()))
    if counts.get("forward", 0) != args.num_forward:
        raise RuntimeError(f"Expected {args.num_forward} forward trajectories, got {counts}")
    if counts.get("reference", 0) != args.num_backward:
        raise RuntimeError(f"Expected {args.num_backward} reference trajectories, got {counts}")

    reference_mask = likelihoods["sources"] == "reference"
    m_result = compute_m_tv(
        likelihoods["log_z_pf"][reference_mask],
        likelihoods["log_r_pb"][reference_mask],
    )
    b_result = compute_b_tv(
        likelihoods["log_z_pf"],
        likelihoods["log_r_pb"],
        alpha=args.alpha,
        num_forward=args.num_forward,
        num_backward=args.num_backward,
    )

    fcs_sample_count = args.fcs_epochs * args.fcs_bucket_size
    base_forward_trajectories = [
        trajectory for trajectory in trajectories if trajectory.source == "forward"
    ]
    fcs_trajectories = base_forward_trajectories[:fcs_sample_count]
    additional_forward_count = fcs_sample_count - len(fcs_trajectories)
    if additional_forward_count > 0:
        if spec.kind == "rna":
            assert rna_evaluation is not None
            additional_forward = collect_rna_conditional_trajectories(
                rna_evaluation,
                backward_policy,
                additional_forward_count,
                0,
                device,
                rng,
            )
        else:
            assert isinstance(environment, (HyperGridEnv, RegularTreeEnv))
            additional_forward = collect_forward_trajectories(
                environment, forward_policy, additional_forward_count, device
            )
        if not all(trajectory.source == "forward" for trajectory in additional_forward):
            raise RuntimeError("Additional FCS bucket trajectories must all be forward samples")
        fcs_trajectories.extend(additional_forward)
    fcs_exact_log_weights = exact_fcs_terminal_log_weights(
        environment,
        fcs_trajectories,
        forward_policy,
        device,
        hypergrid_terminal_probabilities=hypergrid_terminal_probabilities,
        rna_evaluation=rna_evaluation,
    )
    fcs_exact_result = compute_fcs(
        fcs_trajectories,
        fcs_exact_log_weights,
        epochs=args.fcs_epochs,
        bucket_size=args.fcs_bucket_size,
    )
    fcs_mc_log_weights, fcs_mc_backward_trajectories = estimate_fcs_mc_terminal_log_weights(
        environment,
        fcs_trajectories,
        forward_policy,
        backward_policy,
        device,
        epochs=args.fcs_epochs,
        bucket_size=args.fcs_bucket_size,
        budgets=args.fcs_mc_budgets,
    )
    fcs_mc_results = {
        budget: compute_fcs(
            fcs_trajectories,
            fcs_mc_log_weights[budget],
            epochs=args.fcs_epochs,
            bucket_size=args.fcs_bucket_size,
        )
        for budget in args.fcs_mc_budgets
    }
    unique_states = int(round(fcs_exact_result.mean_unique_states * args.fcs_epochs))
    if any(
        not math.isclose(
            result.mean_unique_states,
            fcs_exact_result.mean_unique_states,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for result in fcs_mc_results.values()
    ):
        raise RuntimeError("Exact and MC FCS variants must use the same sampled set S")

    ratios = likelihoods["log_ratios"]
    checkpoint_index = spec.checkpoints.index(checkpoint)
    row: dict[str, Any] = {
        "series_id": spec.series_id,
        "environment": spec.environment,
        "method": spec.method,
        "environment_kind": spec.kind,
        "tree_g": spec.tree_g,
        "tree_h": spec.tree_h,
        "training_seed": 0,
        "checkpoint": checkpoint,
        "checkpoint_index": checkpoint_index,
        "phase": "calibration" if checkpoint_index < args.calibration_checkpoints else "test",
        "repeat": repeat,
        "mc_seed": repetition_seed,
        "num_forward": args.num_forward,
        "num_backward": args.num_backward,
        "m_num_trajectories": args.num_backward,
        "m_sampling_source": "reference",
        "b_num_trajectories": args.num_forward + args.num_backward,
        "fcs_num_trajectories": fcs_sample_count,
        "fcs_mc_budgets": ",".join(str(value) for value in args.fcs_mc_budgets),
        "fcs_mc_max_budget": max(args.fcs_mc_budgets),
        "fcs_mc_num_backward_trajectories": fcs_mc_backward_trajectories,
        "fcs_mc_k1_backward_budget": unique_states,
        "fcs_mc_k10_backward_budget": unique_states * 10,
        "fcs_forward_paths_evaluated": 0,
        "true_tv": true_tv,
        "ground_truth_kind": spec.ground_truth_kind,
        "checkpoint_log_z": checkpoint_log_z,
        "estimator_log_z": estimator_log_z,
        "m_tv": m_result.value,
        "m_tv_unclipped": m_result.unclipped,
        "m_optimal_c": m_result.optimal_c,
        "b_tv": b_result.value,
        "b_tv_unclipped": b_result.unclipped,
        "b_tv_empirical": b_result.empirical,
        "b_confidence_term": b_result.confidence,
        "b_optimal_c": b_result.optimal_c,
        "fcs_exact": fcs_exact_result.value,
        "fcs_mc_k1": fcs_mc_results[1].value,
        "fcs_mc_k10": fcs_mc_results[10].value,
        "fcs_epochs": args.fcs_epochs,
        "fcs_bucket_size": args.fcs_bucket_size,
        "fcs_exact_epoch_std": fcs_exact_result.epoch_std,
        "fcs_mc_k1_epoch_std": fcs_mc_results[1].epoch_std,
        "fcs_mc_k10_epoch_std": fcs_mc_results[10].epoch_std,
        "fcs_mean_unique_states": fcs_exact_result.mean_unique_states,
        "log_ratio_mean": float(np.mean(ratios)),
        "log_ratio_std": float(np.std(ratios, ddof=1)),
        "log_ratio_min": float(np.min(ratios)),
        "log_ratio_max": float(np.max(ratios)),
        "max_abs_log_ratio": float(np.max(np.abs(ratios))),
        "num_forward_logprob_clipped": likelihoods["num_forward_logprob_clipped"],
        "num_backward_logprob_clipped": likelihoods["num_backward_logprob_clipped"],
        "max_abs_training_floor_log_ratio_difference": likelihoods[
            "max_abs_training_floor_log_ratio_difference"
        ],
        "repeat_seconds": time.perf_counter() - started,
    }
    row.update(exact_diagnostics)
    return row


# %% 10. TV monitoring, calibration, and plots — TVAnalysis

# Some exclusion keys occur only in historical run metadata. They preserve
# comparison grouping and do not register or import the removed methods.
METHOD_SPECIFIC = {
    "method",
    "use_state_flow",
    "lamda",
    "weighting",
    "alpha",
    "reward_scale",
    "local_search_steps",
    "local_search_prob",
    "pessimistic_steps",
    "pessimistic_lr_scale",
    "target_TV",
    "verification_budget",
    "confidence_level",
    "patience",
    "moving_average_factor",
    "stabilizing_factor",
    "forward_traj",
    "backward_traj",
    "teacher_traj",
}


class TVAnalysis:
    """Explicit, reusable workflow for frozen-policy TV monitoring.

    Use evaluate(), calibrate(), plot_comparisons(), sweep_supports(), then
    finish(). The shared session owns all run discovery and export paths;
    this object owns checkpoint contexts and resulting tables."""

    def __init__(
        self, session, samples=None, repeats=None, checkpoints=None, device=None, alpha=0.05
    ):
        """Configure monitoring without discovering runs or evaluating checkpoints.

        ``session`` owns run selection, source tracking, issues, and exports.
        Explicit sample/repeat/checkpoint/device values override the corresponding
        GF_TV_SAMPLES, GF_TV_REPEATS, GF_TV_CHECKPOINTS, and GF_DEVICE variables.
        Each repeat uses seed 42 + repeat, preserving the notebook protocol.
        """
        self.session = session
        self.samples = int(os.environ.get("GF_TV_SAMPLES", "100") if samples is None else samples)
        self.repeats = int(os.environ.get("GF_TV_REPEATS", "1") if repeats is None else repeats)
        checkpoint_selection = (
            os.environ.get("GF_TV_CHECKPOINTS") if checkpoints is None else checkpoints
        )
        if isinstance(checkpoint_selection, str):
            checkpoint_selection = checkpoint_selection.split(",") if checkpoint_selection else None
        self.checkpoints = (
            tuple((int(value) for value in checkpoint_selection))
            if checkpoint_selection is not None
            else None
        )
        self.device = torch.device(
            os.environ.get("GF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.alpha = float(alpha)
        if self.samples < 1 or self.repeats < 1:
            raise ValueError("Positive TV sample/repeat counts required")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.shared_rna_support = None
        self.rna_support_selection = "per_run_support"
        self.raw = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.checkpoint_summary = pd.DataFrame()
        self.comparison_summary = pd.DataFrame()
        self.comparison_runs = pd.DataFrame()
        self.sweep = pd.DataFrame()
        self.contexts = []
        self._evaluated = False

    def _require_evaluation(self):
        if not self._evaluated:
            raise RuntimeError(
                "Call evaluate() before calibration, plots, support sweeps, or finish()."
            )

    def _selected_rna_support(self, path, size=None):
        """Load the selected, reward-ranked RNA support with its source digest."""
        path = Path(path)
        self.session.record(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                seqs = data["sequences"].astype(str)
                values = np.asarray(data["rewards"], dtype=np.float64)
            if len(seqs) != len(values) or len(set(seqs)) != len(seqs):
                raise ValueError("Invalid oracle support lengths/duplicates")
            if any((len(s) != 14 or set(s) - set("UCGA") for s in seqs)):
                raise ValueError("Invalid RNA14 sequence in oracle artifact")
            retained = {s: (RNAOps.encode(s), float(r)) for (s, r) in zip(seqs, values)}
        else:
            retained = load_terminal_state_file(path)
        ordered = sorted(retained, key=lambda key: (-retained[key][1], key))
        if size is not None:
            ordered = ordered[:size]
        if not ordered:
            raise ValueError("Empty RNA support")
        sequences = tuple(sorted(ordered))
        rewards = np.array([retained[s][1] for s in sequences], dtype=float)
        if not np.all(np.isfinite(rewards)) or np.any(rewards <= 0):
            raise ValueError("Support rewards must be positive and finite")
        observations = np.stack([retained[s][0] for s in sequences]).astype(np.float32)
        alphabet = {c: i for (i, c) in enumerate(RNAOps.alphabet)}
        characters = np.array([[alphabet[c] for c in s] for s in sequences], dtype=np.int8)
        substrings = {""}
        for s in sequences:
            for left in range(RNAOps.length):
                for right in range(left + 1, RNAOps.length + 1):
                    if left != 0 or right != RNAOps.length:
                        substrings.add(s[left:right])
        substrings = tuple(sorted(substrings, key=lambda s: (len(s), s)))
        indexes = {s: i for (i, s) in enumerate(substrings)}
        prepend = np.full((len(sequences), RNAOps.length, RNAOps.length), -1, dtype=np.int32)
        append = np.full_like(prepend, -1)
        for i, s in enumerate(sequences):
            for span in range(2, RNAOps.length + 1):
                for left in range(RNAOps.length - span + 1):
                    right = left + span - 1
                    prepend[i, left, right] = indexes[s[left + 1 : right + 1]]
                    append[i, left, right] = indexes[s[left:right]]
        return RNASupport(
            str(path),
            str(path),
            sequences,
            {s: i for (i, s) in enumerate(sequences)},
            observations,
            rewards,
            rewards / rewards.sum(),
            substrings,
            indexes,
            prepend,
            append,
            characters,
            {path.name: self.session.sha(path)},
            canonical_rna_support_hash(sequences, rewards),
            len(retained),
        )

    def _evaluation_arguments(self, samples, repeats, epochs=1, bucket=100):
        """Build the unchanged evaluation budgets and seed sequence."""
        return argparse.Namespace(
            seeds=list(range(42, 42 + repeats)),
            num_forward=samples,
            num_backward=samples,
            alpha=self.alpha,
            fcs_epochs=epochs,
            fcs_bucket_size=bucket,
            fcs_mc_budgets=(1, 10),
            calibration_checkpoints=5,
        )

    def _checkpoint_policies(self, run, spec, step, env):
        """Restore student policies and the saved source flow/log partition."""
        config = run["config"]
        folder = run["path"] / ("model_" + str(step))
        (forward, backward) = create_policies(
            (
                int(env.observation_space.shape[0])
                if hasattr(env, "observation_space")
                else RNAOps.state_dim
            ),
            int(env.action_space.n) if hasattr(env, "action_space") else RNAOps.action_dim,
            config,
            self.device,
        )
        for policy, filename in [
            (forward, "forward_policy.pth"),
            (backward, "backward_policy.pth"),
        ]:
            path = folder / filename
            self.session.record(path)
            policy.load_state_dict(_torch_load(path, self.device))
            policy.eval()
        zp = folder / "logZ.pth"
        if zp.exists():
            self.session.record(zp)
            logz = float(_torch_load(zp, self.device).item())
        else:
            from train import (
                get_arguments,
                create_model,
                create_environment as training_environment,
            )

            args = get_arguments([])
            for k, v in config.items():
                setattr(args, k, v)
            args.wandb_project = None
            fullenv = training_environment(args)
            model = create_model(fullenv, args, model_dir=None)
            flowpath = folder / "logF.pth"
            self.session.record(flowpath)
            model.logF.load_state_dict(_torch_load(flowpath, model.device))
            logz = float(model.get_logZ().detach().cpu())
            fullenv.close()
            del model
        return (forward, backward, logz)

    def _one_evaluation(
        self,
        run,
        spec,
        step,
        env,
        forward,
        backward,
        logz,
        support,
        samples=None,
        repeats=None,
        epochs=1,
        bucket=100,
    ):
        """Evaluate one frozen checkpoint, preserving target and provenance fields."""
        samples = self.samples if samples is None else samples
        repeats = self.repeats if repeats is None else repeats
        topology = build_hypergrid_topology(env) if spec.kind == "hypergrid" else None
        (exact, effective_z, rna_eval, terminal_probs, diagnostics) = compute_exact_checkpoint(
            spec, env, forward, self.device, 4096, topology, support
        )
        evaluation_z = effective_z if spec.kind == "rna" else logz
        rows = []
        for repeat in range(repeats):
            row = run_repetition(
                spec,
                step,
                repeat,
                env,
                forward,
                backward,
                logz,
                evaluation_z,
                exact,
                diagnostics,
                rna_eval,
                terminal_probs,
                self._evaluation_arguments(samples, repeats, epochs, bucket),
                self.device,
            )
            row.update(
                {
                    "run_key": run["run_key"],
                    "cohort": run["cohort"],
                    "training_seed": run["seed"],
                    "b_tv_legacy": float(
                        np.clip(row["b_tv_empirical"] + 0.5 * row["b_confidence_term"], 0, 1)
                    ),
                    "b_tv_paper": row["b_tv"],
                    "ground_truth_kind": (
                        "conditional_tv_on_saved_support" if spec.kind == "rna" else "global_tv"
                    ),
                }
            )
            row["b_paper_confidence_term"] = row["b_confidence_term"]
            row["btv_formula"] = "manuscript_theorem_3.11"
            row["target_definition"] = (
                "stored_float64_truth_dist"
                if spec.kind == "hypergrid"
                else (
                    "saved_support_reward_conditional"
                    if spec.kind == "rna"
                    else "uniform_leaf_rewards"
                )
            )
            row["operational_target_mismatch"] = (
                run["sim"] == "hypergrid-mild"
                and self.session.hypergrid_training_target(run)
                == "legacy_float32_operational_rewards"
            )
            row["training_target_provenance"] = (
                self.session.hypergrid_training_target(run)
                if spec.kind == "hypergrid"
                else "not_hypergrid"
            )
            if support is not None:
                row.update(
                    {
                        "support_size": support.size,
                        "support_sha256": support.canonical_hash,
                        "support_source": support.source_file,
                    }
                )
            rows.append(row)
        return rows

    def _resolve_shared_rna_support(self, selected_runs, profile, explicit=None):
        """Resolve the documented fixed support for the long RNA profile."""
        if explicit:
            path = self.session.repository_path(explicit)
            if not (path / "terminal_states.pkl").is_file():
                raise FileNotFoundError(
                    "GF_RNA_SUPPORT must contain terminal_states.pkl: " + str(path)
                )
            return (path, "explicit")
        if profile != "rna-long":
            return (None, "per_run_support")
        current = {
            str(path.relative_to(self.session.root)).replace("\\", "/"): self.session.sha(path)
            for path in (self.session.root / "code").rglob("*.py")
            if "notebooks" not in path.parts and "__pycache__" not in path.parts
        }
        candidates = []
        for run in selected_runs:
            config = run["config"]
            if (
                run["sim"] != "rna14"
                or str(config.get("method", "")).lower() != "stableteacher"
                or config.get("seed") != 0
                or (int(config.get("total_iterations", 0)) != 300000)
            ):
                continue
            if (
                all((run["source_hashes"].get(key) == value for (key, value) in current.items()))
                and (run["path"] / "terminal_states.pkl").is_file()
            ):
                candidates.append(run["path"])
        if len(candidates) != 1:
            raise ValueError(
                f"rna-long requires one current-source StableTeacher seed0 300000-round support; found {len(candidates)}. Supply GF_RNA_SUPPORT explicitly if selecting a different documented support."
            )
        return (candidates[0], "sole_current_StableTeacher_seed0_300000")

    def evaluate(self):
        """Evaluate all selected checkpoints and export tv_raw_repeats.csv.

        RNA values are conditional on the selected saved support. All sampler,
        normalization, seed, and strict-source rules match the original workflow."""
        self.shared_rna_support = None
        self.rna_support_selection = "per_run_support"
        try:
            (self.shared_rna_support, self.rna_support_selection) = (
                self._resolve_shared_rna_support(
                    self.session.runs, self.session.profile, os.environ.get("GF_RNA_SUPPORT")
                )
            )
        except Exception as exc:
            self.session.issues.append({"path": "rna_shared_support", "issue": str(exc)})
            if self.session.strict:
                raise
        if self.shared_rna_support is not None:
            self.session.record(self.shared_rna_support / "terminal_states.pkl")
            print(
                "Shared RNA support:",
                self.shared_rna_support,
                "selection:",
                self.rna_support_selection,
            )
        rows = []
        self.contexts = []
        for run in self.session.runs:
            sim = run["sim"]
            config = run["config"]
            if not (sim.startswith("hypergrid") or sim in ("regular-tree", "rna14")):
                self.session.issues.append(
                    {
                        "path": str(run["path"]),
                        "issue": "Exact TV routine supports Hypergrid, Regular Tree, and RNA conditional supports; skipped this environment.",
                    }
                )
                continue
            total = int(config.get("total_iterations", 0))
            steps = (
                self.checkpoints
                if self.checkpoints is not None
                else tuple((round(total * i / 10) for i in range(1, 11)))
            )
            if len(set(steps)) != len(steps) or min(steps, default=0) <= 0:
                self.session.issues.append(
                    {
                        "path": str(run["path"]),
                        "issue": "Training budget too small for ten distinct checkpoints; use smoke-tv or GF_TV_CHECKPOINTS.",
                    }
                )
                continue
            kind = (
                "hypergrid"
                if sim.startswith("hypergrid")
                else "tree" if sim == "regular-tree" else "rna"
            )
            spec = SeriesSpec(
                run["run_key"],
                sim,
                run["method"],
                kind,
                run["path"],
                steps,
                preset=sim.split("-")[-1] if kind == "hypergrid" else None,
                tree_g=config.get("tree_g"),
                tree_h=config.get("tree_h"),
            )
            try:
                env = create_environment(spec)
                support = None
                if kind == "rna":
                    supportdir = (
                        self.shared_rna_support
                        if self.shared_rna_support is not None
                        else run["path"]
                    )
                    support = self._selected_rna_support(supportdir / "terminal_states.pkl", 10000)
                    if support.size != 10000:
                        self.session.issues.append(
                            {
                                "path": str(supportdir),
                                "issue": f"RNA support has {support.size} states; paper requires 10000 for this analysis.",
                            }
                        )
                for step in steps:
                    try:
                        (forward, backward, logz) = self._checkpoint_policies(run, spec, step, env)
                        rows.extend(
                            self._one_evaluation(
                                run, spec, step, env, forward, backward, logz, support
                            )
                        )
                        print(sim, run["method"], step, "evaluated")
                    except Exception as exc:
                        self.session.issues.append(
                            {
                                "path": str(run["path"] / ("model_" + str(step))),
                                "issue": f"{type(exc).__name__}: {exc}",
                            }
                        )
                self.contexts.append((run, spec, env, support))
            except Exception as exc:
                self.session.issues.append(
                    {"path": str(run["path"]), "issue": f"{type(exc).__name__}: {exc}"}
                )
        self.raw = pd.DataFrame(rows)
        self.raw.to_csv(self.session.figure_root / "tv_raw_repeats.csv", index=False)
        display(self.raw.head())
        self._evaluated = True
        return self.raw

    def calibrate(self):
        """Fit on the first five checkpoints and report held-out calibration.

        Writes the unchanged Table 3/7 CSVs and per-series monitoring plots.
        An incomplete ten-checkpoint/repeat series is reported and withheld."""
        self._require_evaluation()
        table_rows = []
        checkpoint_rows = []
        if not self.raw.empty:
            for runkey, group in self.raw.groupby("run_key"):
                measures = [
                    "true_tv",
                    "m_tv",
                    "b_tv_paper",
                    "b_tv_legacy",
                    "fcs_exact",
                    "fcs_mc_k1",
                    "fcs_mc_k10",
                ]
                means = group.groupby("checkpoint")[measures].mean().sort_index()
                counts = group.groupby("checkpoint").size()
                (fig, ax) = plt.subplots(figsize=(7, 3.5))
                for metric, label in [
                    ("true_tv", "Exact TV"),
                    ("m_tv", "MTV"),
                    ("b_tv_paper", "BTV (paper equation)"),
                ]:
                    ax.plot(means.index, means[metric], label=label)
                    spread = group.groupby("checkpoint")[metric].agg(["min", "max"])
                    ax.fill_between(spread.index, spread["min"], spread["max"], alpha=0.15)
                ax.plot(
                    means.index,
                    means.b_tv_paper.cummin(),
                    ls="--",
                    label="Best-so-far BTV (descriptive)",
                )
                ax.set(
                    xlabel="Training round",
                    ylabel="TV / monitoring value",
                    title=f"{group.environment.iloc[0]} / {group.method.iloc[0]} / seed {group.training_seed.iloc[0]}",
                )
                ax.legend(fontsize=8)
                self.session.save_figure(fig, "tv_" + runkey)
                if len(means) != 10 or not (counts == self.repeats).all():
                    self.session.issues.append(
                        {
                            "path": runkey,
                            "issue": "Incomplete ten-checkpoint/repeat series; Table 3/7 calibration withheld.",
                        }
                    )
                    continue
                x = means.m_tv.to_numpy()
                y = means.true_tv.to_numpy()
                design = np.column_stack([np.ones(5), x[:5]])
                if np.linalg.matrix_rank(design) < 2:
                    self.session.issues.append(
                        {
                            "path": runkey,
                            "issue": "Calibration MTV is constant; affine calibration is not identifiable.",
                        }
                    )
                    continue
                (intercept, slope) = np.linalg.lstsq(design, y[:5], rcond=None)[0]
                means["calibrated_m_tv"] = intercept + slope * x
                corr = lambda a, b: (
                    float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
                )
                table_rows.append(
                    {
                        "run_key": runkey,
                        "environment": group.environment.iloc[0],
                        "method": group.method.iloc[0],
                        "seed": group.training_seed.iloc[0],
                        "fcs_variant": "exact_terminal_marginals",
                        "best_btv_paper": means.b_tv_paper.min(),
                        "best_btv_legacy": means.b_tv_legacy.min(),
                        "corr_mtv_true_tv": corr(x, y),
                        "corr_fcs_exact_true_tv": corr(means.fcs_exact, y),
                        "rmse_calibrated_mtv": np.sqrt(
                            np.mean((means.calibrated_m_tv.to_numpy()[5:] - y[5:]) ** 2)
                        ),
                        "rmse_fcs_exact": np.sqrt(
                            np.mean((means.fcs_exact.to_numpy()[5:] - y[5:]) ** 2)
                        ),
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                    }
                )
                means["fcs"] = means["fcs_mc_k1"]
                means["fcs_variant"] = "monte_carlo_k1"
                means["run_key"] = runkey
                means["phase"] = ["calibration"] * 5 + ["test"] * 5
                checkpoint_rows.append(means.reset_index())
        summary = pd.DataFrame(table_rows)
        summary.to_csv(self.session.figure_root / "table3_calibration.csv", index=False)
        display(summary)
        (
            pd.concat(checkpoint_rows, ignore_index=True) if checkpoint_rows else pd.DataFrame()
        ).to_csv(self.session.figure_root / "table7_checkpoints.csv", index=False)
        self.summary = summary
        self.checkpoint_summary = (
            pd.concat(checkpoint_rows, ignore_index=True) if checkpoint_rows else pd.DataFrame()
        )
        return self.summary

    def _comparison_signature(self, config):
        """Group methods using the same non-method scientific configuration."""
        return self.session.signature(
            {key: value for (key, value) in config.items() if key not in METHOD_SPECIFIC}
        )

    def _aggregate_comparison(self, frame, selected_runs, shared_rna_support=False):
        """Average MC repeats within each run before aggregating independent seeds."""
        notices = []
        run_lookup = {run["run_key"]: run for run in selected_runs}
        selected = (
            frame[
                frame.environment.str.startswith("hypergrid") | (frame.environment == "rna14")
            ].copy()
            if not frame.empty
            else pd.DataFrame()
        )
        if selected.empty:
            return (pd.DataFrame(), pd.DataFrame(), notices)
        eligible = {
            run["run_key"]
            for run in selected_runs
            if run["sim"].startswith("hypergrid")
            and int(run["config"].get("total_iterations", 0)) == 30000
            or (run["sim"] == "rna14" and int(run["config"].get("total_iterations", 0)) == 300000)
        }
        excluded = selected[~selected.run_key.isin(eligible)]
        if not excluded.empty:
            notices.append(
                {
                    "path": "figure10",
                    "issue": "Shorter/other-budget runs excluded: this generic long-run comparison selects 30000 Hypergrid or 300000 RNA rounds; it does not reproduce final Figure 10.",
                }
            )
        selected = selected[selected.run_key.isin(eligible)].copy()
        if selected.empty:
            return (pd.DataFrame(), pd.DataFrame(), notices)
        if not shared_rna_support and (selected.environment == "rna14").any():
            notices.append(
                {
                    "path": "figure10/rna14",
                    "issue": "RNA comparison withheld: set GF_RNA_SUPPORT to one fixed StableTeacher support shared across algorithms and seeds.",
                }
            )
            selected = selected[selected.environment != "rna14"].copy()
        if selected.empty:
            return (pd.DataFrame(), pd.DataFrame(), notices)
        selected["support_identity"] = selected.apply(
            lambda row: (
                str(row.get("support_sha256"))
                if row.environment == "rna14"
                else "stored_global_reward_target"
            ),
            axis=1,
        )
        selected["comparison_config"] = selected.run_key.map(
            lambda key: self._comparison_signature(run_lookup[key]["config"])
        )
        keys = [
            "environment",
            "support_identity",
            "comparison_config",
            "cohort",
            "method",
            "run_key",
            "training_seed",
            "checkpoint",
            "num_backward",
        ]
        by_run = (
            selected.groupby(keys, dropna=False)
            .agg(
                mtv=("m_tv", "mean"),
                evaluation_repeats=("repeat", "count"),
                target_mismatch=("operational_target_mismatch", "max"),
            )
            .reset_index()
        )
        rows = []
        groupkeys = [
            "environment",
            "support_identity",
            "comparison_config",
            "cohort",
            "method",
            "checkpoint",
            "num_backward",
        ]
        for groupkey, group in by_run.groupby(groupkeys, dropna=False):
            row = dict(zip(groupkeys, groupkey))
            if group.training_seed.duplicated().any():
                notices.append(
                    {
                        "path": "figure10/" + row["cohort"],
                        "issue": "Duplicate trained runs for one seed at the same checkpoint: aggregate withheld.",
                    }
                )
                continue
            seeds = sorted(group.training_seed.astype(int).tolist())
            expected = [0] if row["environment"] == "rna14" else self.session.expected_seeds
            complete = seeds == expected
            row.update(
                {
                    "mtv_mean": float(group.mtv.mean()),
                    "mtv_std": float(group.mtv.std(ddof=self.session.sd_ddof)),
                    "n_training_seeds": len(seeds),
                    "observed_seeds": ",".join(map(str, seeds)),
                    "expected_seeds": ",".join(map(str, expected)),
                    "complete_seed_set": complete,
                    "target_mismatch": bool(group.target_mismatch.any()),
                    "min_evaluation_repeats": int(group.evaluation_repeats.min()),
                    "max_evaluation_repeats": int(group.evaluation_repeats.max()),
                }
            )
            rows.append(row)
        summary = pd.DataFrame(rows)
        if not summary.empty:
            for _, group in summary.groupby(["cohort", "num_backward"]):
                if not group.complete_seed_set.all():
                    notices.append(
                        {
                            "path": "figure10/" + group.cohort.iloc[0],
                            "issue": "Incomplete training-seed cohort; observed counts remain in CSV and plot labels. No absent seeds are imputed.",
                        }
                    )
        return (summary, by_run, notices)

    def plot_comparisons(self):
        """Export generic long-run comparisons using the legacy figure10 filenames.

        These aggregate training seeds after averaging evaluation repeats; they
        do not reproduce the final paper Figure 10 fixed-policy MC uncertainty."""
        self._require_evaluation()
        (figure10, figure10_runs, notices) = self._aggregate_comparison(
            self.raw, self.session.runs, self.shared_rna_support is not None
        )
        self.session.issues.extend(notices)
        figure10.to_csv(self.session.figure_root / "figure10_seed_summary.csv", index=False)
        figure10_runs.to_csv(
            self.session.figure_root / "figure10_per_training_run.csv", index=False
        )
        if not figure10.empty:
            for budget, budget_data in figure10.groupby("num_backward"):
                panelgroups = list(
                    budget_data.groupby(
                        ["environment", "support_identity", "comparison_config"],
                        sort=True,
                        dropna=False,
                    )
                )
                columns = min(4, len(panelgroups))
                rows_count = (len(panelgroups) + columns - 1) // columns
                (fig, axes) = plt.subplots(
                    rows_count,
                    columns,
                    figsize=(4.5 * columns, 3.5 * rows_count),
                    squeeze=False,
                    layout="constrained",
                )
                method_colors = {
                    method: plt.get_cmap("tab10")(index % 10)
                    for (index, method) in enumerate(sorted(budget_data.method.unique()))
                }
                for ax, ((environment, support_identity, comparison_config), group) in zip(
                    axes.flat, panelgroups
                ):
                    for (cohort, method), trace in group.groupby(["cohort", "method"]):
                        trace = trace.sort_values("checkpoint")
                        minimum = int(trace.n_training_seeds.min())
                        maximum = int(trace.n_training_seeds.max())
                        count_label = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
                        label = f"{method} (n={count_label})"
                        if not trace.complete_seed_set.all():
                            label += " [incomplete]"
                        if group[group.method == method].cohort.nunique() > 1:
                            label += " " + cohort[:6]
                        x = trace.checkpoint.to_numpy(dtype=float) / 100
                        mean = trace.mtv_mean.to_numpy(dtype=float)
                        std = trace.mtv_std.to_numpy(dtype=float)
                        ax.plot(
                            x,
                            mean,
                            label=label,
                            color=method_colors[method],
                            marker="o" if len(trace) == 1 else None,
                        )
                        valid = np.isfinite(std)
                        if valid.any():
                            ax.fill_between(
                                x,
                                mean - std,
                                mean + std,
                                where=valid,
                                alpha=0.12,
                                color=method_colors[method],
                            )
                    title = environment
                    if environment == "rna14":
                        title += "\nShared support " + support_identity[:8] + "; seed 0"
                    else:
                        title += "\nStored global target"
                    if group.target_mismatch.any():
                        title += "; legacy H16 source"
                    ax.set(
                        title=title, xlabel="Training rounds (x100)", ylabel="MTV", ylim=(0, 1.02)
                    )
                    ax.legend(fontsize=7)
                for ax in list(axes.flat)[len(panelgroups) :]:
                    ax.set_visible(False)
                fig.suptitle(
                    f"Generic comparison | {int(budget):,} backward samples\nTraining-seed means and sample SD; not final Figure 10"
                )
                warning = (
                    "Stored-target diagnostic: shared Hypergrid support protocol requires review."
                    if budget_data.environment.str.startswith("hypergrid").any()
                    else "Conditional on the explicit shared RNA support."
                )
                if budget_data.target_mismatch.any():
                    warning += "\nLegacy H16 checkpoints used different operational reward boundaries; rerun required."
                fig.text(0.5, -0.025, warning, ha="center", va="top", fontsize=8, wrap=True)
                self.session.save_figure(fig, f"figure10_mtv_m{int(budget)}")
        self.comparison_summary = figure10
        self.comparison_runs = figure10_runs
        return self.comparison_summary

    def sweep_supports(self):
        """Run the opt-in RNA support-size/FCS sweep (GF_RNA_SWEEP=1).

        The oracle column requires independently prepared top-200,000 metadata;
        missing data stay unavailable and are recorded in the session issues."""
        self._require_evaluation()
        sweep = []
        if os.environ.get("GF_RNA_SWEEP", "0") == "1":
            normalization_path = self.session.root / "code/envs/rna/L14_RNA1_info.csv"
            self.session.record(normalization_path)
            stats = pd.read_csv(normalization_path).set_index("reward_exp").loc[40]
            # The cached exponent-40 moment predates the additive environment
            # reward epsilon. Include its mass over all RNA14 terminal states.
            global_log_z = float(np.logaddexp(float(stats["logz_scaled"]), np.log(4**14 * 1e-12)))
            normalization_source = "bundled_full_state_exponent40_statistics"
            oracle_path = os.environ.get("GF_RNA_ORACLE_SUPPORT")
            oracle_support = None
            if oracle_path:
                oracle_path = self.session.repository_path(oracle_path)
                metadata_path = oracle_path.with_suffix(".metadata.json")
                metadata = self.session.read_json(metadata_path)
                if (
                    metadata.get("top_k") != 200000
                    or metadata.get("state_count") != 4**14
                    or metadata.get("normalization") != "all_4^14_states_from_same_cache"
                ):
                    raise ValueError(
                        "Oracle metadata must establish top200000 ranking and full RNA14 normalization"
                    )
                if metadata.get("artifact_sha256") != self.session.sha(oracle_path):
                    raise ValueError("Oracle support digest does not match its provenance metadata")
                if metadata.get("reward_epsilon") != 1e-12:
                    raise ValueError(
                        "Oracle metadata must include the RNA environment's additive 1e-12 reward; "
                        "regenerate legacy artifacts with rna_oracle_preparation.ipynb"
                    )
                if metadata.get("reward_exponent") != 40 or not np.isclose(
                    metadata.get("reward_scale", np.nan), float(stats["scale"]), rtol=1e-12
                ):
                    raise ValueError(
                        "Oracle reward transformation differs from the training environment"
                    )
                oracle_support = self._selected_rna_support(oracle_path)
                if oracle_support.size != 200000:
                    raise ValueError("Oracle must contain exactly 200000 distinct states")
                global_log_z = float(metadata["global_log_reward_sum"])
                normalization_source = "exhaustive_same_cache_oracle_preparation"
            else:
                self.session.issues.append(
                    {
                        "path": "table8/oracle",
                        "issue": "Oracle artifact not supplied: run the opt-in oracle preparation notebook. Top-K masses use bundled full-state statistics; the Oracle column remains unavailable.",
                    }
                )
            for run, spec, env, support in self.contexts:
                if (
                    spec.kind != "rna"
                    or str(run["config"].get("method", "")).lower() != "stableteacher"
                ):
                    continue
                if int(run["config"].get("total_iterations", 0)) != 300000:
                    self.session.issues.append(
                        {
                            "path": run["run_key"],
                            "issue": "Table 8/9 require the long 300000-round StableTeacher checkpoint; this shorter run is excluded.",
                        }
                    )
                    continue
                try:
                    step = spec.checkpoints[-1]
                    (forward, backward, logz) = self._checkpoint_policies(run, spec, step, env)
                    source = Path(support.source_file)
                    supports = []
                    for k in (10, 100, 1000, 10000):
                        sub = self._selected_rna_support(source, k)
                        if sub.size != k:
                            raise ValueError(
                                f"Saved support has fewer than the required {k} states"
                            )
                        supports.append((str(k), sub))
                    if oracle_support is not None:
                        supports.append(("oracle_top200000", oracle_support))
                    for label, sub in supports:
                        for samples, epochs, bucket in [(m, 1, 100) for m in (10, 100, 1000)] + [
                            (100, e, b) for (e, b) in [(5, 100), (1, 1000), (5, 1000)]
                        ]:
                            records = self._one_evaluation(
                                run,
                                spec,
                                step,
                                env,
                                forward,
                                backward,
                                logz,
                                sub,
                                samples,
                                self.repeats,
                                epochs,
                                bucket,
                            )
                            fraction = float(np.exp(logsumexp(np.log(sub.rewards)) - global_log_z))
                            for row in records:
                                row.update(
                                    {
                                        "top_k": label,
                                        "captured_reward_fraction": fraction,
                                        "reward_fraction_denominator": "all_4^14_RNA14_states",
                                        "normalization_source": normalization_source,
                                        "global_log_reward_sum": global_log_z,
                                        "fcs": row["fcs_mc_k1"],
                                        "fcs_variant": "monte_carlo_k1",
                                    }
                                )
                            sweep.extend(records)
                except Exception as exc:
                    self.session.issues.append(
                        {"path": run["run_key"], "issue": f"RNA support sweep failed: {exc}"}
                    )
        sweepframe = pd.DataFrame(sweep)
        sweepframe.to_csv(self.session.figure_root / "table8_9_subgraph_sweep.csv", index=False)
        if not sweepframe.empty:
            induced = (
                sweepframe[(sweepframe.fcs_epochs == 1) & (sweepframe.fcs_bucket_size == 1000)]
                .groupby(["run_key", "top_k"])
                .fcs_mean_unique_states.agg(["mean", "std", "count"])
                .reset_index()
            )
            induced["fcs_variant"] = "monte_carlo_k1"
            induced.to_csv(self.session.figure_root / "table9_induced_subspace.csv", index=False)
            display(induced)
            grouping = [
                "run_key",
                "top_k",
                "num_backward",
                "fcs_epochs",
                "fcs_bucket_size",
                "fcs_variant",
                "normalization_source",
            ]
            sweep_summary = (
                sweepframe.groupby(grouping)
                .agg(
                    mtv_mean=("m_tv", "mean"),
                    mtv_std=("m_tv", "std"),
                    true_tv=("true_tv", "mean"),
                    fcs_mc_k1_mean=("fcs_mc_k1", "mean"),
                    fcs_mc_k1_std=("fcs_mc_k1", "std"),
                    fcs_exact_mean=("fcs_exact", "mean"),
                    fcs_exact_std=("fcs_exact", "std"),
                    captured_reward_fraction=("captured_reward_fraction", "mean"),
                    n_repeats=("repeat", "count"),
                )
                .reset_index()
            )
            sweep_summary.to_csv(
                self.session.figure_root / "table8_monitoring_summary.csv", index=False
            )
            fit_rows = []
            for key, group in sweep_summary.groupby(
                ["run_key", "num_backward", "fcs_epochs", "fcs_bucket_size"]
            ):
                row = dict(zip(["run_key", "num_backward", "fcs_epochs", "fcs_bucket_size"], key))
                row["n_supports"] = len(group)
                row["fcs_variant"] = "monte_carlo_k1"
                for metric in ["mtv_mean", "fcs_mc_k1_mean"]:
                    row[metric + "_linear_fit_r2"] = (
                        float(np.corrcoef(group[metric], group.true_tv)[0, 1] ** 2)
                        if len(group) > 2
                        and np.std(group[metric]) > 0
                        and (np.std(group.true_tv) > 0)
                        else np.nan
                    )
                fit_rows.append(row)
            pd.DataFrame(fit_rows).to_csv(
                self.session.figure_root / "table8_linear_fit_r2.csv", index=False
            )
        self.sweep = sweepframe
        return self.sweep

    def finish(self):
        """Write the TV review/manifest with configuration and support provenance.

        Strict sessions reject missing or incomplete checkpoint/repeat outputs."""
        self._require_evaluation()
        extra = {
            "analysis_scope": "Generic monitoring; legacy figure10_* exports average MC repeats within runs then summarize training seeds on the selected support. They do not reproduce final Figure 10 fixed-policy MC uncertainty or its RNA terminal-cache union.",
            "tv_samples_per_direction": self.samples,
            "repeats": self.repeats,
            "evaluation_seeds": list(range(42, 42 + self.repeats)),
            "alpha": self.alpha,
            "paper_confidence_multiplier": 2,
            "fcs_main_text": "exact_terminal_probabilities",
            "fcs_appendix": "monte_carlo_k1",
            "oracle_definition": "ground_truth_top_200000_reward_states",
            "rna_shared_support": (
                str(self.shared_rna_support) if self.shared_rna_support is not None else None
            ),
            "rna_support_selection": self.rna_support_selection,
            "rna_shared_support_sha256": (
                self.session.sha(self.shared_rna_support / "terminal_states.pkl")
                if self.shared_rna_support is not None
                else None
            ),
        }
        complete = not (
            not self.contexts
            or len(self.contexts)
            != sum(
                (
                    run["sim"].startswith("hypergrid") or run["sim"] in ("regular-tree", "rna14")
                    for run in self.session.runs
                )
            )
            or len(self.raw)
            != sum((len(spec.checkpoints) for (_, spec, _, _) in self.contexts)) * self.repeats
        )
        return self.session.finish("tv", extra=extra, complete=complete)
