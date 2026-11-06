"""Runtime profiling utilities for GFlowNet training runs."""

from __future__ import annotations

import json
import os
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional

import torch


PROFILE_FILENAME = "runtime_profile.json"


def profile_runtime(phase: str):
    """Measure a method call as part of a named runtime phase."""

    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with self._measure_runtime_phase(phase):
                return method(self, *args, **kwargs)

        return wrapped

    return decorator


class RuntimeProfilingMixin:
    """Profile sampling and training performed by ``GFlowNetBase.learn``.

    Subclasses provide their optimizer implementation as ``_train_impl``.
    They may optionally provide ``_collect_rollouts_impl`` when rollout
    collection differs from the base implementation. CPU time uses the process
    clock, which accumulates work across all threads in the current process.
    CUDA time is the sum of kernel durations reported by PyTorch Profiler.
    """

    _PROFILE_PHASES = ("sampling", "training", "reference_flow")
    _GPU_PROFILE_MARKER_PREFIX = "stablegfn.runtime_profile"
    # 101 is intentionally odd so alternating sampling/training phases are
    # both selected by periodic sparse profiling.
    _GPU_PROFILE_INTERVAL = 101

    def _synchronize_profile_device(self) -> None:
        """Wait for queued CUDA work so wall-clock measurements are accurate."""
        device = getattr(self, "device", None)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    def _ensure_runtime_profile_state(self) -> Dict[str, Any]:
        state = getattr(self, "_runtime_profile_state", None)
        if state is None:
            state = {
                "active": False,
                "wall_elapsed": {
                    phase: 0.0 for phase in self._PROFILE_PHASES
                },
                "cpu_elapsed": {
                    phase: 0.0 for phase in self._PROFILE_PHASES
                },
                "gpu_kernel_elapsed": {
                    phase: 0.0 for phase in self._PROFILE_PHASES
                },
                "gpu_profiled_call_counts": {
                    phase: 0 for phase in self._PROFILE_PHASES
                },
                "call_counts": {phase: 0 for phase in self._PROFILE_PHASES},
                "measure_depth": 0,
                "torch_profiler": None,
            }
            self._runtime_profile_state = state
        return state

    @contextmanager
    def _measure_runtime_phase(self, phase: str):
        if not getattr(self, 'runtime_profile', True):
            yield
            return
        state = self._ensure_runtime_profile_state()
        if phase not in state["wall_elapsed"]:
            state["wall_elapsed"][phase] = 0.0
            state["cpu_elapsed"][phase] = 0.0
            state["gpu_kernel_elapsed"][phase] = 0.0
            state["call_counts"][phase] = 0

        is_top_level = state["measure_depth"] == 0
        state["measure_depth"] += 1
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            profiler = state.get("torch_profiler")
            if profiler is None:
                yield
            else:
                marker = self._gpu_profile_marker(phase)
                with torch.profiler.record_function(marker):
                    yield
        finally:
            state["cpu_elapsed"][phase] += time.process_time() - cpu_started
            state["wall_elapsed"][phase] += time.perf_counter() - wall_started
            state["call_counts"][phase] += 1
            state["measure_depth"] -= 1
            if is_top_level:
                self._flush_gpu_profile_phase(phase)

    def collect_rollouts(self):
        with self._measure_runtime_phase("sampling"):
            implementation = getattr(self, "_collect_rollouts_impl", None)
            if implementation is not None:
                return implementation()
            return super().collect_rollouts()

    def train(self):
        with self._measure_runtime_phase("training"):
            return self._train_impl()

    def learn(self, total_iterations, i_start=0, e_start=0):
        """Run training and persist a profile even if the run is interrupted."""
        if not getattr(self, 'runtime_profile', True):
            return super().learn(total_iterations=total_iterations, i_start=i_start, e_start=e_start)
        state = self._ensure_runtime_profile_state()
        state["active"] = True
        state["wall_elapsed"] = {
            phase: 0.0 for phase in self._PROFILE_PHASES
        }
        state["cpu_elapsed"] = {
            phase: 0.0 for phase in self._PROFILE_PHASES
        }
        state["gpu_kernel_elapsed"] = {
            phase: 0.0 for phase in self._PROFILE_PHASES
        }
        state["gpu_profiled_call_counts"] = {
            phase: 0 for phase in self._PROFILE_PHASES
        }
        state["call_counts"] = {phase: 0 for phase in self._PROFILE_PHASES}
        state["measure_depth"] = 0
        state["torch_profiler"] = None
        state["run_id"] = uuid.uuid4().hex
        state["started_at_utc"] = self._utc_now()
        state["started_at_perf_counter"] = time.perf_counter()
        state["started_at_process_time"] = time.process_time()
        state["requested_total_iterations"] = int(total_iterations)
        state["start_iteration"] = int(i_start)
        state["start_episode"] = int(e_start)
        self._start_gpu_kernel_profiler()

        status = "completed"
        error: Optional[BaseException] = None
        try:
            return super().learn(
                total_iterations=total_iterations,
                i_start=i_start,
                e_start=e_start,
            )
        except BaseException as exc:
            error = exc
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            raise
        finally:
            try:
                self._finish_runtime_profile(status=status, error=error)
            except Exception as profile_error:
                if error is None:
                    raise
                warnings.warn(
                    f"Unable to save runtime profile after {status} run: "
                    f"{profile_error}",
                    RuntimeWarning,
                )

    def _finish_runtime_profile(
        self,
        status: str,
        error: Optional[BaseException],
    ) -> None:
        state = self._ensure_runtime_profile_state()
        self._synchronize_profile_device()
        total_wall_seconds = (
            time.perf_counter() - state["started_at_perf_counter"]
        )
        total_cpu_seconds = (
            time.process_time() - state["started_at_process_time"]
        )
        self._stop_gpu_kernel_profiler()

        wall_timings = self._build_timing_breakdown(
            total_wall_seconds,
            state["wall_elapsed"],
        )
        cpu_timings = self._build_timing_breakdown(
            total_cpu_seconds,
            state["cpu_elapsed"],
        )

        gpu_available = state["gpu_kernel_profiling_available"]
        uses_cuda = (
            torch.cuda.is_available()
            and str(getattr(self, "device", "")).startswith("cuda")
        )
        if gpu_available and uses_cuda:
            gpu_elapsed, missing_gpu_phases = (
                self._estimate_gpu_kernel_elapsed()
            )
            if missing_gpu_phases:
                gpu_available = False
                state["gpu_kernel_profiling_available"] = False
                state["gpu_kernel_profiling_reason"] = (
                    "No CUDA kernel sample was collected for phases with "
                    f"work: {', '.join(missing_gpu_phases)}"
                )
        elif gpu_available:
            gpu_elapsed = {
                phase: 0.0 for phase in self._PROFILE_PHASES
            }

        if gpu_available:
            total_gpu_seconds = (
                gpu_elapsed["training"] + gpu_elapsed["sampling"]
            )
            gpu_timings = self._build_timing_breakdown(
                total_gpu_seconds,
                gpu_elapsed,
            )
            compute_timings = {
                key: cpu_timings[key] + gpu_timings[key]
                for key in cpu_timings
            }
        else:
            gpu_timings = {
                key: None
                for key in (
                    "total",
                    "training",
                    "sampling",
                    "reference_flow",
                    "training_excluding_reference_flow",
                    "other",
                )
            }
            compute_timings = dict(gpu_timings)

        profile = {
            "run_id": state["run_id"],
            "algorithm": self.__class__.__name__,
            "status": status,
            "started_at_utc": state["started_at_utc"],
            "finished_at_utc": self._utc_now(),
            "requested_total_iterations": state["requested_total_iterations"],
            "start_iteration": state["start_iteration"],
            "end_iteration": int(getattr(self, "logger", {}).get("iteration", 0)),
            "start_episode": state["start_episode"],
            "end_episode": int(getattr(self, "logger", {}).get("episode", 0)),
            "timing_semantics": {
                "timings_seconds": (
                    "aggregate CPU time across all threads in the current "
                    "process plus summed CUDA kernel durations"
                ),
                "cpu_timings_seconds": (
                    "process CPU time across all threads; child processes "
                    "are not included"
                ),
                "gpu_kernel_timings_seconds": (
                    "estimated sum of CUDA kernel durations from sparse "
                    "samples; overlapping kernels are counted independently"
                ),
                "wall_timings_seconds": (
                    "host-observed elapsed time; total waits for CUDA once at "
                    "run completion, while phase boundaries do not synchronize"
                ),
            },
            "timings_seconds": compute_timings,
            "cpu_timings_seconds": cpu_timings,
            "gpu_kernel_timings_seconds": gpu_timings,
            "wall_timings_seconds": wall_timings,
            "gpu_kernel_profiling": {
                "available": gpu_available,
                "reason": state["gpu_kernel_profiling_reason"],
                "estimated": (
                    gpu_available and uses_cuda
                ),
                "sampling_interval_top_level_calls": (
                    self._GPU_PROFILE_INTERVAL
                ),
                "profiled_call_counts": dict(
                    state["gpu_profiled_call_counts"]
                ),
            },
            "call_counts": dict(state["call_counts"]),
        }
        if error is not None:
            profile["error_type"] = type(error).__name__
            profile["error_message"] = str(error)

        self._append_runtime_profile(profile)
        state["active"] = False

    def _start_gpu_kernel_profiler(self) -> None:
        state = self._ensure_runtime_profile_state()
        device = getattr(self, "device", None)
        uses_cuda = torch.cuda.is_available() and str(device).startswith("cuda")
        if not uses_cuda:
            state["gpu_kernel_profiling_available"] = True
            state["gpu_kernel_profiling_reason"] = "CPU-only run"
            return
        if not getattr(self, 'cuda_profile', True):
            state["gpu_kernel_profiling_available"] = False
            state["gpu_kernel_profiling_reason"] = "Disabled by cuda_profile=False"
            return

        try:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=self._gpu_profiler_schedule,
                on_trace_ready=self._consume_gpu_profile_trace,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            profiler.start()
        except Exception as exc:
            state["gpu_kernel_profiling_available"] = False
            state["gpu_kernel_profiling_reason"] = (
                f"PyTorch CUDA profiler unavailable: {exc}"
            )
            warnings.warn(
                state["gpu_kernel_profiling_reason"],
                RuntimeWarning,
            )
            return

        state["torch_profiler"] = profiler
        state["gpu_kernel_profiling_available"] = True
        state["gpu_kernel_profiling_reason"] = "PyTorch CUDA profiler"

    def _flush_gpu_profile_phase(self, phase: str) -> None:
        state = self._ensure_runtime_profile_state()
        profiler = state.get("torch_profiler")
        if profiler is None:
            return

        state["pending_gpu_phase"] = phase
        try:
            profiler.step()
        except Exception as exc:
            self._disable_gpu_kernel_profiler(
                f"Unable to collect CUDA kernel times: {exc}"
            )
        finally:
            state["pending_gpu_phase"] = None

    def _consume_gpu_profile_trace(self, profiler) -> None:
        state = self._ensure_runtime_profile_state()
        if not state["gpu_kernel_profiling_available"]:
            return

        try:
            averages = profiler.key_averages()
            events_by_key = {
                getattr(event, "key", None): event for event in averages
            }
            for phase in self._PROFILE_PHASES:
                event = events_by_key.get(self._gpu_profile_marker(phase))
                if event is None:
                    continue
                kernel_us = self._event_device_time_us(
                    event,
                    self_only=False,
                )
                state["gpu_kernel_elapsed"][phase] += kernel_us / 1e6
                state["gpu_profiled_call_counts"][phase] += int(
                    getattr(event, "count", 1)
                )
        except Exception as exc:
            self._disable_gpu_kernel_profiler(
                f"Unable to aggregate CUDA kernel times: {exc}"
            )

    def _stop_gpu_kernel_profiler(self) -> None:
        state = self._ensure_runtime_profile_state()
        profiler = state.get("torch_profiler")
        if profiler is None:
            return
        try:
            profiler.stop()
        except Exception as exc:
            self._disable_gpu_kernel_profiler(
                f"Unable to finalize CUDA kernel times: {exc}"
            )
        finally:
            state["torch_profiler"] = None

    def _disable_gpu_kernel_profiler(self, reason: str) -> None:
        state = self._ensure_runtime_profile_state()
        state["gpu_kernel_profiling_available"] = False
        state["gpu_kernel_profiling_reason"] = reason
        warnings.warn(reason, RuntimeWarning)

    def _gpu_profile_marker(self, phase: str) -> str:
        return f"{self._GPU_PROFILE_MARKER_PREFIX}::{phase}"

    def _gpu_profiler_schedule(self, step: int):
        action = torch.profiler.ProfilerAction
        if step == 0:
            return action.RECORD
        if step == 1:
            return action.RECORD_AND_SAVE
        if step % self._GPU_PROFILE_INTERVAL == (
            self._GPU_PROFILE_INTERVAL - 1
        ):
            return action.RECORD_AND_SAVE
        return action.NONE

    def _estimate_gpu_kernel_elapsed(self):
        state = self._ensure_runtime_profile_state()
        estimated = {}
        missing_phases = []
        for phase in self._PROFILE_PHASES:
            total_calls = state["call_counts"][phase]
            sampled_calls = state["gpu_profiled_call_counts"][phase]
            if total_calls == 0:
                estimated[phase] = 0.0
            elif sampled_calls == 0:
                estimated[phase] = 0.0
                missing_phases.append(phase)
            else:
                estimated[phase] = (
                    state["gpu_kernel_elapsed"][phase]
                    * total_calls
                    / sampled_calls
                )
        return estimated, missing_phases

    @staticmethod
    def _event_device_time_us(event, self_only: bool) -> float:
        if self_only:
            attribute_names = (
                "self_device_time_total",
                "self_cuda_time_total",
            )
        else:
            attribute_names = (
                "device_time_total",
                "cuda_time_total",
            )

        for attribute_name in attribute_names:
            value = getattr(event, attribute_name, None)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _build_timing_breakdown(
        total_seconds: float,
        phase_seconds: Dict[str, float],
    ) -> Dict[str, float]:
        training_seconds = phase_seconds["training"]
        sampling_seconds = phase_seconds["sampling"]
        reference_flow_seconds = phase_seconds["reference_flow"]
        return {
            "total": total_seconds,
            "training": training_seconds,
            "sampling": sampling_seconds,
            "reference_flow": reference_flow_seconds,
            "training_excluding_reference_flow": max(
                0.0,
                training_seconds - reference_flow_seconds,
            ),
            "other": max(
                0.0,
                total_seconds - training_seconds - sampling_seconds,
            ),
        }

    def _append_runtime_profile(self, profile: Dict[str, Any]) -> None:
        output_dir = getattr(self, "model_dir", None) or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        profile_path = os.path.join(output_dir, PROFILE_FILENAME)

        document = {"schema_version": 2, "runs": []}
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if (
                    isinstance(existing, dict)
                    and existing.get("schema_version") in (1, 2)
                    and isinstance(existing.get("runs"), list)
                ):
                    document["runs"] = existing["runs"]
                else:
                    warnings.warn(
                        f"Replacing incompatible runtime profile at {profile_path}",
                        RuntimeWarning,
                    )
            except (OSError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f"Replacing unreadable runtime profile at {profile_path}: {exc}",
                    RuntimeWarning,
                )

        document["runs"].append(profile)
        temporary_path = f"{profile_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2)
                handle.write("\n")
            os.replace(temporary_path, profile_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
