"""Ordered online mode tracking and end-of-run metric persistence."""

from __future__ import annotations

import json
import os
import pickle


MODE_METRICS_FILENAME = "mode_metrics.json"
MODE_RECORDS_FILENAME = "mode_records.pkl"


def record_discovered_mode(
    discovered_modes,
    discovered_mode_rewards,
    mode_key,
    reward,
):
    """Record a first-seen mode without losing encounter order or reward."""
    if mode_key is None or mode_key in discovered_modes:
        return False

    discovered_modes.add(mode_key)
    discovered_mode_rewards[mode_key] = (
        None if reward is None else float(reward)
    )
    return True


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_and_store_mode_metrics(model, environment, output_dir):
    """Evaluate ordered modes and store their records and named counts.

    Environments opt into this hook by defining ``MODE_METRIC_NAMES``.  The
    mode records are written before evaluation so a scheduler timeout during
    the potentially expensive chemistry calculation still leaves enough data
    for later recomputation.
    """
    base_environment = getattr(environment, "unwrapped", environment)
    metric_names = getattr(base_environment, "MODE_METRIC_NAMES", None)
    if metric_names is None:
        return None

    discovered_mode_rewards = getattr(
        model,
        "discovered_mode_rewards",
        None,
    )
    if discovered_mode_rewards is None:
        raise AttributeError(
            "The model does not expose ordered discovered_mode_rewards."
        )

    ordered_records = list(discovered_mode_rewards.items())
    os.makedirs(output_dir, exist_ok=True)
    records_path = os.path.join(output_dir, MODE_RECORDS_FILENAME)
    with open(records_path, "wb") as handle:
        pickle.dump(
            {
                "schema_version": 1,
                "records": ordered_records,
            },
            handle,
        )

    environment_name = getattr(
        base_environment,
        "MODE_METRIC_ENVIRONMENT",
        type(base_environment).__name__,
    )
    metrics_path = os.path.join(output_dir, MODE_METRICS_FILENAME)
    base_payload = {
        "schema_version": 1,
        "environment": environment_name,
        "num_tracked_modes": len(ordered_records),
        "metric_order": list(metric_names),
        "records_file": MODE_RECORDS_FILENAME,
    }

    thresholds = {}
    raw_threshold = getattr(
        base_environment,
        "evaluation_mode_r_raw_threshold",
        None,
    )
    if raw_threshold is not None:
        thresholds["raw_reward"] = float(raw_threshold)
        thresholds["reward_comparison"] = ">="
    similarity_threshold = getattr(
        base_environment,
        "tanimoto_similarity_threshold",
        None,
    )
    if similarity_threshold is not None:
        thresholds["tanimoto_similarity"] = float(similarity_threshold)
    if thresholds:
        base_payload["thresholds"] = thresholds

    try:
        samples = [state for state, _ in ordered_records]
        rewards = [reward for _, reward in ordered_records]
        if any(reward is None for reward in rewards):
            raise ValueError(
                "Every tracked mode must retain its scaled reward for "
                "end-of-run evaluation."
            )

        counts = tuple(
            base_environment.get_error(samples, rewards=rewards)
        )
        if len(counts) != len(metric_names):
            raise ValueError(
                f"Expected {len(metric_names)} mode metrics, "
                f"got {len(counts)}."
            )
        integer_counts = [int(value) for value in counts]
        payload = {
            **base_payload,
            "status": "ok",
            "counts_tuple": integer_counts,
            "metrics": dict(zip(metric_names, integer_counts)),
        }
    except Exception as error:
        payload = {
            **base_payload,
            "status": "error",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_json(metrics_path, payload)
        raise

    _write_json(metrics_path, payload)
    return payload
