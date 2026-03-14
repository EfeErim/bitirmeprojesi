"""Versioned migration helpers for the public JSON config surface."""

from __future__ import annotations

import copy
from typing import Any, Dict

CONFIG_SCHEMA_VERSION_KEY = "config_schema_version"
CURRENT_CONFIG_SCHEMA_VERSION = 1

_LEGACY_OOD_ALIAS_KEYS = (
    "threshold_factor",
    "primary_score_method",
    "radial_l2_enabled",
    "sure_enabled",
    "conformal_enabled",
    "conformal_alpha",
    "conformal_method",
    "ber_enabled",
    "ber_lambda_old",
    "ber_lambda_new",
    "radial_beta_range",
    "radial_beta_steps",
    "sure_semantic_percentile",
    "sure_confidence_percentile",
    "conformal_raps_lambda",
    "conformal_raps_k_reg",
    "energy_temperature_mode",
    "energy_temperature",
    "energy_temperature_range",
    "energy_temperature_steps",
    "knn_backend",
    "knn_chunk_size",
)


def _read_config_schema_version(payload: Dict[str, Any]) -> int:
    raw = payload.get(CONFIG_SCHEMA_VERSION_KEY, 0)
    try:
        version = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{CONFIG_SCHEMA_VERSION_KEY} must be an integer-compatible value.") from None
    if version < 0:
        raise ValueError(f"{CONFIG_SCHEMA_VERSION_KEY} must be non-negative.")
    return version


def _ensure_nested_dict(payload: Dict[str, Any], *path: str) -> Dict[str, Any]:
    current = payload
    for key in path:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    return current


def _migrate_legacy_top_level_ood_aliases(payload: Dict[str, Any]) -> None:
    top_level_ood = payload.get("ood")
    if not isinstance(top_level_ood, dict):
        return

    alias_keys = [key for key in _LEGACY_OOD_ALIAS_KEYS if key in top_level_ood]
    if not alias_keys:
        return

    continual_ood = _ensure_nested_dict(payload, "training", "continual", "ood")
    for key in alias_keys:
        if continual_ood.get(key) is None:
            continual_ood[key] = copy.deepcopy(top_level_ood[key])


def _migrate_legacy_checkpoint_interval(payload: Dict[str, Any]) -> None:
    colab_training = payload.get("colab", {}).get("training")
    if not isinstance(colab_training, dict):
        return
    if "checkpoint_every_n_steps" not in colab_training and "checkpoint_interval" in colab_training:
        colab_training["checkpoint_every_n_steps"] = copy.deepcopy(colab_training["checkpoint_interval"])


def _migrate_v0_to_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    _migrate_legacy_top_level_ood_aliases(payload)
    _migrate_legacy_checkpoint_interval(payload)
    payload[CONFIG_SCHEMA_VERSION_KEY] = 1
    return payload


def migrate_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    migrated = copy.deepcopy(payload or {})
    version = _read_config_schema_version(migrated)
    if version > CURRENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {CONFIG_SCHEMA_VERSION_KEY}={version}; "
            f"current loader supports up to {CURRENT_CONFIG_SCHEMA_VERSION}."
        )

    while version < CURRENT_CONFIG_SCHEMA_VERSION:
        if version == 0:
            migrated = _migrate_v0_to_v1(migrated)
            version = 1
            continue
        raise ValueError(f"Unsupported migration path from {CONFIG_SCHEMA_VERSION_KEY}={version}.")

    migrated[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION
    return migrated


def project_compatibility_aliases(merged_config: Dict[str, Any]) -> None:
    merged_config[CONFIG_SCHEMA_VERSION_KEY] = CURRENT_CONFIG_SCHEMA_VERSION

    top_level_ood = _ensure_nested_dict(merged_config, "ood")
    continual_ood = _ensure_nested_dict(merged_config, "training", "continual", "ood")
    for key in _LEGACY_OOD_ALIAS_KEYS:
        if key in continual_ood:
            top_level_ood[key] = copy.deepcopy(continual_ood[key])

    colab_training = _ensure_nested_dict(merged_config, "colab", "training")
    checkpoint_every_n_steps = colab_training.get("checkpoint_every_n_steps")
    if checkpoint_every_n_steps is not None:
        colab_training["checkpoint_interval"] = copy.deepcopy(checkpoint_every_n_steps)
