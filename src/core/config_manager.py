#!/usr/bin/env python3
"""Minimal JSON config loader for the slimmed training + inference repo."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from src.shared.json_utils import deep_merge, read_json_dict
from src.training.quantization import assert_no_prohibited_4bit_flags

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    return read_json_dict(path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(base, override)


class ConfigurationManager:
    """Load `config/base.json` and optionally merge `config/<environment>.json`."""

    def __init__(self, config_dir: str = "config", environment: Optional[str] = None) -> None:
        self.config_dir = Path(config_dir)
        self._environment = environment
        self._base_config: Optional[Dict[str, Any]] = None
        self._merged_config: Optional[Dict[str, Any]] = None

    def load_base_config(self) -> Dict[str, Any]:
        base_path = self.config_dir / "base.json"
        self._base_config = _read_json(base_path)
        return dict(self._base_config)

    def load_config_file(self, filename: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
        config_path = self.config_dir / filename
        if not config_path.exists():
            return {}
        payload = _read_json(config_path)
        if schema_name and schema_name not in payload:
            return {schema_name: payload}
        return payload

    def get_environment_config(self, env: str) -> Dict[str, Any]:
        if not env:
            return {}
        env_path = self.config_dir / f"{env}.json"
        if not env_path.exists():
            return {}
        return _read_json(env_path)

    def load_all_configs(self) -> Dict[str, Any]:
        merged = self.load_base_config()
        if self._environment:
            merged = _deep_merge(merged, self.get_environment_config(self._environment))
        self._normalize_training_surface(merged)
        self._normalize_ood_surface(merged)
        assert_no_prohibited_4bit_flags(merged)
        self._merged_config = merged
        return dict(merged)

    def reload_config(self) -> Dict[str, Any]:
        self._base_config = None
        self._merged_config = None
        return self.load_all_configs()

    def get_config(self, key: str, default: Any = None) -> Any:
        config = self._merged_config if self._merged_config is not None else self.load_all_configs()
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def validate_merged_config(self) -> bool:
        config = self._merged_config if self._merged_config is not None else self.load_all_configs()
        required = ("training", "router", "ood")
        if any(section not in config for section in required):
            return False
        continual = config.get("training", {}).get("continual")
        return isinstance(continual, dict)

    def _normalize_ood_surface(self, merged_config: Dict[str, Any]) -> None:
        training = merged_config.setdefault("training", {})
        continual = training.setdefault("continual", {})
        continual_ood = continual.setdefault("ood", {})
        top_level_ood = merged_config.setdefault("ood", {})

        legacy_threshold = top_level_ood.get("threshold_factor")
        canonical_threshold = continual_ood.get("threshold_factor")

        if canonical_threshold is None and legacy_threshold is not None:
            continual_ood["threshold_factor"] = float(legacy_threshold)
            canonical_threshold = float(legacy_threshold)
        if canonical_threshold is not None:
            top_level_ood["threshold_factor"] = float(canonical_threshold)

    def _normalize_training_surface(self, merged_config: Dict[str, Any]) -> None:
        training = merged_config.setdefault("training", {})
        continual = training.setdefault("continual", {})

        optimization = continual.setdefault("optimization", {})
        scheduler = optimization.setdefault("scheduler", {})
        data = continual.setdefault("data", {})
        early_stopping = continual.setdefault("early_stopping", {})
        evaluation = continual.setdefault("evaluation", {})

        continual["seed"] = int(continual.get("seed", 42))
        continual["deterministic"] = bool(continual.get("deterministic", False))

        optimization["grad_accumulation_steps"] = int(optimization.get("grad_accumulation_steps", 1))
        optimization["max_grad_norm"] = float(optimization.get("max_grad_norm", 0.0))
        optimization["mixed_precision"] = str(optimization.get("mixed_precision", "auto"))
        optimization["label_smoothing"] = float(optimization.get("label_smoothing", 0.0))

        scheduler["name"] = str(scheduler.get("name", "none"))
        scheduler["warmup_ratio"] = float(scheduler.get("warmup_ratio", 0.0))
        scheduler["min_lr"] = float(scheduler.get("min_lr", 0.0))
        scheduler["step_on"] = str(scheduler.get("step_on", "batch"))

        data["sampler"] = str(data.get("sampler", "shuffle"))
        data["loader_error_policy"] = str(data.get("loader_error_policy", "tolerant"))
        data["target_size"] = int(data.get("target_size", 224))
        data["cache_size"] = int(data.get("cache_size", 1000))
        data["validate_images_on_init"] = bool(data.get("validate_images_on_init", True))

        evaluation_best_metric = str(evaluation.get("best_metric", early_stopping.get("metric", "val_loss")))
        evaluation["best_metric"] = evaluation_best_metric
        evaluation["emit_ood_gate"] = bool(evaluation.get("emit_ood_gate", True))
        evaluation["require_ood_for_gate"] = bool(evaluation.get("require_ood_for_gate", False))

        early_metric = str(early_stopping.get("metric", evaluation_best_metric))
        inferred_mode = "min" if early_metric in {"val_loss", "generalization_gap"} else "max"
        early_stopping["enabled"] = bool(early_stopping.get("enabled", False))
        early_stopping["metric"] = early_metric
        early_stopping["mode"] = str(early_stopping.get("mode", inferred_mode))
        early_stopping["patience"] = int(early_stopping.get("patience", 3))
        early_stopping["min_delta"] = float(early_stopping.get("min_delta", 0.0))

        colab = merged_config.setdefault("colab", {})
        colab_training = colab.setdefault("training", {})
        legacy_checkpoint_interval = colab_training.get("checkpoint_interval")
        checkpoint_every_n_steps = colab_training.get("checkpoint_every_n_steps", legacy_checkpoint_interval)

        colab_training["num_workers"] = int(colab_training.get("num_workers", 2))
        colab_training["pin_memory"] = bool(colab_training.get("pin_memory", True))
        colab_training["stdout_progress_batch_interval"] = int(colab_training.get("stdout_progress_batch_interval", 50))
        colab_training["checkpoint_every_n_steps"] = int(
            checkpoint_every_n_steps if checkpoint_every_n_steps is not None else 200
        )
        colab_training["checkpoint_interval"] = int(colab_training["checkpoint_every_n_steps"])
        colab_training["checkpoint_on_exception"] = bool(colab_training.get("checkpoint_on_exception", True))


_CONFIG_LOCK = RLock()
_CONFIG_CACHE: Dict[tuple[str, Optional[str]], ConfigurationManager] = {}


def _get_manager(config_dir: str = "config", environment: Optional[str] = None) -> ConfigurationManager:
    key = (str(Path(config_dir)), environment)
    with _CONFIG_LOCK:
        manager = _CONFIG_CACHE.get(key)
        if manager is None:
            manager = ConfigurationManager(config_dir=config_dir, environment=environment)
            _CONFIG_CACHE[key] = manager
        return manager


def get_config(environment: Optional[str] = None, config_dir: str = "config") -> Dict[str, Any]:
    return _get_manager(config_dir=config_dir, environment=environment).load_all_configs()


def reload_configuration(environment: Optional[str] = None, config_dir: str = "config") -> Dict[str, Any]:
    return _get_manager(config_dir=config_dir, environment=environment).reload_config()
