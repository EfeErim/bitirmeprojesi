"""Normalization helpers for the public continual-training config contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.shared.json_utils import deep_merge
from src.training.services.ood_score_selection import normalize_requested_primary_score_method

DEFAULT_BACKBONE_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_FUSION_LAYERS = [2, 5, 8, 11]
DEFAULT_FUSION_OUTPUT_DIM = 768
DEFAULT_DEVICE = "cuda"


def _build_default_continual_surface(*, model_name: str, device: Any) -> Dict[str, Any]:
    return {
        "backbone": {"model_name": str(model_name)},
        "adapter": {
            "target_modules_strategy": "all_linear_transformer",
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
        "fusion": {
            "layers": list(DEFAULT_FUSION_LAYERS),
            "output_dim": DEFAULT_FUSION_OUTPUT_DIM,
            "dropout": 0.1,
            "gating": "softmax",
        },
        "ood": {
            "threshold_factor": 2.0,
            "primary_score_method": "auto",
            "ber_enabled": False,
            "ber_lambda_old": 0.1,
            "ber_lambda_new": 0.1,
            "ber_warmup_steps": 50,
            "energy_temperature_mode": "fixed",
            "energy_temperature": 1.0,
            "energy_temperature_range": [0.5, 3.0],
            "energy_temperature_steps": 16,
            "radial_l2_enabled": True,
            "radial_beta_range": [0.5, 2.0],
            "radial_beta_steps": 16,
            "knn_backend": "auto",
            "knn_chunk_size": 2048,
            "sure_enabled": True,
            "sure_semantic_percentile": 95.0,
            "sure_confidence_percentile": 90.0,
            "conformal_enabled": True,
            "conformal_alpha": 0.05,
            "conformal_method": "threshold",
            "conformal_raps_lambda": 0.0,
            "conformal_raps_k_reg": 1,
        },
        "optimization": {
            "grad_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "mixed_precision": "auto",
            "label_smoothing": 0.0,
            "scheduler": {
                "name": "cosine",
                "warmup_ratio": 0.1,
                "min_lr": 1e-6,
                "step_on": "batch",
            },
        },
        "data": {
            "sampler": "shuffle",
            "loader_error_policy": "tolerant",
            "target_size": 224,
            "cache_size": 1000,
            "validate_images_on_init": True,
        },
        "early_stopping": {
            "enabled": True,
            "metric": "val_loss",
            "mode": "min",
            "patience": 5,
            "min_delta": 0.0,
        },
        "evaluation": {
            "best_metric": "val_loss",
            "emit_ood_gate": True,
            "require_ood_for_gate": True,
            "ood_fallback_strategy": "held_out_benchmark",
            "ood_benchmark_auto_run": True,
            "ood_benchmark_min_classes": 3,
        },
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "batch_size": 8,
        "device": str(device),
        "strict_model_loading": False,
        "seed": 42,
        "deterministic": True,
    }


def _coerce_legacy_flat_config(flat_config: Dict[str, Any], *, model_name: str, device: Any) -> Dict[str, Any]:
    return {
        "backbone": {"model_name": str(flat_config.get("model_name", model_name))},
        "adapter": {
            "target_modules_strategy": str(flat_config.get("target_modules_strategy", "all_linear_transformer")),
            "lora_r": int(flat_config.get("lora_r", 16)),
            "lora_alpha": int(flat_config.get("lora_alpha", 16)),
            "lora_dropout": float(flat_config.get("lora_dropout", 0.1)),
        },
        "fusion": {
            "layers": [
                int(value)
                for value in flat_config.get("fusion_layers", DEFAULT_FUSION_LAYERS)
            ],
            "output_dim": int(flat_config.get("fusion_output_dim", DEFAULT_FUSION_OUTPUT_DIM)),
            "dropout": float(flat_config.get("fusion_dropout", 0.1)),
            "gating": str(flat_config.get("fusion_gating", "softmax")),
        },
        "ood": {
            "threshold_factor": float(flat_config.get("ood_threshold_factor", 2.0)),
            "primary_score_method": normalize_requested_primary_score_method(
                flat_config.get("primary_score_method", "auto")
            ),
        },
        "learning_rate": float(flat_config.get("learning_rate", 1e-4)),
        "weight_decay": float(flat_config.get("weight_decay", 0.01)),
        "num_epochs": int(flat_config.get("num_epochs", 10)),
        "batch_size": int(flat_config.get("batch_size", 8)),
        "device": str(flat_config.get("device", device)),
        "strict_model_loading": bool(flat_config.get("strict_model_loading", False)),
    }


def normalize_continual_training_config(
    training_continual: Optional[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_BACKBONE_MODEL_NAME,
    device: Any = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    normalized = deep_merge(
        _build_default_continual_surface(model_name=model_name, device=device),
        dict(training_continual or {}),
    )

    backbone = normalized.setdefault("backbone", {})
    adapter = normalized.setdefault("adapter", {})
    fusion = normalized.setdefault("fusion", {})
    ood = normalized.setdefault("ood", {})
    optimization = normalized.setdefault("optimization", {})
    scheduler = optimization.setdefault("scheduler", {})
    data = normalized.setdefault("data", {})
    early_stopping = normalized.setdefault("early_stopping", {})
    evaluation = normalized.setdefault("evaluation", {})

    backbone["model_name"] = str(backbone.get("model_name", model_name))

    adapter["target_modules_strategy"] = str(adapter.get("target_modules_strategy", "all_linear_transformer"))
    adapter["lora_r"] = int(adapter.get("lora_r", 16))
    adapter["lora_alpha"] = int(adapter.get("lora_alpha", 16))
    adapter["lora_dropout"] = float(adapter.get("lora_dropout", 0.1))

    raw_layers = fusion.get("layers", DEFAULT_FUSION_LAYERS)
    fusion["layers"] = [int(value) for value in list(raw_layers or DEFAULT_FUSION_LAYERS)]
    fusion["output_dim"] = int(fusion.get("output_dim", DEFAULT_FUSION_OUTPUT_DIM))
    fusion["dropout"] = float(fusion.get("dropout", 0.1))
    fusion["gating"] = str(fusion.get("gating", "softmax"))

    normalized["learning_rate"] = float(normalized.get("learning_rate", 1e-4))
    normalized["weight_decay"] = float(normalized.get("weight_decay", 0.01))
    normalized["num_epochs"] = int(normalized.get("num_epochs", 10))
    normalized["batch_size"] = int(normalized.get("batch_size", 8))
    normalized["device"] = str(normalized.get("device", device))
    normalized["strict_model_loading"] = bool(normalized.get("strict_model_loading", False))
    normalized["seed"] = int(normalized.get("seed", 42))
    normalized["deterministic"] = bool(normalized.get("deterministic", False))

    ood["threshold_factor"] = float(ood.get("threshold_factor", 2.0))
    ood["primary_score_method"] = normalize_requested_primary_score_method(ood.get("primary_score_method", "auto"))
    ood["ber_enabled"] = bool(ood.get("ber_enabled", False))
    ood["ber_lambda_old"] = float(ood.get("ber_lambda_old", 0.1))
    ood["ber_lambda_new"] = float(ood.get("ber_lambda_new", 0.1))
    ood["ber_warmup_steps"] = int(ood.get("ber_warmup_steps", 50))
    ood["energy_temperature_mode"] = str(ood.get("energy_temperature_mode", "fixed"))
    ood["energy_temperature"] = float(ood.get("energy_temperature", 1.0))
    raw_energy_range = list(ood.get("energy_temperature_range", [0.5, 3.0]))
    if len(raw_energy_range) < 2:
        raw_energy_range = [0.5, 3.0]
    ood["energy_temperature_range"] = [float(raw_energy_range[0]), float(raw_energy_range[1])]
    ood["energy_temperature_steps"] = int(ood.get("energy_temperature_steps", 16))
    ood["radial_l2_enabled"] = bool(ood.get("radial_l2_enabled", True))
    raw_beta_range = list(ood.get("radial_beta_range", [0.5, 2.0]))
    if len(raw_beta_range) < 2:
        raw_beta_range = [0.5, 2.0]
    ood["radial_beta_range"] = [float(raw_beta_range[0]), float(raw_beta_range[1])]
    ood["radial_beta_steps"] = int(ood.get("radial_beta_steps", 16))
    ood["knn_backend"] = str(ood.get("knn_backend", "auto"))
    ood["knn_chunk_size"] = int(ood.get("knn_chunk_size", 2048))
    ood["sure_enabled"] = bool(ood.get("sure_enabled", True))
    ood["sure_semantic_percentile"] = float(ood.get("sure_semantic_percentile", 95.0))
    ood["sure_confidence_percentile"] = float(ood.get("sure_confidence_percentile", 90.0))
    ood["conformal_enabled"] = bool(ood.get("conformal_enabled", True))
    ood["conformal_alpha"] = float(ood.get("conformal_alpha", 0.05))
    ood["conformal_method"] = str(ood.get("conformal_method", "threshold"))
    ood["conformal_raps_lambda"] = float(ood.get("conformal_raps_lambda", 0.0))
    ood["conformal_raps_k_reg"] = int(ood.get("conformal_raps_k_reg", 1))

    optimization["grad_accumulation_steps"] = int(optimization.get("grad_accumulation_steps", 4))
    optimization["max_grad_norm"] = float(optimization.get("max_grad_norm", 1.0))
    optimization["mixed_precision"] = str(optimization.get("mixed_precision", "auto"))
    optimization["label_smoothing"] = float(optimization.get("label_smoothing", 0.0))

    scheduler["name"] = str(scheduler.get("name", "cosine"))
    scheduler["warmup_ratio"] = float(scheduler.get("warmup_ratio", 0.1))
    scheduler["min_lr"] = float(scheduler.get("min_lr", 1e-6))
    scheduler["step_on"] = str(scheduler.get("step_on", "batch"))

    data["sampler"] = str(data.get("sampler", "shuffle"))
    data["loader_error_policy"] = str(data.get("loader_error_policy", "tolerant"))
    data["target_size"] = int(data.get("target_size", 224))
    data["cache_size"] = int(data.get("cache_size", 1000))
    data["validate_images_on_init"] = bool(data.get("validate_images_on_init", True))

    evaluation_best_metric = str(evaluation.get("best_metric", early_stopping.get("metric", "val_loss")))
    evaluation["best_metric"] = evaluation_best_metric
    evaluation["emit_ood_gate"] = bool(evaluation.get("emit_ood_gate", True))
    evaluation["require_ood_for_gate"] = bool(evaluation.get("require_ood_for_gate", True))
    evaluation["ood_fallback_strategy"] = str(evaluation.get("ood_fallback_strategy", "held_out_benchmark"))
    evaluation["ood_benchmark_auto_run"] = bool(evaluation.get("ood_benchmark_auto_run", True))
    evaluation["ood_benchmark_min_classes"] = int(evaluation.get("ood_benchmark_min_classes", 3))

    early_metric = str(early_stopping.get("metric", evaluation_best_metric))
    inferred_mode = "min" if early_metric in {"val_loss", "generalization_gap"} else "max"
    early_stopping["enabled"] = bool(early_stopping.get("enabled", True))
    early_stopping["metric"] = early_metric
    early_stopping["mode"] = str(early_stopping.get("mode", inferred_mode))
    early_stopping["patience"] = int(early_stopping.get("patience", 5))
    early_stopping["min_delta"] = float(early_stopping.get("min_delta", 0.0))

    return normalized


def extract_continual_training_config(
    config: Optional[Dict[str, Any]],
    *,
    model_name: str = DEFAULT_BACKBONE_MODEL_NAME,
    device: Any = DEFAULT_DEVICE,
) -> Dict[str, Any]:
    payload = dict(config or {})
    training = payload.get("training")
    if isinstance(training, dict) and isinstance(training.get("continual"), dict):
        source = dict(training.get("continual", {}))
    elif any(
        key in payload
        for key in ("backbone", "adapter", "fusion", "ood", "optimization", "data", "early_stopping", "evaluation")
    ):
        source = payload
    else:
        source = _coerce_legacy_flat_config(payload, model_name=model_name, device=device)
    return normalize_continual_training_config(source, model_name=model_name, device=device)
