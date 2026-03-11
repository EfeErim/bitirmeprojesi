#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import torch
import torch.nn as nn

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in some local sandboxes
    np = None  # type: ignore[assignment]

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features
from src.ood.continual_ood import ContinualOODDetector
from src.training.ber_loss import BERLoss
from src.training.quantization import assert_no_prohibited_4bit_flags
from src.training.services.config_surface import (
    DEFAULT_BACKBONE_MODEL_NAME,
    normalize_continual_training_config,
)
from src.training.services.ood_calibration import calibrate_trainer_ood
from src.training.services.persistence import (
    build_trainer_metadata_payload,
    compute_config_hash,
    load_trainer_adapter,
    save_trainer_adapter,
)
from src.training.services.persistence import (
    restore_training_state as restore_trainer_training_state,
)
from src.training.services.persistence import (
    snapshot_training_state as snapshot_trainer_training_state,
)
from src.training.services.runtime import (
    autocast_context,
    build_grad_scaler,
    build_idx_to_class,
    compute_grad_norm,
    configure_runtime_reproducibility,
    configure_training_plan_state,
    ensure_scheduler,
    resolve_runtime_device,
    step_scheduler,
)
from src.training.services.runtime import (
    setup_optimizer as setup_trainer_optimizer,
)
from src.training.services.trainer_runtime import (
    add_trainer_classes,
    execute_train_batch,
    flush_pending_gradients,
    has_pending_gradients,
    initialize_trainer_engine,
    predict_with_ood_result,
    refresh_optimizer_after_model_change,
)
from src.training.types import TrainBatchStats, TrainingCheckpointPayload

AUTO_MODEL_FACTORY: Any = None
try:
    from transformers import AutoModel as _loaded_auto_model

    AUTO_MODEL_FACTORY = _loaded_auto_model
except Exception:  # pragma: no cover - test fallback
    pass

# Preserve the historical module-level names so tests and callers can monkeypatch
# dependency injection points directly.
AutoModel = AUTO_MODEL_FACTORY

PEFT_LORA_CONFIG: Any = None
PEFT_MODEL_CLASS: Any = None
PEFT_GET_MODEL: Any = None
try:
    from peft import LoraConfig as _loaded_lora_config
    from peft import PeftModel as _loaded_peft_model
    from peft import get_peft_model as _loaded_peft_get_model

    PEFT_LORA_CONFIG = _loaded_lora_config
    PEFT_MODEL_CLASS = _loaded_peft_model
    PEFT_GET_MODEL = _loaded_peft_get_model
except Exception:  # pragma: no cover - test fallback
    pass

LoraConfig = PEFT_LORA_CONFIG
PeftModel = PEFT_MODEL_CLASS
get_peft_model = PEFT_GET_MODEL


logger = logging.getLogger(__name__)

EXCLUDED_TARGET_TOKEN = ("classifier", "router", "head", "pooler")
PREFERRED_TARGET_TOKEN = (
    "transformer",
    "encoder",
    "block",
    "layer",
    "attention",
    "attn",
    "mlp",
    "ffn",
)
_CONFIG_EXTRA_EXCLUDED_KEYS = {
    "backbone",
    "adapter",
    "fusion",
    "ood",
    "optimization",
    "early_stopping",
    "evaluation",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "batch_size",
    "device",
    "strict_model_loading",
    "seed",
    "deterministic",
}
_META_TENSOR_COPY_ERROR = "Cannot copy out of meta tensor; no data!"


def _resolve_radial_beta_range(ood: Dict[str, Any]) -> tuple[float, float]:
    raw_range = ood.get("radial_beta_range", [0.5, 2.0])
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) < 2:
        raw_range = [0.5, 2.0]
    return float(raw_range[0]), float(raw_range[1])


def _resolve_early_stopping_surface(
    early_stopping: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> tuple[str, str]:
    metric = str(early_stopping.get("metric", evaluation.get("best_metric", "val_loss")))
    mode = str(
        early_stopping.get(
            "mode",
            "min" if metric in {"val_loss", "generalization_gap"} else "max",
        )
    )
    return metric, mode


def _collect_extra_training_fields(normalized: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in normalized.items()
        if key not in _CONFIG_EXTRA_EXCLUDED_KEYS
    }


@dataclass
class ContinualSDLoRAConfig:
    """Runtime configuration for v6 continual SD-LoRA training."""

    backbone_model_name: str = DEFAULT_BACKBONE_MODEL_NAME
    target_modules_strategy: str = "all_linear_transformer"
    fusion_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    fusion_output_dim: int = 768
    fusion_dropout: float = 0.1
    fusion_gating: str = "softmax"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 8
    device: str = "cuda"
    strict_model_loading: bool = False
    ood_threshold_factor: float = 2.0
    ood_primary_score_method: str = "ensemble"
    seed: int = 42
    deterministic: bool = True
    grad_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str = "auto"
    label_smoothing: float = 0.0
    scheduler_name: str = "cosine"
    scheduler_warmup_ratio: float = 0.1
    scheduler_min_lr: float = 1e-6
    scheduler_step_on: str = "batch"
    early_stopping_enabled: bool = True
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    evaluation_best_metric: str = "val_loss"
    evaluation_emit_ood_gate: bool = True
    evaluation_require_ood_for_gate: bool = True
    evaluation_ood_fallback_strategy: str = "held_out_benchmark"
    evaluation_ood_benchmark_auto_run: bool = True
    evaluation_ood_benchmark_min_classes: int = 3
    # --- Bi-directional Energy Regularization (BER) ---
    ber_enabled: bool = False
    ber_lambda_old: float = 0.1
    ber_lambda_new: float = 0.1
    ber_warmup_steps: int = 50
    # --- Radially Scaled L2 Normalization ---
    radial_l2_enabled: bool = True
    radial_beta_min: float = 0.5
    radial_beta_max: float = 2.0
    radial_beta_steps: int = 16
    # --- SURE+ Double Scoring ---
    sure_enabled: bool = True
    sure_semantic_percentile: float = 95.0
    sure_confidence_percentile: float = 90.0
    # --- Conformal Prediction ---
    conformal_enabled: bool = True
    conformal_alpha: float = 0.05
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.target_modules_strategy != "all_linear_transformer":
            raise ValueError("v6 requires target_modules_strategy='all_linear_transformer'.")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive.")
        if not self.fusion_layers:
            raise ValueError("fusion_layers must not be empty.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        if self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be non-negative.")
        if self.label_smoothing < 0.0:
            raise ValueError("label_smoothing must be non-negative.")
        if self.mixed_precision not in {"off", "auto", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: off, auto, fp16, bf16.")
        if self.scheduler_name not in {"none", "linear", "cosine"}:
            raise ValueError("scheduler.name must be one of: none, linear, cosine.")
        if self.scheduler_step_on not in {"batch", "epoch"}:
            raise ValueError("scheduler.step_on must be 'batch' or 'epoch'.")
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping.patience must be non-negative.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping.min_delta must be non-negative.")
        if self.evaluation_ood_fallback_strategy not in {"held_out_benchmark", "none"}:
            raise ValueError("evaluation_ood_fallback_strategy must be 'held_out_benchmark' or 'none'.")
        if self.evaluation_ood_benchmark_min_classes < 1:
            raise ValueError("evaluation_ood_benchmark_min_classes must be at least 1.")
        if self.ood_primary_score_method not in {"ensemble", "energy", "knn"}:
            raise ValueError("ood.primary_score_method must be one of: ensemble, energy, knn.")
        if self.ber_lambda_old < 0.0 or self.ber_lambda_new < 0.0:
            raise ValueError("BER lambda values must be non-negative.")
        if self.ber_warmup_steps < 0:
            raise ValueError("ber_warmup_steps must be non-negative.")
        if self.radial_beta_min <= 0.0 or self.radial_beta_max <= 0.0:
            raise ValueError("Radial beta range values must be positive.")
        if self.radial_beta_min >= self.radial_beta_max:
            raise ValueError("radial_beta_min must be less than radial_beta_max.")
        if self.radial_beta_steps < 2:
            raise ValueError("radial_beta_steps must be at least 2.")
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if not (0.0 < self.sure_semantic_percentile <= 100.0):
            raise ValueError("sure_semantic_percentile must be in (0, 100].")
        if not (0.0 < self.sure_confidence_percentile <= 100.0):
            raise ValueError("sure_confidence_percentile must be in (0, 100].")

    def as_contract_dict(self) -> Dict[str, Any]:
        """Return normalized config payload used in metadata persistence."""
        payload = {
            "backbone": {"model_name": self.backbone_model_name},
            "adapter": {
                "target_modules_strategy": self.target_modules_strategy,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "fusion": {
                "layers": self.fusion_layers,
                "output_dim": self.fusion_output_dim,
                "dropout": self.fusion_dropout,
                "gating": self.fusion_gating,
            },
            "ood": {
                "threshold_factor": self.ood_threshold_factor,
                "primary_score_method": self.ood_primary_score_method,
                "ber_enabled": self.ber_enabled,
                "ber_lambda_old": self.ber_lambda_old,
                "ber_lambda_new": self.ber_lambda_new,
                "ber_warmup_steps": self.ber_warmup_steps,
                "radial_l2_enabled": self.radial_l2_enabled,
                "radial_beta_range": [self.radial_beta_min, self.radial_beta_max],
                "radial_beta_steps": self.radial_beta_steps,
                "sure_enabled": self.sure_enabled,
                "sure_semantic_percentile": self.sure_semantic_percentile,
                "sure_confidence_percentile": self.sure_confidence_percentile,
                "conformal_enabled": self.conformal_enabled,
                "conformal_alpha": self.conformal_alpha,
            },
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "strict_model_loading": self.strict_model_loading,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "optimization": {
                "grad_accumulation_steps": self.grad_accumulation_steps,
                "max_grad_norm": self.max_grad_norm,
                "mixed_precision": self.mixed_precision,
                "label_smoothing": self.label_smoothing,
                "scheduler": {
                    "name": self.scheduler_name,
                    "warmup_ratio": self.scheduler_warmup_ratio,
                    "min_lr": self.scheduler_min_lr,
                    "step_on": self.scheduler_step_on,
                },
            },
            "early_stopping": {
                "enabled": self.early_stopping_enabled,
                "metric": self.early_stopping_metric,
                "mode": self.early_stopping_mode,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
            },
            "evaluation": {
                "best_metric": self.evaluation_best_metric,
                "emit_ood_gate": self.evaluation_emit_ood_gate,
                "require_ood_for_gate": self.evaluation_require_ood_for_gate,
                "ood_fallback_strategy": self.evaluation_ood_fallback_strategy,
                "ood_benchmark_auto_run": self.evaluation_ood_benchmark_auto_run,
                "ood_benchmark_min_classes": self.evaluation_ood_benchmark_min_classes,
            },
        }
        if self.extra:
            payload.update(dict(self.extra))
        return payload

    @classmethod
    def from_training_config(cls, training_continual: Dict[str, Any]) -> "ContinualSDLoRAConfig":
        """Build from `training.continual` dictionary."""
        normalized = normalize_continual_training_config(
            training_continual,
            model_name=str(training_continual.get("backbone", {}).get("model_name", DEFAULT_BACKBONE_MODEL_NAME)),
            device=training_continual.get("device", "cuda"),
        )
        assert_no_prohibited_4bit_flags(normalized)
        backbone = normalized.get("backbone", {})
        adapter = normalized.get("adapter", {})
        fusion = normalized.get("fusion", {})
        ood = normalized.get("ood", {})
        optimization = normalized.get("optimization", {})
        scheduler = optimization.get("scheduler", {}) if isinstance(optimization, dict) else {}
        early_stopping = normalized.get("early_stopping", {})
        evaluation = normalized.get("evaluation", {})
        radial_beta_min, radial_beta_max = _resolve_radial_beta_range(ood)
        early_stopping_metric, early_stopping_mode = _resolve_early_stopping_surface(
            early_stopping,
            evaluation,
        )

        config = cls(
            backbone_model_name=str(backbone.get("model_name", DEFAULT_BACKBONE_MODEL_NAME)),
            target_modules_strategy=str(adapter.get("target_modules_strategy", "all_linear_transformer")),
            fusion_layers=[int(v) for v in fusion.get("layers", [2, 5, 8, 11])],
            fusion_output_dim=int(fusion.get("output_dim", 768)),
            fusion_dropout=float(fusion.get("dropout", 0.1)),
            fusion_gating=str(fusion.get("gating", "softmax")),
            lora_r=int(adapter.get("lora_r", 16)),
            lora_alpha=int(adapter.get("lora_alpha", 16)),
            lora_dropout=float(adapter.get("lora_dropout", 0.1)),
            learning_rate=float(normalized.get("learning_rate", 1e-4)),
            weight_decay=float(normalized.get("weight_decay", 0.01)),
            num_epochs=int(normalized.get("num_epochs", 10)),
            batch_size=int(normalized.get("batch_size", 8)),
            device=str(normalized.get("device", "cuda")),
            strict_model_loading=bool(normalized.get("strict_model_loading", False)),
            ood_threshold_factor=float(ood.get("threshold_factor", 2.0)),
            ood_primary_score_method=str(ood.get("primary_score_method", "ensemble")).strip().lower() or "ensemble",
            ber_enabled=bool(ood.get("ber_enabled", False)),
            ber_lambda_old=float(ood.get("ber_lambda_old", 0.1)),
            ber_lambda_new=float(ood.get("ber_lambda_new", 0.1)),
            ber_warmup_steps=int(ood.get("ber_warmup_steps", 50)),
            radial_l2_enabled=bool(ood.get("radial_l2_enabled", True)),
            radial_beta_min=radial_beta_min,
            radial_beta_max=radial_beta_max,
            radial_beta_steps=int(ood.get("radial_beta_steps", 16)),
            sure_enabled=bool(ood.get("sure_enabled", True)),
            sure_semantic_percentile=float(ood.get("sure_semantic_percentile", 95.0)),
            sure_confidence_percentile=float(ood.get("sure_confidence_percentile", 90.0)),
            conformal_enabled=bool(ood.get("conformal_enabled", True)),
            conformal_alpha=float(ood.get("conformal_alpha", 0.05)),
            seed=int(normalized.get("seed", 42)),
            deterministic=bool(normalized.get("deterministic", True)),
            grad_accumulation_steps=int(optimization.get("grad_accumulation_steps", 4)),
            max_grad_norm=float(optimization.get("max_grad_norm", 1.0)),
            mixed_precision=str(optimization.get("mixed_precision", "auto")),
            label_smoothing=float(optimization.get("label_smoothing", 0.0)),
            scheduler_name=str(scheduler.get("name", "cosine")),
            scheduler_warmup_ratio=float(scheduler.get("warmup_ratio", 0.1)),
            scheduler_min_lr=float(scheduler.get("min_lr", 1e-6)),
            scheduler_step_on=str(scheduler.get("step_on", "batch")),
            early_stopping_enabled=bool(early_stopping.get("enabled", True)),
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=int(early_stopping.get("patience", 5)),
            early_stopping_min_delta=float(early_stopping.get("min_delta", 0.0)),
            evaluation_best_metric=str(evaluation.get("best_metric", "val_loss")),
            evaluation_emit_ood_gate=bool(evaluation.get("emit_ood_gate", True)),
            evaluation_require_ood_for_gate=bool(evaluation.get("require_ood_for_gate", True)),
            evaluation_ood_fallback_strategy=str(evaluation.get("ood_fallback_strategy", "held_out_benchmark")),
            evaluation_ood_benchmark_auto_run=bool(evaluation.get("ood_benchmark_auto_run", True)),
            evaluation_ood_benchmark_min_classes=int(evaluation.get("ood_benchmark_min_classes", 3)),
            extra=_collect_extra_training_fields(normalized),
        )
        config.validate()
        return config


class ContinualSDLoRATrainer:
    """Single-engine continual SD-LoRA trainer for v6 runtime."""

    def __init__(self, config: ContinualSDLoRAConfig):
        self.config = config
        self.config.validate()

        self.device = resolve_runtime_device(self.config.device)
        self.backbone: Optional[nn.Module] = None
        self.adapter_model: Optional[nn.Module] = None
        self.classifier: Optional[nn.Linear] = None
        self.fusion: Optional[MultiScaleFeatureFusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.current_epoch = 0
        self.optimizer_steps = 0
        self.class_to_idx: Dict[str, int] = {}
        self.target_modules_resolved: List[str] = []
        self.ood_detector = ContinualOODDetector(
            threshold_factor=self.config.ood_threshold_factor,
            primary_score_method=self.config.ood_primary_score_method,
            radial_l2_enabled=self.config.radial_l2_enabled,
            radial_beta_range=(self.config.radial_beta_min, self.config.radial_beta_max),
            radial_beta_steps=self.config.radial_beta_steps,
            sure_enabled=self.config.sure_enabled,
            sure_semantic_percentile=self.config.sure_semantic_percentile,
            sure_confidence_percentile=self.config.sure_confidence_percentile,
            conformal_enabled=self.config.conformal_enabled,
            conformal_alpha=self.config.conformal_alpha,
        )
        self.ber_loss: Optional[BERLoss] = None
        self._is_initialized = False
        self._contract = self.config.as_contract_dict()
        self._config_hash = compute_config_hash(self._contract)
        self._peft_available = LoraConfig is not None
        self._adapter_wrapped = False
        self._planned_scheduler_steps = 0
        self._planned_epochs = int(max(1, self.config.num_epochs))
        self._accumulation_counter = 0
        self._last_grad_norm = 0.0
        self.best_metric_state: Dict[str, Any] = {}
        self._idx_to_class: Dict[int, str] = {}
        self._trainable_params_cache: Optional[List[torch.nn.Parameter]] = None
        self._ood_calibration_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        configure_runtime_reproducibility(self.config, np_module=np)
        self._refresh_class_index_cache()

    def _refresh_class_index_cache(self) -> None:
        self._idx_to_class = build_idx_to_class(self.class_to_idx)

    def _class_index_cache_stale(self) -> bool:
        if len(self._idx_to_class) != len(self.class_to_idx):
            return True
        return any(self._idx_to_class.get(idx) != name for name, idx in self.class_to_idx.items())

    def _refresh_optimizer_after_model_change(self) -> None:
        refresh_optimizer_after_model_change(self)

    def _reported_grad_norm(self, *, gradients_unscaled: bool) -> float:
        del gradients_unscaled
        # Non-step microbatches only need a stable telemetry value. Reuse the
        # last measured post-step norm instead of scanning all gradients again.
        return float(getattr(self, "_last_grad_norm", 0.0))

    def set_preferred_ood_calibration_loader(self, loader: Iterable[Dict[str, torch.Tensor]]) -> None:
        self._ood_calibration_loader = loader

    def _ensure_ood_calibrated(self, *, operation: str) -> None:
        issue = self.ood_detector.calibration_issue()
        if issue is None:
            return
        if self._ood_calibration_loader is not None:
            self.calibrate_ood(self._ood_calibration_loader)
            issue = self.ood_detector.calibration_issue()
            if issue is None:
                return
            raise RuntimeError(
                f"{issue} Automatic OOD calibration before {operation} did not produce usable class statistics."
            )
        raise RuntimeError(
            f"{issue} No calibration loader is available for automatic OOD calibration before {operation}."
        )

    def configure_training_plan(self, *, total_batches: int, num_epochs: Optional[int] = None) -> None:
        configure_training_plan_state(self, total_batches=total_batches, num_epochs=num_epochs)

    def _ensure_scheduler(self) -> None:
        ensure_scheduler(self)

    def _step_scheduler(self) -> None:
        step_scheduler(self)

    def initialize_engine(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        initialize_trainer_engine(
            self,
            class_to_idx=class_to_idx,
            auto_model_factory=AutoModel,
            fusion_cls=MultiScaleFeatureFusion,
        )

    def _raise_missing_peft(self) -> None:
        message = (
            "peft is required for SD-LoRA adapter wrapping but is not available. "
            "Install a compatible `peft` package and retry."
        )
        logger.error(message)
        raise RuntimeError(message)

    @staticmethod
    def _module_has_meta_tensors(module: nn.Module) -> bool:
        return any(param.is_meta for param in module.parameters()) or any(buffer.is_meta for buffer in module.buffers())

    @staticmethod
    def _is_dispatch_managed_module(module: nn.Module) -> bool:
        if getattr(module, "hf_device_map", None) is not None:
            return True
        if hasattr(module, "_hf_hook"):
            return True
        for child in module.modules():
            if child is module:
                continue
            if hasattr(child, "_hf_hook"):
                return True
        return False

    def _prepare_module_for_device(self, module: nn.Module, module_name: str) -> nn.Module:
        has_meta_tensors = self._module_has_meta_tensors(module)
        dispatch_managed = self._is_dispatch_managed_module(module)

        if has_meta_tensors and dispatch_managed:
            logger.warning(
                "%s has meta tensors with HF/Accelerate dispatch hooks; skipping explicit .to(%s).",
                module_name,
                self.device,
            )
            return module

        if has_meta_tensors:
            raise RuntimeError(
                f"{module_name} contains meta tensors before device move and is not dispatch-managed. "
                "Disable meta/offload loading or materialize weights before initialize_engine()."
            )

        try:
            return module.to(self.device)
        except NotImplementedError as exc:
            if _META_TENSOR_COPY_ERROR in str(exc) and self._is_dispatch_managed_module(module):
                logger.warning(
                    "%s raised meta tensor NotImplementedError during .to(%s) but is dispatch-managed; "
                    "leaving module placement unchanged.",
                    module_name,
                    self.device,
                )
                return module
            raise

    @staticmethod
    def _supports_low_cpu_mem_usage_kwarg() -> bool:
        try:
            signature = inspect.signature(get_peft_model)
        except (TypeError, ValueError):
            return False

        if "low_cpu_mem_usage" in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    def _apply_lora(self, model: nn.Module, target_modules: Sequence[str]) -> nn.Module:
        if LoraConfig is None or get_peft_model is None:
            self._raise_missing_peft()

        suffixes = sorted({name.split(".")[-1] for name in target_modules})
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=suffixes,
            bias="none",
        )

        supports_low_cpu_mem_usage = self._supports_low_cpu_mem_usage_kwarg()
        if supports_low_cpu_mem_usage:
            wrapped = cast(nn.Module, get_peft_model(model, lora_config, low_cpu_mem_usage=False))
        else:
            wrapped = cast(nn.Module, get_peft_model(model, lora_config))

        for name, param in wrapped.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
        self._adapter_wrapped = True
        return wrapped

    def resolve_target_modules(self, model: nn.Module) -> List[str]:
        """Resolve all transformer linear modules excluding classifier/router heads."""
        preferred: List[str] = []
        fallback: List[str] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            lowered = module_name.lower()
            if any(token in lowered for token in EXCLUDED_TARGET_TOKEN):
                continue
            fallback.append(module_name)
            if any(token in lowered for token in PREFERRED_TARGET_TOKEN):
                preferred.append(module_name)

        resolved = sorted(set(preferred or fallback))
        if not resolved:
            raise RuntimeError("No linear target modules resolved for all_linear_transformer strategy.")
        return resolved

    def add_classes(self, new_class_names: Iterable[str]) -> Dict[str, int]:
        return add_trainer_classes(self, new_class_names)

    def setup_optimizer(self) -> None:
        setup_trainer_optimizer(self)

    def _extract_hidden_states(self, images: torch.Tensor) -> Sequence[torch.Tensor]:
        if self.adapter_model is None:
            raise RuntimeError("Adapter model not initialized.")

        output = self.adapter_model(images, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            return list(hidden_states)

        if hasattr(output, "last_hidden_state"):
            return [output.last_hidden_state]
        if torch.is_tensor(output):
            return [output]
        raise RuntimeError(
            "Backbone output does not expose hidden states, last_hidden_state, or tensor output. "
            "Cannot derive feature representations for fusion."
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.fusion is None:
            raise RuntimeError("Fusion module is not initialized.")
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)
        states = self._extract_hidden_states(images)
        selected = select_multiscale_features(states, self.config.fusion_layers)
        return self.fusion(selected)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized.")
        features = self.encode(images)
        return self.classifier(features)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
        images = batch["images"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with autocast_context(self.device, self.config.mixed_precision):
            logits = self.forward_logits(images)
            if self.ber_loss is not None:
                total_loss, self._last_ber_components = self.ber_loss(
                    logits,
                    labels,
                    label_smoothing=float(self.config.label_smoothing),
                )
                return total_loss
            return nn.functional.cross_entropy(
                logits,
                labels,
                label_smoothing=float(self.config.label_smoothing),
            )

    def set_train_mode(self) -> None:
        self._set_module_training_mode(training=True)

    def set_eval_mode(self) -> None:
        self._set_module_training_mode(training=False)

    def _set_module_training_mode(self, *, training: bool) -> None:
        for module in (self.adapter_model, self.classifier, self.fusion):
            if module is not None:
                module.train(training)

    def _compute_grad_norm(self) -> float:
        return compute_grad_norm(self.optimizer)

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> TrainBatchStats:
        return execute_train_batch(self, batch)

    def has_pending_gradients(self) -> bool:
        return has_pending_gradients(self)

    def flush_pending_gradients(self) -> Optional[float]:
        return flush_pending_gradients(self)

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self._ood_calibration_loader = loader
        return calibrate_trainer_ood(self, loader)

    def predict_with_ood(self, images: torch.Tensor) -> Dict[str, Any]:
        return predict_with_ood_result(self, images)

    def _metadata_payload(self) -> Dict[str, Any]:
        return build_trainer_metadata_payload(self)

    def snapshot_training_state(self) -> TrainingCheckpointPayload:
        return snapshot_trainer_training_state(self, np_module=np)

    def restore_training_state(self, payload: TrainingCheckpointPayload | Dict[str, Any]) -> TrainingCheckpointPayload:
        return restore_trainer_training_state(self, payload, np_module=np)

    def save_adapter(self, output_dir: str) -> Path:
        self._ensure_ood_calibrated(operation="save_adapter()")
        return save_trainer_adapter(self, output_dir)

    def load_adapter(self, adapter_dir: str) -> Dict[str, Any]:
        return load_trainer_adapter(self, adapter_dir, peft_model_cls=PeftModel)
