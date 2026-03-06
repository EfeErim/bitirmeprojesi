#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import inspect
import json
import logging
import math
import random
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in some local sandboxes
    np = None  # type: ignore[assignment]

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features
from src.ood.continual_ood import ClassCalibration, ContinualOODDetector
from src.training.quantization import assert_no_prohibited_4bit_flags
from src.training.types import TrainBatchStats, TrainingCheckpointPayload

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover - test fallback
    AutoModel = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover - test fallback
    LoraConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]

    def get_peft_model(model: nn.Module, _cfg: Any) -> nn.Module:  # type: ignore[no-redef]
        return model


logger = logging.getLogger(__name__)

_DEFAULT_PLAN_TARGETS = {
    "accuracy": 0.93,
    "ood_auroc": 0.92,
    "ood_false_positive_rate": 0.05,
}

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


@dataclass
class ContinualSDLoRAConfig:
    """Runtime configuration for v6 continual SD-LoRA training."""

    backbone_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    target_modules_strategy: str = "all_linear_transformer"
    fusion_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    fusion_output_dim: int = 768
    fusion_dropout: float = 0.1
    fusion_gating: str = "softmax"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 1
    batch_size: int = 8
    device: str = "cuda"
    strict_model_loading: bool = False
    ood_threshold_factor: float = 2.0
    seed: int = 42
    deterministic: bool = False
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 0.0
    mixed_precision: str = "auto"
    label_smoothing: float = 0.0
    scheduler_name: str = "none"
    scheduler_warmup_ratio: float = 0.0
    scheduler_min_lr: float = 0.0
    scheduler_step_on: str = "batch"
    early_stopping_enabled: bool = False
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    evaluation_best_metric: str = "val_loss"
    evaluation_emit_ood_gate: bool = True
    evaluation_require_ood_for_gate: bool = False
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
            "ood": {"threshold_factor": self.ood_threshold_factor},
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
            },
        }
        if self.extra:
            payload.update(dict(self.extra))
        return payload

    @classmethod
    def from_training_config(cls, training_continual: Dict[str, Any]) -> "ContinualSDLoRAConfig":
        """Build from `training.continual` dictionary."""
        assert_no_prohibited_4bit_flags(training_continual)
        backbone = training_continual.get("backbone", {})
        adapter = training_continual.get("adapter", {})
        fusion = training_continual.get("fusion", {})
        ood = training_continual.get("ood", {})
        optimization = training_continual.get("optimization", {})
        scheduler = optimization.get("scheduler", {}) if isinstance(optimization, dict) else {}
        early_stopping = training_continual.get("early_stopping", {})
        evaluation = training_continual.get("evaluation", {})

        config = cls(
            backbone_model_name=str(backbone.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")),
            target_modules_strategy=str(adapter.get("target_modules_strategy", "all_linear_transformer")),
            fusion_layers=[int(v) for v in fusion.get("layers", [2, 5, 8, 11])],
            fusion_output_dim=int(fusion.get("output_dim", 768)),
            fusion_dropout=float(fusion.get("dropout", 0.1)),
            fusion_gating=str(fusion.get("gating", "softmax")),
            lora_r=int(adapter.get("lora_r", 16)),
            lora_alpha=int(adapter.get("lora_alpha", 32)),
            lora_dropout=float(adapter.get("lora_dropout", 0.1)),
            learning_rate=float(training_continual.get("learning_rate", 1e-4)),
            weight_decay=float(training_continual.get("weight_decay", 0.0)),
            num_epochs=int(training_continual.get("num_epochs", 1)),
            batch_size=int(training_continual.get("batch_size", 8)),
            device=str(training_continual.get("device", "cuda")),
            strict_model_loading=bool(training_continual.get("strict_model_loading", False)),
            ood_threshold_factor=float(ood.get("threshold_factor", 2.0)),
            seed=int(training_continual.get("seed", 42)),
            deterministic=bool(training_continual.get("deterministic", False)),
            grad_accumulation_steps=int(optimization.get("grad_accumulation_steps", 1)),
            max_grad_norm=float(optimization.get("max_grad_norm", 0.0)),
            mixed_precision=str(optimization.get("mixed_precision", "auto")),
            label_smoothing=float(optimization.get("label_smoothing", 0.0)),
            scheduler_name=str(scheduler.get("name", "none")),
            scheduler_warmup_ratio=float(scheduler.get("warmup_ratio", 0.0)),
            scheduler_min_lr=float(scheduler.get("min_lr", 0.0)),
            scheduler_step_on=str(scheduler.get("step_on", "batch")),
            early_stopping_enabled=bool(early_stopping.get("enabled", False)),
            early_stopping_metric=str(early_stopping.get("metric", evaluation.get("best_metric", "val_loss"))),
            early_stopping_mode=str(
                early_stopping.get(
                    "mode",
                    "min" if str(early_stopping.get("metric", evaluation.get("best_metric", "val_loss"))) in {"val_loss", "generalization_gap"} else "max",
                )
            ),
            early_stopping_patience=int(early_stopping.get("patience", 3)),
            early_stopping_min_delta=float(early_stopping.get("min_delta", 0.0)),
            evaluation_best_metric=str(evaluation.get("best_metric", "val_loss")),
            evaluation_emit_ood_gate=bool(evaluation.get("emit_ood_gate", True)),
            evaluation_require_ood_for_gate=bool(evaluation.get("require_ood_for_gate", False)),
            extra={k: v for k, v in training_continual.items() if k not in {
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
            }},
        )
        config.validate()
        return config


class ContinualSDLoRATrainer:
    """Single-engine continual SD-LoRA trainer for v6 runtime."""

    def __init__(self, config: ContinualSDLoRAConfig):
        self.config = config
        self.config.validate()

        self.device = torch.device(
            self.config.device if torch.cuda.is_available() and str(self.config.device).startswith("cuda") else "cpu"
        )
        self.backbone: Optional[nn.Module] = None
        self.adapter_model: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None
        self.fusion: Optional[MultiScaleFeatureFusion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.current_epoch = 0
        self.optimizer_steps = 0
        self.class_to_idx: Dict[str, int] = {}
        self.target_modules_resolved: List[str] = []
        self.ood_detector = ContinualOODDetector(threshold_factor=self.config.ood_threshold_factor)
        self._is_initialized = False
        self._contract = self.config.as_contract_dict()
        self._config_hash = self._compute_config_hash(self._contract)
        self._peft_available = LoraConfig is not None
        self._adapter_wrapped = False
        self._peft_warning_emitted = False
        self._planned_scheduler_steps = 0
        self._planned_epochs = int(max(1, self.config.num_epochs))
        self._accumulation_counter = 0
        self.best_metric_state: Dict[str, Any] = {}
        self._configure_runtime_reproducibility()

    @staticmethod
    def _compute_config_hash(contract: Dict[str, Any]) -> str:
        serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _configure_runtime_reproducibility(self) -> None:
        seed = int(self.config.seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if np is not None:
            np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.config.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def configure_training_plan(self, *, total_batches: int, num_epochs: Optional[int] = None) -> None:
        epochs = int(max(1, num_epochs if num_epochs is not None else self.config.num_epochs))
        self._planned_epochs = epochs
        optimizer_steps = max(
            1,
            math.ceil((max(1, int(total_batches)) * epochs) / max(1, int(self.config.grad_accumulation_steps))),
        )
        if self.config.scheduler_step_on == "epoch":
            optimizer_steps = epochs
        self._planned_scheduler_steps = int(optimizer_steps)
        if self.optimizer is not None:
            self._ensure_scheduler()

    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda":
            return None
        mode = str(self.config.mixed_precision).lower()
        if mode == "off":
            return None
        if mode == "bf16":
            return torch.bfloat16
        if mode == "fp16":
            return torch.float16
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16

    def _autocast_context(self) -> Any:
        dtype = self._resolve_amp_dtype()
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _amp_scaler_enabled(self) -> bool:
        return self._resolve_amp_dtype() == torch.float16 and self.device.type == "cuda"

    def _ensure_scheduler(self) -> None:
        if self.optimizer is None or self.scheduler is not None or self.config.scheduler_name == "none":
            return

        total_units = max(
            1,
            int(self._planned_scheduler_steps if self._planned_scheduler_steps > 0 else self._planned_epochs),
        )
        if self.config.scheduler_name == "linear":
            warmup_steps = int(max(0, round(total_units * float(self.config.scheduler_warmup_ratio))))
            min_lr_scale = (
                float(self.config.scheduler_min_lr) / float(self.config.learning_rate)
                if self.config.learning_rate > 0
                else 0.0
            )
            min_lr_scale = max(0.0, min(1.0, min_lr_scale))

            def _lr_lambda(step_idx: int) -> float:
                step = int(max(0, step_idx))
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                remaining = max(1, total_units - warmup_steps)
                progress = float(step - warmup_steps) / float(remaining)
                progress = max(0.0, min(1.0, progress))
                return max(min_lr_scale, 1.0 - progress * (1.0 - min_lr_scale))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
            return

        if self.config.scheduler_name == "cosine":
            eta_min = float(self.config.scheduler_min_lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_units,
                eta_min=eta_min,
            )

    def _step_scheduler(self) -> None:
        if self.scheduler is None:
            return
        self.scheduler.step()

    @staticmethod
    def load_plan_targets(spec_path: Optional[Path] = None) -> Dict[str, float]:
        """Load optional metric targets, falling back to hardcoded defaults."""
        if spec_path is None:
            return dict(_DEFAULT_PLAN_TARGETS)

        resolved = Path(spec_path)
        if not resolved.exists():
            return dict(_DEFAULT_PLAN_TARGETS)

        payload = json.loads(resolved.read_text(encoding="utf-8"))
        targets = payload.get("targets", {}) if isinstance(payload, dict) else {}
        return {
            "accuracy": float(targets.get("continual_accuracy", _DEFAULT_PLAN_TARGETS["accuracy"])),
            "ood_auroc": float(targets.get("ood_auroc", _DEFAULT_PLAN_TARGETS["ood_auroc"])),
            "ood_false_positive_rate": float(
                targets.get("ood_false_positive_rate", _DEFAULT_PLAN_TARGETS["ood_false_positive_rate"])
            ),
        }

    @staticmethod
    def compute_plan_metrics(
        *,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        ood_labels: Optional[Sequence[int]] = None,
        ood_scores: Optional[Sequence[float]] = None,
    ) -> Dict[str, Optional[float]]:
        """Compute plan metrics in a deterministic and serializable structure."""
        if len(y_true) == 0:
            raise ValueError("y_true must not be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        y_true_t = torch.tensor(list(y_true), dtype=torch.long)
        y_pred_t = torch.tensor(list(y_pred), dtype=torch.long)
        accuracy = float((y_true_t == y_pred_t).float().mean().item())

        ood_auroc: Optional[float] = None
        ood_fpr: Optional[float] = None
        ood_total: int = 0
        in_dist_total: int = 0
        if ood_labels is not None and ood_scores is not None:
            if len(ood_labels) != len(ood_scores):
                raise ValueError("ood_labels and ood_scores must have same length")
            if len(ood_labels) > 0:
                labels_t = torch.tensor(list(ood_labels), dtype=torch.long)
                scores_t = torch.tensor(list(ood_scores), dtype=torch.float32)
                ood_total = int((labels_t == 1).sum().item())
                in_dist_total = int((labels_t == 0).sum().item())
                if ood_total > 0 and in_dist_total > 0:
                    try:
                        ood_auroc = float(roc_auc_score(labels_t.cpu().numpy(), scores_t.cpu().numpy()))
                    except Exception:
                        ood_auroc = None
                    in_scores = scores_t[labels_t == 0]
                    if in_scores.numel() > 0:
                        threshold = torch.quantile(in_scores, 0.95)
                        ood_fpr = float((in_scores > threshold).float().mean().item())

        return {
            "accuracy": accuracy,
            "ood_auroc": ood_auroc,
            "ood_false_positive_rate": ood_fpr,
            "classification_samples": int(y_true_t.numel()),
            "ood_samples": ood_total,
            "in_distribution_samples": in_dist_total,
        }

    @classmethod
    def validate_plan_metrics(
        cls,
        metrics: Dict[str, Optional[float]],
        targets: Optional[Dict[str, float]] = None,
        *,
        require_ood: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate pass/fail against adapter plan targets with explicit gating conditions."""
        target_values = dict(targets or cls.load_plan_targets())

        checks: Dict[str, Dict[str, Any]] = {}

        acc_value = metrics.get("accuracy")
        checks["accuracy"] = {
            "value": acc_value,
            "target": target_values["accuracy"],
            "operator": ">=",
            "asserted": acc_value is not None,
            "passed": bool(acc_value is not None and float(acc_value) >= float(target_values["accuracy"])),
        }

        auroc_value = metrics.get("ood_auroc")
        checks["ood_auroc"] = {
            "value": auroc_value,
            "target": target_values["ood_auroc"],
            "operator": ">=",
            "asserted": auroc_value is not None,
            "passed": bool(auroc_value is not None and float(auroc_value) >= float(target_values["ood_auroc"])),
        }

        fpr_value = metrics.get("ood_false_positive_rate")
        checks["ood_false_positive_rate"] = {
            "value": fpr_value,
            "target": target_values["ood_false_positive_rate"],
            "operator": "<=",
            "asserted": fpr_value is not None,
            "passed": bool(fpr_value is not None and float(fpr_value) <= float(target_values["ood_false_positive_rate"])),
        }

        missing_checks = [name for name, detail in checks.items() if not detail["asserted"]]
        if require_ood:
            gating_status = "failed" if missing_checks else "ready"
            gating_reason = "missing_required_metrics" if missing_checks else "all_required_metrics_present"
        else:
            gating_status = "soft" if missing_checks else "ready"
            gating_reason = "missing_optional_metrics" if missing_checks else "all_metrics_present"

        all_asserted_passed = all(detail["passed"] for detail in checks.values() if detail["asserted"])
        hard_fail = require_ood and bool(missing_checks)
        passed = bool(all_asserted_passed and not hard_fail)

        return {
            "passed": passed,
            "require_ood": bool(require_ood),
            "targets": target_values,
            "checks": checks,
            "gating": {
                "status": gating_status,
                "reason": gating_reason,
                "missing_metrics": missing_checks,
            },
        }

    @classmethod
    def write_plan_metric_artifact(
        cls,
        *,
        output_path: Path,
        metrics: Dict[str, Optional[float]],
        targets: Optional[Dict[str, float]] = None,
        require_ood: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a deterministic metric artifact used for explicit gating."""
        evaluation = cls.validate_plan_metrics(metrics, targets, require_ood=require_ood)
        artifact = {
            "schema_version": "v6_plan_metric_gate",
            "metrics": metrics,
            "evaluation": evaluation,
            "context": dict(context or {}),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact

    def initialize_engine(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        """Load frozen backbone, apply adapters, initialize heads."""
        if class_to_idx:
            self.class_to_idx = dict(class_to_idx)

        if AutoModel is None:
            raise RuntimeError("transformers AutoModel is unavailable for continual trainer initialization.")

        loaded_backbone = AutoModel.from_pretrained(self.config.backbone_model_name)
        self.backbone = self._prepare_module_for_device(
            loaded_backbone,
            module_name="backbone",
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = int(getattr(getattr(self.backbone, "config", None), "hidden_size", self.config.fusion_output_dim))
        self.fusion = MultiScaleFeatureFusion(
            input_dim=hidden_size,
            output_dim=self.config.fusion_output_dim,
            num_scales=max(1, len(self.config.fusion_layers)),
            dropout=self.config.fusion_dropout,
            gating=self.config.fusion_gating,
        ).to(self.device)

        self.target_modules_resolved = self.resolve_target_modules(self.backbone)
        adapter_model = self._apply_lora(self.backbone, self.target_modules_resolved)
        self.adapter_model = self._prepare_module_for_device(
            adapter_model,
            module_name="adapter_model",
        )
        self.classifier = nn.Linear(self.config.fusion_output_dim, max(1, len(self.class_to_idx))).to(self.device)

        self._is_initialized = True
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_scaler_enabled())
        self.optimizer_steps = 0
        self._accumulation_counter = 0
        logger.info(
            "Continual engine initialized: backbone=%s, targets=%s",
            self.config.backbone_model_name,
            len(self.target_modules_resolved),
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
            if "Cannot copy out of meta tensor; no data!" in str(exc) and self._is_dispatch_managed_module(module):
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
        if LoraConfig is None:
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
            wrapped = get_peft_model(model, lora_config, low_cpu_mem_usage=False)
        else:
            wrapped = get_peft_model(model, lora_config)

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
        """Add new classes and expand classifier output."""
        for name in new_class_names:
            if name not in self.class_to_idx:
                self.class_to_idx[name] = len(self.class_to_idx)
        if self.classifier is None:
            return dict(self.class_to_idx)

        old_classifier = self.classifier
        old_out = int(getattr(old_classifier, "out_features", 0))
        new_out = max(1, len(self.class_to_idx))
        if new_out == old_out:
            return dict(self.class_to_idx)

        replacement = nn.Linear(old_classifier.in_features, new_out).to(self.device)
        if old_out > 0:
            replacement.weight.data[:old_out] = old_classifier.weight.data[:old_out]
            replacement.bias.data[:old_out] = old_classifier.bias.data[:old_out]
        self.classifier = replacement
        return dict(self.class_to_idx)

    def setup_optimizer(self) -> None:
        if not self._is_initialized or self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("initialize_engine() must be called before setup_optimizer().")
        trainable_params = [p for p in self.adapter_model.parameters() if p.requires_grad]
        trainable_params.extend([p for p in self.classifier.parameters() if p.requires_grad])
        trainable_params.extend([p for p in self.fusion.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_scaler_enabled())
        self._ensure_scheduler()
        self.optimizer.zero_grad(set_to_none=True)

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
        return [images]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.fusion is None:
            raise RuntimeError("Fusion module is not initialized.")
        images = images.to(self.device)
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
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)
        with self._autocast_context():
            logits = self.forward_logits(images)
            return nn.functional.cross_entropy(
                logits,
                labels,
                label_smoothing=float(self.config.label_smoothing),
            )

    def set_train_mode(self) -> None:
        if self.adapter_model is not None:
            self.adapter_model.train()
        if self.classifier is not None:
            self.classifier.train()
        if self.fusion is not None:
            self.fusion.train()

    def set_eval_mode(self) -> None:
        if self.adapter_model is not None:
            self.adapter_model.eval()
        if self.classifier is not None:
            self.classifier.eval()
        if self.fusion is not None:
            self.fusion.eval()

    def _compute_grad_norm(self) -> float:
        if self.optimizer is None:
            return 0.0
        total_norm_sq = 0.0
        has_grad = False
        for group in self.optimizer.param_groups:
            for param in group.get("params", []):
                if param is None or param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    grad = grad.coalesce().values()
                grad_norm = float(torch.norm(grad, p=2).item())
                total_norm_sq += grad_norm * grad_norm
                has_grad = True
        if not has_grad:
            return 0.0
        return float(total_norm_sq ** 0.5)

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> TrainBatchStats:
        if self.optimizer is None:
            self.setup_optimizer()
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
        self.set_train_mode()
        step_started_at = time.perf_counter()
        accumulation_steps = int(max(1, self.config.grad_accumulation_steps))
        if self._accumulation_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss = self.training_step(batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss encountered.")

        raw_loss_value = float(loss.detach().item())
        scaled_loss = loss / float(accumulation_steps)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accumulation_counter += 1
        should_step = self._accumulation_counter >= accumulation_steps
        grad_norm = self._compute_grad_norm()
        if should_step:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [param for group in self.optimizer.param_groups for param in group.get("params", []) if param is not None],
                    max_norm=float(self.config.max_grad_norm),
                )
            grad_norm = self._compute_grad_norm()
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer_steps += 1
            if self.config.scheduler_step_on == "batch":
                self._step_scheduler()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_counter = 0

        step_time_sec = float(max(1e-9, time.perf_counter() - step_started_at))
        batch_size = int(batch.get("labels", torch.empty(0)).shape[0]) if isinstance(batch, dict) else 0
        if batch_size <= 0 and isinstance(batch, dict) and "images" in batch:
            batch_size = int(batch["images"].shape[0])
        samples_per_sec = float(batch_size / step_time_sec) if batch_size > 0 else 0.0
        lr_value = float(self.optimizer.param_groups[0].get("lr", self.config.learning_rate))
        return TrainBatchStats(
            loss=raw_loss_value,
            lr=lr_value,
            grad_norm=float(grad_norm),
            step_time_sec=step_time_sec,
            samples_per_sec=samples_per_sec,
            batch_size=int(batch_size),
            accumulation_step=int(self._accumulation_counter if self._accumulation_counter > 0 else accumulation_steps),
            optimizer_steps=int(self.optimizer_steps),
            optimizer_step_applied=bool(should_step),
        )

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        feats: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        self.adapter_model.eval()  # type: ignore[union-attr]
        self.classifier.eval()  # type: ignore[union-attr]
        self.fusion.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                features = self.encode(images)
                logits = self.classifier(features)  # type: ignore[operator]
                feats.append(features.detach().cpu())
                logits_list.append(logits.detach().cpu())
                labels_list.append(labels.detach().cpu())

        if not feats:
            raise ValueError("Cannot calibrate OOD with an empty loader.")
        return self.ood_detector.calibrate(
            features=torch.cat(feats, dim=0),
            logits=torch.cat(logits_list, dim=0),
            labels=torch.cat(labels_list, dim=0),
        )

    def predict_with_ood(self, images: torch.Tensor) -> Dict[str, Any]:
        self.adapter_model.eval()  # type: ignore[union-attr]
        self.classifier.eval()  # type: ignore[union-attr]
        self.fusion.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            features = self.encode(images.to(self.device))
            logits = self.classifier(features)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1)
            confidence, indices = probs.max(dim=1)
            ood = self.ood_detector.score(features=features, logits=logits, predicted_labels=indices)

        idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        predicted_idx = int(indices[0].item()) if indices.numel() else 0
        return {
            "status": "success",
            "disease": {
                "class_index": predicted_idx,
                "name": idx_to_class.get(predicted_idx, str(predicted_idx)),
                "confidence": float(confidence[0].item()) if confidence.numel() else 0.0,
            },
            "ood_analysis": {
                "ensemble_score": float(ood["ensemble_score"][0].item()),
                "class_threshold": float(ood["class_threshold"][0].item()),
                "is_ood": bool(ood["is_ood"][0].item()),
                "mahalanobis_z": float(ood["mahalanobis_z"][0].item()),
                "energy_z": float(ood["energy_z"][0].item()),
                "calibration_version": int(ood["calibration_version"][0].item()),
            },
        }

    def _metadata_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": "v6",
            "engine": "continual_sd_lora",
            "trainer_config": self.config.as_contract_dict(),
            "config_hash": self._config_hash,
            "backbone": {
                "model_name": self.config.backbone_model_name,
                "frozen": True,
            },
            "fusion": {
                "layers": self.config.fusion_layers,
                "output_dim": self.config.fusion_output_dim,
                "dropout": self.config.fusion_dropout,
                "gating": self.config.fusion_gating,
            },
            "class_to_idx": self.class_to_idx,
            "ood_calibration": {"version": self.ood_detector.calibration_version},
            "ood_state": self._serialize_ood_state(),
            "target_modules_resolved": list(self.target_modules_resolved),
            "adapter_runtime": {
                "peft_available": bool(self._peft_available),
                "adapter_wrapped": bool(self._adapter_wrapped),
                "degraded_without_peft": bool(not self._adapter_wrapped),
            },
        }

    def _serialize_ood_state(self) -> Dict[str, Any]:
        class_stats: Dict[str, Any] = {}
        for class_id, stats in self.ood_detector.class_stats.items():
            class_stats[str(class_id)] = {
                "mean": stats.mean.detach().cpu().tolist(),
                "var": stats.var.detach().cpu().tolist(),
                "mahalanobis_mu": float(stats.mahalanobis_mu),
                "mahalanobis_sigma": float(stats.mahalanobis_sigma),
                "energy_mu": float(stats.energy_mu),
                "energy_sigma": float(stats.energy_sigma),
                "threshold": float(stats.threshold),
            }
        return {
            "threshold_factor": float(self.ood_detector.threshold_factor),
            "calibration_version": int(self.ood_detector.calibration_version),
            "class_stats": class_stats,
        }

    def _restore_ood_state(self, payload: Dict[str, Any]) -> None:
        threshold_factor = float(payload.get("threshold_factor", self.config.ood_threshold_factor))
        self.ood_detector = ContinualOODDetector(threshold_factor=threshold_factor)
        self.ood_detector.calibration_version = int(payload.get("calibration_version", 0))

        class_stats = payload.get("class_stats", {})
        if not isinstance(class_stats, dict):
            return
        for class_id_raw, stats in class_stats.items():
            if not isinstance(stats, dict):
                continue
            class_id = int(class_id_raw)
            self.ood_detector.class_stats[class_id] = ClassCalibration(
                mean=torch.tensor(stats.get("mean", []), dtype=torch.float32),
                var=torch.tensor(stats.get("var", []), dtype=torch.float32),
                mahalanobis_mu=float(stats.get("mahalanobis_mu", 0.0)),
                mahalanobis_sigma=float(stats.get("mahalanobis_sigma", 1.0)),
                energy_mu=float(stats.get("energy_mu", 0.0)),
                energy_sigma=float(stats.get("energy_sigma", 1.0)),
                threshold=float(stats.get("threshold", 0.0)),
            )

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        if np is not None:
            payload["numpy"] = np.random.get_state()
        return payload

    @staticmethod
    def _restore_rng_state(payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        try:
            torch_state = payload.get("torch")
            if torch_state is not None:
                torch.set_rng_state(torch_state)
        except Exception:
            pass
        try:
            cuda_state = payload.get("torch_cuda")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            pass
        try:
            python_state = payload.get("python")
            if python_state is not None:
                random.setstate(python_state)
        except Exception:
            pass
        try:
            numpy_state = payload.get("numpy")
            if numpy_state is not None and np is not None:
                np.random.set_state(numpy_state)
        except Exception:
            pass

    def snapshot_training_state(self) -> TrainingCheckpointPayload:
        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Cannot snapshot training state before initialization.")
        return TrainingCheckpointPayload(
            schema_version="v6_training_checkpoint",
            created_at=datetime.utcnow().isoformat() + "Z",
            trainer_config=self.config.as_contract_dict(),
            config_hash=self._config_hash,
            class_to_idx=dict(self.class_to_idx),
            target_modules_resolved=list(self.target_modules_resolved),
            model_state={
                "adapter_model": self.adapter_model.state_dict(),
                "classifier": self.classifier.state_dict(),
                "fusion": self.fusion.state_dict(),
            },
            optimizer_state=self.optimizer.state_dict() if self.optimizer is not None else None,
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
            scaler_state=self.scaler.state_dict() if self.scaler.is_enabled() else None,
            ood_state=self._serialize_ood_state(),
            rng_state=self._capture_rng_state(),
            best_metric_state=dict(self.best_metric_state),
            current_epoch=int(self.current_epoch),
            optimizer_steps=int(self.optimizer_steps),
        )

    def restore_training_state(self, payload: TrainingCheckpointPayload | Dict[str, Any]) -> TrainingCheckpointPayload:
        checkpoint = (
            payload
            if isinstance(payload, TrainingCheckpointPayload)
            else TrainingCheckpointPayload.from_dict(payload)
        )
        class_to_idx = {str(k): int(v) for k, v in checkpoint.class_to_idx.items()}
        self.class_to_idx = dict(class_to_idx)
        self.target_modules_resolved = [str(v) for v in checkpoint.target_modules_resolved]

        needs_initialize = self.adapter_model is None or self.classifier is None or self.fusion is None
        if needs_initialize:
            self.initialize_engine(class_to_idx=self.class_to_idx)
        else:
            self._is_initialized = True

        model_state = checkpoint.model_state
        adapter_state = model_state.get("adapter_model")
        classifier_state = model_state.get("classifier")
        fusion_state = model_state.get("fusion")

        if adapter_state is not None and self.adapter_model is not None:
            self.adapter_model.load_state_dict(adapter_state, strict=False)
        if classifier_state is not None and self.classifier is not None:
            self.classifier.load_state_dict(classifier_state)
        if fusion_state is not None and self.fusion is not None:
            self.fusion.load_state_dict(fusion_state)

        if self.optimizer is None:
            self.setup_optimizer()
        optimizer_state = checkpoint.optimizer_state
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.scheduler_state
        if scheduler_state is not None:
            self._ensure_scheduler()
            if self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.scaler_state
        if scaler_state is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)

        ood_state = checkpoint.ood_state
        if isinstance(ood_state, dict):
            self._restore_ood_state(ood_state)
        self._restore_rng_state(checkpoint.rng_state)
        self.current_epoch = int(checkpoint.current_epoch)
        self.optimizer_steps = int(checkpoint.optimizer_steps)
        self.best_metric_state = dict(checkpoint.best_metric_state)
        return checkpoint

    def save_adapter(self, output_dir: str) -> Path:
        out = Path(output_dir)
        adapter_dir = out / "continual_sd_lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Cannot save adapter before initialization.")

        if hasattr(self.adapter_model, "save_pretrained"):
            self.adapter_model.save_pretrained(adapter_dir)
        else:
            torch.save(self.adapter_model.state_dict(), adapter_dir / "adapter_model.pt")

        torch.save(self.classifier.state_dict(), adapter_dir / "classifier.pth")
        torch.save(self.fusion.state_dict(), adapter_dir / "fusion.pth")

        meta = self._metadata_payload()
        (adapter_dir / "adapter_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return adapter_dir

    def load_adapter(self, adapter_dir: str) -> Dict[str, Any]:
        root = Path(adapter_dir)
        if root.is_dir() and (root / "continual_sd_lora_adapter").exists():
            root = root / "continual_sd_lora_adapter"
        meta_path = root / "adapter_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"adapter_meta.json not found in {root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.class_to_idx = {str(k): int(v) for k, v in meta.get("class_to_idx", {}).items()}
        self.target_modules_resolved = [str(v) for v in meta.get("target_modules_resolved", [])]

        self.initialize_engine(class_to_idx=self.class_to_idx)
        adapter_config_path = root / "adapter_config.json"
        if adapter_config_path.exists() and PeftModel is not None and self.backbone is not None:
            loaded_adapter = PeftModel.from_pretrained(self.backbone, str(root), is_trainable=False)
            self.adapter_model = self._prepare_module_for_device(loaded_adapter, module_name="adapter_model")
            self._adapter_wrapped = True
        elif (root / "adapter_model.pt").exists() and self.adapter_model is not None:
            self.adapter_model.load_state_dict(torch.load(root / "adapter_model.pt", map_location=self.device), strict=False)
        classifier_path = root / "classifier.pth"
        fusion_path = root / "fusion.pth"
        if classifier_path.exists():
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))  # type: ignore[union-attr]
        if fusion_path.exists():
            self.fusion.load_state_dict(torch.load(fusion_path, map_location=self.device))  # type: ignore[union-attr]
        ood_state = meta.get("ood_state", {})
        if isinstance(ood_state, dict) and ood_state:
            self._restore_ood_state(ood_state)
        else:
            self.ood_detector.calibration_version = int(meta.get("ood_calibration", {}).get("version", 0))
        return meta
