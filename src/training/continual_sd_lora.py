#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency in some local sandboxes
    np = None  # type: ignore[assignment]

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features
from src.ood.continual_ood import ContinualOODDetector
from src.training.quantization import assert_no_prohibited_4bit_flags
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
    build_train_batch_stats,
    clip_gradients,
    compute_grad_norm,
    configure_runtime_reproducibility,
    configure_training_plan_state,
    ensure_scheduler,
    step_scheduler,
)
from src.training.services.runtime import (
    setup_optimizer as setup_trainer_optimizer,
)
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
                    (
                        "min"
                        if str(early_stopping.get("metric", evaluation.get("best_metric", "val_loss")))
                        in {"val_loss", "generalization_gap"}
                        else "max"
                    ),
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
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.current_epoch = 0
        self.optimizer_steps = 0
        self.class_to_idx: Dict[str, int] = {}
        self.target_modules_resolved: List[str] = []
        self.ood_detector = ContinualOODDetector(threshold_factor=self.config.ood_threshold_factor)
        self._is_initialized = False
        self._contract = self.config.as_contract_dict()
        self._config_hash = compute_config_hash(self._contract)
        self._peft_available = LoraConfig is not None
        self._adapter_wrapped = False
        self._peft_warning_emitted = False
        self._planned_scheduler_steps = 0
        self._planned_epochs = int(max(1, self.config.num_epochs))
        self._accumulation_counter = 0
        self.best_metric_state: Dict[str, Any] = {}
        self._idx_to_class: Dict[int, str] = {}
        self._trainable_params_cache: Optional[List[torch.nn.Parameter]] = None
        configure_runtime_reproducibility(self.config, np_module=np)
        self._refresh_class_index_cache()

    def _refresh_class_index_cache(self) -> None:
        self._idx_to_class = build_idx_to_class(self.class_to_idx)

    def _class_index_cache_stale(self) -> bool:
        if len(self._idx_to_class) != len(self.class_to_idx):
            return True
        return any(self._idx_to_class.get(idx) != name for name, idx in self.class_to_idx.items())

    def configure_training_plan(self, *, total_batches: int, num_epochs: Optional[int] = None) -> None:
        configure_training_plan_state(self, total_batches=total_batches, num_epochs=num_epochs)

    def _ensure_scheduler(self) -> None:
        ensure_scheduler(self)

    def _step_scheduler(self) -> None:
        step_scheduler(self)

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
        self.scaler = build_grad_scaler(self.device, self.config.mixed_precision)
        self.optimizer_steps = 0
        self._accumulation_counter = 0
        self._trainable_params_cache = None
        self._refresh_class_index_cache()
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
        self._refresh_class_index_cache()
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
        self._trainable_params_cache = None
        return dict(self.class_to_idx)

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
        with autocast_context(self.device, self.config.mixed_precision):
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
        return compute_grad_norm(self.optimizer)

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
            clip_gradients(self)
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

        return build_train_batch_stats(
            batch=batch,
            optimizer=self.optimizer,
            config=self.config,
            loss=raw_loss_value,
            grad_norm=grad_norm,
            step_started_at=step_started_at,
            accumulation_counter=self._accumulation_counter,
            accumulation_steps=accumulation_steps,
            optimizer_steps=self.optimizer_steps,
            optimizer_step_applied=bool(should_step),
        )

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        feats: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Cannot calibrate OOD before adapter, classifier, and fusion are initialized.")
        self.adapter_model.eval()
        self.classifier.eval()
        self.fusion.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                features = self.encode(images)
                logits = self.classifier(features)
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
        if self.adapter_model is None or self.classifier is None or self.fusion is None:
            raise RuntimeError("Cannot predict before adapter, classifier, and fusion are initialized.")
        self.adapter_model.eval()
        self.classifier.eval()
        self.fusion.eval()
        with torch.no_grad():
            features = self.encode(images.to(self.device))
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)
            confidence, indices = probs.max(dim=1)
            ood = self.ood_detector.score(features=features, logits=logits, predicted_labels=indices)

        if self._class_index_cache_stale():
            self._refresh_class_index_cache()
        predicted_idx = int(indices[0].item()) if indices.numel() else 0
        return {
            "status": "success",
            "disease": {
                "class_index": predicted_idx,
                "name": self._idx_to_class.get(predicted_idx, str(predicted_idx)),
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
        return build_trainer_metadata_payload(self)

    def snapshot_training_state(self) -> TrainingCheckpointPayload:
        return snapshot_trainer_training_state(self, np_module=np)

    def restore_training_state(self, payload: TrainingCheckpointPayload | Dict[str, Any]) -> TrainingCheckpointPayload:
        return restore_trainer_training_state(self, payload, np_module=np)

    def save_adapter(self, output_dir: str) -> Path:
        return save_trainer_adapter(self, output_dir)

    def load_adapter(self, adapter_dir: str) -> Dict[str, Any]:
        return load_trainer_adapter(self, adapter_dir, peft_model_cls=PeftModel)
