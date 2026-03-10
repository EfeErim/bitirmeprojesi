"""Internal trainer orchestration helpers used by the public trainer facade."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from src.training.ber_loss import BERLoss
from src.training.services.runtime import (
    build_grad_scaler,
    build_adamw_optimizer,
    build_train_batch_stats,
    clip_gradients,
    collect_trainable_parameters,
)

logger = logging.getLogger(__name__)


def refresh_optimizer_after_model_change(trainer: Any) -> None:
    if trainer.optimizer is None:
        return

    previous_optimizer = trainer.optimizer
    previous_scheduler_state = trainer.scheduler.state_dict() if trainer.scheduler is not None else None

    rebuilt_optimizer = build_adamw_optimizer(
        collect_trainable_parameters(trainer),
        lr=trainer.config.learning_rate,
        weight_decay=trainer.config.weight_decay,
        device=trainer.device,
    )

    if len(previous_optimizer.param_groups) == len(rebuilt_optimizer.param_groups):
        for previous_group, rebuilt_group in zip(previous_optimizer.param_groups, rebuilt_optimizer.param_groups):
            for key, value in previous_group.items():
                if key != "params":
                    rebuilt_group[key] = value

    for group in rebuilt_optimizer.param_groups:
        for param in group.get("params", []):
            if param in previous_optimizer.state:
                rebuilt_optimizer.state[param] = copy.deepcopy(previous_optimizer.state[param])

    trainer.optimizer = rebuilt_optimizer
    trainer.scheduler = None
    if previous_scheduler_state is not None:
        trainer._ensure_scheduler()
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(previous_scheduler_state)
    trainer.optimizer.zero_grad(set_to_none=True)


def initialize_trainer_engine(
    trainer: Any,
    *,
    class_to_idx: Dict[str, int] | None,
    auto_model_factory: Any,
    fusion_cls: type[Any],
) -> None:
    """Load frozen backbone, apply LoRA wrappers, and initialize trainable heads."""
    if class_to_idx:
        trainer.class_to_idx = dict(class_to_idx)

    if auto_model_factory is None:
        raise RuntimeError("transformers AutoModel is unavailable for continual trainer initialization.")

    loaded_backbone = auto_model_factory.from_pretrained(trainer.config.backbone_model_name)
    trainer.backbone = trainer._prepare_module_for_device(loaded_backbone, module_name="backbone")
    for param in trainer.backbone.parameters():
        param.requires_grad = False

    hidden_size = int(
        getattr(getattr(trainer.backbone, "config", None), "hidden_size", trainer.config.fusion_output_dim)
    )
    trainer.fusion = fusion_cls(
        input_dim=hidden_size,
        output_dim=trainer.config.fusion_output_dim,
        num_scales=max(1, len(trainer.config.fusion_layers)),
        dropout=trainer.config.fusion_dropout,
        gating=trainer.config.fusion_gating,
    ).to(trainer.device)

    trainer.target_modules_resolved = trainer.resolve_target_modules(trainer.backbone)
    adapter_model = trainer._apply_lora(trainer.backbone, trainer.target_modules_resolved)
    trainer.adapter_model = trainer._prepare_module_for_device(adapter_model, module_name="adapter_model")
    trainer.classifier = nn.Linear(trainer.config.fusion_output_dim, max(1, len(trainer.class_to_idx))).to(
        trainer.device
    )

    if trainer.config.ber_enabled:
        trainer.ber_loss = BERLoss(
            lambda_old=trainer.config.ber_lambda_old,
            lambda_new=trainer.config.ber_lambda_new,
            num_old_classes=0,
            warmup_steps=int(getattr(trainer.config, "ber_warmup_steps", 50)),
        ).to(trainer.device)
    else:
        trainer.ber_loss = None

    trainer._is_initialized = True
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.scaler = build_grad_scaler(trainer.device, trainer.config.mixed_precision)
    trainer.optimizer_steps = 0
    trainer._accumulation_counter = 0
    trainer._trainable_params_cache = None
    trainer._refresh_class_index_cache()
    logger.info(
        "Continual engine initialized: backbone=%s, targets=%s",
        trainer.config.backbone_model_name,
        len(trainer.target_modules_resolved),
    )


def add_trainer_classes(trainer: Any, new_class_names: Iterable[str]) -> Dict[str, int]:
    """Expand the classifier while preserving learned weights for existing classes."""
    num_old = len(trainer.class_to_idx)
    for name in new_class_names:
        if name not in trainer.class_to_idx:
            trainer.class_to_idx[name] = len(trainer.class_to_idx)
    trainer._refresh_class_index_cache()

    if trainer.ber_loss is not None:
        trainer.ber_loss.num_old_classes = num_old

    if trainer.classifier is None:
        return dict(trainer.class_to_idx)

    old_classifier = trainer.classifier
    old_out = int(getattr(old_classifier, "out_features", 0))
    new_out = max(1, len(trainer.class_to_idx))
    if new_out == old_out:
        return dict(trainer.class_to_idx)

    replacement = nn.Linear(old_classifier.in_features, new_out).to(trainer.device)
    if old_out > 0:
        replacement.weight.data[:old_out] = old_classifier.weight.data[:old_out]
        if old_classifier.bias is not None and replacement.bias is not None:
            replacement.bias.data[:old_out] = old_classifier.bias.data[:old_out]
    trainer.classifier = replacement
    trainer._trainable_params_cache = None
    refresh_optimizer_after_model_change(trainer)
    return dict(trainer.class_to_idx)


def has_pending_gradients(trainer: Any) -> bool:
    return int(getattr(trainer, "_accumulation_counter", 0)) > 0


def _apply_optimizer_step(trainer: Any) -> float:
    if trainer.optimizer is None:
        raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
    if trainer.scaler.is_enabled():
        trainer.scaler.unscale_(trainer.optimizer)
    clip_gradients(trainer)
    grad_norm = trainer._compute_grad_norm()
    if trainer.scaler.is_enabled():
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        trainer.optimizer.step()
    trainer.optimizer_steps += 1
    if trainer.config.scheduler_step_on == "batch":
        trainer._step_scheduler()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._accumulation_counter = 0
    return float(grad_norm)


def flush_pending_gradients(trainer: Any) -> float | None:
    if trainer.optimizer is None or not has_pending_gradients(trainer):
        return None
    return _apply_optimizer_step(trainer)


def execute_train_batch(trainer: Any, batch: Dict[str, torch.Tensor]):
    if trainer.optimizer is None:
        trainer.setup_optimizer()
    if trainer.optimizer is None:
        raise RuntimeError("Optimizer is not configured. Call setup_optimizer().")
    trainer.set_train_mode()
    step_started_at = time.perf_counter()
    accumulation_steps = int(max(1, trainer.config.grad_accumulation_steps))
    if trainer._accumulation_counter == 0:
        trainer.optimizer.zero_grad(set_to_none=True)

    loss = trainer.training_step(batch)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss encountered.")

    raw_loss_value = float(loss.detach().item())
    scaled_loss = loss / float(accumulation_steps)
    if trainer.scaler.is_enabled():
        trainer.scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    trainer._accumulation_counter += 1
    should_step = trainer._accumulation_counter >= accumulation_steps
    if should_step:
        grad_norm = _apply_optimizer_step(trainer)
    else:
        grad_norm = trainer._reported_grad_norm(gradients_unscaled=not trainer.scaler.is_enabled())

    ber_components = getattr(trainer, "_last_ber_components", {}) if trainer.ber_loss is not None else {}
    return build_train_batch_stats(
        batch=batch,
        optimizer=trainer.optimizer,
        config=trainer.config,
        loss=raw_loss_value,
        grad_norm=grad_norm,
        step_started_at=step_started_at,
        accumulation_counter=trainer._accumulation_counter,
        accumulation_steps=accumulation_steps,
        optimizer_steps=trainer.optimizer_steps,
        optimizer_step_applied=bool(should_step),
        ber_ce_loss=ber_components.get("ce"),
        ber_old_loss=ber_components.get("ber_old"),
        ber_new_loss=ber_components.get("ber_new"),
    )


def predict_with_ood_result(trainer: Any, images: torch.Tensor) -> Dict[str, Any]:
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot predict before adapter, classifier, and fusion are initialized.")
    trainer._ensure_ood_calibrated(operation="predict_with_ood()")
    trainer.set_eval_mode()
    with torch.inference_mode():
        features = trainer.encode(images.to(trainer.device, non_blocking=True))
        logits = trainer.classifier(features)
        probs = torch.softmax(logits, dim=1)
        confidence, indices = probs.max(dim=1)
        ood = trainer.ood_detector.score(features=features, logits=logits, predicted_labels=indices)

    if trainer._class_index_cache_stale():
        trainer._refresh_class_index_cache()
    predicted_idx = int(indices[0].item()) if indices.numel() else 0

    ood_analysis: Dict[str, Any] = {
        "ensemble_score": float(ood["ensemble_score"][0].item()),
        "class_threshold": float(ood["class_threshold"][0].item()),
        "is_ood": bool(ood["is_ood"][0].item()),
        "mahalanobis_z": float(ood["mahalanobis_z"][0].item()),
        "energy_z": float(ood["energy_z"][0].item()),
        "calibration_version": int(ood["calibration_version"][0].item()),
    }

    if trainer.ood_detector.radial_beta is not None:
        ood_analysis["radial_beta"] = float(trainer.ood_detector.radial_beta)

    if trainer.ood_detector.sure_enabled and "sure_semantic_score" in ood:
        ood_analysis["sure_semantic_score"] = float(ood["sure_semantic_score"][0].item())
        ood_analysis["sure_confidence_score"] = float(ood["sure_confidence_score"][0].item())
        ood_analysis["sure_semantic_ood"] = bool(ood["sure_semantic_ood"][0].item())
        ood_analysis["sure_confidence_reject"] = bool(ood["sure_confidence_reject"][0].item())

    if trainer.ood_detector.conformal_enabled:
        with torch.inference_mode():
            conformal_set = trainer.ood_detector.build_conformal_set(
                features[0],
                logits[0],
                trainer._idx_to_class,
            )
        ood_analysis["conformal_set"] = conformal_set
        ood_analysis["conformal_set_size"] = len(conformal_set)
        ood_analysis["conformal_coverage"] = 1.0 - trainer.ood_detector.conformal_alpha

    return {
        "status": "success",
        "disease": {
            "class_index": predicted_idx,
            "name": trainer._idx_to_class.get(predicted_idx, str(predicted_idx)),
            "confidence": float(confidence[0].item()) if confidence.numel() else 0.0,
        },
        "ood_analysis": ood_analysis,
    }
