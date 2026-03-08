#!/usr/bin/env python3
"""v6 independent crop adapter with continual SD-LoRA lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

import torch

from src.adapter.checkpointing import (
    build_checkpoint_load_result,
    build_runtime_adapter_metadata,
    normalize_trainer_config,
)
from src.adapter.checkpointing import (
    load_training_checkpoint as load_adapter_training_checkpoint,
)
from src.adapter.checkpointing import (
    save_training_checkpoint as save_adapter_training_checkpoint,
)
from src.shared.contracts import AdapterMetadata
from src.shared.json_utils import read_json_dict, write_json
from src.training.services.runtime import resolve_session_num_epochs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
    from src.training.session import ContinualTrainingSession
    from src.training.types import TrainingCheckpointPayload


def _trainer_types() -> tuple[type["ContinualSDLoRAConfig"], type["ContinualSDLoRATrainer"]]:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer

    return ContinualSDLoRAConfig, ContinualSDLoRATrainer


def _training_session_type() -> type["ContinualTrainingSession"]:
    from src.training.session import ContinualTrainingSession

    return ContinualTrainingSession


def _checkpoint_payload_type() -> type["TrainingCheckpointPayload"]:
    from src.training.types import TrainingCheckpointPayload

    return TrainingCheckpointPayload


class IndependentCropAdapter:
    """Per-crop continual adapter runtime surface for v6."""

    def __init__(
        self,
        crop_name: str,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str = "cuda",
    ) -> None:
        self.crop_name = str(crop_name)
        self.model_name = str(model_name)
        self.device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")

        self.engine = "continual_sd_lora"
        self.schema_version = "v6"
        self.class_to_idx: Dict[str, int] = {}
        self.is_trained = False
        self._trainer: Optional["ContinualSDLoRATrainer"] = None
        self._calibration_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None

        logger.info("IndependentCropAdapter initialized for %s on %s", self.crop_name, self.device)

    @property
    def target_modules_resolved(self) -> list[str]:
        if self._trainer is None:
            return []
        return list(self._trainer.target_modules_resolved)

    @property
    def trainer(self) -> "ContinualSDLoRATrainer":
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        return self._trainer

    @property
    def ood_calibration_version(self) -> int:
        if self._trainer is None:
            return 0
        return int(self._trainer.ood_detector.calibration_version)

    def parameters(self):
        if self._trainer is None:
            return iter(())
        modules = [self._trainer.adapter_model, self._trainer.classifier, self._trainer.fusion]
        for module in modules:
            if module is None:
                continue
            yield from module.parameters()

    def _normalize_continual_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        config = dict(config or {})
        continual = config.get("training", {}).get("continual") if isinstance(config.get("training"), dict) else None
        if isinstance(continual, dict):
            normalized = dict(continual)
        elif "backbone" in config or "adapter" in config:
            normalized = config
        else:
            normalized = {
                "backbone": {"model_name": self.model_name},
                "adapter": {
                    "target_modules_strategy": "all_linear_transformer",
                    "lora_r": int(config.get("lora_r", 16)),
                    "lora_alpha": int(config.get("lora_alpha", 32)),
                    "lora_dropout": float(config.get("lora_dropout", 0.1)),
                },
                "fusion": {
                    "layers": [2, 5, 8, 11],
                    "output_dim": int(config.get("fusion_output_dim", 768)),
                    "dropout": float(config.get("fusion_dropout", 0.1)),
                    "gating": str(config.get("fusion_gating", "softmax")),
                },
                "ood": {"threshold_factor": float(config.get("ood_threshold_factor", 2.0))},
                "learning_rate": float(config.get("learning_rate", 1e-4)),
                "weight_decay": float(config.get("weight_decay", 0.0)),
                "num_epochs": int(config.get("num_epochs", 1)),
                "batch_size": int(config.get("batch_size", 8)),
                "device": str(config.get("device", str(self.device))),
                "strict_model_loading": bool(config.get("strict_model_loading", False)),
            }
        normalized.setdefault("backbone", {"model_name": self.model_name})
        normalized["backbone"].setdefault("model_name", self.model_name)
        return normalized

    def initialize_engine(
        self,
        *,
        class_names: Optional[Iterable[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initialize frozen backbone + continual adapter engine."""
        if class_to_idx is not None:
            self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        elif class_names is not None:
            self.class_to_idx = {str(name): idx for idx, name in enumerate(class_names)}

        continual_dict = self._normalize_continual_config(config)
        config_cls, trainer_cls = _trainer_types()
        trainer_config = config_cls.from_training_config(continual_dict)
        self._trainer = trainer_cls(trainer_config)
        self._trainer.initialize_engine(class_to_idx=self.class_to_idx)
        self.is_trained = True

        return {
            "status": "initialized",
            "engine": self.engine,
            "schema_version": self.schema_version,
            "num_classes": len(self.class_to_idx),
        }

    def add_classes(self, new_classes: Iterable[str]) -> Dict[str, Any]:
        """Add new class labels and expand classifier."""
        if self._trainer is None:
            raise RuntimeError("initialize_engine() must run before add_classes().")
        self.class_to_idx = self._trainer.add_classes(new_classes)
        return {
            "status": "classes_added",
            "num_classes": len(self.class_to_idx),
            "class_to_idx": dict(self.class_to_idx),
        }

    def build_training_session(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        *,
        num_epochs: Optional[int] = None,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        observers: Optional[list[Callable[[Dict[str, Any]], None]]] = None,
        stop_policy: Optional[Callable[[], bool]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 0,
        checkpoint_on_exception: bool = False,
    ) -> "ContinualTrainingSession":
        """Build a training session around the initialized trainer."""
        if self._trainer is None:
            raise RuntimeError("initialize_engine() must run before build_training_session().")
        self._calibration_loader = val_loader if val_loader is not None else train_loader
        resolved_epochs = resolve_session_num_epochs(self._trainer.config, num_epochs)
        training_session_cls = _training_session_type()
        return training_session_cls(
            self._trainer,
            train_loader,
            resolved_epochs,
            val_loader=val_loader,
            observers=observers,
            stop_policy=stop_policy,
            resume_state=resume_state,
            run_id=run_id,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            checkpoint_on_exception=checkpoint_on_exception,
        )

    def save_training_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        session_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
    ) -> Path:
        """Persist trainer and session state for fault-tolerant notebook runs."""
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        return save_adapter_training_checkpoint(
            trainer=self._trainer,
            checkpoint_dir=checkpoint_dir,
            session_state=session_state,
            run_id=run_id,
        )

    def load_training_checkpoint(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Load resumable trainer and session state from checkpoint directory."""
        config_cls, trainer_cls = _trainer_types()

        def _trainer_factory(normalized: Dict[str, Any]) -> "ContinualSDLoRATrainer":
            cfg = config_cls.from_training_config(normalized)
            return trainer_cls(cfg)

        self._trainer, trainer_payload, session_payload = load_adapter_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            device=self.device,
            trainer=self._trainer,
            trainer_factory=_trainer_factory,
            checkpoint_payload_factory=_checkpoint_payload_type,
            model_name=self.model_name,
        )
        class_to_idx = trainer_payload.class_to_idx
        if isinstance(class_to_idx, dict):
            self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        self.is_trained = True
        return build_checkpoint_load_result(trainer_payload=trainer_payload, session_payload=session_payload)

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Calibrate OOD statistics for current classes."""
        if self._trainer is None:
            raise RuntimeError("initialize_engine() must run before calibrate_ood().")
        self._calibration_loader = loader
        result = self._trainer.calibrate_ood(loader)
        return {
            "status": "calibrated",
            "ood_calibration": {
                "version": self.ood_calibration_version,
                "num_classes": int(result.get("num_classes", 0)),
            },
        }

    def _ensure_ood_calibrated_for_export(self) -> None:
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        issue = self._trainer.ood_detector.calibration_issue()
        if issue is None:
            return
        if self._calibration_loader is None:
            raise RuntimeError(f"{issue} No calibration loader is available for automatic export calibration.")
        logger.info("Auto-calibrating OOD state before adapter export for %s", self.crop_name)
        self.calibrate_ood(self._calibration_loader)

    def predict_with_ood(self, image: torch.Tensor) -> Dict[str, Any]:
        """Return diagnosis + v6 OOD payload for a single image tensor."""
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return self._trainer.predict_with_ood(image.to(self.device))

    def detect_ood_dynamic(self, image: torch.Tensor) -> Dict[str, Any]:
        """Compatibility helper returning OOD analysis fields only."""
        result = self.predict_with_ood(image)
        ood = result.get("ood_analysis", {})
        return {
            "is_ood": bool(ood.get("is_ood", False)),
            "ensemble_score": float(ood.get("ensemble_score", 0.0)),
            "class_threshold": float(ood.get("class_threshold", 0.0)),
            "calibration_version": int(ood.get("calibration_version", 0)),
        }

    def _metadata_payload(self) -> Dict[str, Any]:
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        from src.training.services.persistence import serialize_ood_state

        return build_runtime_adapter_metadata(
            trainer=self._trainer,
            class_to_idx=self.class_to_idx,
            schema_version=self.schema_version,
            engine=self.engine,
            model_name=self.model_name,
            ood_calibration_version=self.ood_calibration_version,
            target_modules_resolved=list(self.target_modules_resolved),
            serialize_ood_state_fn=serialize_ood_state,
        )

    def save_adapter(self, checkpoint_dir: str) -> Path:
        """Persist adapter assets with v6 metadata schema."""
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        self._ensure_ood_calibrated_for_export()

        root = Path(checkpoint_dir)
        asset_dir = self._trainer.save_adapter(str(root))
        meta_path = asset_dir / "adapter_meta.json"
        if not meta_path.exists():
            metadata = self._metadata_payload()
            write_json(meta_path, metadata)
        return asset_dir

    def load_adapter(self, checkpoint_dir: str) -> None:
        """Load adapter assets and metadata from disk."""
        root = Path(checkpoint_dir)
        asset_dir = root / "continual_sd_lora_adapter"
        if not asset_dir.exists():
            asset_dir = root

        meta_path = asset_dir / "adapter_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"adapter_meta.json not found in {asset_dir}")

        meta = AdapterMetadata.from_dict(read_json_dict(meta_path))
        self.class_to_idx = dict(meta.class_to_idx)

        normalized = normalize_trainer_config(
            meta.trainer_config,
            model_name=self.model_name,
            device=self.device,
            backbone=dict(meta.backbone or {"model_name": self.model_name}),
            fusion=dict(meta.fusion or {"layers": [2, 5, 8, 11]}),
        )

        config_cls, trainer_cls = _trainer_types()
        cfg = config_cls.from_training_config(normalized)
        self._trainer = trainer_cls(cfg)
        self._trainer.load_adapter(str(asset_dir))
        self.is_trained = True

    def get_summary(self) -> Dict[str, Any]:
        """Return concise runtime adapter summary."""
        return {
            "crop_name": self.crop_name,
            "model_name": self.model_name,
            "engine": self.engine,
            "schema_version": self.schema_version,
            "is_trained": self.is_trained,
            "num_classes": len(self.class_to_idx),
            "class_to_idx": dict(self.class_to_idx),
            "ood_calibration_version": self.ood_calibration_version,
        }
