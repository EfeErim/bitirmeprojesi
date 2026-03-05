#!/usr/bin/env python3
"""v6 independent crop adapter with continual SD-LoRA lifecycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer

logger = logging.getLogger(__name__)


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
        self._trainer: Optional[ContinualSDLoRATrainer] = None
        self._last_config: Dict[str, Any] = {}

        logger.info("IndependentCropAdapter initialized for %s on %s", self.crop_name, self.device)

    @property
    def target_modules_resolved(self) -> list[str]:
        if self._trainer is None:
            return []
        return list(self._trainer.target_modules_resolved)

    @property
    def ood_calibration_version(self) -> int:
        if self._trainer is None:
            return 0
        return int(self._trainer.ood_detector.calibration_version)

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
        self._last_config = continual_dict

        trainer_config = ContinualSDLoRAConfig.from_training_config(continual_dict)
        self._trainer = ContinualSDLoRATrainer(trainer_config)
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

    def train_increment(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        *,
        num_epochs: Optional[int] = None,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run continual increment training."""
        if self._trainer is None:
            raise RuntimeError("initialize_engine() must run before train_increment().")
        train_kwargs: Dict[str, Any] = {}
        if num_epochs is not None:
            train_kwargs["num_epochs"] = num_epochs
        if val_loader is not None:
            train_kwargs["val_loader"] = val_loader
        if progress_callback is not None:
            train_kwargs["progress_callback"] = progress_callback
        if should_stop is not None:
            train_kwargs["should_stop"] = should_stop
        if resume_state is not None:
            train_kwargs["resume_state"] = resume_state

        try:
            history = self._trainer.train_increment(train_loader, **train_kwargs)
        except TypeError:
            # Backward compatibility for trainer shims with narrower signatures.
            fallback_attempts = [
                ["resume_state"],
                ["should_stop"],
                ["should_stop", "val_loader"],
                ["should_stop", "val_loader", "progress_callback"],
                ["should_stop", "val_loader", "progress_callback", "resume_state"],
            ]
            history = None
            for drop_keys in fallback_attempts:
                downgraded = dict(train_kwargs)
                for key in drop_keys:
                    downgraded.pop(key, None)
                try:
                    history = self._trainer.train_increment(train_loader, **downgraded)
                    break
                except TypeError:
                    continue
            if history is None:
                raise
        self.is_trained = True
        return {
            "status": "trained",
            "history": history,
            "num_classes": len(self.class_to_idx),
        }

    def save_training_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        progress_state: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
        run_id: str = "",
    ) -> Path:
        """Persist resumable trainer state for fault-tolerant notebook runs."""
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")
        root = Path(checkpoint_dir)
        return self._trainer.save_training_checkpoint(
            str(root),
            progress_state=progress_state,
            history=history,
            run_id=run_id,
        )

    def load_training_checkpoint(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Load resumable trainer state from checkpoint directory."""
        if self._trainer is None:
            normalized = {
                "backbone": {"model_name": self.model_name},
                "adapter": {
                    "target_modules_strategy": "all_linear_transformer",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                },
                "fusion": {"layers": [2, 5, 8, 11]},
                "ood": {"threshold_factor": 2.0},
                "device": str(self.device),
            }
            cfg = ContinualSDLoRAConfig.from_training_config(normalized)
            self._trainer = ContinualSDLoRATrainer(cfg)
        payload = self._trainer.load_training_checkpoint(checkpoint_dir)
        class_to_idx = payload.get("class_to_idx", {})
        if isinstance(class_to_idx, dict):
            self.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
        self.is_trained = True
        return payload

    def calibrate_ood(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Calibrate OOD statistics for current classes."""
        if self._trainer is None:
            raise RuntimeError("initialize_engine() must run before calibrate_ood().")
        result = self._trainer.calibrate_ood(loader)
        return {
            "status": "calibrated",
            "ood_calibration": {
                "version": self.ood_calibration_version,
                "num_classes": int(result.get("num_classes", 0)),
            },
        }

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
        return {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "backbone": {
                "model_name": self._trainer.config.backbone_model_name,
                "frozen": True,
            },
            "fusion": {
                "layers": list(self._trainer.config.fusion_layers),
                "output_dim": int(self._trainer.config.fusion_output_dim),
                "dropout": float(self._trainer.config.fusion_dropout),
                "gating": str(self._trainer.config.fusion_gating),
            },
            "class_to_idx": dict(self.class_to_idx),
            "ood_calibration": {
                "version": self.ood_calibration_version,
            },
            "target_modules_resolved": list(self.target_modules_resolved),
        }

    def save_adapter(self, checkpoint_dir: str) -> None:
        """Persist adapter assets with v6 metadata schema."""
        if self._trainer is None:
            raise RuntimeError("Adapter is not initialized.")

        root = Path(checkpoint_dir)
        asset_dir = self._trainer.save_adapter(str(root))
        metadata = self._metadata_payload()
        (asset_dir / "adapter_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_adapter(self, checkpoint_dir: str) -> None:
        """Load adapter assets and metadata from disk."""
        root = Path(checkpoint_dir)
        asset_dir = root / "continual_sd_lora_adapter"
        if not asset_dir.exists():
            asset_dir = root

        meta_path = asset_dir / "adapter_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"adapter_meta.json not found in {asset_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.class_to_idx = {str(k): int(v) for k, v in meta.get("class_to_idx", {}).items()}

        normalized = {
            "backbone": meta.get("backbone", {"model_name": self.model_name}),
            "adapter": {
                "target_modules_strategy": "all_linear_transformer",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
            "fusion": meta.get("fusion", {"layers": [2, 5, 8, 11]}),
            "ood": {"threshold_factor": 2.0},
            "device": str(self.device),
        }

        cfg = ContinualSDLoRAConfig.from_training_config(normalized)
        self._trainer = ContinualSDLoRATrainer(cfg)
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
