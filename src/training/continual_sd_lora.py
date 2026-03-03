#!/usr/bin/env python3
"""v6 continual SD-LoRA training surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features
from src.ood.continual_ood import ContinualOODDetector
from src.training.quantization import (
    HybridINT8Config,
    assert_no_prohibited_4bit_flags,
    load_hybrid_int8_backbone,
)

try:
    from transformers import AutoModel
except Exception:  # pragma: no cover - test fallback
    AutoModel = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - test fallback
    LoraConfig = None  # type: ignore[assignment]

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
    quantization_mode: str = "int8_hybrid"
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
    strict_quantization_backend: bool = True
    allow_cpu_quantization_fallback: bool = False
    ood_threshold_factor: float = 2.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.quantization_mode.lower() != "int8_hybrid":
            raise ValueError("v6 requires quantization_mode='int8_hybrid'.")
        if self.target_modules_strategy != "all_linear_transformer":
            raise ValueError("v6 requires target_modules_strategy='all_linear_transformer'.")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_r and lora_alpha must be positive.")
        if not self.fusion_layers:
            raise ValueError("fusion_layers must not be empty.")

    def as_contract_dict(self) -> Dict[str, Any]:
        """Return normalized config payload used in metadata persistence."""
        payload = {
            "backbone": {"model_name": self.backbone_model_name},
            "quantization": {"mode": self.quantization_mode},
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
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_training_config(cls, training_continual: Dict[str, Any]) -> "ContinualSDLoRAConfig":
        """Build from `training.continual` dictionary."""
        assert_no_prohibited_4bit_flags(training_continual)
        backbone = training_continual.get("backbone", {})
        quantization = training_continual.get("quantization", {})
        adapter = training_continual.get("adapter", {})
        fusion = training_continual.get("fusion", {})
        ood = training_continual.get("ood", {})

        config = cls(
            backbone_model_name=str(backbone.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")),
            quantization_mode=str(quantization.get("mode", "int8_hybrid")),
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
            strict_quantization_backend=bool(quantization.get("strict_backend", True)),
            allow_cpu_quantization_fallback=bool(quantization.get("allow_cpu_fallback", False)),
            ood_threshold_factor=float(ood.get("threshold_factor", 2.0)),
            extra={k: v for k, v in training_continual.items() if k not in {
                "backbone",
                "quantization",
                "adapter",
                "fusion",
                "ood",
                "learning_rate",
                "weight_decay",
                "num_epochs",
                "batch_size",
                "device",
                "strict_model_loading",
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
        self.current_epoch = 0
        self.class_to_idx: Dict[str, int] = {}
        self.target_modules_resolved: List[str] = []
        self.ood_detector = ContinualOODDetector(threshold_factor=self.config.ood_threshold_factor)
        self._is_initialized = False
        self._contract = self.config.as_contract_dict()
        self._peft_available = LoraConfig is not None
        self._adapter_wrapped = False
        self._peft_warning_emitted = False

    @property
    def quantization_metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.config.quantization_mode,
            "strict_backend": self.config.strict_quantization_backend,
            "allow_cpu_fallback": self.config.allow_cpu_quantization_fallback,
        }

    @staticmethod
    def _is_low_bit_loaded_model(model: nn.Module) -> bool:
        """Return True when model device placement is already managed by low-bit loader."""
        candidates = [
            model,
            getattr(model, "base_model", None),
            getattr(getattr(model, "base_model", None), "model", None),
            getattr(model, "model", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if bool(getattr(candidate, "is_loaded_in_8bit", False)):
                return True
            if bool(getattr(candidate, "is_loaded_in_4bit", False)):
                return True
            quant_method = str(getattr(candidate, "quantization_method", "")).lower()
            if "bitsandbytes" in quant_method:
                return True
        return False

    def initialize_engine(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        """Load frozen backbone, apply adapters, initialize heads."""
        if class_to_idx:
            self.class_to_idx = dict(class_to_idx)

        if AutoModel is None:
            raise RuntimeError("transformers AutoModel is unavailable for continual trainer initialization.")

        int8_cfg = HybridINT8Config(
            mode=self.config.quantization_mode,
            strict_backend=self.config.strict_quantization_backend,
            allow_cpu_fallback=self.config.allow_cpu_quantization_fallback,
        )
        self.backbone = load_hybrid_int8_backbone(
            self.config.backbone_model_name,
            auto_model_cls=AutoModel,
            cfg=int8_cfg,
            strict_model_loading=self.config.strict_model_loading,
        )
        if self._is_low_bit_loaded_model(self.backbone):
            logger.info("Backbone is already low-bit loaded; skipping explicit .to(device).")
        else:
            self.backbone = self.backbone.to(self.device)
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
        if self._is_low_bit_loaded_model(adapter_model):
            self.adapter_model = adapter_model
        else:
            self.adapter_model = adapter_model.to(self.device)
        self.classifier = nn.Linear(self.config.fusion_output_dim, max(1, len(self.class_to_idx))).to(self.device)

        self._is_initialized = True
        logger.info(
            "Continual engine initialized: backbone=%s, targets=%s",
            self.config.backbone_model_name,
            len(self.target_modules_resolved),
        )

    def _warn_missing_peft_once(self) -> None:
        if self._peft_warning_emitted:
            return
        message = (
            "peft is not installed; continuing without LoRA adapter wrapping. "
            "Training will run in degraded mode (fusion/classifier only). Install `peft` to enable SD-LoRA adapters."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        self._peft_warning_emitted = True

    def _apply_lora(self, model: nn.Module, target_modules: Sequence[str]) -> nn.Module:
        if LoraConfig is None:
            self._warn_missing_peft_once()
            self._adapter_wrapped = False
            return model

        suffixes = sorted({name.split(".")[-1] for name in target_modules})
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=suffixes,
            bias="none",
        )
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
        logits = self.forward_logits(images)
        return nn.functional.cross_entropy(logits, labels)

    def train_increment(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        num_epochs: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, List[float]]:
        if self.optimizer is None:
            self.setup_optimizer()

        epochs = int(num_epochs if num_epochs is not None else self.config.num_epochs)
        history = {"train_loss": []}
        for epoch_idx in range(max(1, epochs)):
            self.adapter_model.train()  # type: ignore[union-attr]
            self.classifier.train()  # type: ignore[union-attr]
            self.fusion.train()  # type: ignore[union-attr]
            losses: List[float] = []
            total_batches = len(train_loader) if hasattr(train_loader, "__len__") else 0
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))

                if progress_callback is not None:
                    callback_payload = {
                        "epoch": epoch_idx + 1,
                        "batch": batch_idx + 1,
                        "total_batches": int(total_batches),
                        "batch_loss": float(loss.item()),
                        "epoch_progress": float((batch_idx + 1) / max(1, total_batches)),
                    }
                    progress_callback(callback_payload)

            epoch_loss = float(sum(losses) / max(1, len(losses)))
            history["train_loss"].append(epoch_loss)
            self.current_epoch += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch_done": epoch_idx + 1,
                        "epoch_loss": epoch_loss,
                    }
                )
        return history

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
            "backbone": {
                "model_name": self.config.backbone_model_name,
                "frozen": True,
            },
            "quantization": self.quantization_metadata,
            "fusion": {
                "layers": self.config.fusion_layers,
                "output_dim": self.config.fusion_output_dim,
                "dropout": self.config.fusion_dropout,
                "gating": self.config.fusion_gating,
            },
            "class_to_idx": self.class_to_idx,
            "ood_calibration": {"version": self.ood_detector.calibration_version},
            "target_modules_resolved": list(self.target_modules_resolved),
            "adapter_runtime": {
                "peft_available": bool(self._peft_available),
                "adapter_wrapped": bool(self._adapter_wrapped),
                "degraded_without_peft": bool(not self._adapter_wrapped),
            },
        }

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
        classifier_path = root / "classifier.pth"
        fusion_path = root / "fusion.pth"
        if classifier_path.exists():
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))  # type: ignore[union-attr]
        if fusion_path.exists():
            self.fusion.load_state_dict(torch.load(fusion_path, map_location=self.device))  # type: ignore[union-attr]
        self.ood_detector.calibration_version = int(meta.get("ood_calibration", {}).get("version", 0))
        return meta

