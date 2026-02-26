#!/usr/bin/env python3
"""Shared Phase 3 CoNeC training components and lightweight fallbacks."""

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class CoNeCConfig:
    """Configuration for CoNeC-LoRA training."""

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    model_name: str = "facebook/dinov3-giant"
    learning_rate: float = 5e-5
    num_epochs: int = 10
    batch_size: int = 16
    device: str = "cuda"
    temperature: float = 0.07
    prototype_dim: int = 128
    num_prototypes: int = 10
    contrastive_weight: float = 0.1
    orthogonal_weight: float = 0.01
    target_modules: List[str] = None
    gradient_accumulation_steps: int = 2
    use_amp: bool = True
    memory_efficient_attention: bool = True
    checkpoint_interval: int = 5
    early_stopping_patience: int = 10
    max_memory_gb: Optional[float] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        if getattr(self, 'temperature', 0.0) <= 0:
            raise ValueError("temperature must be positive")
        total_weight = getattr(self, 'contrastive_weight', 0.0) + getattr(self, 'orthogonal_weight', 0.0)
        if total_weight > 1.0:
            raise ValueError("contrastive_weight + orthogonal_weight must be <= 1.0")


class PrototypeManager:
    def __init__(self, num_prototypes: int = 10, prototype_dim: int = 128, device: str = 'cpu'):
        self._prototypes = torch.zeros(num_prototypes, prototype_dim, device=device)

    def get_prototypes(self):
        return self._prototypes

    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        if features is not None and features.ndim == 2 and features.size(1) == self._prototypes.size(1):
            with torch.no_grad():
                self._prototypes[0] = 0.9 * self._prototypes[0] + 0.1 * features.mean(dim=0)

    def set_prototypes(self, prototypes: torch.Tensor):
        self._prototypes = prototypes


class MahalanobisDetector:
    def __init__(self):
        self.enabled = True

    def compute_scores(self, features: torch.Tensor, labels: torch.Tensor):
        return torch.norm(features, dim=1)


class DynamicThresholdManager:
    def __init__(self, threshold: float = 1.0):
        self._threshold = threshold

    def get_threshold(self) -> float:
        return self._threshold


class ColabMemoryMonitor:
    def __init__(self, max_memory_gb: Optional[float] = None, clear_cache_frequency: int = 10):
        self.max_memory_gb = max_memory_gb
        self.clear_cache_frequency = clear_cache_frequency


try:
    from src.debugging.monitoring import ColabMemoryMonitor  # type: ignore[assignment]
except Exception:
    pass

try:
    from src.ood.prototypes import PrototypeComputer, PrototypeConfig

    class _OODPrototypeManager:
        """Compatibility wrapper backed by src.ood.prototypes."""

        def __init__(self, num_prototypes: int = 10, prototype_dim: int = 128, device: str = 'cpu'):
            self._num_prototypes = int(num_prototypes)
            self._prototype_dim = int(prototype_dim)
            self._device = torch.device(device)
            config = PrototypeConfig(feature_dim=self._prototype_dim, device=str(self._device))
            self._computer = PrototypeComputer(config=config)
            self._prototypes = torch.zeros(self._num_prototypes, self._prototype_dim, device=self._device)

        def get_prototypes(self):
            return self._prototypes

        def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
            if features is None or labels is None or features.numel() == 0:
                return
            try:
                feats = features.to(self._device)
                lbls = labels.to(self._device).view(-1)
                if feats.ndim != 2 or feats.shape[1] != self._prototype_dim:
                    return
                for cls in torch.unique(lbls):
                    cls_idx = int(cls.item())
                    if cls_idx < 0 or cls_idx >= self._num_prototypes:
                        continue
                    class_features = feats[lbls == cls]
                    if class_features.numel() == 0:
                        continue
                    class_mean = class_features.mean(dim=0)
                    self._prototypes[cls_idx] = (
                        0.9 * self._prototypes[cls_idx] + 0.1 * class_mean
                    )
            except Exception:
                # Keep previous prototypes on update failures.
                return

        def set_prototypes(self, prototypes: torch.Tensor):
            tensor = prototypes.to(self._device)
            if tensor.ndim == 2 and tensor.shape[1] == self._prototype_dim:
                n = min(self._num_prototypes, tensor.shape[0])
                updated = torch.zeros(self._num_prototypes, self._prototype_dim, device=self._device)
                updated[:n] = tensor[:n]
                self._prototypes = updated

    PrototypeManager = _OODPrototypeManager  # type: ignore[assignment]
except Exception:
    pass

try:
    from src.ood.mahalanobis import MahalanobisDistance

    class _OODMahalanobisDetector:
        """Compatibility wrapper backed by src.ood.mahalanobis."""

        def __init__(self):
            self.enabled = True

        def compute_scores(self, features: torch.Tensor, labels: torch.Tensor):
            if features is None or labels is None or features.numel() == 0:
                return torch.empty(0, device=features.device if torch.is_tensor(features) else 'cpu')

            unique_classes = torch.unique(labels)
            prototypes = []
            class_stds = {}
            class_to_slot = {}

            for slot, cls in enumerate(unique_classes.tolist()):
                cls_features = features[labels == cls]
                proto = cls_features.mean(dim=0)
                std = cls_features.std(dim=0, unbiased=False).clamp(min=1e-6)
                prototypes.append(proto)
                class_stds[slot] = std
                class_to_slot[int(cls)] = slot

            prototype_tensor = torch.stack(prototypes, dim=0)
            md = MahalanobisDistance(prototype_tensor, class_stds, device=str(features.device))

            scores = []
            for feat, label in zip(features, labels):
                slot = class_to_slot[int(label.item())]
                score = md.compute_distance(feat.unsqueeze(0), slot)
                scores.append(score.reshape(-1)[0])

            return torch.stack(scores)

    MahalanobisDetector = _OODMahalanobisDetector  # type: ignore[assignment]
except Exception:
    pass

try:
    from src.ood.dynamic_thresholds import AdaptiveThresholdManager

    class _OODDynamicThresholdManager:
        """Compatibility wrapper backed by src.ood.dynamic_thresholds."""

        def __init__(self, threshold: float = 1.0):
            self._threshold = float(threshold)
            self._manager = AdaptiveThresholdManager(
                initial_thresholds={0: self._threshold},
                adaptation_rate=0.05,
                min_threshold=0.0,
                max_threshold=1e9,
            )

        def update_threshold(self, value: float) -> float:
            updated = self._manager.update_thresholds({0: float(value)})
            self._threshold = float(updated.get(0, self._threshold))
            return self._threshold

        def get_threshold(self) -> float:
            return self._threshold

    DynamicThresholdManager = _OODDynamicThresholdManager  # type: ignore[assignment]
except Exception:
    pass
