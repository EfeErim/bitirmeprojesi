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
    from src.ood.prototypes import PrototypeManager  # type: ignore[assignment]
except Exception:
    pass

try:
    from src.ood.mahalanobis import MahalanobisDetector  # type: ignore[assignment]
except Exception:
    pass

try:
    from src.ood.dynamic_thresholds import DynamicThresholdManager  # type: ignore[assignment]
except Exception:
    pass
