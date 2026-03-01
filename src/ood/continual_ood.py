#!/usr/bin/env python3
"""Continual-learning OOD detector using Mahalanobis + energy ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ClassCalibration:
    mean: torch.Tensor
    var: torch.Tensor
    mahalanobis_mu: float
    mahalanobis_sigma: float
    energy_mu: float
    energy_sigma: float
    threshold: float


class ContinualOODDetector:
    """OOD detector with weighted z-score ensemble: 0.6*M + 0.4*E."""

    def __init__(self, threshold_factor: float = 2.0) -> None:
        self.threshold_factor = float(threshold_factor)
        self.class_stats: Dict[int, ClassCalibration] = {}
        self.calibration_version = 0

    @staticmethod
    def _safe_std(value: torch.Tensor) -> torch.Tensor:
        return value.clamp_min(1e-6)

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(logits, dim=1)

    def calibrate(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if features.ndim != 2:
            raise ValueError("features must be [N, D]")
        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        self.class_stats = {}
        energies = self._energy(logits)

        unique_classes = torch.unique(labels)
        for class_id in unique_classes.tolist():
            mask = labels == class_id
            class_features = features[mask]
            class_logits = logits[mask]
            class_energy = energies[mask]
            if class_features.numel() == 0:
                continue

            mean = class_features.mean(dim=0)
            var = class_features.var(dim=0, unbiased=False).clamp_min(1e-6)

            distances = ((class_features - mean) ** 2 / var).sum(dim=1).sqrt()
            m_mu = float(distances.mean().item())
            m_sigma = float(self._safe_std(distances.std(unbiased=False)).item())

            e_mu = float(class_energy.mean().item())
            e_sigma = float(self._safe_std(class_energy.std(unbiased=False)).item())

            m_z = (distances - m_mu) / m_sigma
            e_z = (class_energy - e_mu) / e_sigma
            ensemble = 0.6 * m_z + 0.4 * e_z
            threshold = float(ensemble.mean().item() + self.threshold_factor * self._safe_std(ensemble.std(unbiased=False)).item())

            self.class_stats[int(class_id)] = ClassCalibration(
                mean=mean,
                var=var,
                mahalanobis_mu=m_mu,
                mahalanobis_sigma=m_sigma,
                energy_mu=e_mu,
                energy_sigma=e_sigma,
                threshold=threshold,
            )

        self.calibration_version += 1
        return {
            "num_classes": float(len(self.class_stats)),
            "calibration_version": float(self.calibration_version),
        }

    def _score_class(self, class_id: int, feature: torch.Tensor, logit: torch.Tensor) -> Dict[str, float]:
        if class_id not in self.class_stats:
            return {
                "mahalanobis_z": 0.0,
                "energy_z": 0.0,
                "ensemble_score": 0.0,
                "class_threshold": float("inf"),
                "is_ood": False,
            }

        stats = self.class_stats[class_id]
        distance = torch.sqrt(((feature - stats.mean) ** 2 / stats.var).sum())
        energy = self._energy(logit.unsqueeze(0))[0]

        m_z = float(((distance - stats.mahalanobis_mu) / stats.mahalanobis_sigma).item())
        e_z = float(((energy - stats.energy_mu) / stats.energy_sigma).item())
        ensemble = 0.6 * m_z + 0.4 * e_z
        threshold = float(stats.threshold)
        return {
            "mahalanobis_z": m_z,
            "energy_z": e_z,
            "ensemble_score": float(ensemble),
            "class_threshold": threshold,
            "is_ood": bool(ensemble > threshold),
        }

    def score(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        predicted_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if features.ndim != 2 or logits.ndim != 2:
            raise ValueError("features/logits must be batched tensors [N, D] and [N, C]")

        if predicted_labels is None:
            predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = predicted_labels.reshape(-1)

        result = {
            "mahalanobis_z": [],
            "energy_z": [],
            "ensemble_score": [],
            "class_threshold": [],
            "is_ood": [],
            "calibration_version": torch.full((features.size(0),), self.calibration_version, dtype=torch.long),
        }

        for feat, logit, cls in zip(features, logits, predicted_labels):
            item = self._score_class(int(cls.item()), feat, logit)
            result["mahalanobis_z"].append(item["mahalanobis_z"])
            result["energy_z"].append(item["energy_z"])
            result["ensemble_score"].append(item["ensemble_score"])
            result["class_threshold"].append(item["class_threshold"])
            result["is_ood"].append(item["is_ood"])

        return {
            "mahalanobis_z": torch.tensor(result["mahalanobis_z"], dtype=torch.float32, device=features.device),
            "energy_z": torch.tensor(result["energy_z"], dtype=torch.float32, device=features.device),
            "ensemble_score": torch.tensor(result["ensemble_score"], dtype=torch.float32, device=features.device),
            "class_threshold": torch.tensor(result["class_threshold"], dtype=torch.float32, device=features.device),
            "is_ood": torch.tensor(result["is_ood"], dtype=torch.bool, device=features.device),
            "calibration_version": result["calibration_version"].to(features.device),
        }
