"""Shared OOD scoring helpers used across detector and calibration paths."""

from __future__ import annotations

from typing import Final

import torch

MAHALANOBIS_WEIGHT: Final[float] = 0.6
ENERGY_WEIGHT: Final[float] = 0.4


def safe_std(value: torch.Tensor) -> torch.Tensor:
    return value.clamp_min(1e-6)


def energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(logits, dim=1)


def mahalanobis_distance(
    features: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    normalized = ((features - mean) ** 2) / var
    return normalized.sum(dim=-1).sqrt()


def ensemble_z_score(
    distances: torch.Tensor,
    energies: torch.Tensor,
    *,
    mahalanobis_mu: float,
    mahalanobis_sigma: float,
    energy_mu: float,
    energy_sigma: float,
) -> torch.Tensor:
    mahalanobis_z = (distances - float(mahalanobis_mu)) / max(float(mahalanobis_sigma), 1e-6)
    energy_z = (energies - float(energy_mu)) / max(float(energy_sigma), 1e-6)
    return (MAHALANOBIS_WEIGHT * mahalanobis_z) + (ENERGY_WEIGHT * energy_z)


def ensemble_threshold(ensemble_scores: torch.Tensor, threshold_factor: float) -> float:
    std = float(safe_std(ensemble_scores.std(unbiased=False)).item())
    return float(ensemble_scores.mean().item() + (float(threshold_factor) * std))
