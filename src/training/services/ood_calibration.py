"""Streamed OOD calibration helpers for continual SD-LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.ood.continual_ood import ClassCalibration, ContinualOODDetector


@dataclass
class _VectorAccumulator:
    total: Optional[torch.Tensor] = None
    total_sq: Optional[torch.Tensor] = None
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().to(device="cpu", dtype=torch.float64)
        summed = values.sum(dim=0)
        squared = (values * values).sum(dim=0)
        if self.total is None:
            self.total = summed
            self.total_sq = squared
        else:
            self.total = self.total + summed
            self.total_sq = self.total_sq + squared
        self.count += int(values.shape[0])

    def mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total is None or self.total_sq is None or self.count <= 0:
            raise ValueError("Cannot finalize empty vector accumulator.")
        mean = self.total / float(self.count)
        var = (self.total_sq / float(self.count)) - (mean * mean)
        return mean, var.clamp_min(1e-6)


@dataclass
class _ScalarAccumulator:
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        self.total += float(values.sum().item())
        self.total_sq += float((values * values).sum().item())
        self.count += int(values.numel())

    def mean_std(self) -> tuple[float, float]:
        if self.count <= 0:
            raise ValueError("Cannot finalize empty scalar accumulator.")
        mean = self.total / float(self.count)
        variance = max(0.0, (self.total_sq / float(self.count)) - (mean * mean))
        return float(mean), max(1e-6, variance ** 0.5)


def _is_reiterable(loader: Iterable[Dict[str, torch.Tensor]]) -> bool:
    try:
        iterator = iter(loader)
    except TypeError:
        return False
    return iterator is not loader


def _collect_materialized_tensors(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features_list: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            features = trainer.encode(images)
            logits = trainer.classifier(features)
            features_list.append(features.detach().cpu())
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
    if not features_list:
        raise ValueError("Cannot calibrate OOD with an empty loader.")
    return (
        torch.cat(features_list, dim=0),
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
    )


def _iter_feature_batches(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
):
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(trainer.device)
            labels = batch["labels"].reshape(-1).to(device="cpu", dtype=torch.long)
            features_device = trainer.encode(images)
            logits_device = trainer.classifier(features_device)
            features = features_device.detach().to(device="cpu", dtype=torch.float64)
            logits = logits_device.detach().to(device="cpu", dtype=torch.float64)
            yield features, logits, labels


def calibrate_trainer_ood(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Calibrate trainer-owned OOD statistics with streamed multi-pass accumulation."""
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot calibrate OOD before adapter, classifier, and fusion are initialized.")

    trainer.adapter_model.eval()
    trainer.classifier.eval()
    trainer.fusion.eval()

    if not _is_reiterable(loader):
        features, logits, labels = _collect_materialized_tensors(trainer, loader)
        return trainer.ood_detector.calibrate(features=features, logits=logits, labels=labels)

    feature_accumulators: Dict[int, _VectorAccumulator] = {}
    energy_accumulators: Dict[int, _ScalarAccumulator] = {}
    total_examples = 0
    for features, logits, labels in _iter_feature_batches(trainer, loader):
        energies = ContinualOODDetector._energy(logits)
        total_examples += int(labels.numel())
        for class_id in torch.unique(labels).tolist():
            class_mask = labels == int(class_id)
            feature_accumulators.setdefault(int(class_id), _VectorAccumulator()).update(features[class_mask])
            energy_accumulators.setdefault(int(class_id), _ScalarAccumulator()).update(energies[class_mask])

    if total_examples <= 0:
        raise ValueError("Cannot calibrate OOD with an empty loader.")

    class_stats: Dict[int, Dict[str, Any]] = {}
    for class_id, feature_accumulator in feature_accumulators.items():
        mean, var = feature_accumulator.mean_var()
        energy_mu, energy_sigma = energy_accumulators[class_id].mean_std()
        class_stats[class_id] = {
            "mean": mean,
            "var": var,
            "energy_mu": energy_mu,
            "energy_sigma": energy_sigma,
        }

    mahalanobis_accumulators: Dict[int, _ScalarAccumulator] = {
        class_id: _ScalarAccumulator()
        for class_id in class_stats
    }
    for features, _logits, labels in _iter_feature_batches(trainer, loader):
        for class_id in torch.unique(labels).tolist():
            class_mask = labels == int(class_id)
            stats = class_stats[int(class_id)]
            class_features = features[class_mask]
            distances = ((class_features - stats["mean"]) ** 2 / stats["var"]).sum(dim=1).sqrt()
            mahalanobis_accumulators[int(class_id)].update(distances)

    for class_id, accumulator in mahalanobis_accumulators.items():
        mahalanobis_mu, mahalanobis_sigma = accumulator.mean_std()
        class_stats[class_id]["mahalanobis_mu"] = mahalanobis_mu
        class_stats[class_id]["mahalanobis_sigma"] = mahalanobis_sigma

    ensemble_accumulators: Dict[int, _ScalarAccumulator] = {
        class_id: _ScalarAccumulator()
        for class_id in class_stats
    }
    for features, logits, labels in _iter_feature_batches(trainer, loader):
        energies = ContinualOODDetector._energy(logits)
        for class_id in torch.unique(labels).tolist():
            class_mask = labels == int(class_id)
            stats = class_stats[int(class_id)]
            class_features = features[class_mask]
            class_energies = energies[class_mask]
            distances = ((class_features - stats["mean"]) ** 2 / stats["var"]).sum(dim=1).sqrt()
            mahalanobis_z = (distances - float(stats["mahalanobis_mu"])) / float(stats["mahalanobis_sigma"])
            energy_z = (class_energies - float(stats["energy_mu"])) / float(stats["energy_sigma"])
            ensemble = (0.6 * mahalanobis_z) + (0.4 * energy_z)
            ensemble_accumulators[int(class_id)].update(ensemble)

    trainer.ood_detector.class_stats = {}
    for class_id, stats in class_stats.items():
        ensemble_mu, ensemble_sigma = ensemble_accumulators[class_id].mean_std()
        threshold = float(ensemble_mu + (trainer.ood_detector.threshold_factor * ensemble_sigma))
        trainer.ood_detector.class_stats[int(class_id)] = ClassCalibration(
            mean=stats["mean"].to(dtype=torch.float32),
            var=stats["var"].to(dtype=torch.float32),
            mahalanobis_mu=float(stats["mahalanobis_mu"]),
            mahalanobis_sigma=float(stats["mahalanobis_sigma"]),
            energy_mu=float(stats["energy_mu"]),
            energy_sigma=float(stats["energy_sigma"]),
            threshold=threshold,
        )

    trainer.ood_detector.calibration_version += 1
    return {
        "num_classes": float(len(trainer.ood_detector.class_stats)),
        "calibration_version": float(trainer.ood_detector.calibration_version),
    }
