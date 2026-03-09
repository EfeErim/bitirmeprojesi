"""Streamed OOD calibration helpers for continual SD-LoRA training.

Supports extended calibration phases for:
  - Radially Scaled L2 Normalization (auto-tune β)
  - SURE+ Double Scoring (semantic + confidence thresholds)
  - Conformal Prediction Guarantees (q̂ calibration)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.ood._scoring_utils import energy_from_logits, ensemble_z_score, mahalanobis_distance
from src.ood.conformal_prediction import calibrate_conformal_qhat, compute_nonconformity_scores
from src.ood.continual_ood import ClassCalibration, ContinualOODDetector
from src.ood.radial_normalization import auto_tune_beta, radial_l2_normalize
from src.ood.sure_scoring import calibrate_sure_thresholds, compute_confidence_score, compute_semantic_score


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
            current_total = self.total
            current_total_sq = self.total_sq
            assert current_total is not None and current_total_sq is not None
            self.total = current_total + summed
            self.total_sq = current_total_sq + squared
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


def _requires_materialized_extended_calibration(detector: ContinualOODDetector) -> bool:
    return bool(
        detector.radial_l2_enabled
        or detector.sure_enabled
        or detector.conformal_enabled
    )


def _move_detector_stats_to_device(detector: ContinualOODDetector, device: torch.device) -> None:
    for stats in detector.class_stats.values():
        stats.mean = stats.mean.to(device=device, dtype=torch.float32)
        stats.var = stats.var.to(device=device, dtype=torch.float32)


def _normalize_features_for_detector(
    detector: ContinualOODDetector,
    features: torch.Tensor,
) -> torch.Tensor:
    if detector.radial_l2_enabled and detector.radial_beta is not None:
        return radial_l2_normalize(features, detector.radial_beta)
    return features


def _class_distances(features: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    return mahalanobis_distance(features, stats["mean"], stats["var"])


def _class_ensemble_scores(
    features: torch.Tensor,
    energies: torch.Tensor,
    stats: Dict[str, Any],
) -> torch.Tensor:
    distances = _class_distances(features, stats)
    return ensemble_z_score(
        distances,
        energies,
        mahalanobis_mu=float(stats["mahalanobis_mu"]),
        mahalanobis_sigma=float(stats["mahalanobis_sigma"]),
        energy_mu=float(stats["energy_mu"]),
        energy_sigma=float(stats["energy_sigma"]),
    )


def _iter_scored_class_batches(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
):
    for features, logits, labels in _iter_feature_batches(trainer, loader):
        features = _normalize_features_for_detector(trainer.ood_detector, features)
        energies = energy_from_logits(logits)
        for cid_raw in torch.unique(labels).tolist():
            cid = int(cid_raw)
            if cid not in trainer.ood_detector.class_stats:
                continue
            mask = labels == cid
            class_stats = trainer.ood_detector.class_stats[cid]
            class_features = features[mask]
            class_logits = logits[mask]
            class_energies = energies[mask]
            ensemble_scores = ensemble_z_score(
                mahalanobis_distance(class_features, class_stats.mean, class_stats.var),
                class_energies,
                mahalanobis_mu=class_stats.mahalanobis_mu,
                mahalanobis_sigma=class_stats.mahalanobis_sigma,
                energy_mu=class_stats.energy_mu,
                energy_sigma=class_stats.energy_sigma,
            )
            yield cid, ensemble_scores, class_logits, class_stats


def _build_calibration_summary(detector: ContinualOODDetector) -> Dict[str, float]:
    summary: Dict[str, float] = {
        "num_classes": float(len(detector.class_stats)),
        "calibration_version": float(detector.calibration_version),
    }
    if detector.radial_beta is not None:
        summary["radial_beta"] = float(detector.radial_beta)
    if detector.conformal_qhat is not None:
        summary["conformal_qhat"] = float(detector.conformal_qhat)
    return summary


def calibrate_trainer_ood(
    trainer: Any,
    loader: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Calibrate trainer-owned OOD statistics.

    The fully extended detector uses global beta and threshold calibration, so it
    falls back to one materialized pass to avoid repeated loader scans.
    """
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot calibrate OOD before adapter, classifier, and fusion are initialized.")

    trainer.adapter_model.eval()
    trainer.classifier.eval()
    trainer.fusion.eval()

    if not _is_reiterable(loader) or _requires_materialized_extended_calibration(trainer.ood_detector):
        features, logits, labels = _collect_materialized_tensors(trainer, loader)
        calibration_result = trainer.ood_detector.calibrate(features=features, logits=logits, labels=labels)
        _move_detector_stats_to_device(trainer.ood_detector, trainer.device)
        return calibration_result

    feature_accumulators: Dict[int, _VectorAccumulator] = {}
    energy_accumulators: Dict[int, _ScalarAccumulator] = {}
    total_examples = 0
    for features, logits, labels in _iter_feature_batches(trainer, loader):
        energies = energy_from_logits(logits)
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

    # --- Phase 1.5: Auto-tune radial β if enabled (requires extra pass) ---
    if trainer.ood_detector.radial_l2_enabled:
        all_features: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        for features, _logits, labels in _iter_feature_batches(trainer, loader):
            all_features.append(features.float())
            all_labels.append(labels)
        if all_features:
            cat_features = torch.cat(all_features, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            trainer.ood_detector.radial_beta = auto_tune_beta(
                cat_features,
                cat_labels,
                beta_range=trainer.ood_detector.radial_beta_range,
                beta_steps=trainer.ood_detector.radial_beta_steps,
            )
            # Recompute mean/var on normalized features
            for class_id in class_stats:
                mask = cat_labels == class_id
                normed = radial_l2_normalize(cat_features[mask], trainer.ood_detector.radial_beta)
                class_stats[class_id]["mean"] = normed.mean(dim=0).to(dtype=torch.float64)
                class_stats[class_id]["var"] = normed.var(dim=0, unbiased=False).clamp_min(1e-6).to(dtype=torch.float64)

    # --- Phase 2: Mahalanobis distance accumulation ---
    mahalanobis_accumulators: Dict[int, _ScalarAccumulator] = {
        class_id: _ScalarAccumulator()
        for class_id in class_stats
    }
    for features, _logits, labels in _iter_feature_batches(trainer, loader):
        features = _normalize_features_for_detector(trainer.ood_detector, features)
        for class_id in torch.unique(labels).tolist():
            class_mask = labels == int(class_id)
            stats = class_stats[int(class_id)]
            class_features = features[class_mask]
            distances = _class_distances(class_features, stats)
            mahalanobis_accumulators[int(class_id)].update(distances)

    for class_id, accumulator in mahalanobis_accumulators.items():
        mahalanobis_mu, mahalanobis_sigma = accumulator.mean_std()
        class_stats[class_id]["mahalanobis_mu"] = mahalanobis_mu
        class_stats[class_id]["mahalanobis_sigma"] = mahalanobis_sigma

    # --- Phase 3: Ensemble z-score accumulation ---
    ensemble_accumulators: Dict[int, _ScalarAccumulator] = {
        class_id: _ScalarAccumulator()
        for class_id in class_stats
    }
    for features, logits, labels in _iter_feature_batches(trainer, loader):
        features = _normalize_features_for_detector(trainer.ood_detector, features)
        energies = energy_from_logits(logits)
        for class_id in torch.unique(labels).tolist():
            class_mask = labels == int(class_id)
            stats = class_stats[int(class_id)]
            class_features = features[class_mask]
            class_energies = energies[class_mask]
            ensemble = _class_ensemble_scores(class_features, class_energies, stats)
            ensemble_accumulators[int(class_id)].update(ensemble)

    trainer.ood_detector.class_stats = {}
    for class_id, stats in class_stats.items():
        ensemble_mu, ensemble_sigma = ensemble_accumulators[class_id].mean_std()
        threshold = float(ensemble_mu + (trainer.ood_detector.threshold_factor * ensemble_sigma))
        trainer.ood_detector.class_stats[int(class_id)] = ClassCalibration(
            mean=stats["mean"].to(device=trainer.device, dtype=torch.float32),
            var=stats["var"].to(device=trainer.device, dtype=torch.float32),
            mahalanobis_mu=float(stats["mahalanobis_mu"]),
            mahalanobis_sigma=float(stats["mahalanobis_sigma"]),
            energy_mu=float(stats["energy_mu"]),
            energy_sigma=float(stats["energy_sigma"]),
            threshold=threshold,
        )

    # --- Phase 4: SURE+ threshold calibration (streamed) ---
    if trainer.ood_detector.sure_enabled and trainer.ood_detector.class_stats:
        all_semantic: List[torch.Tensor] = []
        all_confidence: List[torch.Tensor] = []
        for _cid, ensemble_scores, class_logits, _class_stats in _iter_scored_class_batches(trainer, loader):
            all_semantic.append(compute_semantic_score(ensemble_scores))
            all_confidence.append(compute_confidence_score(class_logits))

        if all_semantic:
            cat_sem = torch.cat(all_semantic)
            cat_conf = torch.cat(all_confidence)
            sem_t, conf_t = calibrate_sure_thresholds(
                cat_sem,
                cat_conf,
                semantic_percentile=trainer.ood_detector.sure_semantic_percentile,
                confidence_percentile=trainer.ood_detector.sure_confidence_percentile,
            )
            for cs in trainer.ood_detector.class_stats.values():
                cs.sure_semantic_threshold = sem_t
                cs.sure_confidence_threshold = conf_t

    # --- Phase 5: Conformal prediction calibration (streamed) ---
    if trainer.ood_detector.conformal_enabled and trainer.ood_detector.class_stats:
        all_nc: List[torch.Tensor] = []
        for _cid, ensemble_scores, _class_logits, class_stats in _iter_scored_class_batches(trainer, loader):
            thresholds = torch.full_like(ensemble_scores, class_stats.threshold)
            all_nc.append(compute_nonconformity_scores(ensemble_scores, thresholds))

        if all_nc:
            trainer.ood_detector.conformal_qhat = calibrate_conformal_qhat(
                torch.cat(all_nc),
                alpha=trainer.ood_detector.conformal_alpha,
            )

    trainer.ood_detector.calibration_version += 1
    return _build_calibration_summary(trainer.ood_detector)
