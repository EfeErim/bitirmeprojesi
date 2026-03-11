#!/usr/bin/env python3
"""Continual-learning OOD detector using a multi-score runtime stack.

The detector keeps the legacy Mahalanobis + energy ensemble path, but also
calibrates raw energy and per-class kNN distance scores from the same feature
materialization. SURE+ and conformal prediction remain ensemble-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.ood._scoring_utils import (
    energy_from_logits,
    ensemble_z_score,
    mahalanobis_distance,
    safe_std,
)
from src.ood.conformal_prediction import (
    build_prediction_set,
    calibrate_conformal_qhat,
    compute_nonconformity_scores,
)
from src.ood.radial_normalization import auto_tune_beta, radial_l2_normalize
from src.ood.sure_scoring import (
    apply_sure_decision,
    calibrate_sure_thresholds,
    compute_confidence_score,
    compute_semantic_score,
)

SUPPORTED_OOD_SCORE_METHODS = ("ensemble", "energy", "knn")
DEFAULT_KNN_K = 10
DEFAULT_KNN_BANK_CAP = 256
_FLOAT_EPS = 1e-6
_BASE_SCORE_FIELDS = (
    "mahalanobis_z",
    "energy_z",
    "ensemble_score",
    "class_threshold",
    "energy_score",
    "energy_threshold",
    "knn_distance",
    "knn_threshold",
    "primary_score",
    "decision_threshold",
    "is_ood",
)
_SURE_SCORE_FIELDS = (
    "sure_semantic_score",
    "sure_confidence_score",
    "sure_semantic_ood",
    "sure_confidence_reject",
)


def normalize_primary_score_method(value: Any) -> str:
    resolved = str(value or "ensemble").strip().lower()
    if resolved not in SUPPORTED_OOD_SCORE_METHODS:
        raise ValueError(
            "primary_score_method must be one of: "
            + ", ".join(SUPPORTED_OOD_SCORE_METHODS)
            + "."
        )
    return resolved


def _distribution_threshold(values: torch.Tensor, threshold_factor: float) -> float:
    std = float(safe_std(values.std(unbiased=False)).item())
    return float(values.mean().item() + (float(threshold_factor) * std))


@dataclass
class ClassCalibration:
    mean: torch.Tensor
    var: torch.Tensor
    mahalanobis_mu: float
    mahalanobis_sigma: float
    energy_mu: float
    energy_sigma: float
    threshold: float
    energy_threshold: float = 0.0
    knn_distance_mu: float = 0.0
    knn_distance_sigma: float = 1.0
    knn_threshold: float = 0.0
    knn_bank: Optional[torch.Tensor] = None
    knn_k: int = DEFAULT_KNN_K
    sure_semantic_threshold: float = 0.0
    sure_confidence_threshold: float = 0.0


class ContinualOODDetector:
    """OOD detector with configurable primary score method.

    All score methods are calibrated together:
    - ``ensemble``: weighted Mahalanobis z + energy z
    - ``energy``: raw energy score thresholded per predicted class
    - ``knn``: mean distance to the class-local k nearest neighbors
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,
        *,
        primary_score_method: str = "ensemble",
        knn_k: int = DEFAULT_KNN_K,
        knn_bank_cap: int = DEFAULT_KNN_BANK_CAP,
        radial_l2_enabled: bool = False,
        radial_beta_range: Tuple[float, float] = (0.5, 2.0),
        radial_beta_steps: int = 16,
        sure_enabled: bool = False,
        sure_semantic_percentile: float = 95.0,
        sure_confidence_percentile: float = 90.0,
        conformal_enabled: bool = False,
        conformal_alpha: float = 0.05,
    ) -> None:
        self.threshold_factor = float(threshold_factor)
        self.primary_score_method = normalize_primary_score_method(primary_score_method)
        self.knn_k = int(knn_k)
        self.knn_bank_cap = int(knn_bank_cap)
        self.class_stats: Dict[int, ClassCalibration] = {}
        self.calibration_version = 0

        self.radial_l2_enabled = bool(radial_l2_enabled)
        self.radial_beta_range = (float(radial_beta_range[0]), float(radial_beta_range[1]))
        self.radial_beta_steps = int(radial_beta_steps)
        self.radial_beta: Optional[float] = None

        self.sure_enabled = bool(sure_enabled)
        self.sure_semantic_percentile = float(sure_semantic_percentile)
        self.sure_confidence_percentile = float(sure_confidence_percentile)

        self.conformal_enabled = bool(conformal_enabled)
        self.conformal_alpha = float(conformal_alpha)
        self.conformal_qhat: Optional[float] = None

    @staticmethod
    def _safe_std(value: torch.Tensor) -> torch.Tensor:
        return safe_std(value)

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return energy_from_logits(logits)

    @staticmethod
    def _calibrated_class_ids(labels: torch.Tensor) -> List[int]:
        return [int(class_id) for class_id in torch.unique(labels).tolist()]

    @staticmethod
    def _build_calibration_summary(
        *,
        class_count: int,
        calibration_version: int,
        primary_score_method: str,
        radial_beta: Optional[float],
        conformal_qhat: Optional[float],
        knn_k: int,
        knn_bank_cap: int,
    ) -> Dict[str, float | str]:
        result: Dict[str, float | str] = {
            "num_classes": float(class_count),
            "calibration_version": float(calibration_version),
            "primary_score_method": str(primary_score_method),
            "knn_k": float(knn_k),
            "knn_bank_cap": float(knn_bank_cap),
        }
        if radial_beta is not None:
            result["radial_beta"] = float(radial_beta)
        if conformal_qhat is not None:
            result["conformal_qhat"] = float(conformal_qhat)
        return result

    @staticmethod
    def _stats_on_input_device(
        stats: ClassCalibration,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
        mean = stats.mean.to(device=reference.device, dtype=target_dtype)
        var = stats.var.to(device=reference.device, dtype=target_dtype)
        return mean, var

    @staticmethod
    def _knn_bank_on_input_device(stats: ClassCalibration, reference: torch.Tensor) -> Optional[torch.Tensor]:
        if stats.knn_bank is None or stats.knn_bank.numel() <= 0:
            return None
        target_dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
        return stats.knn_bank.to(device=reference.device, dtype=target_dtype)

    @staticmethod
    def _default_score_result(primary_score_method: str) -> Dict[str, Any]:
        return {
            "mahalanobis_z": 0.0,
            "energy_z": 0.0,
            "ensemble_score": 0.0,
            "class_threshold": float("inf"),
            "energy_score": 0.0,
            "energy_threshold": float("inf"),
            "knn_distance": 0.0,
            "knn_threshold": float("inf"),
            "primary_score": 0.0,
            "decision_threshold": float("inf"),
            "is_ood": False,
            "score_method": primary_score_method,
            "candidate_scores": {name: 0.0 for name in SUPPORTED_OOD_SCORE_METHODS},
            "candidate_thresholds": {name: float("inf") for name in SUPPORTED_OOD_SCORE_METHODS},
            "sure_semantic_score": None,
            "sure_confidence_score": None,
            "sure_semantic_ood": None,
            "sure_confidence_reject": None,
        }

    @staticmethod
    def _set_sure_thresholds(
        class_stats: Dict[int, ClassCalibration],
        *,
        semantic_threshold: float,
        confidence_threshold: float,
    ) -> None:
        for stats in class_stats.values():
            stats.sure_semantic_threshold = semantic_threshold
            stats.sure_confidence_threshold = confidence_threshold

    def _maybe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.radial_l2_enabled and self.radial_beta is not None:
            return radial_l2_normalize(features, self.radial_beta)
        return features

    def _class_distances(self, features: torch.Tensor, stats: ClassCalibration) -> torch.Tensor:
        mean, var = self._stats_on_input_device(stats, features)
        return mahalanobis_distance(features, mean, var)

    def _class_ensemble_scores(
        self,
        features: torch.Tensor,
        energies: torch.Tensor,
        stats: ClassCalibration,
    ) -> torch.Tensor:
        distances = self._class_distances(features, stats)
        return ensemble_z_score(
            distances,
            energies,
            mahalanobis_mu=stats.mahalanobis_mu,
            mahalanobis_sigma=stats.mahalanobis_sigma,
            energy_mu=stats.energy_mu,
            energy_sigma=stats.energy_sigma,
        )

    @staticmethod
    def _subsample_evenly(features: torch.Tensor, limit: int) -> torch.Tensor:
        if int(features.shape[0]) <= int(limit):
            return features.detach().clone()
        indices = torch.div(
            torch.arange(int(limit), device=features.device) * int(features.shape[0]),
            int(limit),
            rounding_mode="floor",
        )
        return features.index_select(0, indices.to(dtype=torch.long)).detach().clone()

    def _build_knn_bank(self, class_features: torch.Tensor) -> torch.Tensor:
        return self._subsample_evenly(class_features, self.knn_bank_cap)

    @staticmethod
    def _knn_distances_from_bank(
        features: torch.Tensor,
        bank: Optional[torch.Tensor],
        *,
        k: int,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        batch_size = int(features.shape[0])
        if bank is None or bank.numel() <= 0 or batch_size <= 0:
            return torch.zeros(batch_size, dtype=features.dtype, device=features.device)
        if int(bank.shape[0]) <= 1 and exclude_self:
            return torch.zeros(batch_size, dtype=features.dtype, device=features.device)

        query = features.to(dtype=torch.float32)
        ref_bank = bank.to(device=features.device, dtype=torch.float32)
        distances = torch.cdist(query, ref_bank, p=2)

        if exclude_self and int(ref_bank.shape[0]) > 1:
            distances = distances.clone()
            for row_index in range(int(distances.shape[0])):
                zero_indices = torch.nonzero(distances[row_index] <= _FLOAT_EPS, as_tuple=False).reshape(-1)
                if int(zero_indices.numel()) > 0:
                    distances[row_index, int(zero_indices[0].item())] = float("inf")
            neighbor_count = min(max(1, int(k)), max(1, int(ref_bank.shape[0]) - 1))
        else:
            neighbor_count = min(max(1, int(k)), int(ref_bank.shape[0]))

        nearest = torch.topk(distances, k=neighbor_count, largest=False, dim=1).values
        return nearest.mean(dim=1).to(device=features.device, dtype=features.dtype)

    def _class_knn_distances(
        self,
        features: torch.Tensor,
        stats: ClassCalibration,
        *,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        bank = self._knn_bank_on_input_device(stats, features)
        return self._knn_distances_from_bank(
            features,
            bank,
            k=int(stats.knn_k),
            exclude_self=exclude_self,
        )

    def calibration_issue(self) -> Optional[str]:
        if not self.class_stats:
            return "OOD detector has no calibrated class statistics."
        if self.calibration_version <= 0:
            return "OOD detector calibration version is unset."
        if self.primary_score_method == "knn":
            missing_banks = [
                str(class_id)
                for class_id, stats in self.class_stats.items()
                if stats.knn_bank is None or stats.knn_bank.numel() <= 0
            ]
            if missing_banks:
                return "OOD detector is missing kNN calibration banks for configured primary_score_method='knn'."
        return None

    def assert_calibrated(self, *, operation: str) -> None:
        issue = self.calibration_issue()
        if issue is None:
            return
        raise RuntimeError(f"{issue} Run calibrate_ood(...) before {operation}.")

    def calibrate(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        if features.ndim != 2:
            raise ValueError("features must be [N, D]")
        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        if self.radial_l2_enabled:
            self.radial_beta = auto_tune_beta(
                features,
                labels,
                beta_range=self.radial_beta_range,
                beta_steps=self.radial_beta_steps,
            )
            features = radial_l2_normalize(features, self.radial_beta)

        self.class_stats = {}
        energies = self._energy(logits)
        class_ids = self._calibrated_class_ids(labels)

        for class_id in class_ids:
            mask = labels == class_id
            class_features = features[mask]
            class_energy = energies[mask]
            if class_features.numel() == 0:
                continue

            mean = class_features.mean(dim=0)
            var = class_features.var(dim=0, unbiased=False).clamp_min(_FLOAT_EPS)

            distances = mahalanobis_distance(class_features, mean, var)
            mahalanobis_mu = float(distances.mean().item())
            mahalanobis_sigma = float(self._safe_std(distances.std(unbiased=False)).item())

            energy_mu = float(class_energy.mean().item())
            energy_sigma = float(self._safe_std(class_energy.std(unbiased=False)).item())

            ensemble_scores = ensemble_z_score(
                distances,
                class_energy,
                mahalanobis_mu=mahalanobis_mu,
                mahalanobis_sigma=mahalanobis_sigma,
                energy_mu=energy_mu,
                energy_sigma=energy_sigma,
            )
            ensemble_threshold_value = _distribution_threshold(ensemble_scores, self.threshold_factor)
            energy_threshold_value = _distribution_threshold(class_energy, self.threshold_factor)

            knn_bank = self._build_knn_bank(class_features)
            knn_distances = self._knn_distances_from_bank(
                class_features,
                knn_bank,
                k=self.knn_k,
                exclude_self=True,
            )
            knn_distance_mu = float(knn_distances.mean().item())
            knn_distance_sigma = float(self._safe_std(knn_distances.std(unbiased=False)).item())
            knn_threshold_value = _distribution_threshold(knn_distances, self.threshold_factor)

            self.class_stats[int(class_id)] = ClassCalibration(
                mean=mean,
                var=var,
                mahalanobis_mu=mahalanobis_mu,
                mahalanobis_sigma=mahalanobis_sigma,
                energy_mu=energy_mu,
                energy_sigma=energy_sigma,
                threshold=ensemble_threshold_value,
                energy_threshold=energy_threshold_value,
                knn_distance_mu=knn_distance_mu,
                knn_distance_sigma=knn_distance_sigma,
                knn_threshold=knn_threshold_value,
                knn_bank=knn_bank,
                knn_k=self.knn_k,
            )

        if self.sure_enabled and self.class_stats:
            all_ensemble_scores = []
            all_confidence = []
            for class_id in class_ids:
                mask = labels == class_id
                if class_id not in self.class_stats:
                    continue
                stats = self.class_stats[class_id]
                class_features = features[mask]
                class_logits = logits[mask]
                class_energy = energies[mask]

                ensemble_scores = self._class_ensemble_scores(class_features, class_energy, stats)
                all_ensemble_scores.append(ensemble_scores)
                all_confidence.append(compute_confidence_score(class_logits))

            if all_ensemble_scores:
                cat_semantic = compute_semantic_score(torch.cat(all_ensemble_scores))
                cat_confidence = torch.cat(all_confidence)
                sem_thresh, conf_thresh = calibrate_sure_thresholds(
                    cat_semantic,
                    cat_confidence,
                    semantic_percentile=self.sure_semantic_percentile,
                    confidence_percentile=self.sure_confidence_percentile,
                )
                self._set_sure_thresholds(
                    self.class_stats,
                    semantic_threshold=sem_thresh,
                    confidence_threshold=conf_thresh,
                )

        if self.conformal_enabled and self.class_stats:
            all_ensemble_scores = []
            all_class_thresholds = []
            for class_id in class_ids:
                mask = labels == class_id
                if class_id not in self.class_stats:
                    continue
                stats = self.class_stats[class_id]
                class_features = features[mask]
                class_energy = energies[mask]

                ensemble_scores = self._class_ensemble_scores(class_features, class_energy, stats)
                all_ensemble_scores.append(ensemble_scores)
                all_class_thresholds.append(torch.full_like(ensemble_scores, stats.threshold))

            if all_ensemble_scores:
                cat_scores = torch.cat(all_ensemble_scores)
                cat_thresholds = torch.cat(all_class_thresholds)
                nc_scores = compute_nonconformity_scores(cat_scores, cat_thresholds)
                self.conformal_qhat = calibrate_conformal_qhat(nc_scores, alpha=self.conformal_alpha)

        self.calibration_version += 1
        return self._build_calibration_summary(
            class_count=len(self.class_stats),
            calibration_version=self.calibration_version,
            primary_score_method=self.primary_score_method,
            radial_beta=self.radial_beta,
            conformal_qhat=self.conformal_qhat,
            knn_k=self.knn_k,
            knn_bank_cap=self.knn_bank_cap,
        )

    def _score_class(self, class_id: int, feature: torch.Tensor, logit: torch.Tensor) -> Dict[str, Any]:
        if class_id not in self.class_stats:
            return self._default_score_result(self.primary_score_method)

        score_feature = feature.unsqueeze(0) if feature.ndim == 1 else feature
        score_feature = self._maybe_normalize(score_feature)

        stats = self.class_stats[class_id]
        distance = self._class_distances(score_feature, stats)
        energy = self._energy(logit.unsqueeze(0))
        ensemble = self._class_ensemble_scores(score_feature, energy, stats)
        knn_distance = self._class_knn_distances(score_feature, stats)

        mahalanobis_z = float(((distance[0] - stats.mahalanobis_mu) / max(stats.mahalanobis_sigma, _FLOAT_EPS)).item())
        energy_z = float(((energy[0] - stats.energy_mu) / max(stats.energy_sigma, _FLOAT_EPS)).item())
        candidate_scores = {
            "ensemble": float(ensemble[0].item()),
            "energy": float(energy[0].item()),
            "knn": float(knn_distance[0].item()),
        }
        candidate_thresholds = {
            "ensemble": float(stats.threshold),
            "energy": float(stats.energy_threshold),
            "knn": float(stats.knn_threshold),
        }
        primary_score = float(candidate_scores[self.primary_score_method])
        decision_threshold = float(candidate_thresholds[self.primary_score_method])

        result: Dict[str, Any] = {
            "mahalanobis_z": mahalanobis_z,
            "energy_z": energy_z,
            "ensemble_score": candidate_scores["ensemble"],
            "class_threshold": candidate_thresholds["ensemble"],
            "energy_score": candidate_scores["energy"],
            "energy_threshold": candidate_thresholds["energy"],
            "knn_distance": candidate_scores["knn"],
            "knn_threshold": candidate_thresholds["knn"],
            "primary_score": primary_score,
            "decision_threshold": decision_threshold,
            "is_ood": bool(primary_score > decision_threshold),
            "score_method": self.primary_score_method,
            "candidate_scores": candidate_scores,
            "candidate_thresholds": candidate_thresholds,
        }

        if self.sure_enabled:
            confidence_score = float((1.0 - torch.softmax(logit, dim=-1).max()).item())
            sure_decision = apply_sure_decision(
                candidate_scores["ensemble"],
                confidence_score,
                stats.sure_semantic_threshold,
                stats.sure_confidence_threshold,
            )
            result["sure_semantic_score"] = candidate_scores["ensemble"]
            result["sure_confidence_score"] = confidence_score
            result["sure_semantic_ood"] = sure_decision["semantic_ood"]
            result["sure_confidence_reject"] = sure_decision["confidence_reject"]
            if self.primary_score_method == "ensemble":
                result["is_ood"] = sure_decision["combined_reject"]
        else:
            result["sure_semantic_score"] = None
            result["sure_confidence_score"] = None
            result["sure_semantic_ood"] = None
            result["sure_confidence_reject"] = None

        return result

    def build_conformal_set(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        idx_to_class: Dict[int, str],
    ) -> List[str]:
        if not self.conformal_enabled or self.conformal_qhat is None:
            return []

        if features.ndim == 1:
            feat = self._maybe_normalize(features.unsqueeze(0)).squeeze(0)
        else:
            feat = self._maybe_normalize(features)

        return build_prediction_set(
            features=feat,
            logits=logits,
            qhat=self.conformal_qhat,
            class_stats=self.class_stats,
            idx_to_class=idx_to_class,
        )

    def score(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        predicted_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if features.ndim != 2 or logits.ndim != 2:
            raise ValueError("features/logits must be batched tensors [N, D] and [N, C]")

        if predicted_labels is None:
            predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = predicted_labels.reshape(-1)

        result: Dict[str, list] = {field: [] for field in (*_BASE_SCORE_FIELDS, *_SURE_SCORE_FIELDS)}
        candidate_scores: Dict[str, list] = {name: [] for name in SUPPORTED_OOD_SCORE_METHODS}
        candidate_thresholds: Dict[str, list] = {name: [] for name in SUPPORTED_OOD_SCORE_METHODS}

        for feat, logit, cls in zip(features, logits, predicted_labels):
            item = self._score_class(int(cls.item()), feat, logit)
            for field in (*_BASE_SCORE_FIELDS, *_SURE_SCORE_FIELDS):
                result[field].append(item[field])
            for method_name in SUPPORTED_OOD_SCORE_METHODS:
                candidate_scores[method_name].append(item["candidate_scores"][method_name])
                candidate_thresholds[method_name].append(item["candidate_thresholds"][method_name])

        output: Dict[str, Any] = {
            "mahalanobis_z": torch.tensor(result["mahalanobis_z"], dtype=torch.float32, device=features.device),
            "energy_z": torch.tensor(result["energy_z"], dtype=torch.float32, device=features.device),
            "ensemble_score": torch.tensor(result["ensemble_score"], dtype=torch.float32, device=features.device),
            "class_threshold": torch.tensor(result["class_threshold"], dtype=torch.float32, device=features.device),
            "energy_score": torch.tensor(result["energy_score"], dtype=torch.float32, device=features.device),
            "energy_threshold": torch.tensor(result["energy_threshold"], dtype=torch.float32, device=features.device),
            "knn_distance": torch.tensor(result["knn_distance"], dtype=torch.float32, device=features.device),
            "knn_threshold": torch.tensor(result["knn_threshold"], dtype=torch.float32, device=features.device),
            "primary_score": torch.tensor(result["primary_score"], dtype=torch.float32, device=features.device),
            "decision_threshold": torch.tensor(result["decision_threshold"], dtype=torch.float32, device=features.device),
            "is_ood": torch.tensor(result["is_ood"], dtype=torch.bool, device=features.device),
            "calibration_version": torch.full(
                (features.size(0),),
                self.calibration_version,
                dtype=torch.long,
                device=features.device,
            ),
            "primary_score_method": self.primary_score_method,
            "candidate_scores": {
                name: torch.tensor(values, dtype=torch.float32, device=features.device)
                for name, values in candidate_scores.items()
            },
            "candidate_thresholds": {
                name: torch.tensor(values, dtype=torch.float32, device=features.device)
                for name, values in candidate_thresholds.items()
            },
        }
        if self.sure_enabled:
            output["sure_semantic_score"] = torch.tensor(
                [value if value is not None else 0.0 for value in result["sure_semantic_score"]],
                dtype=torch.float32,
                device=features.device,
            )
            output["sure_confidence_score"] = torch.tensor(
                [value if value is not None else 0.0 for value in result["sure_confidence_score"]],
                dtype=torch.float32,
                device=features.device,
            )
            output["sure_semantic_ood"] = torch.tensor(
                [bool(value) if value is not None else False for value in result["sure_semantic_ood"]],
                dtype=torch.bool,
                device=features.device,
            )
            output["sure_confidence_reject"] = torch.tensor(
                [bool(value) if value is not None else False for value in result["sure_confidence_reject"]],
                dtype=torch.bool,
                device=features.device,
            )
        if self.radial_beta is not None:
            output["radial_beta"] = self.radial_beta
        return output
