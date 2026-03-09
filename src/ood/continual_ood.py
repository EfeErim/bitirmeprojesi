#!/usr/bin/env python3
"""Continual-learning OOD detector using Mahalanobis + energy ensemble.

Extended with:
  - Radially Scaled L2 Normalization (tunable β geometry)
  - SURE+ Double Scoring (semantic OOD + confidence rejection)
  - Conformal Prediction Guarantees (dynamic prediction sets with coverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.ood._scoring_utils import (
    energy_from_logits,
    ensemble_threshold,
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

_BASE_SCORE_FIELDS = (
    "mahalanobis_z",
    "energy_z",
    "ensemble_score",
    "class_threshold",
    "is_ood",
)
_SURE_SCORE_FIELDS = (
    "sure_semantic_score",
    "sure_confidence_score",
    "sure_semantic_ood",
    "sure_confidence_reject",
)


@dataclass
class ClassCalibration:
    mean: torch.Tensor
    var: torch.Tensor
    mahalanobis_mu: float
    mahalanobis_sigma: float
    energy_mu: float
    energy_sigma: float
    threshold: float
    # --- SURE+ thresholds ---
    sure_semantic_threshold: float = 0.0
    sure_confidence_threshold: float = 0.0


class ContinualOODDetector:
    """OOD detector with weighted z-score ensemble: 0.6*M + 0.4*E.

    Optionally enhanced with radial L2 normalization, SURE+ double scoring,
    and conformal prediction guarantees.
    """

    def __init__(
        self,
        threshold_factor: float = 2.0,
        *,
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
        self.class_stats: Dict[int, ClassCalibration] = {}
        self.calibration_version = 0

        # Radially Scaled L2 Normalization
        self.radial_l2_enabled = bool(radial_l2_enabled)
        self.radial_beta_range = (float(radial_beta_range[0]), float(radial_beta_range[1]))
        self.radial_beta_steps = int(radial_beta_steps)
        self.radial_beta: Optional[float] = None

        # SURE+ Double Scoring
        self.sure_enabled = bool(sure_enabled)
        self.sure_semantic_percentile = float(sure_semantic_percentile)
        self.sure_confidence_percentile = float(sure_confidence_percentile)

        # Conformal Prediction
        self.conformal_enabled = bool(conformal_enabled)
        self.conformal_alpha = float(conformal_alpha)
        self.conformal_qhat: Optional[float] = None

    @staticmethod
    def _safe_std(value: torch.Tensor) -> torch.Tensor:
        return safe_std(value)

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
        radial_beta: Optional[float],
        conformal_qhat: Optional[float],
    ) -> Dict[str, float]:
        result: Dict[str, float] = {
            "num_classes": float(class_count),
            "calibration_version": float(calibration_version),
        }
        if radial_beta is not None:
            result["radial_beta"] = float(radial_beta)
        if conformal_qhat is not None:
            result["conformal_qhat"] = float(conformal_qhat)
        return result

    @staticmethod
    def _default_score_result() -> Dict[str, Any]:
        return {
            "mahalanobis_z": 0.0,
            "energy_z": 0.0,
            "ensemble_score": 0.0,
            "class_threshold": float("inf"),
            "is_ood": False,
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

    @staticmethod
    def _build_score_output(
        *,
        result: Dict[str, list],
        device: torch.device | str,
        batch_size: int,
        calibration_version: int,
        sure_enabled: bool,
        radial_beta: Optional[float],
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "mahalanobis_z": torch.tensor(result["mahalanobis_z"], dtype=torch.float32, device=device),
            "energy_z": torch.tensor(result["energy_z"], dtype=torch.float32, device=device),
            "ensemble_score": torch.tensor(result["ensemble_score"], dtype=torch.float32, device=device),
            "class_threshold": torch.tensor(result["class_threshold"], dtype=torch.float32, device=device),
            "is_ood": torch.tensor(result["is_ood"], dtype=torch.bool, device=device),
            "calibration_version": torch.full(
                (batch_size,),
                calibration_version,
                dtype=torch.long,
                device=device,
            ),
        }
        if sure_enabled:
            output["sure_semantic_score"] = torch.tensor(
                [s if s is not None else 0.0 for s in result["sure_semantic_score"]],
                dtype=torch.float32,
                device=device,
            )
            output["sure_confidence_score"] = torch.tensor(
                [s if s is not None else 0.0 for s in result["sure_confidence_score"]],
                dtype=torch.float32,
                device=device,
            )
            output["sure_semantic_ood"] = torch.tensor(
                [s if s is not None else False for s in result["sure_semantic_ood"]],
                dtype=torch.bool,
                device=device,
            )
            output["sure_confidence_reject"] = torch.tensor(
                [s if s is not None else False for s in result["sure_confidence_reject"]],
                dtype=torch.bool,
                device=device,
            )
        if radial_beta is not None:
            output["radial_beta"] = radial_beta
        return output

    def _maybe_normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Apply radial L2 normalization if enabled and β is calibrated."""
        if self.radial_l2_enabled and self.radial_beta is not None:
            return radial_l2_normalize(features, self.radial_beta)
        return features

    def _class_distances(
        self,
        features: torch.Tensor,
        stats: ClassCalibration,
    ) -> torch.Tensor:
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

    def calibration_issue(self) -> Optional[str]:
        """Return a human-readable calibration problem, if any."""
        if not self.class_stats:
            return "OOD detector has no calibrated class statistics."
        if self.calibration_version <= 0:
            return "OOD detector calibration version is unset."
        return None

    def assert_calibrated(self, *, operation: str) -> None:
        issue = self.calibration_issue()
        if issue is None:
            return
        raise RuntimeError(f"{issue} Run calibrate_ood(...) before {operation}.")

    def calibrate(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if features.ndim != 2:
            raise ValueError("features must be [N, D]")
        if logits.ndim != 2:
            raise ValueError("logits must be [N, C]")
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        # --- Phase 0: Auto-tune β if radial normalization enabled ---
        if self.radial_l2_enabled:
            self.radial_beta = auto_tune_beta(
                features,
                labels,
                beta_range=self.radial_beta_range,
                beta_steps=self.radial_beta_steps,
            )
            features = radial_l2_normalize(features, self.radial_beta)

        # --- Phase 1: Per-class mean/var + energy stats ---
        self.class_stats = {}
        energies = self._energy(logits)

        class_ids = self._calibrated_class_ids(labels)
        for class_id in class_ids:
            mask = labels == class_id
            class_features = features[mask]
            class_logits = logits[mask]
            class_energy = energies[mask]
            if class_features.numel() == 0:
                continue

            mean = class_features.mean(dim=0)
            var = class_features.var(dim=0, unbiased=False).clamp_min(1e-6)

            distances = mahalanobis_distance(class_features, mean, var)
            m_mu = float(distances.mean().item())
            m_sigma = float(self._safe_std(distances.std(unbiased=False)).item())

            e_mu = float(class_energy.mean().item())
            e_sigma = float(self._safe_std(class_energy.std(unbiased=False)).item())

            ensemble = ensemble_z_score(
                distances,
                class_energy,
                mahalanobis_mu=m_mu,
                mahalanobis_sigma=m_sigma,
                energy_mu=e_mu,
                energy_sigma=e_sigma,
            )
            threshold = ensemble_threshold(ensemble, self.threshold_factor)

            self.class_stats[int(class_id)] = ClassCalibration(
                mean=mean,
                var=var,
                mahalanobis_mu=m_mu,
                mahalanobis_sigma=m_sigma,
                energy_mu=e_mu,
                energy_sigma=e_sigma,
                threshold=threshold,
            )

        # --- Phase 2: SURE+ threshold calibration ---
        if self.sure_enabled and self.class_stats:
            all_ensemble_z = []
            all_confidence = []
            for class_id in class_ids:
                mask = labels == class_id
                if class_id not in self.class_stats:
                    continue
                stats = self.class_stats[class_id]
                class_features = features[mask]
                class_logits = logits[mask]
                class_energy = energies[mask]

                ensemble_z = self._class_ensemble_scores(class_features, class_energy, stats)
                conf_scores = compute_confidence_score(class_logits)

                all_ensemble_z.append(ensemble_z)
                all_confidence.append(conf_scores)

            if all_ensemble_z:
                cat_semantic = compute_semantic_score(torch.cat(all_ensemble_z))
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

        # --- Phase 3: Conformal prediction calibration ---
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

                ensemble_z = self._class_ensemble_scores(class_features, class_energy, stats)

                all_ensemble_scores.append(ensemble_z)
                all_class_thresholds.append(torch.full_like(ensemble_z, stats.threshold))

            if all_ensemble_scores:
                cat_scores = torch.cat(all_ensemble_scores)
                cat_thresholds = torch.cat(all_class_thresholds)
                nc_scores = compute_nonconformity_scores(cat_scores, cat_thresholds)
                self.conformal_qhat = calibrate_conformal_qhat(
                    nc_scores, alpha=self.conformal_alpha
                )

        self.calibration_version += 1
        return self._build_calibration_summary(
            class_count=len(self.class_stats),
            calibration_version=self.calibration_version,
            radial_beta=self.radial_beta,
            conformal_qhat=self.conformal_qhat,
        )

    def _score_class(self, class_id: int, feature: torch.Tensor, logit: torch.Tensor) -> Dict[str, Any]:
        if class_id not in self.class_stats:
            return self._default_score_result()

        score_feature = feature
        if feature.ndim == 1:
            score_feature = self._maybe_normalize(feature.unsqueeze(0)).squeeze(0)
        else:
            score_feature = self._maybe_normalize(feature)

        stats = self.class_stats[class_id]
        distance = self._class_distances(score_feature, stats)
        energy = self._energy(logit.unsqueeze(0))[0]
        ensemble = self._class_ensemble_scores(score_feature, energy, stats)

        m_z = float(((distance - stats.mahalanobis_mu) / max(stats.mahalanobis_sigma, 1e-6)).item())
        e_z = float(((energy - stats.energy_mu) / max(stats.energy_sigma, 1e-6)).item())
        threshold = float(stats.threshold)

        result: Dict[str, Any] = {
            "mahalanobis_z": m_z,
            "energy_z": e_z,
            "ensemble_score": float(ensemble.item()),
            "class_threshold": threshold,
            "is_ood": bool(ensemble.item() > threshold),
        }

        # SURE+ scoring
        if self.sure_enabled:
            semantic_score = float(ensemble.item())  # Semantic score IS the ensemble score
            confidence_score = float((1.0 - torch.softmax(logit, dim=-1).max()).item())
            sure_decision = apply_sure_decision(
                semantic_score,
                confidence_score,
                stats.sure_semantic_threshold,
                stats.sure_confidence_threshold,
            )
            result["sure_semantic_score"] = semantic_score
            result["sure_confidence_score"] = confidence_score
            result["sure_semantic_ood"] = sure_decision["semantic_ood"]
            result["sure_confidence_reject"] = sure_decision["confidence_reject"]
            # Override is_ood with SURE+ combined decision
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
        """Build conformal prediction set for a single sample.

        Args:
            features: Single feature vector ``[D]``.
            logits: Single logit vector ``[C]``.
            idx_to_class: Class index to name mapping.

        Returns:
            List of class names in the prediction set.
        """
        if not self.conformal_enabled or self.conformal_qhat is None:
            return []

        # Apply radial normalization to features
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

        for feat, logit, cls in zip(features, logits, predicted_labels):
            item = self._score_class(int(cls.item()), feat, logit)
            for field in (*_BASE_SCORE_FIELDS, *_SURE_SCORE_FIELDS):
                result[field].append(item[field])

        return self._build_score_output(
            result=result,
            device=features.device,
            batch_size=features.size(0),
            calibration_version=self.calibration_version,
            sure_enabled=self.sure_enabled,
            radial_beta=self.radial_beta,
        )
