"""Conformal Prediction guarantees for OOD detection.

Casts OOD ensemble scores into the split conformal prediction framework,
providing dynamic prediction sets with rigorous statistical coverage
guarantees: P(y_true ∈ C(x)) ≥ 1 − α under exchangeability.

Reference: Conformal Prediction for OOD detection via statistical hypothesis testing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


def compute_nonconformity_scores(
    ensemble_scores: torch.Tensor,
    class_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Convert OOD ensemble scores to non-conformity scores.

    For each calibration sample, the non-conformity score is the signed distance
    from the per-class decision boundary:  s_i = ensemble_i − threshold_{y_i}.

    Positive values indicate the sample exceeds the OOD boundary (more non-conforming).

    Args:
        ensemble_scores: Per-sample ensemble z-scores ``[N]``.
        class_thresholds: Per-sample class thresholds ``[N]`` (looked up from predicted label).

    Returns:
        Non-conformity scores ``[N]``.
    """
    return ensemble_scores - class_thresholds


def calibrate_conformal_qhat(
    nonconformity_scores: torch.Tensor,
    alpha: float = 0.05,
) -> float:
    """Compute the conformal quantile q̂ for split conformal prediction.

    q̂ = Quantile(s, ⌈(1−α)(1+1/n)⌉ / 1)  — the adjusted (1−α) quantile
    of the calibration non-conformity scores.

    Args:
        nonconformity_scores: Calibration non-conformity scores ``[N]``.
        alpha: Desired miscoverage rate (e.g. 0.05 for 95% coverage).

    Returns:
        The conformal quantile threshold q̂.
    """
    if nonconformity_scores.numel() == 0:
        return float("inf")

    n = nonconformity_scores.numel()
    # Adjusted quantile level for finite-sample coverage
    adjusted_quantile = min(1.0, math.ceil((1 - alpha) * (n + 1)) / n)

    return float(
        torch.quantile(
            nonconformity_scores.float(),
            min(max(adjusted_quantile, 0.0), 1.0),
        ).item()
    )


def score_all_classes(
    features: torch.Tensor,
    logits: torch.Tensor,
    class_stats: Dict[int, Any],
) -> Dict[int, float]:
    """Score a single sample against every calibrated class.

    For each known class, computes the ensemble z-score as if that class were
    the predicted class.  This is needed to build the conformal prediction set.

    Args:
        features: Single feature vector ``[D]``.
        logits: Single logit vector ``[C]``.
        class_stats: Mapping ``{class_id: ClassCalibration}``.

    Returns:
        ``{class_id: ensemble_score}`` for every calibrated class.
    """
    scores: Dict[int, float] = {}
    energy = float((-torch.logsumexp(logits.unsqueeze(0), dim=1))[0].item())

    for class_id, stats in class_stats.items():
        mean = stats.mean.to(device=features.device, dtype=features.dtype)
        var = stats.var.to(device=features.device, dtype=features.dtype)
        # Mahalanobis distance against this class
        distance = float(
            torch.sqrt(((features - mean) ** 2 / var).sum()).item()
        )
        m_z = (distance - stats.mahalanobis_mu) / max(stats.mahalanobis_sigma, 1e-6)
        e_z = (energy - stats.energy_mu) / max(stats.energy_sigma, 1e-6)
        ensemble = 0.6 * m_z + 0.4 * e_z
        scores[int(class_id)] = float(ensemble)

    return scores


def build_prediction_set(
    features: torch.Tensor,
    logits: torch.Tensor,
    qhat: float,
    class_stats: Dict[int, Any],
    idx_to_class: Dict[int, str],
) -> List[str]:
    """Build the conformal prediction set for a single test sample.

    A class k is included in the set if its non-conformity score ≤ q̂, i.e.
    the sample is *not sufficiently non-conforming* w.r.t. class k to exclude it.

    By construction, P(y_true ∈ C(x)) ≥ 1 − α under exchangeability.

    Args:
        features: Single feature vector ``[D]``.
        logits: Single logit vector ``[C]``.
        qhat: Conformal quantile from calibration.
        class_stats: ``{class_id: ClassCalibration}`` with per-class OOD stats.
        idx_to_class: ``{class_id: class_name}`` mapping.

    Returns:
        List of class names included in the prediction set, sorted by score (ascending).
    """
    if qhat == float("inf"):
        # No valid calibration — return all classes
        return sorted(idx_to_class.values())

    all_scores = score_all_classes(features, logits, class_stats)

    included: List[tuple[float, str]] = []
    for class_id, ensemble_score in all_scores.items():
        threshold = class_stats[class_id].threshold
        nonconformity = ensemble_score - threshold
        if nonconformity <= qhat:
            class_name = idx_to_class.get(class_id, str(class_id))
            included.append((ensemble_score, class_name))

    # Sort by ensemble score ascending (most conforming first)
    included.sort(key=lambda x: x[0])
    return [name for _, name in included]


def compute_empirical_coverage(
    true_labels: torch.Tensor,
    prediction_sets: List[List[int]],
) -> float:
    """Compute empirical coverage: fraction of samples where true label ∈ prediction set.

    Args:
        true_labels: Ground-truth class indices ``[N]``.
        prediction_sets: List of N prediction sets (each a list of class indices).

    Returns:
        Empirical coverage in [0, 1].
    """
    if len(prediction_sets) == 0:
        return 0.0

    covered = 0
    for label, pred_set in zip(true_labels.tolist(), prediction_sets):
        if int(label) in pred_set:
            covered += 1

    return float(covered) / float(len(prediction_sets))
