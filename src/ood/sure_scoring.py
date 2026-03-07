"""SURE+ Double Scoring Functions for simultaneous OOD detection and failure prediction.

Implements independent semantic shift and predictive confidence scoring with
separate calibrated thresholds, enabling joint evaluation via the DS-F1 metric.

Reference: SURE+ — unifying OOD detection and failure prediction (2026).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def compute_semantic_score(ensemble_z: torch.Tensor) -> torch.Tensor:
    """Compute the semantic shift score (reuses the Mahalanobis+Energy ensemble z-score).

    Higher values indicate greater semantic deviation from known class manifolds.

    Args:
        ensemble_z: Pre-computed ensemble z-scores ``[N]``.

    Returns:
        Semantic OOD scores ``[N]`` (identity mapping — semantic score IS the ensemble score).
    """
    return ensemble_z


def compute_confidence_score(logits: torch.Tensor) -> torch.Tensor:
    """Compute the predictive confidence score (inverse of max softmax probability).

    Higher values indicate lower model confidence — potential misclassification.

    Args:
        logits: Raw logit tensor ``[N, C]``.

    Returns:
        Confidence rejection scores ``[N]``. Range [0, 1); higher = less confident.
    """
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return 1.0 - max_probs


def calibrate_sure_thresholds(
    semantic_scores: torch.Tensor,
    confidence_scores: torch.Tensor,
    semantic_percentile: float = 95.0,
    confidence_percentile: float = 90.0,
) -> Tuple[float, float]:
    """Calibrate independent SURE+ thresholds from in-distribution calibration data.

    Thresholds are set at the specified percentiles of the in-distribution score
    distributions.  Samples exceeding either threshold at inference time are
    flagged for rejection.

    Args:
        semantic_scores: In-distribution semantic scores ``[N]``.
        confidence_scores: In-distribution confidence scores ``[N]``.
        semantic_percentile: Percentile (0–100) for semantic OOD threshold.
        confidence_percentile: Percentile (0–100) for confidence rejection threshold.

    Returns:
        ``(semantic_threshold, confidence_threshold)``
    """
    if semantic_scores.numel() == 0 or confidence_scores.numel() == 0:
        return 0.0, 0.0

    semantic_q = float(semantic_percentile) / 100.0
    confidence_q = float(confidence_percentile) / 100.0

    semantic_threshold = float(
        torch.quantile(semantic_scores.float(), min(max(semantic_q, 0.0), 1.0)).item()
    )
    confidence_threshold = float(
        torch.quantile(confidence_scores.float(), min(max(confidence_q, 0.0), 1.0)).item()
    )
    return semantic_threshold, confidence_threshold


def apply_sure_decision(
    semantic_score: float,
    confidence_score: float,
    semantic_threshold: float,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Apply SURE+ double-scoring decision logic.

    A sample is rejected if *either* score exceeds its independent threshold.

    Args:
        semantic_score: Semantic shift score for the sample.
        confidence_score: Predictive confidence score for the sample.
        semantic_threshold: Calibrated semantic OOD threshold.
        confidence_threshold: Calibrated confidence rejection threshold.

    Returns:
        Dictionary with ``semantic_ood``, ``confidence_reject``, and ``combined_reject`` flags.
    """
    semantic_ood = bool(semantic_score > semantic_threshold)
    confidence_reject = bool(confidence_score > confidence_threshold)
    return {
        "semantic_ood": semantic_ood,
        "confidence_reject": confidence_reject,
        "combined_reject": semantic_ood or confidence_reject,
    }


def compute_ds_f1(
    semantic_ood_labels: torch.Tensor,
    confidence_reject_labels: torch.Tensor,
    semantic_predictions: torch.Tensor,
    confidence_predictions: torch.Tensor,
) -> Dict[str, float]:
    """Compute the DS-F1 metric combining semantic OOD and confidence rejection F1.

    Args:
        semantic_ood_labels: Ground-truth binary OOD labels ``[N]``.
        confidence_reject_labels: Ground-truth binary misclassification labels ``[N]``.
        semantic_predictions: Binary semantic OOD predictions ``[N]``.
        confidence_predictions: Binary confidence rejection predictions ``[N]``.

    Returns:
        Dictionary with ``semantic_f1``, ``confidence_f1``, and ``ds_f1`` (harmonic mean).
    """

    def _binary_f1(labels: torch.Tensor, preds: torch.Tensor) -> float:
        labels = labels.bool()
        preds = preds.bool()
        tp = float((labels & preds).sum().item())
        fp = float((~labels & preds).sum().item())
        fn = float((labels & ~preds).sum().item())
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        if precision + recall < 1e-8:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    semantic_f1 = _binary_f1(semantic_ood_labels, semantic_predictions)
    confidence_f1 = _binary_f1(confidence_reject_labels, confidence_predictions)

    if semantic_f1 + confidence_f1 < 1e-8:
        ds_f1 = 0.0
    else:
        ds_f1 = float(2.0 * semantic_f1 * confidence_f1 / (semantic_f1 + confidence_f1))

    return {
        "semantic_f1": semantic_f1,
        "confidence_f1": confidence_f1,
        "ds_f1": ds_f1,
    }
