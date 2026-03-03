"""Canonical OOD unit suite consolidating continual and dynamic threshold checks."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from src.ood.continual_ood import ContinualOODDetector
from src.ood.dynamic_thresholds import DynamicOODThreshold, calibrate_thresholds_using_validation


def test_continual_ood_calibration_and_score_contract():
    detector = ContinualOODDetector(threshold_factor=2.0)
    features = torch.randn(24, 8)
    logits = torch.randn(24, 3)
    labels = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)

    calibration = detector.calibrate(features, logits, labels)
    score = detector.score(features[:6], logits[:6])

    assert int(calibration["calibration_version"]) == 1
    assert int(calibration["num_classes"]) == 3
    assert {
        "mahalanobis_z",
        "energy_z",
        "ensemble_score",
        "class_threshold",
        "is_ood",
        "calibration_version",
    } <= set(score.keys())
    assert score["ensemble_score"].shape[0] == 6


def test_continual_ood_unknown_class_safe_defaults():
    detector = ContinualOODDetector()
    features = torch.randn(12, 6)
    logits = torch.randn(12, 2)
    labels = torch.tensor([0] * 6 + [1] * 6)
    detector.calibrate(features, logits, labels)

    predicted = torch.full((4,), 99)
    out = detector.score(features[:4], logits[:4], predicted_labels=predicted)

    assert torch.all(out["ensemble_score"] == 0.0)
    assert torch.isinf(out["class_threshold"]).all()
    assert torch.equal(out["is_ood"], torch.tensor([False, False, False, False]))


def test_dynamic_threshold_validation_metrics_shape():
    threshold = DynamicOODThreshold()
    thresholds = {0: 25.0}

    val_loader = MagicMock()
    val_loader.__iter__.return_value = []
    val_loader.__len__.return_value = 0

    model = MagicMock()
    model.eval = MagicMock()

    mahalanobis = MagicMock()
    mahalanobis.compute_distance = MagicMock(return_value=torch.tensor(10.0))

    metrics = threshold.validate_thresholds(thresholds, val_loader, model, mahalanobis)

    assert {
        "false_positive_rate",
        "true_negative_rate",
        "total_in_dist_samples",
        "num_classes_tested",
    } <= set(metrics.keys())


def test_calibrate_thresholds_using_validation_target_fpr_percentile_behavior():
    distances = np.random.normal(10.0, 1.0, 100)

    model = MagicMock()
    model.eval = MagicMock()

    val_loader = MagicMock()

    def _iter():
        batch_size = 20
        for i in range(0, 100, batch_size):
            images = torch.randn(batch_size, 3, 224, 224)
            labels = torch.zeros(batch_size, dtype=torch.long)
            for j in range(batch_size):
                images[j, 0, 0, 0] = float(distances[i + j])
            yield images, labels

    val_loader.__iter__.return_value = _iter()
    val_loader.__len__.return_value = 5

    mahalanobis = MagicMock()
    mahalanobis.compute_distance = lambda features, class_idx: torch.tensor(float(features[0, 0, 0, 0].item()))

    thresholds = calibrate_thresholds_using_validation(
        model=model,
        val_loader=val_loader,
        mahalanobis=mahalanobis,
        target_fpr=0.10,
        device="cpu",
        min_samples=10,
    )

    expected_90th = np.percentile(distances, 90)
    assert 0 in thresholds
    assert abs(thresholds[0] - expected_90th) < 2.0

