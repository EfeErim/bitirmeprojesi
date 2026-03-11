"""Minimal OOD contract tests for the kept continual detector."""

import torch

from src.ood.continual_ood import ContinualOODDetector
from src.training.services.persistence import restore_ood_state


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
        "energy_score",
        "energy_threshold",
        "knn_distance",
        "knn_threshold",
        "primary_score",
        "decision_threshold",
        "is_ood",
        "calibration_version",
    } <= set(score.keys())
    assert score["ensemble_score"].shape[0] == 6
    assert score["candidate_scores"]["knn"].shape[0] == 6


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


def test_restore_legacy_ood_state_defaults_primary_score_to_ensemble():
    restored = restore_ood_state(
        {
            "threshold_factor": 2.0,
            "calibration_version": 1,
            "class_stats": {
                "0": {
                    "mean": [0.0, 0.0],
                    "var": [1.0, 1.0],
                    "mahalanobis_mu": 0.1,
                    "mahalanobis_sigma": 0.2,
                    "energy_mu": 0.3,
                    "energy_sigma": 0.4,
                    "threshold": 0.5,
                }
            },
        },
        default_threshold_factor=2.0,
        device="cpu",
    )

    assert restored.primary_score_method == "ensemble"
    assert restored.class_stats[0].energy_threshold > 0.0
