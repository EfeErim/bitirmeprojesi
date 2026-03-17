"""Minimal OOD contract tests for the kept continual detector."""

import torch

from src.ood._scoring_utils import distribution_threshold, ensemble_threshold
from src.ood.continual_ood import ClassCalibration, ContinualOODDetector
from src.training.services.persistence import restore_ood_state


def test_distribution_threshold_alias_matches_ensemble_threshold():
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    assert distribution_threshold(values, 2.0) == ensemble_threshold(values, 2.0)


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

def test_sure_confidence_reject_does_not_override_primary_ood_decision():
    detector = ContinualOODDetector(threshold_factor=2.0, sure_enabled=True)
    detector.calibration_version = 1
    detector.class_stats[0] = ClassCalibration(
        mean=torch.zeros(2, dtype=torch.float32),
        var=torch.ones(2, dtype=torch.float32),
        mahalanobis_mu=0.0,
        mahalanobis_sigma=1.0,
        energy_mu=-10.0,
        energy_sigma=1.0,
        threshold=5.0,
        energy_threshold=-5.0,
        knn_distance_mu=0.0,
        knn_distance_sigma=1.0,
        knn_threshold=5.0,
        knn_bank=torch.zeros((1, 2), dtype=torch.float32),
        knn_k=1,
        sure_semantic_threshold=10.0,
        sure_confidence_threshold=0.001,
    )

    features = torch.zeros((1, 2), dtype=torch.float32)
    logits = torch.tensor([[6.0, 0.0]], dtype=torch.float32)
    result = detector.score(features, logits, predicted_labels=torch.tensor([0]))

    assert bool(result["sure_confidence_reject"][0].item()) is True
    assert bool(result["sure_semantic_ood"][0].item()) is False
    assert bool(result["is_ood"][0].item()) is False

