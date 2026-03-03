import pytest
import torch

from src.ood.continual_ood import ContinualOODDetector


def test_calibration_increments_version_and_sets_class_stats():
    detector = ContinualOODDetector(threshold_factor=2.0)
    features = torch.randn(20, 8)
    logits = torch.randn(20, 3)
    labels = torch.tensor([0] * 10 + [1] * 10)

    out = detector.calibrate(features, logits, labels)

    assert int(out['calibration_version']) == 1
    assert len(detector.class_stats) == 2


def test_score_payload_contains_required_keys():
    detector = ContinualOODDetector(threshold_factor=1.0)
    features = torch.randn(12, 8)
    logits = torch.randn(12, 2)
    labels = torch.tensor([0] * 6 + [1] * 6)
    detector.calibrate(features, logits, labels)

    score = detector.score(features[:4], logits[:4])

    assert {'mahalanobis_z', 'energy_z', 'ensemble_score', 'class_threshold', 'is_ood', 'calibration_version'} <= set(score.keys())
    assert score['ensemble_score'].shape[0] == 4


def test_calibrate_rejects_invalid_feature_shape():
    detector = ContinualOODDetector()
    features = torch.randn(2, 3, 4)
    logits = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        detector.calibrate(features, logits, labels)


def test_score_rejects_invalid_shapes():
    detector = ContinualOODDetector()
    with pytest.raises(ValueError):
        detector.score(torch.randn(2, 3, 4), torch.randn(2, 2))


def test_score_for_unknown_class_uses_safe_defaults():
    detector = ContinualOODDetector()
    features = torch.randn(10, 8)
    logits = torch.randn(10, 2)
    labels = torch.tensor([0] * 5 + [1] * 5)
    detector.calibrate(features, logits, labels)

    probe_features = torch.randn(3, 8)
    probe_logits = torch.randn(3, 2)
    predicted = torch.full((3,), 99)
    score = detector.score(probe_features, probe_logits, predicted_labels=predicted)

    assert torch.all(score["ensemble_score"] == 0.0)
    assert torch.all(score["is_ood"] == torch.tensor([False, False, False]))
    assert torch.isinf(score["class_threshold"]).all()


def test_score_uses_explicit_predicted_labels_and_keeps_version():
    detector = ContinualOODDetector()
    features = torch.randn(8, 6)
    logits = torch.randn(8, 2)
    labels = torch.tensor([0] * 4 + [1] * 4)
    detector.calibrate(features, logits, labels)

    predicted = torch.tensor([1, 1, 1], dtype=torch.long)
    score = detector.score(features[:3], logits[:3], predicted_labels=predicted)

    assert score["calibration_version"].tolist() == [1, 1, 1]
    assert score["ensemble_score"].shape[0] == 3
