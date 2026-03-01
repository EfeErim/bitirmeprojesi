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
