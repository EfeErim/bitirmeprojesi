import torch

from src.ood.dynamic_thresholds import DynamicOODThreshold
from src.ood.mahalanobis import MahalanobisDistance


def test_mahalanobis_initialization():
    prototypes = torch.randn(3, 768)
    class_stds = {
        0: torch.ones(768) * 0.1,
        1: torch.ones(768) * 0.2,
        2: torch.ones(768) * 0.3,
    }
    mahalanobis = MahalanobisDistance(prototypes, class_stds)
    assert mahalanobis.num_classes == 3
    assert mahalanobis.feature_dim == 768
    assert len(mahalanobis.inv_covariances) == 3


def test_mahalanobis_compute_distance_and_nearest_class():
    prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    class_stds = {
        0: torch.tensor([0.1, 0.1, 0.1]),
        1: torch.tensor([0.2, 0.2, 0.2]),
    }
    mahalanobis = MahalanobisDistance(prototypes, class_stds)

    features = torch.tensor([[1.1, 0.1, 0.1], [0.1, 1.1, 0.1]])
    distance = mahalanobis.compute_distance(features[:1], 0)
    assert distance.shape == (1,)
    assert distance.item() > 0

    nearest, distances = mahalanobis.get_nearest_class(features)
    assert distances.shape == (2,)
    assert nearest.tolist() == [0, 1]


def test_threshold_from_distances_and_fallback():
    threshold_computer = DynamicOODThreshold(threshold_factor=2.0, min_val_samples_per_class=5)
    distances_per_class = {
        0: [1.0, 1.2, 1.1, 0.9, 1.3, 1.0, 1.1],
        1: [2.0, 2.1, 1.9, 2.2, 2.0, 1.8, 2.1],
    }
    thresholds = threshold_computer.compute_thresholds_from_distances(distances_per_class)
    assert 0 in thresholds and 1 in thresholds
    assert thresholds[0] > 1.0

    fallback_computer = DynamicOODThreshold(
        threshold_factor=2.0,
        min_val_samples_per_class=10,
        fallback_threshold=25.0,
    )
    fallback_thresholds = fallback_computer.compute_thresholds_from_distances({0: [1.0, 1.2, 1.1]})
    assert fallback_thresholds[0] == 25.0
