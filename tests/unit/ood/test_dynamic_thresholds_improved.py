import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ood.dynamic_thresholds import DynamicOODThreshold


class _DummyModel(torch.nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        batch_size = x.shape[0]
        features = torch.randn(batch_size, 1, self.feature_dim)
        return type('Output', (), {'last_hidden_state': features})()


class _DummyMahalanobis:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def compute_distance(self, features, class_idx):
        batch_size = features.shape[0]
        return torch.randn(batch_size) + class_idx * 2.0


def _make_loader(num_classes: int, samples_per_class: int) -> DataLoader:
    total = num_classes * samples_per_class
    data = torch.randn(total, 3, 64, 64)
    labels = torch.cat([torch.full((samples_per_class,), idx, dtype=torch.long) for idx in range(num_classes)])
    return DataLoader(TensorDataset(data, labels), batch_size=16)


def test_compute_thresholds_with_sufficient_samples():
    num_classes = 3
    feature_dim = 64
    thresholds = DynamicOODThreshold.compute_thresholds(
        mahalanobis=_DummyMahalanobis(num_classes),
        model=_DummyModel(feature_dim),
        val_loader=_make_loader(num_classes, samples_per_class=40),
        feature_dim=feature_dim,
        device='cpu',
    )
    assert len(thresholds) == num_classes
    assert all(isinstance(value, float) for value in thresholds.values())


def test_compute_thresholds_with_insufficient_samples_uses_fallback_path():
    num_classes = 3
    feature_dim = 64
    thresholds = DynamicOODThreshold.compute_thresholds(
        mahalanobis=_DummyMahalanobis(num_classes),
        model=_DummyModel(feature_dim),
        val_loader=_make_loader(num_classes, samples_per_class=5),
        feature_dim=feature_dim,
        device='cpu',
    )
    assert len(thresholds) == num_classes


def test_custom_configuration_is_accepted():
    threshold_computer = DynamicOODThreshold(
        threshold_factor=2.576,
        min_val_samples_per_class=50,
        fallback_threshold=30.0,
        confidence_level=0.99,
        max_fallback_threshold=60.0,
        min_fallback_threshold=15.0,
    )
    assert threshold_computer.threshold_factor == 2.576
    assert threshold_computer.confidence_level == 0.99


def test_confidence_interval_contains_sample_mean():
    np.random.seed(42)
    values = np.random.normal(10.0, 2.0, 50)
    computer = DynamicOODThreshold()
    lower, upper = computer._compute_confidence_interval(values, 0.95)
    mean = float(np.mean(values))
    assert lower <= mean <= upper
