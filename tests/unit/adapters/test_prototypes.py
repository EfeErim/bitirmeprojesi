#!/usr/bin/env python3
"""
Tests for OOD prototypes module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory, mock_dataset_factory
from src.ood.prototypes import (
    PrototypeComputer,
    PrototypeConfig,
    compute_prototypes,
    update_prototypes_moving_average,
    find_nearest_prototype,
    compute_prototype_accuracy
)


class TestPrototypeComputer:
    """Test prototype computation functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample prototype configuration."""
        return PrototypeConfig(
            feature_dim=128,
            device="cpu",
            use_moving_average=True,
            update_rate=0.1,
            min_samples=5,
            max_prototypes=1000
        )

    @pytest.fixture
    def sample_features(self):
        """Create sample feature vectors."""
        return torch.randn(50, 128)

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for 3 classes."""
        return torch.randint(0, 3, (50,))

    def test_computer_initialization(self, sample_config):
        """Test PrototypeComputer initialization."""
        computer = PrototypeComputer(config=sample_config)
        assert computer.config == sample_config
        assert computer.config.feature_dim == 128
        assert computer.config.device == "cpu"

    def test_compute_prototypes_from_features(self, sample_config, sample_features, sample_labels):
        """Test computing prototypes from features."""
        computer = PrototypeComputer(config=sample_config)

        prototypes, class_stds = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Should have prototypes for each class
        num_classes = len(torch.unique(sample_labels))
        assert prototypes.shape[0] == num_classes
        assert prototypes.shape[1] == 128
        assert len(class_stds) == num_classes

    def test_prototype_caching(self, sample_config, sample_features, sample_labels):
        """Test prototype caching functionality."""
        computer = PrototypeComputer(config=sample_config)

        # First computation - cache miss
        prototypes1, stds1 = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )
        stats1 = computer.get_prototype_cache_stats()

        # Second computation for same classes - should be cache hit
        prototypes2, stds2 = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )
        stats2 = computer.get_prototype_cache_stats()

        # Cache hits should increase
        assert stats2['hits'] > stats1['hits']
        # Prototypes should be identical
        assert torch.allclose(prototypes1, prototypes2)

    def test_get_prototype_for_class(self, sample_config, sample_features, sample_labels):
        """Test retrieving prototype for specific class."""
        computer = PrototypeComputer(config=sample_config)

        # Compute prototypes first
        prototypes, _ = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Get prototype for class 0
        proto0 = computer.get_prototype_for_class(0)
        assert proto0 is not None
        assert proto0.shape[0] == 128

        # Should match the computed prototype
        assert torch.allclose(proto0, prototypes[0])

    def test_clear_prototype_cache(self, sample_config, sample_features, sample_labels):
        """Test clearing prototype cache."""
        computer = PrototypeComputer(config=sample_config)

        # Compute and cache
        computer.compute_prototypes_from_features(sample_features, sample_labels)
        stats_before = computer.get_prototype_cache_stats()
        assert stats_before['cache_size'] > 0

        # Clear cache
        computer.clear_prototype_cache()
        stats_after = computer.get_prototype_cache_stats()
        assert stats_after['cache_size'] == 0
        assert stats_after['hits'] == 0
        assert stats_after['misses'] == 0

    def test_update_prototype(self, sample_config):
        """Test updating a single prototype."""
        computer = PrototypeComputer(config=sample_config)

        # Set initial prototype
        initial_proto = torch.ones(128) * 0.5
        computer.prototypes[0] = initial_proto
        computer.prototype_counts[0] = 10

        # New features for class 0
        new_features = torch.randn(5, 128)

        # Update prototype
        updated = computer.update_prototype(0, new_features)

        assert updated is not None
        assert updated.shape == (128,)
        # Should be different from initial (moving average)
        assert not torch.allclose(updated, initial_proto)

    def test_prototype_accuracy(self, sample_config, sample_features, sample_labels):
        """Test prototype-based classification accuracy."""
        computer = PrototypeComputer(config=sample_config)

        # Compute prototypes
        prototypes, _ = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Compute accuracy on training data
        accuracy = compute_prototype_accuracy(sample_features, sample_labels, prototypes)
        assert 0 <= accuracy <= 1

        # Should be reasonably high for training data
        assert accuracy > 0.5  # At least better than random for 3 classes

    def test_find_nearest_prototype(self, sample_config):
        """Test finding nearest prototype."""
        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Test feature closest to prototype 0
        feature = torch.tensor([0.9, 0.1, 0.0])
        nearest_idx, distance = find_nearest_prototype(feature, prototypes)

        assert nearest_idx == 0
        assert distance < 0.5

        # Test feature exactly at prototype 1
        feature = torch.tensor([0.0, 1.0, 0.0])
        nearest_idx, distance = find_nearest_prototype(feature, prototypes)
        assert nearest_idx == 1
        assert distance == 0.0

    def test_prototype_consistency(self, sample_config, sample_features, sample_labels):
        """Test that prototypes are consistent across multiple computations."""
        computer = PrototypeComputer(config=sample_config)

        # Compute multiple times
        prototypes1, _ = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )
        prototypes2, _ = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Should be identical (deterministic)
        assert torch.allclose(prototypes1, prototypes2)

    def test_prototype_with_insufficient_samples(self, sample_config):
        """Test handling of classes with insufficient samples."""
        computer = PrototypeComputer(config=sample_config, min_samples=5)

        # Create features with only 2 samples for class 0
        features = torch.randn(10, 128)
        labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])

        prototypes, class_stds = computer.compute_prototypes_from_features(
            features, labels
        )

        # Should still create prototype but may have special handling
        assert prototypes is not None

    def test_prototype_std_computation(self, sample_config, sample_features, sample_labels):
        """Test that class standard deviations are computed correctly."""
        computer = PrototypeComputer(config=sample_config)

        prototypes, class_stds = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Should have std for each class
        num_classes = len(torch.unique(sample_labels))
        assert len(class_stds) == num_classes

        # All stds should be non-negative
        for std in class_stds:
            assert std >= 0

    def test_moving_average_update(self, sample_config):
        """Test moving average update behavior."""
        computer = PrototypeComputer(
            config=sample_config,
            use_moving_average=True,
            update_rate=0.5
        )

        # Set initial prototype
        computer.prototypes[0] = torch.ones(128) * 10.0
        computer.prototype_counts[0] = 100

        # New features with mean at 0
        new_features = torch.zeros(10, 128)

        updated = computer.update_prototype(0, new_features)

        # With update_rate=0.5, new_proto = old * 0.5 + new_mean * 0.5
        expected = torch.ones(128) * 5.0
        assert torch.allclose(updated, expected, atol=1e-5)

    def test_prototype_normalization(self, sample_config, sample_features, sample_labels):
        """Test that prototypes can be normalized."""
        computer = PrototypeComputer(config=sample_config)

        prototypes, _ = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Normalize prototypes
        normalized = torch.nn.functional.normalize(prototypes, p=2, dim=1)

        # Should have unit norm
        norms = torch.norm(normalized, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_prototype_distance_computation(self, sample_config):
        """Test computing distances to prototypes."""
        computer = PrototypeComputer(config=sample_config)

        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        features = torch.tensor([
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])

        distances = computer.compute_distances(features, prototypes)

        assert distances.shape == (3, 3)
        # Diagonal should have smallest distances (closest prototype)
        for i in range(3):
            assert distances[i, i] <= distances[i, (i+1)%3]
            assert distances[i, i] <= distances[i, (i+2)%3]

    def test_prototype_save_and_load(self, sample_config, sample_features, sample_labels, tmp_path):
        """Test saving and loading prototypes."""
        computer = PrototypeComputer(config=sample_config)

        # Compute prototypes
        prototypes, class_stds = computer.compute_prototypes_from_features(
            sample_features, sample_labels
        )

        # Save
        save_path = tmp_path / "prototypes.pt"
        computer.save_prototypes(str(save_path))

        assert save_path.exists()

        # Load into new computer
        new_computer = PrototypeComputer(config=sample_config)
        new_computer.load_prototypes(str(save_path))

        # Should have same prototypes
        assert torch.allclose(new_computer.prototypes[0], computer.prototypes[0])
        assert new_computer.prototype_counts[0] == computer.prototype_counts[0]


class TestPrototypeUtilityFunctions:
    """Test prototype utility functions."""

    def test_compute_prototypes_function(self):
        """Test compute_prototypes function."""
        features = torch.randn(100, 128)
        labels = torch.randint(0, 5, (100,))

        prototypes, class_stds = compute_prototypes(features, labels)

        num_classes = len(torch.unique(labels))
        assert prototypes.shape[0] == num_classes
        assert prototypes.shape[1] == 128
        assert len(class_stds) == num_classes

    def test_update_prototypes_moving_average_function(self):
        """Test update_prototypes_moving_average function."""
        old_prototypes = torch.randn(5, 128)
        new_features_batch = torch.randn(10, 128)
        new_labels = torch.randint(0, 5, (10,))
        update_rate = 0.1

        updated = update_prototypes_moving_average(
            old_prototypes, new_features_batch, new_labels, update_rate
        )

        assert updated.shape == old_prototypes.shape
        # Should be different from old prototypes
        assert not torch.allclose(updated, old_prototypes)

    def test_find_nearest_prototype_function(self):
        """Test find_nearest_prototype function."""
        prototypes = torch.eye(5)  # 5x5 identity matrix

        # Test with vector close to prototype 2
        feature = torch.tensor([0.1, 0.1, 0.8, 0.0, 0.0])
        nearest_idx, distance = find_nearest_prototype(feature, prototypes)

        assert nearest_idx == 2
        assert distance > 0  # Should have some distance

    def test_compute_prototype_accuracy_function(self):
        """Test compute_prototype_accuracy function."""
        # Create simple 2-class problem
        prototypes = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        features = torch.tensor([
            [0.9, 0.1],  # Close to class 0
            [0.8, 0.2],
            [0.1, 0.9],  # Close to class 1
            [0.2, 0.8]
        ])
        labels = torch.tensor([0, 0, 1, 1])

        accuracy = compute_prototype_accuracy(features, labels, prototypes)
        assert accuracy == 1.0  # All correct

        # Introduce some errors
        features2 = torch.tensor([
            [0.9, 0.1],
            [0.1, 0.9],  # Misclassified
            [0.1, 0.9],
            [0.9, 0.1]   # Misclassified
        ])
        labels2 = torch.tensor([0, 0, 1, 1])
        accuracy2 = compute_prototype_accuracy(features2, labels2, prototypes)
        assert accuracy2 == 0.5  # 2 out of 4 correct

    def test_prototype_config_validation(self):
        """Test PrototypeConfig validation."""
        # Valid config
        config = PrototypeConfig(
            feature_dim=128,
            update_rate=0.1,
            min_samples=5
        )
        assert config.feature_dim == 128
        assert 0 < config.update_rate <= 1
        assert config.min_samples > 0

        # Invalid: negative update rate
        with pytest.raises(ValueError):
            PrototypeConfig(feature_dim=128, update_rate=-0.1)

        # Invalid: zero min_samples
        with pytest.raises(ValueError):
            PrototypeConfig(feature_dim=128, min_samples=0)

    def test_prototype_with_different_metrics(self, sample_features, sample_labels):
        """Test prototype computation with different distance metrics."""
        from src.ood.prototypes import compute_prototypes

        # Euclidean (default)
        prototypes_euclidean, _ = compute_prototypes(
            sample_features, sample_labels, metric='euclidean'
        )

        # Cosine
        prototypes_cosine, _ = compute_prototypes(
            sample_features, sample_labels, metric='cosine'
        )

        # Both should produce valid prototypes
        assert prototypes_euclidean.shape[0] > 0
        assert prototypes_cosine.shape[0] > 0
