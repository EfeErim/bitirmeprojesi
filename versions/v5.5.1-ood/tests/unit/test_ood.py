#!/usr/bin/env python3
"""
Unit tests for OOD detection components
"""

import pytest
import torch
import numpy as np
from src.ood.mahalanobis import MahalanobisDistance
from src.ood.dynamic_thresholds import DynamicOODThreshold

class TestMahalanobisDistance:
    """Test cases for MahalanobisDistance."""
    
    def test_initialization(self):
        """Test Mahalanobis distance initialization."""
        prototypes = torch.randn(3, 768)
        class_stds = {
            0: torch.ones(768) * 0.1,
            1: torch.ones(768) * 0.2,
            2: torch.ones(768) * 0.3
        }
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        assert mahalanobis.num_classes == 3
        assert mahalanobis.feature_dim == 768
        assert len(mahalanobis.inv_covariances) == 3
    
    def test_compute_distance(self):
        """Test distance computation."""
        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        class_stds = {
            0: torch.tensor([0.1, 0.1, 0.1]),
            1: torch.tensor([0.2, 0.2, 0.2]),
            2: torch.tensor([0.3, 0.3, 0.3])
        }
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        # Feature close to class 0
        features = torch.tensor([[1.1, 0.1, 0.1]])
        distance = mahalanobis.compute_distance(features, 0)
        
        assert distance.shape == (1,)
        assert distance.item() > 0
    
    def test_compute_all_distances(self):
        """Test computing distances to all classes."""
        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        
        class_stds = {
            0: torch.tensor([0.1, 0.1, 0.1]),
            1: torch.tensor([0.2, 0.2, 0.2])
        }
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        features = torch.tensor([[1.1, 0.1, 0.1]])
        distances = mahalanobis.compute_all_distances(features)
        
        assert distances.shape == (1, 2)
        # Should be closer to class 0
        assert distances[0, 0] < distances[0, 1]
    
    def test_get_nearest_class(self):
        """Test nearest class prediction."""
        prototypes = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float32)
        
        class_stds = {
            0: torch.tensor([0.1, 0.1, 0.1]),
            1: torch.tensor([0.2, 0.2, 0.2])
        }
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        features = torch.tensor([
            [1.1, 0.1, 0.1],  # Close to class 0
            [0.1, 1.1, 0.1]   # Close to class 1
        ])
        
        nearest, distances = mahalanobis.get_nearest_class(features)
        
        assert nearest.shape == (2,)
        assert nearest[0].item() == 0
        assert nearest[1].item() == 1

class TestDynamicOODThreshold:
    """Test cases for DynamicOODThreshold."""
    
    def test_threshold_factor(self):
        """Test threshold factor setting."""
        threshold = DynamicOODThreshold(threshold_factor=2.5)
        assert threshold.threshold_factor == 2.5
    
    def test_compute_thresholds_from_distances(self):
        """Test threshold computation from distances."""
        threshold_computer = DynamicOODThreshold(
            threshold_factor=2.0,
            min_val_samples_per_class=5
        )
        
        # Create mock distances per class
        distances_per_class = {
            0: [1.0, 1.2, 1.1, 0.9, 1.3, 1.0, 1.1],  # mean~1.09, std~0.13
            1: [2.0, 2.1, 1.9, 2.2, 2.0, 1.8, 2.1]   # mean~2.01, std~0.12
        }
        
        thresholds = threshold_computer.compute_thresholds_from_distances(distances_per_class)
        
        assert 0 in thresholds
        assert 1 in thresholds
        # Threshold should be mean + 2*std
        expected_0 = 1.09 + 2 * 0.13
        assert abs(thresholds[0] - expected_0) < 0.01
    
    def test_fallback_threshold(self):
        """Test fallback threshold for insufficient samples."""
        threshold_computer = DynamicOODThreshold(
            threshold_factor=2.0,
            min_val_samples_per_class=10,
            fallback_threshold=25.0
        )
        
        # Insufficient samples
        distances_per_class = {0: [1.0, 1.2, 1.1]}  # Only 3 samples
        
        thresholds = threshold_computer.compute_thresholds_from_distances(distances_per_class)
        
        assert thresholds[0] == 25.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])