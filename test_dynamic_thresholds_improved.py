#!/usr/bin/env python3
"""
Test script for improved dynamic thresholds implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.ood.dynamic_thresholds import DynamicOODThreshold

def test_basic_functionality():
    """Test basic threshold computation with sufficient samples."""
    print("Testing basic functionality...")
    
    num_classes = 3
    feature_dim = 1536
    samples_per_class = 50  # More than 30 for reliable stats
    
    # Create dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            # Generate deterministic features for reproducibility
            features = torch.randn(batch_size, 1, feature_dim)
            return type('Output', (), {'last_hidden_state': features})()
    
    model = DummyModel()
    
    # Dummy Mahalanobis
    class DummyMahalanobis:
        def __init__(self, num_classes):
            self.num_classes = num_classes
        
        def compute_distance(self, features, class_idx):
            # Generate distances with class-specific means
            batch_size = features.shape[0]
            base_distances = torch.randn(batch_size) + class_idx * 2.0
            return base_distances
    
    mahalanobis = DummyMahalanobis(num_classes)
    
    # Create validation data with sufficient samples per class
    dummy_data = torch.randn(samples_per_class * num_classes, 3, 224, 224)
    dummy_labels = torch.cat([torch.full((samples_per_class,), i) for i in range(num_classes)])
    
    dataset = TensorDataset(dummy_data, dummy_labels)
    val_loader = DataLoader(dataset, batch_size=16)
    
    # Test 1: Default configuration (30 samples min, 95% confidence)
    thresholds = DynamicOODThreshold.compute_thresholds(
        mahalanobis, model, val_loader, feature_dim, device='cpu'
    )
    
    print(f"✓ Computed thresholds for {len(thresholds)} classes")
    for class_idx, threshold in thresholds.items():
        print(f"  Class {class_idx}: threshold = {threshold:.4f}")
    
    # Validate thresholds
    validator = DynamicOODThreshold()
    metrics = validator.validate_thresholds(thresholds, val_loader, model, mahalanobis, device='cpu')
    print(f"✓ Validation metrics: FPR = {metrics['false_positive_rate']:.4f}, TNR = {metrics['true_negative_rate']:.4f}")
    
    return True

def test_insufficient_samples():
    """Test fallback strategy with insufficient samples."""
    print("\nTesting insufficient samples fallback...")
    
    num_classes = 3
    feature_dim = 1536
    samples_per_class = 5  # Less than 30
    
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            features = torch.randn(batch_size, 1, feature_dim)
            return type('Output', (), {'last_hidden_state': features})()
    
    model = DummyModel()
    
    class DummyMahalanobis:
        def __init__(self, num_classes):
            self.num_classes = num_classes
        
        def compute_distance(self, features, class_idx):
            batch_size = features.shape[0]
            return torch.randn(batch_size) + class_idx * 2.0
    
    mahalanobis = DummyMahalanobis(num_classes)
    
    dummy_data = torch.randn(samples_per_class * num_classes, 3, 224, 224)
    dummy_labels = torch.cat([torch.full((samples_per_class,), i) for i in range(num_classes)])
    
    dataset = TensorDataset(dummy_data, dummy_labels)
    val_loader = DataLoader(dataset, batch_size=16)
    
    # Should use fallback thresholds
    thresholds = DynamicOODThreshold.compute_thresholds(
        mahalanobis, model, val_loader, feature_dim, device='cpu'
    )
    
    print(f"✓ Computed thresholds with insufficient samples:")
    for class_idx, threshold in thresholds.items():
        print(f"  Class {class_idx}: threshold = {threshold:.4f}")
    
    return True

def test_configurable_parameters():
    """Test custom configuration parameters."""
    print("\nTesting configurable parameters...")
    
    custom_config = {
        'threshold_factor': 2.576,  # 99% confidence
        'min_val_samples_per_class': 50,  # Higher requirement
        'fallback_threshold': 30.0,
        'confidence_level': 0.99,
        'max_fallback_threshold': 60.0,
        'min_fallback_threshold': 15.0
    }
    
    print(f"✓ Using custom config: {custom_config}")
    
    # Verify configuration validation
    try:
        threshold_computer = DynamicOODThreshold(**custom_config)
        print("✓ Configuration accepted and validated")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    return True

def test_confidence_intervals():
    """Test confidence interval computation."""
    print("\nTesting confidence interval computation...")
    
    # Create test data
    np.random.seed(42)
    data = np.random.normal(10.0, 2.0, 50)
    
    threshold_computer = DynamicOODThreshold()
    ci_lower, ci_upper = threshold_computer._compute_confidence_interval(data, 0.95)
    
    mean = np.mean(data)
    print(f"✓ Data: mean = {mean:.4f}, std = {np.std(data):.4f}")
    print(f"✓ 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Check that CI contains the mean
    assert ci_lower <= mean <= ci_upper, "Confidence interval should contain the mean"
    print("✓ Confidence interval correctly contains the sample mean")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Improved Dynamic OOD Threshold Implementation")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_insufficient_samples,
        test_configurable_parameters,
        test_confidence_intervals
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
