#!/usr/bin/env python3
"""
Pytest tests for Stage 2 optimizations - batch processing, caching, OpenCV, and prototypes.
"""

import pytest
import torch
from pathlib import Path
import sys

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory, mock_pipeline_data
from src.router.vlm_pipeline import VLMPipeline
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.utils.data_loader import LRUCache
from src.ood.prototypes import PrototypeComputer


class TestBatchProcessing:
    """Test batch processing optimization."""

    def test_route_batch_exists(self):
        """Test that route_batch method exists."""
        config = {
            'vlm_enabled': True,
            'vlm_confidence_threshold': 0.8,
            'vlm_max_detections': 10
        }
        pipeline = VLMPipeline(config=config, device='cpu')
        assert hasattr(pipeline, 'route_batch'), "route_batch method missing"

    def test_route_batch_returns_correct_shape(self):
        """Test that route_batch returns correct output shapes."""
        config = {
            'vlm_enabled': True,
            'vlm_confidence_threshold': 0.8,
            'vlm_max_detections': 10
        }
        pipeline = VLMPipeline(config=config, device='cpu')

        # Test with dummy batch
        batch_size = 3
        dummy_batch = torch.randn(batch_size, 3, 224, 224)
        crops_out, confs = pipeline.route_batch(dummy_batch)

        assert len(crops_out) == batch_size, f"Should return {batch_size} crop predictions"
        assert len(confs) == batch_size, f"Should return {batch_size} confidence scores"
        assert all(0 <= c <= 1 for c in confs), "Confidences should be in [0,1]"


class TestCachingMechanism:
    """Test caching optimization."""

    def test_pipeline_has_cache_attributes(self):
        """Test that IndependentMultiCropPipeline has caching attributes."""
        import inspect
        source = inspect.getsource(IndependentMultiCropPipeline.__init__)

        assert 'router_cache' in source, "router_cache not found in __init__"
        assert 'adapter_cache' in source, "adapter_cache not found in __init__"
        assert 'cache_enabled' in source, "cache_enabled not found in __init__"

    def test_pipeline_has_cache_methods(self):
        """Test that cache-related methods exist."""
        assert hasattr(IndependentMultiCropPipeline, 'clear_cache'), "clear_cache method missing"
        assert hasattr(IndependentMultiCropPipeline, 'get_cache_stats'), "get_cache_stats method missing"
        assert hasattr(IndependentMultiCropPipeline, '_generate_cache_key'), "_generate_cache_key method missing"

    def test_lru_cache_functionality(self):
        """Test LRU cache implementation."""
        cache = LRUCache(capacity=5)
        test_value = torch.randn(3, 224, 224)
        cache.put("test_key", test_value)
        retrieved = cache.get("test_key")
        assert retrieved is not None, "LRU cache should return stored value"
        assert torch.equal(retrieved, test_value), "Retrieved value should match stored value"

    def test_cache_eviction(self):
        """Test that LRU cache evicts old entries."""
        cache = LRUCache(capacity=3)
        for i in range(4):
            cache.put(f"key_{i}", torch.randn(3, 224, 224))

        # The first key should be evicted
        assert cache.get("key_0") is None, "Oldest entry should be evicted"


class TestOpenCVIntegration:
    """Test OpenCV integration."""

    def test_opencv_import_available(self):
        """Test that OpenCV is available."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_data_loader_uses_opencv(self):
        """Test that data_loader module uses OpenCV."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

        from src.utils import data_loader
        import inspect
        source = inspect.getsource(data_loader)
        has_cv2 = 'cv2' in source or 'import cv2' in source
        assert has_cv2, "OpenCV (cv2) not found in data_loader module"


class TestPrototypeOptimization:
    """Test prototype computation optimization."""

    def test_prototype_computer_exists(self):
        """Test that PrototypeComputer class exists."""
        from src.ood.prototypes import PrototypeComputer
        assert PrototypeComputer is not None

    def test_prototype_methods_exist(self):
        """Test that PrototypeComputer has required methods."""
        pc = PrototypeComputer(feature_dim=128, device='cpu')
        assert hasattr(pc, 'compute_prototypes_from_features'), "Missing optimized method"
        assert hasattr(pc, 'get_prototype_for_class'), "Missing caching method"
        assert hasattr(pc, 'get_prototype_cache_stats'), "Missing cache stats method"

    def test_compute_prototypes_from_features(self):
        """Test prototype computation with vectorized implementation."""
        pc = PrototypeComputer(feature_dim=128, device='cpu')

        # Create dummy data
        num_samples = 100
        num_classes = 5
        features = torch.randn(num_samples, 128)
        labels = torch.randint(0, num_classes, (num_samples,))

        # Compute prototypes
        prototypes, class_stds = pc.compute_prototypes_from_features(features, labels)

        # Check shapes
        assert prototypes.shape[0] == num_classes, f"Should have {num_classes} class prototypes"
        assert prototypes.shape[1] == 128, "Prototype dimension should match feature_dim"
        assert len(class_stds) == num_classes, f"Should have {num_classes} class stds"

    def test_prototype_caching(self):
        """Test prototype caching functionality."""
        pc = PrototypeComputer(feature_dim=128, device='cpu')

        # Get cache stats before
        initial_stats = pc.get_prototype_cache_stats()
        initial_hits = initial_stats.get('hits', 0)
        initial_misses = initial_stats.get('misses', 0)

        # Compute prototypes for a class
        features = torch.randn(10, 128)
        labels = torch.zeros(10, dtype=torch.long)
        prototypes1, _ = pc.compute_prototypes_from_features(features, labels)

        # Check cache was used
        stats_after = pc.get_prototype_cache_stats()
        assert stats_after.get('misses', 0) > initial_misses, "Cache miss should increase"

        # Request same class again
        prototypes2, _ = pc.get_prototype_for_class(0)
        assert prototypes2 is not None, "Should retrieve cached prototype"

        # Check cache hit
        stats_final = pc.get_prototype_cache_stats()
        assert stats_final.get('hits', 0) > initial_hits, "Cache hit should increase"
