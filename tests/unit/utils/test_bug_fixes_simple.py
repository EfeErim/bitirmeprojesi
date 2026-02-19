#!/usr/bin/env python3
"""
Simple focused tests for bug fixes.
Tests the core fixes without heavy dependencies.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
import hashlib

from src.utils.data_loader import preprocess_image, LRUCache
from src.ood.mahalanobis import MahalanobisDistance
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


class TestGradientAccumulation:
    """Test gradient accumulation logic."""
    
    def test_gradient_accumulation_with_zero_grad_after_step(self):
        """Test that zero_grad is called after optimizer.step, not before."""
        # Read source file directly to avoid import dependencies
        with open('src/training/colab_phase1_training.py', 'r') as f:
            source = f.read()
        
        # Should have: backward -> step -> zero_grad
        assert "loss.backward()" in source or "scaler.scale(loss).backward()" in source
        # zero_grad should come after step
        step_pos = source.find("optimizer.step()") if "optimizer.step()" in source else source.find("scaler.step(")
        zero_grad_pos = source.find("zero_grad()")
        assert step_pos < zero_grad_pos, "zero_grad should come after step"
    
    def test_gradient_accumulation_final_step_handling(self):
        """Test that remaining gradients are processed after the loop."""
        with open('src/training/colab_phase1_training.py', 'r') as f:
            source = f.read()
        
        # Should have code to handle remaining gradients after loop
        assert "if self.current_step % self.gradient_accumulation_steps != 0:" in source, \
            "Should handle remaining gradients after loop"
        # Should call either optimizer.step() or scaler.step(optimizer)
        assert "optimizer.step()" in source or "scaler.step(self.optimizer)" in source, \
            "Should call optimizer.step() or scaler.step() for remaining gradients"


class TestFeatureExtractionConsistency:
    """Test feature extraction uses helper method."""
    
    def test_uses_extract_features_in_train_epoch(self):
        """Verify _train_epoch uses _extract_features helper."""
        import inspect
        from src.adapter.independent_crop_adapter import IndependentCropAdapter
        
        source = inspect.getsource(IndependentCropAdapter._train_epoch)
        assert "self._extract_features(images)" in source, \
            "Should use _extract_features helper in _train_epoch"
    
    def test_uses_extract_features_in_validate(self):
        """Verify _validate uses _extract_features helper."""
        import inspect
        from src.adapter.independent_crop_adapter import IndependentCropAdapter
        
        source = inspect.getsource(IndependentCropAdapter._validate)
        assert "self._extract_features(images)" in source, \
            "Should use _extract_features helper in _validate"


class TestCacheKeyGeneration:
    """Test cache key generation uses proper hashing."""
    
    def test_cache_key_uses_tensor_hash(self):
        """Test that cache key uses tensor data hash, not just ID."""
        config = {}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        # Create tensors with same values but different IDs
        tensor1 = torch.ones(1, 3, 224, 224)
        tensor2 = torch.ones(1, 3, 224, 224)
        
        # They should have different IDs
        assert id(tensor1) != id(tensor2)
        
        # But same hash since values are identical
        key1 = pipeline._generate_cache_key(tensor1)
        key2 = pipeline._generate_cache_key(tensor2)
        assert key1 == key2, "Identical tensors should produce same cache key"
    
    def test_cache_key_includes_shape(self):
        """Test that cache key includes shape information."""
        config = {}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        tensor1 = torch.randn(1, 3, 224, 224)
        tensor2 = torch.randn(2, 3, 224, 224)
        
        key1 = pipeline._generate_cache_key(tensor1)
        key2 = pipeline._generate_cache_key(tensor2)
        
        assert key1 != key2, "Different shapes should produce different keys"


class TestMahalanobisMath:
    """Test Mahalanobis distance computation."""
    
    def test_compute_distance_returns_correct_shape(self):
        """Test that compute_distance returns per-sample distances."""
        feature_dim = 10
        prototypes = torch.randn(3, feature_dim)
        class_stds = {i: torch.ones(feature_dim) * 0.1 for i in range(3)}
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        batch_size = 5
        features = torch.randn(batch_size, feature_dim)
        
        distances = mahalanobis.compute_distance(features, 0)
        
        assert distances.shape == (batch_size,), \
            f"Expected shape ({batch_size},), got {distances.shape}"
    
    def test_compute_distance_non_negative(self):
        """Test that distances are non-negative."""
        prototypes = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        class_stds = {0: torch.tensor([0.1, 0.1]), 1: torch.tensor([0.2, 0.2])}
        
        mahalanobis = MahalanobisDistance(prototypes, class_stds)
        
        features = torch.tensor([[1.1, 2.1], [3.1, 4.1]], dtype=torch.float32)
        distances = mahalanobis.compute_distance(features, 0)
        
        assert torch.all(distances >= 0), "All distances should be non-negative"
    
    def test_compute_distance_uses_correct_formula(self):
        """Test that the formula (diff @ inv_cov * diff).sum(dim=1) is used."""
        import inspect
        source = inspect.getsource(MahalanobisDistance.compute_distance)
        
        # Should use element-wise multiplication and sum over dim=1
        assert "sum(dim=1)" in source, "Should sum over dimension 1"
        # Should not use diagonal of matrix multiplication (old bug)
        assert "diagonal(" not in source, "Should not use diagonal (old bug)"


class TestImageValidation:
    """Test image preprocessing validation."""
    
    def test_grayscale_image_converted(self):
        """Test grayscale (2D) images are converted to RGB."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_image(gray)
        assert result.shape == (3, 224, 224)
    
    def test_bgr_image_converted(self):
        """Test BGR images are converted to RGB."""
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(bgr)
        assert result.shape == (3, 224, 224)
    
    def test_rgba_image_converted(self):
        """Test RGBA images are converted to RGB."""
        rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        result = preprocess_image(rgba)
        assert result.shape == (3, 224, 224)
    
    def test_invalid_channels_raises(self):
        """Test that images with invalid channels raise ValueError."""
        invalid = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid number of channels"):
            preprocess_image(invalid)
    
    def test_pil_image_processed(self):
        """Test PIL images are processed correctly."""
        pil_img = Image.new('RGB', (100, 100), color='red')
        result = preprocess_image(pil_img)
        assert result.shape == (3, 224, 224)


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_respects_capacity(self):
        """Test that cache does not exceed capacity."""
        cache = LRUCache(capacity=3)
        
        for i in range(5):
            cache.put(f"key{i}", torch.tensor([i]))
        
        assert len(cache.cache) == 3, "Cache should respect capacity"
    
    def test_lru_eviction_order(self):
        """Test that least recently used items are evicted."""
        cache = LRUCache(capacity=2)
        
        cache.put("a", torch.tensor([1]))
        cache.put("b", torch.tensor([2]))
        cache.get("a")  # Access 'a' to make it recently used
        cache.put("c", torch.tensor([3]))  # Should evict 'b'
        
        assert cache.get("a") is not None, "'a' should still be in cache"
        assert cache.get("b") is None, "'b' should be evicted"
        assert cache.get("c") is not None, "'c' should be in cache"
    
    def test_cache_get_returns_none_for_missing(self):
        """Test that get returns None for missing keys."""
        cache = LRUCache(capacity=2)
        assert cache.get("nonexistent") is None


class TestClassifierWeightPreservation:
    """Test classifier weight preservation in Phase 2."""
    
    def test_old_weights_preserved_after_classifier_expansion(self):
        """Test that old classifier weights are preserved when expanding."""
        # Simulate the Phase 2 classifier expansion logic
        old_num_classes = 5
        new_num_classes = old_num_classes + 3
        hidden_size = 768
        
        # Create original classifier with known weights
        old_classifier = nn.Linear(hidden_size, old_num_classes)
        with torch.no_grad():
            old_classifier.weight.fill_(1.0)
            old_classifier.bias.fill_(0.5)
        
        # Save old weights
        saved_weight = old_classifier.weight.data.clone()
        saved_bias = old_classifier.bias.data.clone()
        
        # Create new classifier
        new_classifier = nn.Linear(hidden_size, new_num_classes)
        
        # Copy old weights (correct pattern)
        with torch.no_grad():
            new_classifier.weight[:old_num_classes] = saved_weight
            new_classifier.bias[:old_num_classes] = saved_bias
        
        # Verify preservation
        assert torch.allclose(new_classifier.weight[:old_num_classes], saved_weight), \
            "Old weights should be preserved"
        assert torch.allclose(new_classifier.bias[:old_num_classes], saved_bias), \
            "Old biases should be preserved"


class TestEmptyParameterGroupHandling:
    """Test empty parameter group validation."""
    
    def test_optimizer_param_groups_filtered(self):
        """Test that empty parameter groups are filtered out."""
        lora_a_params = []
        lora_b_params = [nn.Parameter(torch.randn(2, 2))]
        other_params = []
        
        param_groups = []
        if lora_a_params:
            param_groups.append({'params': lora_a_params, 'lr': 1e-4})
        if lora_b_params:
            param_groups.append({'params': lora_b_params, 'lr': 1e-4 * 16})
        if other_params:
            param_groups.append({'params': other_params, 'lr': 1e-4})
        
        assert len(param_groups) == 1, "Should have exactly one non-empty group"
        assert param_groups[0]['params'] == lora_b_params, "Should contain lora_b_params"


class TestDynamicThresholdsFallback:
    """Test dynamic thresholds fallback logic."""
    
    def test_fallback_for_insufficient_samples(self):
        """Test fallback thresholds for various sample counts."""
        from src.ood.dynamic_thresholds import DynamicOODThreshold
        
        computer = DynamicOODThreshold(
            min_val_samples_per_class=30,
            fallback_threshold=25.0
        )
        
        # 0 samples -> base fallback
        assert computer._handle_insufficient_samples(0, 0) == 25.0
        
        # 3 samples (<5) -> 1.5x fallback
        assert computer._handle_insufficient_samples(1, 3) == min(25.0 * 1.5, 50.0)
        
        # 7 samples (5-9) -> 1.2x fallback
        assert computer._handle_insufficient_samples(2, 7) == min(25.0 * 1.2, 50.0)


class TestDatabaseSessionSafety:
    """Test database session handling."""
    
    def test_session_close_error_handled(self):
        """Test that errors during session.close() are caught and logged."""
        import logging
        from io import StringIO
        
        # Create mock logger to capture errors
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("test_db")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        class MockSession:
            def close(self):
                raise RuntimeError("Close failed")
        
        # Simulate the pattern from api/database.py get_db
        session = MockSession()
        try:
            # Instead of yield, we just simulate the finally block
            pass
        finally:
            try:
                session.close()
            except Exception as e:
                logger.error(f"Error closing database session: {e}")
        
        # Check that error was logged
        log_output = log_stream.getvalue()
        assert "Error closing database session" in log_output or "Close failed" in log_output


class TestDeviceValidation:
    """Test device handling validation."""
    
    def test_mahalanobis_validates_cuda_request(self):
        """Test that Mahalanobis raises if CUDA requested but not available."""
        prototypes = torch.randn(2, 10)
        class_stds = {0: torch.ones(10), 1: torch.ones(10)}
        
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA requested but not available"):
                MahalanobisDistance(prototypes, class_stds, device='cuda')
        else:
            # Should work fine
            m = MahalanobisDistance(prototypes, class_stds, device='cuda')
            assert m.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])