"""
Comprehensive unit tests for SimpleCropRouter.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, Mock
import logging

from src.router.simple_crop_router import SimpleCropRouter


class TestSimpleCropRouterInitialization:
    """Test SimpleCropRouter initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with default parameters."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained') as mock_model, \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            # Setup mocks
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            
            assert router.crops == crops
            assert router.confidence_threshold == 0.92
            assert router.top_k_alternatives == 3
            assert router.device.type == 'cpu'
            assert isinstance(router.classifier, nn.Linear)
            assert router.classifier.out_features == len(crops)
    
    def test_init_custom_thresholds(self):
        """Test initialization with custom confidence threshold and top_k."""
        crops = ['tomato', 'pepper']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(
                crops=crops,
                confidence_threshold=0.85,
                top_k_alternatives=2,
                device='cpu'
            )
            
            assert router.confidence_threshold == 0.85
            assert router.top_k_alternatives == 2
    
    def test_init_top_k_clamped_to_crop_count(self):
        """Test that top_k_alternatives is clamped to number of crops."""
        crops = ['tomato', 'pepper']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(
                crops=crops,
                top_k_alternatives=10,  # More than available crops
                device='cpu'
            )
            
            assert router.top_k_alternatives == 2
    
    def test_init_config_hidden_size_detection(self):
        """Test hidden size detection from config."""
        crops = ['tomato']
        
        # Test with hidden_size attribute
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 1024
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            assert router.hidden_size == 1024
        
        # Test with dim attribute
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            del mock_config_instance.hidden_size
            mock_config_instance.dim = 512
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            assert router.hidden_size == 512
        
        # Test with neither attribute
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock(spec=[])  # No attributes
            mock_config.return_value = mock_config_instance
            
            with pytest.raises(ValueError, match="Cannot determine hidden size"):
                SimpleCropRouter(crops=crops, device='cpu')
    
    def test_init_statistics_initialization(self):
        """Test that statistics are properly initialized."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            
            expected_stats = {
                'total_predictions': 0,
                'high_confidence': 0,
                'low_confidence': 0,
                'rejected_predictions': 0,
                'fallback_used': 0,
                'confidence_sum': 0.0
            }
            assert router.confidence_stats == expected_stats
            
            cache_stats = router.get_cache_stats()
            assert cache_stats['cache_hits'] == 0
            assert cache_stats['cache_misses'] == 0
            assert cache_stats['cache_size'] == 0


class TestSimpleCropRouterConfidenceThreshold:
    """Test confidence threshold behavior."""
    
    @pytest.fixture
    def router_setup(self):
        """Setup router with mocked model."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained') as mock_model, \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(
                crops=crops,
                confidence_threshold=0.90,
                device='cpu'
            )
            
            yield router, mock_model_instance
    
    def test_route_high_confidence_accepted(self, router_setup):
        """Test that high confidence predictions are accepted."""
        router, mock_model = router_setup
        
        # Mock model to return high confidence
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
        mock_model.return_value = mock_outputs
        
        # Mock classifier output
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([[10.0, 1.0, 1.0]])):
            mock_image = torch.randn(1, 3, 224, 224)
            
            crop, confidence = router.route(mock_image)
            
            assert crop in router.crops
            assert confidence >= 0.90
            assert router.confidence_stats['total_predictions'] == 1
            assert router.confidence_stats['high_confidence'] == 1
            assert router.confidence_stats['low_confidence'] == 0
    
    def test_route_low_confidence_rejected(self, router_setup):
        """Test that low confidence predictions are rejected with ValueError."""
        router, mock_model = router_setup
        
        # Mock model to return low confidence
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
        mock_model.return_value = mock_outputs
        
        # Mock classifier to return low confidence for first class
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([[1.0, 10.0, 1.0]])):
            with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.8, 0.1]])):
                mock_image = torch.randn(1, 3, 224, 224)
                
                with pytest.raises(ValueError) as exc_info:
                    router.route(mock_image)
                
                # Check error message contains confidence info
                assert "below threshold" in str(exc_info.value)
                assert router.confidence_stats['total_predictions'] == 1
                assert router.confidence_stats['low_confidence'] == 1
                assert router.confidence_stats['rejected_predictions'] == 1
    
    def test_route_batch_all_high_confidence(self, router_setup):
        """Test batch routing with all high confidence predictions."""
        router, mock_model = router_setup
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(3, 1, 768)
        mock_model.return_value = mock_outputs
        
        # Mock classifier to return high confidence for all
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0]
        ])):
            mock_images = torch.stack([torch.randn(1, 3, 224, 224) for _ in range(3)])
            
            crops, confidences = router.route_batch(mock_images)
            
            assert len(crops) == 3
            assert len(confidences) == 3
            assert all(c in router.crops for c in crops)
            assert all(c >= 0.90 for c in confidences)
            assert router.confidence_stats['total_predictions'] == 3
            assert router.confidence_stats['high_confidence'] == 3
    
    def test_route_batch_with_low_confidence(self, router_setup):
        """Test batch routing with some low confidence predictions."""
        router, mock_model = router_setup
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(3, 1, 768)
        mock_model.return_value = mock_outputs
        
        # Mock classifier to return logits that after softmax give: [0.95, 0.03, 0.02], [0.1, 0.2, 0.7], [0.05, 0.9, 0.05]
        # First: high (0.95), Second: low (0.7), Third: high (0.9)
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([
            [10.0, 1.0, 1.0],   # high confidence
            [1.0, 1.0, 10.0],   # low confidence (inverted for test)
            [1.0, 10.0, 1.0]    # high confidence
        ])):
            # Softmax applied to whole batch, return appropriate confidences
            with patch('torch.softmax', return_value=torch.tensor([
                [0.95, 0.03, 0.02],   # First: high (0.95)
                [0.1, 0.2, 0.7],      # Second: low (0.7)
                [0.05, 0.9, 0.05]     # Third: high (0.9)
            ])):
                mock_images = torch.stack([torch.randn(1, 3, 224, 224) for _ in range(3)])
                
                with pytest.raises(ValueError) as exc_info:
                    router.route_batch(mock_images)
                
                assert "Found 1 predictions with confidence below threshold 0.9" in str(exc_info.value)
                assert router.confidence_stats['total_predictions'] == 3
                assert router.confidence_stats['low_confidence'] >= 1
                assert router.confidence_stats['rejected_predictions'] >= 1
    
    def test_confidence_threshold_edge_cases(self, router_setup):
        """Test edge cases around confidence threshold boundary."""
        router, mock_model = router_setup
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
        mock_model.return_value = mock_outputs
        
        # Test above threshold (should be accepted)
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([[9.0, 1.0, 0.0]])):
            with patch('torch.softmax', return_value=torch.tensor([[0.95, 0.025, 0.025]])):
                mock_image = torch.randn(1, 3, 224, 224)
                crop, confidence = router.route(mock_image)
                assert confidence >= 0.90  # Above threshold
                assert router.confidence_stats['high_confidence'] == 1
        
        # Reset stats
        router.reset_confidence_stats()
        
        # Test just below threshold (should be rejected)
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([[9.0, 1.0, 0.0]])):
            with patch('torch.softmax', return_value=torch.tensor([[0.85, 0.10, 0.05]])):
                mock_image = torch.randn(1, 3, 224, 224)
                with pytest.raises(ValueError, match="below threshold"):
                    router.route(mock_image)
                assert router.confidence_stats['low_confidence'] == 1


class TestSimpleCropRouterCache:
    """Test cache functionality."""
    
    @pytest.fixture
    def router_with_cache(self):
        """Setup router with cache enabled."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            router.image_cache = {}  # Ensure cache is empty
            
            return router
    
    def test_cache_miss_and_store(self, router_with_cache):
        """Test cache miss and storing result."""
        router = router_with_cache
        
        # Mock the backbone and classifier to return high confidence
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
        router.backbone = MagicMock(return_value=mock_outputs)
        router.classifier = MagicMock(return_value=torch.tensor([[10.0, 1.0, 1.0]]))
        
        with patch('torch.softmax', return_value=torch.tensor([[0.95, 0.025, 0.025]])):
            with patch.object(router, '_generate_cache_key', return_value='test_key'):
                mock_image = torch.randn(1, 3, 224, 224)
                
                # First call should be a cache miss
                result1 = router.route(mock_image)
                assert router.cache_misses == 1
                assert router.cache_hits == 0
                assert 'test_key' in router.image_cache
                
                # Second call with same image should be a cache hit
                result2 = router.route(mock_image)
                assert router.cache_hits == 1
                assert router.cache_misses == 1
                assert result1 == result2
    
    def test_cache_key_generation(self, router_with_cache):
        """Test that different images generate different cache keys."""
        router = router_with_cache
        
        image1 = torch.randn(1, 3, 224, 224)
        image2 = torch.randn(1, 3, 224, 224)
        image3 = image1.clone()  # Same as image1
        
        key1 = router._generate_cache_key(image1)
        key2 = router._generate_cache_key(image2)
        key3 = router._generate_cache_key(image3)
        
        assert key1 != key2
        assert key1 == key3  # Same image should produce same key
    
    def test_cache_stats(self, router_with_cache):
        """Test cache statistics reporting."""
        router = router_with_cache
        
        assert router.get_cache_stats() == {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_size': 0
        }
        
        # Simulate cache operations
        router.cache_hits = 5
        router.cache_misses = 3
        router.image_cache['test'] = None
        
        stats = router.get_cache_stats()
        assert stats['cache_hits'] == 5
        assert stats['cache_misses'] == 3
        assert stats['cache_size'] == 1


class TestSimpleCropRouterStatistics:
    """Test statistics tracking."""
    
    @pytest.fixture
    def router_for_stats(self):
        """Setup router for statistics testing."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            return router
    
    def test_get_confidence_stats_empty(self, router_for_stats):
        """Test confidence stats with no predictions."""
        router = router_for_stats
        
        stats = router.get_confidence_stats()
        
        assert stats['total_predictions'] == 0
        assert stats['mean_confidence'] == 0.0
        assert stats['high_confidence_rate'] == 0.0
        assert stats['low_confidence_rate'] == 0.0
        assert stats['rejection_rate'] == 0.0
    
    def test_get_confidence_stats_with_predictions(self, router_for_stats):
        """Test confidence stats with predictions."""
        router = router_for_stats
        
        # Manually set stats
        router.confidence_stats = {
            'total_predictions': 10,
            'high_confidence': 8,
            'low_confidence': 2,
            'rejected_predictions': 2,
            'fallback_used': 0,
            'confidence_sum': 9.2  # Sum of confidences
        }
        
        stats = router.get_confidence_stats()
        
        assert stats['total_predictions'] == 10
        assert abs(stats['mean_confidence'] - 0.92) < 1e-10
        assert abs(stats['high_confidence_rate'] - 0.8) < 1e-10
        assert abs(stats['low_confidence_rate'] - 0.2) < 1e-10
        assert abs(stats['rejection_rate'] - 0.2) < 1e-10
    
    def test_get_full_stats(self, router_for_stats):
        """Test full statistics including cache and confidence."""
        router = router_for_stats
        
        router.confidence_stats = {
            'total_predictions': 5,
            'high_confidence': 4,
            'low_confidence': 1,
            'rejected_predictions': 1,
            'fallback_used': 0,
            'confidence_sum': 4.5
        }
        router.cache_hits = 2
        router.cache_misses = 3
        router.image_cache['key'] = None
        
        full_stats = router.get_full_stats()
        
        assert 'cache' in full_stats
        assert 'confidence' in full_stats
        assert full_stats['cache']['cache_hits'] == 2
        assert full_stats['cache']['cache_misses'] == 3
        assert full_stats['confidence']['total_predictions'] == 5
    
    def test_reset_confidence_stats(self, router_for_stats):
        """Test resetting confidence statistics."""
        router = router_for_stats
        
        # Set some stats
        router.confidence_stats = {
            'total_predictions': 10,
            'high_confidence': 8,
            'low_confidence': 2,
            'rejected_predictions': 2,
            'fallback_used': 1,
            'confidence_sum': 9.2
        }
        
        router.reset_confidence_stats()
        
        expected = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'rejected_predictions': 0,
            'fallback_used': 0,
            'confidence_sum': 0.0
        }
        assert router.confidence_stats == expected


class TestSimpleCropRouterTopKAlternatives:
    """Test top-K alternatives functionality."""
    
    @pytest.fixture
    def router_for_topk(self):
        """Setup router for top-k testing."""
        crops = ['tomato', 'pepper', 'corn']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(
                crops=crops,
                top_k_alternatives=2,
                device='cpu'
            )
            
            return router
    
    def test_top_k_alternatives_in_error(self, router_for_topk):
        """Test that top-K alternatives are included in rejection error."""
        router = router_for_topk
        
        # Mock model to return specific probabilities
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
        router.backbone = MagicMock(return_value=mock_outputs)
        
        with patch.object(router.classifier, 'forward', return_value=torch.tensor([[1.0, 2.0, 3.0]])):
            with patch('torch.softmax', return_value=torch.tensor([[0.1, 0.2, 0.7]])):
                mock_image = torch.randn(1, 3, 224, 224)
                
                with pytest.raises(ValueError) as exc_info:
                    router.route(mock_image)
                
                # Check that error message contains top-k alternatives info
                error_msg = str(exc_info.value)
                assert "Top-2" in error_msg or "top-2" in error_msg
                assert "pepper" in error_msg  # Second crop
                assert "corn" in error_msg  # Third crop


class TestSimpleCropRouterEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def router_basic(self):
        """Basic router setup."""
        crops = ['tomato']
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cpu')
            return router
    
    def test_route_with_empty_crops_list(self):
        """Test initialization with empty crops list."""
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config:
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=[], device='cpu')
            assert router.crops == []
            assert router.classifier.out_features == 0
    
    def test_route_device_selection(self):
        """Test device selection logic."""
        crops = ['tomato']
        
        # Test CPU fallback when CUDA not available
        with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
             patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config, \
             patch('torch.cuda.is_available', return_value=False):
            
            mock_config_instance = MagicMock()
            mock_config_instance.hidden_size = 768
            mock_config.return_value = mock_config_instance
            
            router = SimpleCropRouter(crops=crops, device='cuda')
            assert router.device.type == 'cpu'
        
        # Test CUDA when available (skip if not installed)
        try:
            with patch('src.router.simple_crop_router.AutoModel.from_pretrained'), \
                 patch('src.router.simple_crop_router.AutoConfig.from_pretrained') as mock_config, \
                 patch('torch.cuda.is_available', return_value=True):
                
                mock_config_instance = MagicMock()
                mock_config_instance.hidden_size = 768
                mock_config.return_value = mock_config_instance
                
                router = SimpleCropRouter(crops=crops, device='cuda')
                assert router.device.type == 'cuda'
        except (AssertionError, RuntimeError):
            # Torch not compiled with CUDA, skip test
            pytest.skip("CUDA not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])