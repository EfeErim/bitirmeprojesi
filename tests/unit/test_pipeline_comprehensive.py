"""
Comprehensive unit tests for IndependentMultiCropPipeline.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, Mock
import threading
import time
import sys

# Mock PEFT imports before importing pipeline
sys.modules['peft'] = MagicMock()
sys.modules['peft.LoraConfig'] = MagicMock()
sys.modules['peft.get_peft_model'] = MagicMock()
sys.modules['peft.SDLoRAConfig'] = MagicMock()

# Mock missing compute_class_prototypes function
sys.modules['src.ood.prototypes'] = MagicMock()
sys.modules['src.ood.prototypes'].compute_class_prototypes = MagicMock(return_value=(torch.zeros(10, 768), {}))

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


class TestIndependentMultiCropPipelineInitialization:
    """Test pipeline initialization."""
    
    def test_init_basic(self):
        """Test basic initialization with default config."""
        config = {'crops': ['tomato', 'pepper']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            assert pipeline.crops == ['tomato', 'pepper']
            assert pipeline.device.type == 'cpu'
            assert pipeline.router is None
            assert pipeline.adapters == {}
            assert pipeline.cache_enabled is True
            assert pipeline.cache_size == 1000
    
    def test_init_custom_cache_settings(self):
        """Test initialization with custom cache settings."""
        config = {
            'crops': ['tomato'],
            'cache_enabled': False,
            'cache_size': 500
        }
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            assert pipeline.cache_enabled is False
            assert pipeline.cache_size == 500
    
    def test_init_empty_crops(self):
        """Test initialization with empty crops list."""
        config = {'crops': []}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            assert pipeline.crops == []


class TestIndependentMultiCropPipelineCache:
    """Test caching functionality."""
    
    @pytest.fixture
    def pipeline_with_cache(self):
        """Setup pipeline with mocked components."""
        config = {'crops': ['tomato', 'pepper'], 'cache_enabled': True}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            
            # Mock adapter
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
            }
            pipeline.adapters['tomato'] = mock_adapter
            
            return pipeline
    
    def test_cache_hit_and_miss(self, pipeline_with_cache):
        """Test cache hit and miss behavior."""
        pipeline = pipeline_with_cache
        mock_image = torch.randn(1, 3, 224, 224)
        
        # First call - cache miss
        result1 = pipeline.process_image(mock_image)
        assert pipeline.cache_misses == 1
        assert pipeline.cache_hits == 0
        assert result1['status'] == 'success'
        
        # Second call with same image - cache hit
        result2 = pipeline.process_image(mock_image)
        assert pipeline.cache_misses == 1
        assert pipeline.cache_hits == 1
        assert result2 == result1
    
    def test_cache_disabled(self):
        """Test behavior when cache is disabled."""
        config = {'crops': ['tomato'], 'cache_enabled': False}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock components
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
            }
            pipeline.adapters['tomato'] = mock_adapter
            
            mock_image = torch.randn(1, 3, 224, 224)
            
            # Multiple calls should not use cache
            for _ in range(3):
                pipeline.process_image(mock_image)
            
            assert pipeline.cache_hits == 0
            assert pipeline.cache_misses == 3
            assert mock_adapter.predict_with_ood.call_count == 3
    
    def test_cache_size_limit(self, pipeline_with_cache):
        """Test that cache respects size limit."""
        pipeline = pipeline_with_cache
        pipeline.cache_size = 2
        
        # Create different images
        images = [torch.randn(1, 3, 224, 224) for _ in range(5)]
        
        for img in images:
            pipeline.process_image(img)
        
        # Cache should not exceed size limit
        assert len(pipeline.adapter_cache) <= 2
        assert pipeline.cache_misses == 5
    
    def test_cache_key_generation(self, pipeline_with_cache):
        """Test cache key generation."""
        pipeline = pipeline_with_cache
        
        image1 = torch.randn(1, 3, 224, 224)
        image2 = torch.randn(1, 3, 224, 224)
        image3 = image1.clone()
        
        key1 = pipeline._generate_cache_key(image1)
        key2 = pipeline._generate_cache_key(image2)
        key3 = pipeline._generate_cache_key(image3)
        
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hex length
        assert key1 != key2
        assert key1 == key3
    
    def test_clear_cache(self, pipeline_with_cache):
        """Test cache clearing."""
        pipeline = pipeline_with_cache
        mock_image = torch.randn(1, 3, 224, 224)
        
        # Populate cache
        pipeline.process_image(mock_image)
        pipeline.process_image(mock_image)
        
        assert pipeline.cache_hits > 0 or len(pipeline.adapter_cache) > 0
        
        pipeline.clear_cache()
        
        assert len(pipeline.adapter_cache) == 0
        assert pipeline.cache_hits == 0
        assert pipeline.cache_misses == 0


class TestIndependentMultiCropPipelineErrorHandling:
    """Test error handling in pipeline."""
    
    @pytest.fixture
    def pipeline_for_errors(self):
        """Setup pipeline for error testing."""
        config = {'crops': ['tomato']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            
            # Mock adapter that raises error
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.side_effect = RuntimeError("Model error")
            pipeline.adapters['tomato'] = mock_adapter
            
            return pipeline
    
    def test_adapter_error_handling(self, pipeline_for_errors):
        """Test that adapter errors are caught and returned as error result."""
        pipeline = pipeline_for_errors
        mock_image = torch.randn(1, 3, 224, 224)
        
        result = pipeline.process_image(mock_image)
        
        assert result['status'] == 'error'
        assert 'Model error' in result['message']
        assert result['crop'] == 'tomato'
        assert result['crop_confidence'] == 0.95
    
    def test_missing_adapter_error(self):
        """Test handling when no adapter available for predicted crop."""
        config = {'crops': ['tomato', 'pepper']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router to return crop without adapter
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('corn', 0.95)  # corn not in adapters
            
            mock_image = torch.randn(1, 3, 224, 224)
            
            result = pipeline.process_image(mock_image)
            
            assert result['status'] == 'error'
            assert 'No adapter available' in result['message']
            assert result['crop'] == 'corn'
    
    def test_router_not_initialized_error(self):
        """Test error when router is not initialized."""
        config = {'crops': ['tomato']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            pipeline.router = None
            
            mock_image = torch.randn(1, 3, 224, 224)
            
            with pytest.raises(RuntimeError, match="Router not initialized"):
                pipeline.process_image(mock_image)
    
    def test_router_confidence_rejection(self):
        """Test router confidence rejection triggers fallback."""
        config = {'crops': ['tomato']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router to raise ValueError (low confidence)
            pipeline.router = MagicMock()
            pipeline.router.route.side_effect = ValueError(
                "Confidence below threshold",
                predicted_crop='tomato',
                confidence=0.5,
                top_k_alternatives=[('pepper', 0.3), ('corn', 0.2)]
            )
            
            mock_image = torch.randn(1, 3, 224, 224)
            
            result = pipeline.process_image(mock_image)
            
            assert result['status'] == 'error'
            assert 'Confidence below threshold' in result['message']
            assert 'crop' in result


class TestIndependentMultiCropPipelineBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.fixture
    def pipeline_for_batch(self):
        """Setup pipeline for batch testing."""
        config = {'crops': ['tomato', 'pepper'], 'cache_enabled': False}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router batch method
            pipeline.router = MagicMock()
            pipeline.router.route_batch.return_value = (
                ['tomato', 'pepper', 'tomato'],
                [0.95, 0.87, 0.92]
            )
            
            # Mock adapters
            def create_mock_adapter(crop_name):
                adapter = MagicMock()
                adapter.predict_with_ood.return_value = {
                    'status': 'success',
                    'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                    'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
                }
                return adapter
            
            pipeline.adapters['tomato'] = create_mock_adapter('tomato')
            pipeline.adapters['pepper'] = create_mock_adapter('pepper')
            
            return pipeline
    
    def test_batch_process_success(self, pipeline_for_batch):
        """Test successful batch processing."""
        pipeline = pipeline_for_batch
        
        images = [torch.randn(1, 3, 224, 224) for _ in range(3)]
        results = pipeline.batch_process(images)
        
        assert len(results) == 3
        assert all(r['status'] == 'success' for r in results)
        assert pipeline.router.route_batch.called_once()
    
    def test_batch_process_with_metadata(self, pipeline_for_batch):
        """Test batch processing with metadata."""
        pipeline = pipeline_for_batch
        
        images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
        metadata_list = [
            {'device_model': 'Pixel 7'},
            {'device_model': 'iPhone 14'}
        ]
        
        results = pipeline.batch_process(images, metadata_list)
        
        assert len(results) == 3  # Still 3 images from router batch
        # Metadata should be passed to _handle_ood_detection if OOD triggered
    
    def test_batch_process_empty_list(self, pipeline_for_batch):
        """Test batch processing with empty list."""
        pipeline = pipeline_for_batch
        
        results = pipeline.batch_process([])
        assert results == []
    
    def test_batch_process_router_error(self):
        """Test batch processing when router raises error."""
        config = {'crops': ['tomato']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            pipeline.router = MagicMock()
            pipeline.router.route_batch.side_effect = ValueError("Batch routing failed")
            
            images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
            
            with pytest.raises(ValueError):
                pipeline.batch_process(images)
    
    def test_batch_process_mixed_success_and_failure(self):
        """Test batch processing with some adapters failing."""
        config = {'crops': ['tomato', 'pepper']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router
            pipeline.router = MagicMock()
            pipeline.router.route_batch.return_value = (
                ['tomato', 'pepper', 'tomato'],
                [0.95, 0.87, 0.92]
            )
            
            # Tomato adapter succeeds, pepper adapter fails
            tomato_adapter = MagicMock()
            tomato_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
            }
            
            pepper_adapter = MagicMock()
            pepper_adapter.predict_with_ood.side_effect = RuntimeError("Pepper model error")
            
            pipeline.adapters['tomato'] = tomato_adapter
            pipeline.adapters['pepper'] = pepper_adapter
            
            images = [torch.randn(1, 3, 224, 224) for _ in range(3)]
            results = pipeline.batch_process(images)
            
            assert len(results) == 3
            # First tomato should succeed
            assert results[0]['status'] == 'success'
            # Second pepper should fail
            assert results[1]['status'] == 'error'
            # Third tomato should succeed
            assert results[2]['status'] == 'success'


class TestIndependentMultiCropPipelineOODHandling:
    """Test OOD detection handling."""
    
    @pytest.fixture
    def pipeline_for_ood(self):
        """Setup pipeline for OOD testing."""
        config = {'crops': ['tomato']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock router
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            
            # Mock adapter with OOD triggered
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'unknown', 'confidence': 0.6},
                'ood_analysis': {
                    'is_ood': True,
                    'ood_score': 30.5,
                    'threshold': 25.0,
                    'dynamic_threshold_applied': True,
                    'ood_type': 'NEW_DISEASE_CANDIDATE'
                }
            }
            pipeline.adapters['tomato'] = mock_adapter
            
            return pipeline
    
    def test_ood_detection_triggers_handler(self, pipeline_for_ood):
        """Test that OOD detection triggers _handle_ood_detection."""
        pipeline = pipeline_for_ood
        
        with patch.object(pipeline, '_handle_ood_detection') as mock_handler:
            mock_image = torch.randn(1, 3, 224, 224)
            result = pipeline.process_image(mock_image)
            
            assert result['ood_analysis']['is_ood'] is True
            mock_handler.assert_called_once()
    
    def test_handle_ood_detection(self, pipeline_for_ood):
        """Test _handle_ood_detection method."""
        pipeline = pipeline_for_ood
        
        result = {
            'ood_analysis': {
                'is_ood': True,
                'ood_score': 30.5,
                'threshold': 25.0
            }
        }
        metadata = {'device_model': 'test'}
        
        # Should not raise exception
        pipeline._handle_ood_detection(result, metadata)
        
        # Check result is updated with recommendations
        assert 'recommendations' in result
        assert result['recommendations']['expert_consultation'] is True


class TestIndependentMultiCropPipelineRegistration:
    """Test crop adapter registration."""
    
    @pytest.fixture
    def pipeline_for_registration(self):
        """Setup pipeline for registration testing."""
        config = {'crops': ['tomato', 'pepper', 'corn']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            return pipeline
    
    def test_register_valid_crop(self, pipeline_for_registration):
        """Test registering a valid crop adapter."""
        pipeline = pipeline_for_registration
        
        with patch('src.adapter.independent_crop_adapter.IndependentCropAdapter') as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter_class.return_value = mock_adapter
            
            result = pipeline.register_crop('tomato', '/path/to/adapter')
            
            assert result is True
            assert 'tomato' in pipeline.adapters
            mock_adapter_class.assert_called_once_with(crop_name='tomato', device=pipeline.device)
            mock_adapter.load_adapter.assert_called_once_with('/path/to/adapter')
    
    def test_register_invalid_crop(self, pipeline_for_registration):
        """Test registering an unsupported crop."""
        pipeline = pipeline_for_registration
        
        result = pipeline.register_crop('wheat', '/path/to/adapter')
        
        assert result is False
        assert 'wheat' not in pipeline.adapters
    
    def test_register_load_failure(self, pipeline_for_registration):
        """Test handling when adapter loading fails."""
        pipeline = pipeline_for_registration
        
        with patch('src.adapter.independent_crop_adapter.IndependentCropAdapter') as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter.load_adapter.side_effect = Exception("Load failed")
            mock_adapter_class.return_value = mock_adapter
            
            result = pipeline.register_crop('tomato', '/path/to/adapter')
            
            assert result is False
            assert 'tomato' not in pipeline.adapters
    
    def test_register_clears_cache(self, pipeline_for_registration):
        """Test that registering a new adapter clears caches."""
        pipeline = pipeline_for_registration
        pipeline.router_cache['test'] = None
        pipeline.adapter_cache['test'] = None
        
        with patch('src.adapter.independent_crop_adapter.IndependentCropAdapter') as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter_class.return_value = mock_adapter
            
            pipeline.register_crop('tomato', '/path/to/adapter')
            
            assert len(pipeline.router_cache) == 0
            assert len(pipeline.adapter_cache) == 0


class TestIndependentMultiCropPipelineStatus:
    """Test status reporting."""
    
    @pytest.fixture
    def pipeline_for_status(self):
        """Setup pipeline for status testing."""
        config = {'crops': ['tomato', 'pepper']}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock adapters
            mock_adapter1 = MagicMock()
            mock_adapter1.is_trained = True
            mock_adapter1.current_phase = 1
            mock_adapter1.class_to_idx = {'healthy': 0, 'disease1': 1}
            
            mock_adapter2 = MagicMock()
            mock_adapter2.is_trained = False
            mock_adapter2.current_phase = None
            mock_adapter2.class_to_idx = None
            
            pipeline.adapters['tomato'] = mock_adapter1
            pipeline.adapters['pepper'] = mock_adapter2
            
            return pipeline
    
    def test_get_crop_status(self, pipeline_for_status):
        """Test getting crop status."""
        status = pipeline_for_status.get_crop_status()
        
        assert 'tomato' in status
        assert status['tomato']['is_trained'] is True
        assert status['tomato']['current_phase'] == 1
        assert status['tomato']['num_classes'] == 2
        
        assert 'pepper' in status
        assert status['pepper']['is_trained'] is False
        assert status['pepper']['current_phase'] is None
        assert status['pepper']['num_classes'] == 0
    
    def test_get_cache_stats(self, pipeline_for_status):
        """Test getting cache statistics."""
        pipeline = pipeline_for_status
        pipeline.cache_hits = 10
        pipeline.cache_misses = 5
        pipeline.router_cache['key1'] = None
        pipeline.adapter_cache['key2'] = None
        
        stats = pipeline.get_cache_stats()
        
        assert stats['cache_hits'] == 10
        assert stats['cache_misses'] == 5
        assert stats['hit_rate'] == 10 / 15
        assert stats['router_cache_size'] == 1
        assert stats['adapter_cache_size'] == 1
        assert stats['cache_enabled'] is True


class TestIndependentMultiCropPipelineThreadSafety:
    """Test thread safety of concurrent operations."""
    
    @pytest.fixture
    def thread_safe_pipeline(self):
        """Setup pipeline for thread safety testing."""
        config = {'crops': ['tomato'], 'cache_enabled': True}
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock components
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
            }
            pipeline.adapters['tomato'] = mock_adapter
            
            return pipeline
    
    def test_concurrent_cache_access(self, thread_safe_pipeline):
        """Test concurrent access to cache."""
        pipeline = thread_safe_pipeline
        mock_image = torch.randn(1, 3, 224, 224)
        
        results = []
        errors = []
        
        def worker():
            try:
                result = pipeline.process_image(mock_image)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        # All results should be consistent
        assert all(r['status'] == 'success' for r in results)
    
    def test_concurrent_batch_processing(self, thread_safe_pipeline):
        """Test concurrent batch processing."""
        pipeline = thread_safe_pipeline
        
        # Override router for batch
        pipeline.router.route_batch.return_value = (['tomato'] * 5, [0.95] * 5)
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def worker(batch_size):
            try:
                images = [torch.randn(1, 3, 224, 224) for _ in range(batch_size)]
                batch_results = pipeline.batch_process(images)
                with lock:
                    results.extend(batch_results)
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(3,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15  # 5 threads * 3 images each


class TestIndependentMultiCropPipelineTTL:
    """Test TTL functionality (if implemented)."""
    
    @pytest.fixture
    def pipeline_with_ttl(self):
        """Setup pipeline with TTL cache."""
        config = {
            'crops': ['tomato'],
            'cache_enabled': True,
            'cache_ttl_seconds': 1  # 1 second TTL
        }
        with patch('torch.cuda.is_available', return_value=False):
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Mock components
            pipeline.router = MagicMock()
            pipeline.router.route.return_value = ('tomato', 0.95)
            
            mock_adapter = MagicMock()
            mock_adapter.predict_with_ood.return_value = {
                'status': 'success',
                'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.98},
                'ood_analysis': {'is_ood': False, 'ood_score': 0.1, 'threshold': 25.0}
            }
            pipeline.adapters['tomato'] = mock_adapter
            
            return pipeline
    
    def test_cache_ttl_expiration(self, pipeline_with_ttl):
        """Test that cache entries expire after TTL."""
        pipeline = pipeline_with_ttl
        mock_image = torch.randn(1, 3, 224, 224)
        
        # First call - cache miss
        result1 = pipeline.process_image(mock_image)
        assert pipeline.cache_misses == 1
        
        # Second call immediately - should be cache hit
        result2 = pipeline.process_image(mock_image)
        assert pipeline.cache_hits == 1
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Third call after TTL - should be cache miss again
        result3 = pipeline.process_image(mock_image)
        assert pipeline.cache_misses == 2
        assert pipeline.cache_hits == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])