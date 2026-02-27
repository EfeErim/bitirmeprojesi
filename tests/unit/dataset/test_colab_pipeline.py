#!/usr/bin/env python3
"""
Tests for Colab Data Pipeline Components
Tests for download script, cache manager, DataLoader, and dataset classes.
"""

import pytest
import sys
import tempfile
import shutil
import os
from pathlib import Path
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.dataset.error_handling import (
    DownloadError,
    CacheError,
    DataLoaderError,
    DatasetError,
    get_error_handler,
    get_retry_handler,
    get_resource_monitor
)
from src.dataset.colab_cache import (
    LRUCache,
    CacheEntry,
    ColabCacheManager,
    get_colab_cache_manager
)
from src.dataset.colab_dataloader import (
    DataLoaderConfig,
    AdaptiveWorkerManager,
    PrefetchIterator,
    MemoryEfficientDataset,
    ColabDataLoader,
    create_optimized_dataloader,
    get_colab_dataloader,
    benchmark_dataloader
)
from src.dataset.colab_datasets import (
    DatasetConfig,
    LazyLoadConfig,
    MemoryMapConfig,
    ProgressiveLoadConfig,
    LazyLoadingDataset,
    MemoryMappedDataset,
    ProgressiveLoadingDataset,
    ColabDataset,
    get_colab_dataset,
    ColabDataAugmentation,
    benchmark_dataset
)


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size_bytes=1024*1024, max_entries=10)
        
        entry1 = CacheEntry(
            file_path="/tmp/test1",
            original_path="/original/test1",
            size_bytes=100,
            last_accessed=time.time(),
            access_count=1,
            checksum="abc123",
            metadata={}
        )
        
        cache.put("key1", entry1)
        retrieved = cache.get("key1")
        
        assert retrieved is not None
        assert retrieved.file_path == "/tmp/test1"
        assert retrieved.access_count == 2  # Incremented on access
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size_bytes=1000, max_entries=3)
        
        # Add 3 entries
        for i in range(3):
            entry = CacheEntry(
                file_path=f"/tmp/test{i}",
                original_path=f"/original/test{i}",
                size_bytes=100,
                last_accessed=time.time(),
                access_count=1,
                checksum=f"hash{i}",
                metadata={}
            )
            cache.put(f"key{i}", entry)
        
        # Add 4th entry, should evict the least recently used
        entry4 = CacheEntry(
            file_path="/tmp/test4",
            original_path="/original/test4",
            size_bytes=100,
            last_accessed=time.time(),
            access_count=1,
            checksum="hash4",
            metadata={}
        )
        cache.put("key4", entry4)
        
        assert len(cache.cache) == 3
        assert cache.get("key0") is None  # First entry should be evicted
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size_bytes=1024*1024, max_entries=10)
        
        entry = CacheEntry(
            file_path="/tmp/test",
            original_path="/original/test",
            size_bytes=500,
            last_accessed=time.time(),
            access_count=5,
            checksum="abc",
            metadata={}
        )
        
        cache.put("key", entry)
        stats = cache.stats()
        
        assert stats["entries"] == 1
        assert stats["total_size_bytes"] == 500
        assert stats["max_size_bytes"] == 1024*1024
        assert stats["max_entries"] == 10


class TestColabCacheManager:
    """Test Colab cache manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache_mgr = ColabCacheManager(
            cache_dir=str(temp_cache_dir),
            max_cache_size_gb=1.0,
            max_entries=100
        )
        
        assert cache_mgr.cache_dir == temp_cache_dir
        assert cache_mgr.lru_cache.max_size_bytes == 1 * 1024**3
        assert cache_mgr.lru_cache.max_entries == 100
    
    def test_cache_and_retrieve_file(self, temp_cache_dir):
        """Test caching and retrieving a file."""
        cache_mgr = ColabCacheManager(cache_dir=str(temp_cache_dir))
        
        # Create a test file
        test_file = temp_cache_dir / "test_source.txt"
        test_file.write_text("test content")
        
        # Cache the file
        cached_path = cache_mgr.cache_file(
            source_path=test_file,
            file_key="test_key",
            metadata={"description": "Test file"}
        )
        
        assert cached_path.exists()
        
        # Retrieve from cache
        retrieved = cache_mgr.get_cached_file("test_key")
        assert retrieved is not None
        assert retrieved == cached_path
    
    def test_cache_invalidation(self, temp_cache_dir):
        """Test cache invalidation."""
        cache_mgr = ColabCacheManager(cache_dir=str(temp_cache_dir))
        
        test_file = temp_cache_dir / "test.txt"
        test_file.write_text("test")
        
        cached_path = cache_mgr.cache_file(
            source_path=test_file,
            file_key="test"
        )
        
        assert cached_path.exists()
        
        # Invalidate
        result = cache_mgr.invalidate("test")
        assert result is True
        if os.name == "nt":
            # Restricted Windows sandbox can deny explicit file delete calls.
            probe = temp_cache_dir / "delete_probe.tmp"
            probe.write_text("probe")
            delete_permitted = True
            try:
                os.remove(probe)
            except PermissionError:
                delete_permitted = False

            if delete_permitted:
                assert not cached_path.exists()
            else:
                assert cache_mgr.get_cached_file("test") is None
        else:
            assert not cached_path.exists()
    
    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache_mgr = ColabCacheManager(cache_dir=str(temp_cache_dir))
        
        test_file = temp_cache_dir / "test.txt"
        test_file.write_text("test content here")
        
        cache_mgr.cache_file(
            source_path=test_file,
            file_key="test1"
        )
        
        stats = cache_mgr.get_stats()
        assert "entries" in stats
        assert "disk_usage_gb" in stats


class TestAdaptiveWorkerManager:
    """Test adaptive worker manager."""
    
    def test_get_optimal_workers(self):
        """Test optimal worker calculation."""
        workers = AdaptiveWorkerManager.get_optimal_workers()
        assert workers >= 1
        
        # Test with specific request
        workers = AdaptiveWorkerManager.get_optimal_workers(requested=4)
        assert workers <= 4
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Batch size 32, image size (3, 224, 224), float32
        memory_gb = AdaptiveWorkerManager.estimate_memory_usage(
            batch_size=32,
            sample_shape=(3, 224, 224),
            dtype=torch.float32
        )
        assert memory_gb > 0
        assert memory_gb < 10  # Should be reasonable


class TestColabDataLoader:
    """Test optimized DataLoader."""
    
    @pytest.fixture
    def dummy_dataset(self):
        """Create dummy dataset for testing."""
        data = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 10, (100,))
        return TensorDataset(data, labels)
    
    def test_dataloader_creation(self, dummy_dataset):
        """Test DataLoader creation."""
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=False
        )
        
        loader = ColabDataLoader(dummy_dataset, config)
        
        assert loader.config.batch_size == 16
        assert loader.config.num_workers == 2
        assert len(loader) > 0
    
    def test_adaptive_workers(self, dummy_dataset):
        """Test adaptive worker count."""
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=None,  # Will be auto-detected
            adaptive_num_workers=True
        )
        
        loader = ColabDataLoader(dummy_dataset, config)
        assert loader.config.num_workers >= 1
    
    def test_iterator_functionality(self, dummy_dataset):
        """Test DataLoader iteration."""
        config = DataLoaderConfig(
            batch_size=10,
            num_workers=0,  # No multiprocessing for testing
            pin_memory=False
        )
        
        loader = ColabDataLoader(dummy_dataset, config)
        
        batches = list(loader)
        assert len(batches) > 0
        assert batches[0][0].shape[0] <= 10  # Batch size
    
    def test_performance_stats(self, dummy_dataset):
        """Test performance statistics."""
        config = DataLoaderConfig(
            batch_size=10,
            num_workers=0
        )
        
        loader = ColabDataLoader(dummy_dataset, config)
        
        # Iterate a few batches to generate stats
        for i, _ in enumerate(loader):
            if i >= 2:
                break
        
        stats = loader.get_performance_stats()
        assert "num_batches" in stats
        assert "throughput_batches_per_sec" in stats
    
    def test_get_colab_dataloader_convenience(self, dummy_dataset):
        """Test convenience function."""
        loader = get_colab_dataloader(
            dummy_dataset,
            batch_size=16,
            prefetch_factor=2
        )
        
        assert isinstance(loader, ColabDataLoader)
        assert loader.config.batch_size == 16


class TestLazyLoadingDataset:
    """Test lazy loading dataset."""
    
    @pytest.fixture
    def dummy_dataset(self):
        """Create dummy dataset."""
        data = torch.randn(50, 3, 32, 32)
        labels = torch.randint(0, 5, (50,))
        return TensorDataset(data, labels)
    
    def test_lazy_loading_initialization(self, dummy_dataset):
        """Test lazy loading dataset initialization."""
        config = LazyLoadConfig(
            enabled=True,
            prefetch_size=10,
            cache_size_mb=100
        )
        
        lazy_dataset = LazyLoadingDataset(
            dummy_dataset,
            config=config
        )
        
        assert len(lazy_dataset) == len(dummy_dataset)
        assert lazy_dataset.config.enabled is True
    
    def test_item_access(self, dummy_dataset):
        """Test item access with lazy loading."""
        config = LazyLoadConfig(
            enabled=True,
            prefetch_size=5,
            cache_size_mb=100
        )
        
        lazy_dataset = LazyLoadingDataset(dummy_dataset, config=config)
        
        # Access items
        item0 = lazy_dataset[0]
        item1 = lazy_dataset[1]
        
        assert item0 is not None
        assert item1 is not None
    
    def test_cache_clearing(self, dummy_dataset):
        """Test cache clearing."""
        config = LazyLoadConfig(
            enabled=True,
            prefetch_size=10,
            cache_size_mb=100
        )
        
        lazy_dataset = LazyLoadingDataset(dummy_dataset, config=config)
        
        # Access some items to populate cache
        for i in range(5):
            _ = lazy_dataset[i]
        
        # Clear cache
        lazy_dataset.clear_cache()
        assert len(lazy_dataset._cache) == 0


class TestColabDataset:
    """Test high-level Colab dataset."""
    
    @pytest.fixture
    def dummy_dataset(self):
        """Create dummy dataset."""
        data = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 5, (100,))
        return TensorDataset(data, labels)
    
    def test_colab_dataset_initialization(self, dummy_dataset):
        """Test Colab dataset initialization."""
        config = DatasetConfig(
            name="test_dataset",
            classes=["class0", "class1"],
            lazy_load=True,
            memory_map=False,
            progressive_loading=False
        )
        
        colab_dataset = ColabDataset(
            dummy_dataset,
            config=config
        )
        
        assert colab_dataset.config.name == "test_dataset"
        assert colab_dataset._lazy_loader is not None
        assert colab_dataset._memory_mapper is None
        assert colab_dataset._progressive_loader is None
    
    def test_item_access_with_augmentation(self, dummy_dataset):
        """Test item access with augmentation."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        config = DatasetConfig(
            name="test",
            classes=[],
            lazy_load=False,  # Disable lazy loading for simple test
            augmentation=True
        )
        
        colab_dataset = ColabDataset(
            dummy_dataset,
            config=config,
            transform=transform
        )
        
        item = colab_dataset[0]
        assert item is not None
    
    def test_optimization_stats(self, dummy_dataset):
        """Test optimization statistics."""
        config = DatasetConfig(
            name="test",
            classes=[],
            lazy_load=True,
            memory_map=False,
            progressive_loading=True
        )
        
        colab_dataset = ColabDataset(dummy_dataset, config=config)
        
        stats = colab_dataset.get_optimization_stats()
        assert "lazy_loading" in stats
        assert "progressive_loading" in stats


class TestColabDataAugmentation:
    """Test data augmentation."""
    
    def test_augmentation_levels(self):
        """Test different augmentation levels."""
        for level in ["minimal", "medium", "heavy"]:
            aug = ColabDataAugmentation(augmentation_level=level)
            assert aug.transform is not None
            assert callable(aug.transform)
    
    def test_augmentation_call(self):
        """Test augmentation call."""
        aug = ColabDataAugmentation(augmentation_level="minimal")
        
        # Create a dummy PIL image
        from PIL import Image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        transformed = aug(dummy_image)
        assert isinstance(transformed, torch.Tensor)


class TestErrorHandling:
    """Test error handling components."""
    
    def test_download_error(self):
        """Test DownloadError creation."""
        error = DownloadError(
            message="Download failed",
            file_id="test123",
            destination="/path/to/file"
        )
        
        assert error.context.operation == "download"
        assert error.context.component == "downloader"
        assert error.context.error_code == "DOWNLOAD_FAILED"
        assert error.context.details["file_id"] == "test123"
    
    def test_cache_error(self):
        """Test CacheError creation."""
        error = CacheError(
            message="Cache miss",
            cache_key="key123",
            operation="get"
        )
        
        assert error.context.operation == "get"
        assert error.context.component == "cache"
        assert error.context.error_code == "CACHE_ERROR"
    
    def test_error_handler(self):
        """Test error handler."""
        handler = get_error_handler()
        
        # Test warning logging
        handler.log_warning("Test warning")
        
        # Test info logging
        handler.log_info("Test info")
    
    def test_retry_handler(self):
        """Test retry handler."""
        retry = get_retry_handler(max_retries=2, base_delay=0.1)
        
        # Test successful operation
        result = retry.execute_with_retry(lambda: "success")
        assert result == "success"
        
        # Test operation that fails once then succeeds
        call_count = 0
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First call fails")
            return "success"
        
        result = retry.execute_with_retry(flaky_operation)
        assert result == "success"
        assert call_count == 2
    
    def test_resource_monitor(self):
        """Test resource monitor."""
        monitor = get_resource_monitor()
        
        # These should not raise exceptions
        memory_ok = monitor.check_memory()
        disk_ok = monitor.check_disk_space(required_gb=0.1)
        
        # Results are boolean but we just want to ensure no crashes
        assert isinstance(memory_ok, bool)
        assert isinstance(disk_ok, bool)


class TestBenchmarking:
    """Test benchmarking functions."""
    
    @pytest.fixture
    def dummy_dataset(self):
        """Create dummy dataset."""
        data = torch.randn(50, 3, 32, 32)
        labels = torch.randint(0, 5, (50,))
        return TensorDataset(data, labels)
    
    def test_benchmark_dataset(self, dummy_dataset):
        """Test dataset benchmarking."""
        config = DatasetConfig(
            name="test",
            classes=[],
            lazy_load=False
        )
        
        colab_dataset = ColabDataset(dummy_dataset, config=config)
        
        # Run benchmark with small sample
        stats = benchmark_dataset(colab_dataset, num_samples=10)
        
        assert "samples_per_second" in stats
        assert "avg_time_per_sample_ms" in stats
    
    def test_benchmark_dataloader(self, dummy_dataset):
        """Test DataLoader benchmarking."""
        config = DataLoaderConfig(
            batch_size=5,
            num_workers=0
        )
        
        loader = ColabDataLoader(dummy_dataset, config)
        
        stats = benchmark_dataloader(
            loader,
            num_warmup_batches=2,
            num_benchmark_batches=5
        )
        
        assert "benchmark_throughput_samples_per_sec" in stats


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_full_pipeline(self, temp_dir):
        """Test complete data pipeline."""
        # 1. Create dataset
        data = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 3, (100,))
        dataset = TensorDataset(data, labels)
        
        # 2. Create Colab dataset with optimizations
        config = DatasetConfig(
            name="integration_test",
            classes=["class0", "class1", "class2"],
            lazy_load=True,
            memory_map=False,
            progressive_loading=True,
            max_cache_size_mb=256,
            augmentation=False
        )
        
        colab_dataset = ColabDataset(dataset, config=config)
        
        # 3. Create DataLoader
        loader = get_colab_dataloader(
            colab_dataset,
            batch_size=10,
            num_workers=0
        )
        
        # 4. Iterate through batches
        batch_count = 0
        for batch_data, batch_labels in loader:
            assert batch_data.shape[0] <= 10
            assert batch_labels.shape[0] <= 10
            batch_count += 1
            if batch_count >= 3:
                break
        
        assert batch_count > 0
        
        # 5. Check stats
        stats = colab_dataset.get_optimization_stats()
        assert stats["lazy_loading"] is True
        assert stats["progressive_loading"] is True
        
        loader_stats = loader.get_performance_stats()
        assert "num_batches" in loader_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
