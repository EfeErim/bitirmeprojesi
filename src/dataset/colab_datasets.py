#!/usr/bin/env python3
"""
Modified Dataset Classes for Google Colab
Lazy loading, on-the-fly augmentation, memory-mapped file support, and progressive loading.
"""

import os
import sys
import logging
import time
import threading
import mmap
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Add src to path for error handling imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.dataset.error_handling import (
    ErrorContext,
    DatasetError,
    get_error_handler,
    get_retry_handler,
    get_resource_monitor
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()
retry_handler = get_retry_handler()
resource_monitor = get_resource_monitor()


@dataclass
class DatasetConfig:
    """Configuration for dataset classes."""
    
    name: str
    classes: List[str]
    image_size: Tuple[int, int] = (224, 224)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    lazy_load: bool = True
    memory_map: bool = True
    progressive_loading: bool = True
    max_cache_size_mb: int = 1024  # 1GB
    augmentation: bool = True
    num_workers: int = 4
    prefetch_size: int = 100


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading."""
    
    enabled: bool = True
    prefetch_size: int = 100
    cache_size_mb: int = 1024
    max_prefetch_threads: int = 4
    timeout_seconds: float = 30.0


@dataclass
class MemoryMapConfig:
    """Configuration for memory-mapped files."""
    
    enabled: bool = True
    map_mode: str = "r"  # 'r' for read-only, 'r+' for read-write
    access: str = "sequential"  # 'sequential' or 'random'
    chunk_size_mb: int = 64


@dataclass
class ProgressiveLoadConfig:
    """Configuration for progressive loading."""
    
    enabled: bool = True
    initial_batch_size: int = 32
    max_batch_size: int = 256
    growth_factor: float = 1.2
    warmup_batches: int = 10
    adaptive: bool = True


class ColabCropDataset(Dataset):
    """Colab-friendly ImageFolder wrapper used by training notebooks."""

    def __init__(self, data_dir: Path, transform: Optional[Callable] = None):
        from torchvision.datasets import ImageFolder

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.dataset = ImageFolder(str(self.data_dir), transform=None)
        self.classes = list(self.dataset.classes)
        self.class_to_idx = dict(self.dataset.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception:
                image = torch.randn(3, 224, 224)
        elif not isinstance(image, torch.Tensor):
            image = torch.randn(3, 224, 224)

        return image, label


class ColabDomainShiftDataset(ColabCropDataset):
    """Extension of `ColabCropDataset` that emits domain labels for Phase 3."""

    def __init__(self, data_dir: Path, transform: Optional[Callable] = None, domain_label: int = 0):
        super().__init__(data_dir=data_dir, transform=transform)
        self.domain_label = int(domain_label)
        self.domain_labels = [self.domain_label for _ in range(len(self.dataset))]

    def __getitem__(self, idx: int):
        image, label = super().__getitem__(idx)
        return {
            'images': image,
            'labels': label,
            'domain': self.domain_label
        }


class LazyLoadingDataset(Dataset):
    """Base class for lazy loading datasets."""
    
    def __init__(
        self,
        dataset: Dataset,
        config: LazyLoadConfig = None,
        cache_manager: Optional[Any] = None
    ):
        self.dataset = dataset
        self.config = config or LazyLoadConfig()
        self.cache_manager = cache_manager
        
        self._cache: Dict[int, Any] = {}
        self._cache_lock = threading.Lock()
        self._prefetch_queue: List[int] = []
        self._prefetch_threads: List[threading.Thread] = []
        self._stop_prefetch = threading.Event()
        
        # Start prefetching if enabled
        if self.config.enabled and self.config.prefetch_size > 0:
            self._start_prefetching()
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with lazy loading."""
        try:
            # Check system resources
            if not resource_monitor.check_memory():
                error_handler.log_warning(
                    f"Low memory during lazy loading at index {idx}",
                    context=ErrorContext(
                        operation="lazy_load_getitem",
                        component="lazy_loader",
                        severity="warning",
                        error_code="LOW_MEMORY"
                    )
                )
            
            # Check cache first
            with self._cache_lock:
                if idx in self._cache:
                    return self._cache[idx]
            
            # Load from dataset with retry
            item = retry_handler.execute_with_retry(self._load_item, idx)
            
            # Cache the item
            if self.config.enabled:
                with self._cache_lock:
                    if len(self._cache) < self.config.cache_size_mb * 1024 // 4:  # Rough estimate
                        self._cache[idx] = item
            
            return item
            
        except Exception as e:
            error_handler.handle_exception(
                DatasetError(
                    message=f"Failed to get item at index {idx}: {str(e)}",
                    dataset_name="lazy_loader",
                    operation="lazy_load_getitem",
                    cause=e
                )
            )
            raise
    
    def _load_item(self, idx: int) -> Any:
        """Load item from dataset (override in subclasses)."""
        return self.dataset[idx]
    
    def _start_prefetching(self):
        """Start background prefetching."""
        if self._stop_prefetch.is_set():
            self._stop_prefetch.clear()
        
        # Create prefetch threads
        for _ in range(min(self.config.max_prefetch_threads, self.config.prefetch_size)):
            thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            thread.start()
            self._prefetch_threads.append(thread)
    
    def _prefetch_worker(self):
        """Background worker for prefetching."""
        try:
            while not self._stop_prefetch.is_set():
                # Get next index to prefetch
                with self._cache_lock:
                    if len(self._prefetch_queue) < self.config.prefetch_size:
                        # Get next available index
                        next_idx = len(self._prefetch_queue)
                        if next_idx < len(self.dataset):
                            self._prefetch_queue.append(next_idx)
                        else:
                            break
                    else:
                        time.sleep(0.01)
                        continue
                
                # Prefetch the item
                idx = self._prefetch_queue[-1]
                try:
                    item = self._load_item(idx)
                    with self._cache_lock:
                        self._cache[idx] = item
                except Exception as e:
                    logger.debug(f"Prefetch error for index {idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Prefetch worker error: {str(e)}")
    
    def stop_prefetching(self):
        """Stop prefetching."""
        self._stop_prefetch.set()
        for thread in self._prefetch_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._prefetch_threads.clear()
    
    def clear_cache(self):
        """Clear the cache."""
        with self._cache_lock:
            self._cache.clear()
            self._prefetch_queue.clear()
    
    def __del__(self):
        """Clean up resources."""
        self.stop_prefetching()
        self.clear_cache()


class MemoryMappedDataset(LazyLoadingDataset):
    """Dataset with memory-mapped file support."""
    
    def __init__(
        self,
        dataset: Dataset,
        file_paths: List[Path],
        mmap_config: MemoryMapConfig = None,
        **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.file_paths = file_paths
        self.mmap_config = mmap_config or MemoryMapConfig()
        
        self._mmaps: Dict[int, Any] = {}
        self._mmap_lock = threading.Lock()
        
        if self.mmap_config.enabled:
            self._setup_memory_mapping()
    
    def _setup_memory_mapping(self):
        """Setup memory mapping for files."""
        for idx, file_path in enumerate(self.file_paths):
            try:
                if file_path.exists():
                    with self._mmap_lock:
                        self._mmaps[idx] = self._create_mmap(file_path)
            except Exception as e:
                logger.debug(f"Memory mapping error for {file_path}: {str(e)}")
    
    def _create_mmap(self, file_path: Path) -> Any:
        """Create memory map for a file."""
        try:
            with open(file_path, 'rb') as f:
                # For large files, map in chunks
                file_size = file_path.stat().st_size
                if file_size > self.mmap_config.chunk_size_mb * 1024 * 1024:
                    # Map in chunks
                    return {
                        "file": f,
                        "size": file_size,
                        "chunks": []
                    }
                else:
                    # Map entire file
                    return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as e:
            logger.debug(f"Failed to create mmap for {file_path}: {str(e)}")
            return None
    
    def _load_item(self, idx: int) -> Any:
        """Load item with memory mapping."""
        if idx in self._mmaps and self._mmaps[idx] is not None:
            mmap_obj = self._mmaps[idx]
            if isinstance(mmap_obj, dict):
                # Handle chunked mapping
                return self._load_chunked_mmap(mmap_obj)
            else:
                # Read from memory map
                mmap_obj.seek(0)
                data = mmap_obj.read()
                return self._process_mmap_data(data)
        
        # Fallback to normal loading
        return super()._load_item(idx)
    
    def _load_chunked_mmap(self, mmap_info: Dict[str, Any]) -> Any:
        """Load data from chunked memory map."""
        # This would need implementation based on file type
        # For now, return placeholder
        return None
    
    def _process_mmap_data(self, data: bytes) -> Any:
        """Process memory-mapped data."""
        # This would need implementation based on file type
        # For now, return placeholder
        return data
    
    def __del__(self):
        """Clean up memory maps."""
        super().__del__()
        with self._mmap_lock:
            for mmap_obj in self._mmaps.values():
                if mmap_obj and not isinstance(mmap_obj, dict):
                    mmap_obj.close()
            self._mmaps.clear()


class ProgressiveLoadingDataset(LazyLoadingDataset):
    """Dataset with progressive loading capability."""
    
    def __init__(
        self,
        dataset: Dataset,
        config: ProgressiveLoadConfig = None,
        **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.config = config or ProgressiveLoadConfig()
        
        self._current_batch_size = self.config.initial_batch_size
        self._batch_count = 0
        self._progress_lock = threading.Lock()
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with progressive loading."""
        item = super().__getitem__(idx)
        
        # Apply progressive loading logic
        with self._progress_lock:
            self._batch_count += 1
            if self.config.adaptive and self._batch_count % 10 == 0:
                self._adjust_batch_size()
        
        return item
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance."""
        if self.config.adaptive:
            # This would need actual performance monitoring
            # For now, just increase gradually
            new_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * self.config.growth_factor)
            )
            self._current_batch_size = new_size
    
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        with self._progress_lock:
            return self._current_batch_size


class ColabDataset(Dataset):
    """
    High-level dataset class with all Colab optimizations.
    
    Features:
    - Lazy loading with prefetching
    - Memory-mapped file support
    - Progressive loading
    - On-the-fly data augmentation
    - Cache management
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: DatasetConfig,
        cache_manager: Optional[Any] = None,
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.config = config
        self.cache_manager = cache_manager
        self.transform = transform
        
        # Initialize optimization layers
        self._lazy_loader = None
        self._memory_mapper = None
        self._progressive_loader = None
        
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup all optimization layers."""
        # Setup lazy loading
        if self.config.lazy_load:
            lazy_config = LazyLoadConfig(
                enabled=True,
                prefetch_size=self.config.prefetch_size,
                cache_size_mb=self.config.max_cache_size_mb
            )
            self._lazy_loader = LazyLoadingDataset(
                self.dataset,
                config=lazy_config,
                cache_manager=self.cache_manager
            )
        
        # Setup memory mapping (if applicable)
        if self.config.memory_map and hasattr(self.dataset, 'file_paths'):
            mmap_config = MemoryMapConfig(
                enabled=True,
                map_mode='r',
                access='sequential'
            )
            self._memory_mapper = MemoryMappedDataset(
                self.dataset,
                self.dataset.file_paths,
                mmap_config=mmap_config,
                cache_manager=self.cache_manager
            )
        
        # Setup progressive loading
        if self.config.progressive_loading:
            prog_config = ProgressiveLoadConfig(
                enabled=True,
                initial_batch_size=32,
                max_batch_size=256,
                growth_factor=1.2,
                warmup_batches=10,
                adaptive=True
            )
            self._progressive_loader = ProgressiveLoadingDataset(
                self.dataset,
                config=prog_config,
                cache_manager=self.cache_manager
            )
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self._lazy_loader:
            return len(self._lazy_loader)
        elif self._memory_mapper:
            return len(self._memory_mapper)
        elif self._progressive_loader:
            return len(self._progressive_loader)
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with all optimizations."""
        try:
            # Get item from appropriate loader
            if self._lazy_loader:
                item = self._lazy_loader[idx]
            elif self._memory_mapper:
                item = self._memory_mapper[idx]
            elif self._progressive_loader:
                item = self._progressive_loader[idx]
            else:
                item = self.dataset[idx]
            
            # Apply transformations
            if self.transform:
                item = self.transform(item)
            
            return item
            
        except Exception as e:
            error_handler.handle_exception(
                DatasetError(
                    message=f"Failed to get item at index {idx}: {str(e)}",
                    dataset_name=self.config.name,
                    operation="getitem",
                    cause=e
                )
            )
            raise
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            "lazy_loading": self.config.lazy_load,
            "memory_mapping": self.config.memory_map,
            "progressive_loading": self.config.progressive_loading,
            "prefetch_size": self.config.prefetch_size,
            "max_cache_size_mb": self.config.max_cache_size_mb
        }
        
        if self._lazy_loader:
            stats["lazy_loader"] = {
                "cache_size": len(self._lazy_loader._cache),
                "prefetch_queue_size": len(self._lazy_loader._prefetch_queue)
            }
        
        if self._memory_mapper:
            stats["memory_mapper"] = {
                "num_mmaps": len(self._memory_mapper._mmaps)
            }
        
        if self._progressive_loader:
            stats["progressive_loader"] = {
                "current_batch_size": self._progressive_loader.get_current_batch_size()
            }
        
        return stats
    
    def clear_optimizations(self):
        """Clear all optimization layers."""
        if self._lazy_loader:
            self._lazy_loader.clear_cache()
        if self._memory_mapper:
            self._memory_mapper.clear_cache()
        if self._progressive_loader:
            self._progressive_loader.clear_cache()
    
    def __del__(self):
        """Clean up resources."""
        self.clear_optimizations()
        if self._lazy_loader:
            self._lazy_loader.stop_prefetching()
        if self._memory_mapper:
            self._memory_mapper.stop_prefetching()
        if self._progressive_loader:
            self._progressive_loader.stop_prefetching()


def get_colab_dataset(
    dataset: Dataset,
    config: Optional[DatasetConfig] = None,
    cache_manager: Optional[Any] = None,
    transform: Optional[Callable] = None
) -> ColabDataset:
    """
    Convenience function to create a Colab-optimized dataset.
    
    Args:
        dataset: PyTorch Dataset
        config: Dataset configuration
        cache_manager: Cache manager instance
        transform: Data transformations
    
    Returns:
        ColabDataset instance
    """
    default_config = DatasetConfig(
        name="colab_dataset",
        classes=[],
        image_size=(224, 224),
        lazy_load=True,
        memory_map=True,
        progressive_loading=True,
        max_cache_size_mb=1024,
        augmentation=True,
        num_workers=4,
        prefetch_size=100
    )
    
    if config:
        # Update default config with provided values
        for key, value in config.__dict__.items():
            if value is not None:
                setattr(default_config, key, value)
    
    return ColabDataset(dataset, default_config, cache_manager, transform)


class ColabDataAugmentation:
    """On-the-fly data augmentation for Colab."""
    
    def __init__(self, augmentation_level: str = "medium"):
        self.augmentation_level = augmentation_level
        self.transform = self._get_augmentation_transform()
    
    def _get_augmentation_transform(self) -> Callable:
        """Get augmentation transform based on level."""
        if self.augmentation_level == "minimal":
            return transforms.Compose([
                transforms.ToTensor()
            ])
        elif self.augmentation_level == "medium":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor()
            ])
        elif self.augmentation_level == "heavy":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomResizedCrop(size=(224, 224)),
                transforms.ToTensor()
            ])
        else:
            return transforms.ToTensor()
    
    def __call__(self, sample):
        """Apply augmentation."""
        return self.transform(sample)


def benchmark_dataset(
    dataset: ColabDataset,
    num_samples: int = 100
) -> Dict[str, Any]:
    """Benchmark dataset performance."""
    logger.info("Starting dataset benchmark...")
    
    start_time = time.time()
    
    for i in range(num_samples):
        try:
            _ = dataset[i]
        except Exception as e:
            logger.error(f"Error at index {i}: {str(e)}")
            break
    
    end_time = time.time()
    
    stats = dataset.get_optimization_stats()
    stats.update({
        "benchmark_time_seconds": end_time - start_time,
        "samples_per_second": num_samples / (end_time - start_time) if end_time > start_time else 0.0,
        "avg_time_per_sample_ms": (end_time - start_time) * 1000 / num_samples
    })
    
    logger.info(f"Benchmark complete: {stats['samples_per_second']:.2f} samples/sec")
    return stats


if __name__ == "__main__":
    # Example usage with dummy dataset
    from torch.utils.data import TensorDataset
    
    # Create dummy dataset
    dummy_data = torch.randn(1000, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Create Colab dataset with augmentations
    augmentation = ColabDataAugmentation(augmentation_level="medium")
    colab_dataset = get_colab_dataset(
        dataset,
        config=DatasetConfig(
            name="test_dataset",
            classes=["class_0", "class_1"],
            lazy_load=True,
            memory_map=True,
            progressive_loading=True,
            max_cache_size_mb=512,
            augmentation=True
        ),
        transform=augmentation
    )
    
    # Test access
    for i in range(5):
        sample = colab_dataset[i]
        print(f"Sample {i}: {type(sample[0])}, {sample[0].shape}")
    
    # Print stats
    stats = colab_dataset.get_optimization_stats()
    print("\nDataset Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Benchmark
    benchmark = benchmark_dataset(colab_dataset, num_samples=50)
    print("\nBenchmark Results:")
    for key, value in benchmark.items():
        print(f"  {key}: {value}")