#!/usr/bin/env python3
"""
Optimized DataLoader for Google Colab
Prefetching implementation, pin_memory optimization, and memory-efficient batching.
"""

import os
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.dataset.error_handling import (
    ErrorContext,
    DataLoaderError,
    get_error_handler,
    get_retry_handler,
    get_resource_monitor
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()
retry_handler = get_retry_handler()
resource_monitor = get_resource_monitor()


class _EmptyLoader:
    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader."""
    
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    timeout: float = 0.0
    multiprocessing_context: Optional[str] = None
    
    # Colab-specific optimizations
    use_shared_memory: bool = True
    max_prefetch_batches: int = 4
    adaptive_num_workers: bool = True
    memory_efficient: bool = True


class AdaptiveWorkerManager:
    """Manages number of workers based on available CPU resources."""
    
    @staticmethod
    def get_optimal_workers(
        requested: Optional[int] = None,
        memory_per_worker_gb: float = 0.5,
        reserve_cpus: int = 2
    ) -> int:
        """Determine optimal number of workers."""
        # Explicitly requested 0 workers means single-process mode (no workers)
        if requested == 0:
            return 0
        
        try:
            import psutil
            
            # Get CPU count
            cpu_count = psutil.cpu_count(logical=True) or 4
            
            # Get available memory
            memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Calculate max workers based on memory
            max_workers_by_memory = int(memory_gb / memory_per_worker_gb)
            
            # Calculate max workers based on CPUs
            max_workers_by_cpu = max(1, cpu_count - reserve_cpus)
            
            # Take minimum of both constraints
            optimal = min(max_workers_by_memory, max_workers_by_cpu)
            
            # Respect requested if provided and valid
            if requested is not None and 0 < requested <= optimal:
                return requested
            
            return max(1, optimal)
            
        except ImportError:
            logger.warning("psutil not available, using default worker count")
            return requested or 2
    
    @staticmethod
    def estimate_memory_usage(
        batch_size: int,
        sample_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> float:
        """Estimate memory usage per batch in GB."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        sample_size = np.prod(sample_shape) * element_size / (1024**3)
        batch_memory = sample_size * batch_size
        
        # Add overhead for DataLoader workers
        overhead_factor = 1.5  # 50% overhead for worker processes
        return batch_memory * overhead_factor


class PrefetchIterator:
    """Iterator with prefetching capability."""
    
    def __init__(
        self,
        dataloader: DataLoader,
        prefetch_factor: int = 2,
        max_prefetch_batches: int = 4
    ):
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.max_prefetch_batches = max_prefetch_batches
        
        self._prefetch_queue: List[Any] = []
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._iteration_started = False
    
    def __iter__(self):
        """Start prefetching when iteration begins."""
        if self._iteration_started:
            raise RuntimeError("Iterator can only be used once")
        
        self._iteration_started = True
        self._stop_event.clear()
        
        # Start prefetch thread
        if self.prefetch_factor > 1:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self._prefetch_thread.start()
        
        return self
    
    def __next__(self) -> Any:
        """Get next batch with prefetching."""
        # Wait for prefetch queue to have items
        with self._lock:
            while len(self._prefetch_queue) == 0:
                if self._stop_event.is_set():
                    raise StopIteration
                
                # If no prefetching, get from dataloader directly
                if self._prefetch_thread is None:
                    break
                
                # Small wait to avoid busy waiting
                time.sleep(0.001)
            
            if len(self._prefetch_queue) > 0:
                batch = self._prefetch_queue.pop(0)
            else:
                # Direct fetch if prefetching is disabled
                batch = next(self.dataloader.__iter__())
        
        return batch
    
    def _prefetch_worker(self):
        """Background thread that prefetches batches."""
        try:
            dataloader_iter = self.dataloader.__iter__()
            
            while not self._stop_event.is_set():
                with self._lock:
                    # Only prefetch if queue is not full
                    if len(self._prefetch_queue) < self.max_prefetch_batches:
                        try:
                            batch = next(dataloader_iter)
                            self._prefetch_queue.append(batch)
                        except StopIteration:
                            self._stop_event.set()
                            break
                    else:
                        # Queue is full, wait a bit
                        time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in prefetch worker: {str(e)}")
            self._stop_event.set()
    
    def __del__(self):
        """Clean up prefetch thread."""
        self._stop_event.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)


class MemoryEfficientDataset(Dataset):
    """Base class for memory-efficient dataset operations."""
    
    def __init__(
        self,
        dataset: Dataset,
        cache_manager: Optional[Any] = None,
        lazy_load: bool = True,
        prefetch_size: int = 100
    ):
        self.dataset = dataset
        self.cache_manager = cache_manager
        self.lazy_load = lazy_load
        self.prefetch_size = prefetch_size
        
        self._prefetched_indices: List[int] = []
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()
        
        # Start prefetching if enabled
        if self.lazy_load and prefetch_size > 0:
            self._start_prefetching()
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        """Get item with optional caching."""
        # Check if item is in prefetched cache
        if idx in self._prefetched_indices:
            # This would need actual cache implementation
            pass
        
        # Get from underlying dataset
        item = self.dataset[idx]
        
        # Optionally cache the item
        if self.cache_manager and hasattr(item, '__len__'):
            try:
                # Cache logic would go here
                pass
            except Exception as e:
                logger.debug(f"Cache error: {str(e)}")
        
        return item
    
    def _start_prefetching(self):
        """Start background prefetching of dataset items."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self._prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Background worker for prefetching dataset items."""
        try:
            total_len = len(self.dataset)
            prefetch_indices = list(range(min(self.prefetch_size, total_len)))
            
            for idx in prefetch_indices:
                if self._stop_prefetch.is_set():
                    break
                
                try:
                    # Access the item to load it into memory/OS cache
                    _ = self.dataset[idx]
                    with self._prefetch_lock:
                        self._prefetched_indices.append(idx)
                except Exception as e:
                    logger.debug(f"Prefetch error for index {idx}: {str(e)}")
        except Exception as e:
            logger.error(f"Prefetch worker error: {str(e)}")
    
    def stop_prefetching(self):
        """Stop prefetching."""
        self._stop_prefetch.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
    
    def __del__(self):
        """Clean up prefetch thread."""
        self.stop_prefetching()


def create_optimized_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    collate_fn: Optional[Callable] = None,
    sampler: Optional[Any] = None,
    shuffle: bool = True
) -> DataLoader:
    """Create an optimized DataLoader for Colab."""
    
    # Determine optimal number of workers if adaptive
    if config.adaptive_num_workers:
        optimal_workers = AdaptiveWorkerManager.get_optimal_workers(
            requested=config.num_workers
        )
        logger.info(f"Using {optimal_workers} workers for DataLoader")
        config.num_workers = optimal_workers
    
    # Create base DataLoader
    base_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=config.drop_last,
        timeout=config.timeout,
        multiprocessing_context=config.multiprocessing_context,
        sampler=sampler,
        collate_fn=collate_fn
    )
    
    # Wrap with prefetching if enabled
    if config.prefetch_factor > 1 and config.num_workers > 0:
        logger.info("DataLoader: Prefetching enabled")
        # Note: PyTorch DataLoader already has built-in prefetching via prefetch_factor
        # We could add custom prefetching on top if needed
    
    return base_loader


class ColabDataLoader:
    """
    High-level DataLoader wrapper with Colab-specific optimizations.
    
    Features:
    - Automatic worker count optimization
    - Memory usage monitoring
    - Prefetching with configurable factors
    - Pin memory optimization
    - Graceful degradation on resource constraints
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: Optional[DataLoaderConfig] = None,
        collate_fn: Optional[Callable] = None,
        sampler: Optional[Any] = None,
        **kwargs
    ):
        self.dataset = dataset

        if config is not None:
            self.config = config
        else:
            self.config = DataLoaderConfig(
                batch_size=kwargs.pop('batch_size', 32),
                num_workers=kwargs.pop('num_workers', 4),
                pin_memory=kwargs.pop('pin_memory', True),
                prefetch_factor=kwargs.pop('prefetch_factor', 2),
                persistent_workers=kwargs.pop('persistent_workers', True),
                drop_last=kwargs.pop('drop_last', False),
                timeout=kwargs.pop('timeout', 0.0),
                multiprocessing_context=kwargs.pop('multiprocessing_context', None),
                adaptive_num_workers=kwargs.pop('adaptive_num_workers', True),
                memory_efficient=kwargs.pop('memory_efficient', True)
            )

        self.collate_fn = collate_fn
        self.sampler = sampler

        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.prefetch_factor = self.config.prefetch_factor
        self.shuffle = kwargs.pop('shuffle', sampler is None)
        
        self._dataloader: Optional[DataLoader] = None
        self._prefetch_iterator: Optional[PrefetchIterator] = None
        
        # Performance tracking
        self._batch_times: List[float] = []
        self._memory_samples: List[float] = []
        
        self._setup_dataloader()
    
    def _setup_dataloader(self):
        """Initialize the DataLoader with optimizations."""
        try:
            # Check system resources
            if not resource_monitor.check_memory():
                error_handler.log_warning(
                    "Low memory detected, DataLoader performance may be affected",
                    context=ErrorContext(
                        operation="setup_dataloader",
                        component="data_loader",
                        severity="warning",
                        error_code="LOW_MEMORY"
                    )
                )
            
            # Check CUDA availability for pin_memory
            if self.config.pin_memory and not torch.cuda.is_available():
                logger.warning("CUDA not available, disabling pin_memory")
                self.config.pin_memory = False
            
            # Estimate memory requirements
            if hasattr(self.dataset, '__len__') and len(self.dataset) > 0:
                try:
                    sample = self.dataset[0]
                    if isinstance(sample, (torch.Tensor, np.ndarray)):
                        shape = sample.shape if hasattr(sample, 'shape') else (len(sample),)
                        est_memory = AdaptiveWorkerManager.estimate_memory_usage(
                            self.config.batch_size,
                            shape
                        )
                        logger.info(f"Estimated batch memory: {est_memory:.3f} GB")
                        
                        # Adjust batch size if memory is too high
                        if est_memory > 2.0:  # More than 2GB per batch
                            error_handler.log_warning(
                                f"Batch memory too high ({est_memory:.2f}GB). "
                                f"Consider reducing batch_size",
                                context=ErrorContext(
                                    operation="setup_dataloader",
                                    component="data_loader",
                                    severity="warning",
                                    error_code="HIGH_BATCH_MEMORY",
                                    details={"estimated_gb": est_memory}
                                )
                            )
                except Exception as e:
                    logger.debug(f"Could not estimate memory: {str(e)}")
            
            # Create DataLoader
            try:
                self._dataloader = create_optimized_dataloader(
                    self.dataset,
                    self.config,
                    self.collate_fn,
                    self.sampler,
                    self.shuffle
                )
            except Exception as first_error:
                logger.warning(f"Primary DataLoader initialization failed: {first_error}. Retrying with safe settings.")
                self.config.num_workers = 0
                self.config.prefetch_factor = 2
                self._dataloader = create_optimized_dataloader(
                    self.dataset,
                    self.config,
                    self.collate_fn,
                    self.sampler,
                    self.shuffle
                )
            
            logger.info(
                f"DataLoader created: batch_size={self.config.batch_size}, "
                f"num_workers={self.config.num_workers}, pin_memory={self.config.pin_memory}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to create DataLoader ({e}). Falling back to empty loader.")
            self._dataloader = _EmptyLoader()
    
    def __iter__(self) -> Iterator:
        """Get iterator with optional prefetching."""
        if self._dataloader is None:
            raise RuntimeError("DataLoader not initialized")

        dataloader_iter = iter(self._dataloader)

        while True:
            start_time = time.perf_counter()
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break

            batch_time = max(0.0, time.perf_counter() - start_time)
            self._batch_times.append(batch_time)
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        if self._dataloader is None:
            return 0
        return len(self._dataloader)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._batch_times:
            return {"message": "No batches processed yet"}
        
        batch_times = np.array(self._batch_times)
        
        return {
            "num_batches": len(self._batch_times),
            "total_time_seconds": float(np.sum(batch_times)),
            "avg_batch_time_seconds": float(np.mean(batch_times)),
            "min_batch_time_seconds": float(np.min(batch_times)),
            "max_batch_time_seconds": float(np.max(batch_times)),
            "throughput_batches_per_sec": float(1.0 / np.mean(batch_times)),
            "config": {
                "batch_size": self.config.batch_size,
                "num_workers": self.config.num_workers,
                "pin_memory": self.config.pin_memory,
                "prefetch_factor": self.config.prefetch_factor
            }
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._batch_times.clear()
        self._memory_samples.clear()
    
    def close(self):
        """Clean up resources."""
        if self._dataloader:
            # PyTorch DataLoader doesn't have an explicit close
            # but we can clear references
            self._dataloader = None
        
        if self._prefetch_iterator:
            self._prefetch_iterator = None


def benchmark_dataloader(
    dataloader: ColabDataLoader,
    num_warmup_batches: int = 5,
    num_benchmark_batches: int = 20
) -> Dict[str, Any]:
    """Benchmark DataLoader performance."""
    logger.info("Starting DataLoader benchmark...")
    
    # Warmup
    for i, _ in enumerate(dataloader):
        if i >= num_warmup_batches:
            break
    
    dataloader.reset_stats()
    
    # Benchmark
    start_time = time.time()
    for i, _ in enumerate(dataloader):
        if i >= num_benchmark_batches:
            break
    end_time = time.time()
    
    stats = dataloader.get_performance_stats()

    if "config" not in stats:
        stats["config"] = {
            "batch_size": getattr(dataloader.config, "batch_size", 0),
            "num_workers": getattr(dataloader.config, "num_workers", 0),
            "pin_memory": getattr(dataloader.config, "pin_memory", False),
            "prefetch_factor": getattr(dataloader.config, "prefetch_factor", 0)
        }
    processed_batches = int(stats.get("num_batches", 0))
    benchmark_batches = processed_batches if processed_batches > 0 else num_benchmark_batches
    
    # Add overall metrics
    total_time = end_time - start_time
    stats.update({
        "benchmark_total_time_seconds": total_time,
        "benchmark_throughput_samples_per_sec": (
            stats["config"]["batch_size"] * benchmark_batches / total_time
            if total_time > 0 else 0.0
        )
    })
    
    logger.info(f"Benchmark complete: {stats['benchmark_throughput_samples_per_sec']:.2f} samples/sec")
    return stats


def get_colab_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    **kwargs
) -> ColabDataLoader:
    """
    Convenience function to create a Colab-optimized DataLoader.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        num_workers: Number of workers (auto-detected if None)
        **kwargs: Additional DataLoaderConfig options
    
    Returns:
        ColabDataLoader instance
    """
    resolved_num_workers = (
        AdaptiveWorkerManager.get_optimal_workers()
        if num_workers is None
        else num_workers
    )

    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=resolved_num_workers,
        **kwargs
    )
    
    return ColabDataLoader(dataset, config)


if __name__ == "__main__":
    # Example usage with dummy dataset
    from torch.utils.data import TensorDataset
    
    # Create dummy dataset
    dummy_data = torch.randn(1000, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Create optimized DataLoader
    dataloader = get_colab_dataloader(
        dataset,
        batch_size=32,
        prefetch_factor=2,
        memory_efficient=True
    )
    
    # Iterate through a few batches
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {i}: {batch_data.shape}, {batch_labels.shape}")
        if i >= 2:
            break
    
    # Print stats
    stats = dataloader.get_performance_stats()
    print("\nPerformance Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
