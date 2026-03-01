# Google Colab Data Pipeline

This document describes the optimized data pipeline components for Google Colab, designed to handle large datasets efficiently with limited resources.

## Components Overview

### 1. Data Download Script (`scripts/download_data_colab.py`)

Resumable downloads from Google Drive with progress tracking and verification.

**Key Features:**
- Resumable downloads using `gdown`
- Progress tracking with `tqdm`
- Checksum validation (SHA256)
- Support for multiple sources (Google Drive, HTTP, S3)
- Automatic retry with exponential backoff
- Disk space validation before download

**Usage Example:**

```python
from scripts.download_data_colab import get_colab_downloader

# Get configured downloader
downloader = get_colab_downloader()

# Download from Google Drive
downloaded_file = downloader.download_file(
    file_id="YOUR_FILE_ID",
    destination="data/dataset.zip",
    checksum="expected_sha256_hash_here",
    description="Training Dataset"
)

# Download multiple files
file_list = [
    {"id": "file1_id", "destination": "data/file1.zip", "description": "File 1"},
    {"id": "file2_id", "destination": "data/file2.zip", "description": "File 2"}
]

results = downloader.download_multiple_files(file_list)
```

### 2. Local Caching Strategy (`src/dataset/colab_cache.py`)

LRU cache implementation for Google Drive I/O optimization with local SSD caching.

**Key Features:**
- LRU (Least Recently Used) eviction policy
- Configurable cache size (default 10GB)
- Automatic cleanup based on TTL (default 7 days)
- Background cleanup thread
- Checksum validation
- Performance monitoring
- Thread-safe operations

**Usage Example:**

```python
from src.dataset.colab_cache import get_colab_cache_manager

# Get cache manager (10GB max by default)
cache_mgr = get_colab_cache_manager(
    cache_dir="./.cache",
    max_cache_size_gb=10.0
)

# Cache a file
cached_path = cache_mgr.cache_file(
    source_path=Path("/path/to/source/file.jpg"),
    file_key="unique_file_key",
    metadata={"dataset": "train", "split": "class_a"}
)

# Retrieve from cache
cached = cache_mgr.get_cached_file("unique_file_key")

# Get statistics
stats = cache_mgr.get_stats()
print(f"Cache entries: {stats['entries']}")
print(f"Disk usage: {stats['disk_usage_gb']:.2f} GB")

# Invalidate specific entry
cache_mgr.invalidate("unique_file_key")

# Clear old entries (keep recent 24h)
cache_mgr.clear_cache(keep_recent=True)
```

### 3. Optimized DataLoader (`src/dataset/colab_dataloader.py`)

Prefetching implementation, pin_memory optimization, and memory-efficient batching.

**Key Features:**
- Automatic worker count optimization based on CPU and memory
- Memory usage estimation and batch size warnings
- Prefetching with configurable factors
- Pin memory optimization for GPU training
- Performance monitoring and benchmarking
- Graceful degradation on resource constraints

**Usage Example:**

```python
from src.dataset.colab_dataloader import (
    get_colab_dataloader,
    DataLoaderConfig,
    benchmark_dataloader
)

# Create optimized DataLoader
dataloader = get_colab_dataloader(
    dataset=my_dataset,
    batch_size=32,
    num_workers=None,  # Auto-detect optimal
    pin_memory=True,
    prefetch_factor=2,
    memory_efficient=True
)

# Iterate through batches
for batch_data, batch_labels in dataloader:
    # Training step
    pass

# Get performance statistics
stats = dataloader.get_performance_stats()
print(f"Throughput: {stats['throughput_batches_per_sec']:.2f} batches/sec")

# Benchmark DataLoader
benchmark = benchmark_dataloader(
    dataloader,
    num_warmup_batches=5,
    num_benchmark_batches=20
)
```

**Configuration Options:**

```python
config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,           # Auto-detected if None
    pin_memory=True,         # Use pinned memory for GPU
    prefetch_factor=2,       # Prefetch 2x batches
    persistent_workers=True, # Keep workers alive
    drop_last=False,
    timeout=0.0,
    adaptive_num_workers=True,  # Auto-tune workers
    memory_efficient=True,      # Memory optimizations
    max_prefetch_batches=4      # Max prefetch queue size
)
```

### 4. Modified Dataset Classes (`src/dataset/colab_datasets.py`)

Lazy loading, on-the-fly augmentation, memory-mapped file support, and progressive loading.

**Key Features:**
- Lazy loading with prefetching
- Memory-mapped file support for large datasets
- Progressive batch size adjustment
- On-the-fly data augmentation
- Cache integration
- Thread-safe operations

**Usage Example:**

```python
from src.dataset.colab_datasets import (
    get_colab_dataset,
    DatasetConfig,
    ColabDataAugmentation
)
from torchvision import transforms

# Define dataset configuration
config = DatasetConfig(
    name="plant_disease_dataset",
    classes=["healthy", "disease1", "disease2"],
    image_size=(224, 224),
    lazy_load=True,
    memory_map=True,
    progressive_loading=True,
    max_cache_size_mb=1024,  # 1GB cache
    augmentation=True
)

# Define augmentation transform
augmentation = ColabDataAugmentation(augmentation_level="medium")

# Create optimized dataset
colab_dataset = get_colab_dataset(
    dataset=my_base_dataset,
    config=config,
    transform=augmentation
)

# Access items
sample = colab_dataset[0]

# Get optimization statistics
stats = colab_dataset.get_optimization_stats()
print(f"Lazy loader cache size: {stats['lazy_loader']['cache_size']}")
print(f"Current batch size: {stats['progressive_loader']['current_batch_size']}")

# Benchmark dataset
benchmark = benchmark_dataset(colab_dataset, num_samples=100)
print(f"Samples/sec: {benchmark['samples_per_second']:.2f}")
```

**Augmentation Levels:**

```python
# Minimal (only ToTensor)
aug_minimal = ColabDataAugmentation(augmentation_level="minimal")

# Medium (horizontal flip, rotation, color jitter)
aug_medium = ColabDataAugmentation(augmentation_level="medium")

# Heavy (multiple transforms including random crops)
aug_heavy = ColabDataAugmentation(augmentation_level="heavy")
```

## Error Handling

All components include comprehensive error handling with custom exceptions and logging.

**Error Types:**

- `DownloadError`: Download operation failures
- `CacheError`: Cache operation failures
- `DataLoaderError`: DataLoader initialization/iteration failures
- `DatasetError`: Dataset operation failures

**Error Handling Features:**

- Automatic retry with exponential backoff
- Resource monitoring (memory, disk space)
- Detailed error context and traceback
- Configurable logging levels
- Graceful degradation

**Example:**

```python
from src.dataset.error_handling import (
    get_error_handler,
    DownloadError,
    ErrorContext
)

error_handler = get_error_handler()

try:
    # Your code here
    pass
except Exception as e:
    error_handler.handle_exception(
        DownloadError(
            message="Failed to download dataset",
            file_id="abc123",
            destination="data/file.zip",
            cause=e
        )
    )
```

## Complete Pipeline Example

```python
import torch
from torch.utils.data import TensorDataset
from pathlib import Path

# 1. Download data
from scripts.download_data_colab import get_colab_downloader
downloader = get_colab_downloader()
downloaded = downloader.download_file(
    file_id="YOUR_FILE_ID",
    destination="data/dataset.zip",
    checksum="expected_hash"
)

# 2. Setup cache
from src.dataset.colab_cache import get_colab_cache_manager
cache_mgr = get_colab_cache_manager(cache_dir="./.cache")

# 3. Create dataset (assuming you have extracted data)
base_dataset = TensorDataset(
    torch.randn(1000, 3, 224, 224),
    torch.randint(0, 10, (1000,))
)

# 4. Apply Colab optimizations
from src.dataset.colab_datasets import get_colab_dataset, DatasetConfig
config = DatasetConfig(
    name="my_dataset",
    classes=[f"class_{i}" for i in range(10)],
    lazy_load=True,
    memory_map=False,
    progressive_loading=True,
    max_cache_size_mb=512
)
colab_dataset = get_colab_dataset(base_dataset, config=config)

# 5. Create DataLoader
from src.dataset.colab_dataloader import get_colab_dataloader
dataloader = get_colab_dataloader(
    colab_dataset,
    batch_size=32,
    num_workers=None  # Auto-detect
)

# 6. Use in training
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Training step
        pass
```

## Performance Tips for Colab

1. **Cache Size**: Set `max_cache_size_gb` based on available disk space (Colab typically has ~68GB SSD)
2. **Batch Size**: Monitor memory usage and adjust batch size accordingly
3. **Workers**: Use auto-detection (`num_workers=None`) for optimal performance
4. **Prefetching**: Enable prefetching (`prefetch_factor=2`) for I/O bound datasets
5. **Memory Mapping**: Enable for large image datasets stored as files
6. **Lazy Loading**: Always enable for datasets larger than available RAM
7. **Progressive Loading**: Useful when batch size needs to adapt during training

## Testing

Run the comprehensive test suite:

```bash
# Run all Colab pipeline tests
pytest tests/unit/dataset/test_colab_pipeline.py -v

# Run specific test class
pytest tests/unit/dataset/test_colab_pipeline.py::TestColabCacheManager -v

# Run with coverage
pytest tests/unit/dataset/test_colab_pipeline.py --cov=src/dataset
```

## Troubleshooting

### Low Memory Warnings
- Reduce `batch_size` in DataLoader
- Reduce `max_cache_size_mb` in dataset config
- Disable memory mapping if enabled

### Slow Downloads
- Check network connectivity
- Verify Google Drive file permissions
- Use resumable downloads (automatic)

### DataLoader Stuck
- Reduce `num_workers` (Colab has limited CPU)
- Disable prefetching (`prefetch_factor=1`)
- Check for deadlocks in custom collate functions

### Cache Issues
- Clear cache with `cache_mgr.clear_cache()`
- Check disk space with `!df -h`
- Verify file permissions in cache directory

## API Reference

### `scripts/download_data_colab.py`

- `DriveDownloader`: Google Drive downloader with resumable capability
- `MultiSourceDownloader`: Supports multiple sources (Drive, HTTP, S3)
- `get_colab_downloader()`: Factory function for Colab-configured downloader

### `src/dataset/colab_cache.py`

- `LRUCache`: Least Recently Used cache implementation
- `CacheEntry`: Cache entry dataclass
- `ColabCacheManager`: Main cache manager with background cleanup
- `PerformanceMonitor`: Performance tracking
- `get_colab_cache_manager()`: Factory function

### `src/dataset/colab_dataloader.py`

- `DataLoaderConfig`: Configuration dataclass
- `AdaptiveWorkerManager`: Auto-tune worker count
- `PrefetchIterator`: Iterator with prefetching
- `MemoryEfficientDataset`: Base for memory-efficient datasets
- `ColabDataLoader`: High-level DataLoader wrapper
- `create_optimized_dataloader()`: Factory function
- `get_colab_dataloader()`: Convenience function
- `benchmark_dataloader()`: Performance benchmarking

### `src/dataset/colab_datasets.py`

- `DatasetConfig`: Dataset configuration
- `LazyLoadingDataset`: Lazy loading with prefetching
- `MemoryMappedDataset`: Memory-mapped file support
- `ProgressiveLoadingDataset`: Progressive batch size adjustment
- `ColabDataset`: High-level dataset with all optimizations
- `ColabDataAugmentation`: On-the-fly augmentation
- `get_colab_dataset()`: Convenience function
- `benchmark_dataset()`: Dataset benchmarking

### `src/dataset/error_handling.py`

- `ColabDataPipelineError`: Base exception
- `DownloadError`, `CacheError`, `DataLoaderError`, `DatasetError`: Specific exceptions
- `ErrorContext`: Error context dataclass
- `ColabDataPipelineErrorHandler`: Centralized error handler
- `RetryHandler`: Retry logic with exponential backoff
- `ResourceMonitor`: System resource monitoring
- `get_error_handler()`, `get_retry_handler()`, `get_resource_monitor()`: Factory functions

## License

Same as the main project license.