#!/usr/bin/env python3
"""
Benchmark script for Stage 2 performance optimizations (v5.5.2-router)
Measures improvements in:
- Batch processing (SimpleCropRouter)
- Caching (IndependentMultiCropPipeline)
- OpenCV vs PIL (data loading)
- Prototype computation (vectorization)
"""

import time
import torch
import numpy as np
import logging
from pathlib import Path
import sys
import argparse

# Add project root to path to import from src package
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import CropDataset, preprocess_image, LRUCache
from router.simple_crop_router import SimpleCropRouter
from pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from ood.prototypes import PrototypeComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_open_cv_vs_pil(num_images: int = 100, image_size: Tuple[int, int] = (224, 224)):
    """Compare OpenCV vs PIL loading speed."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: OpenCV vs PIL")
    logger.info("=" * 60)
    
    from PIL import Image
    import cv2
    
    # Create dummy images
    temp_dir = Path("temp_benchmark_images")
    temp_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        img_path = temp_dir / f"test_{i}.jpg"
        if not img_path.exists():
            dummy_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), dummy_img)
        image_paths.append(img_path)
    
    # Benchmark PIL
    start = time.time()
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size)
    pil_time = time.time() - start
    logger.info(f"PIL: {pil_time:.4f}s for {num_images} images ({num_images/pil_time:.2f} img/s)")
    
    # Benchmark OpenCV
    start = time.time()
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
    cv2_time = time.time() - start
    logger.info(f"OpenCV: {cv2_time:.4f}s for {num_images} images ({num_images/cv2_time:.2f} img/s)")
    
    speedup = pil_time / cv2_time
    logger.info(f"Speedup: {speedup:.2f}x faster")
    
    # Cleanup
    for img_path in image_paths:
        img_path.unlink()
    temp_dir.rmdir()
    
    return {
        'pil_time': pil_time,
        'cv2_time': cv2_time,
        'speedup': speedup
    }

def benchmark_batch_processing(num_images: int = 100, batch_size: int = 32):
    """Benchmark batch vs single image routing."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Batch Processing (SimpleCropRouter)")
    logger.info("=" * 60)
    
    # Create dummy model and data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize router (use small model for testing)
    crops = ['tomato', 'pepper', 'corn']
    router = SimpleCropRouter(crops=crops, model_name='facebook/dinov3-base', device=device)
    
    # Create dummy images
    dummy_images = [torch.randn(1, 3, 224, 224) for _ in range(num_images)]
    
    # Benchmark single image processing
    start = time.time()
    for img in dummy_images:
        router.route(img)
    single_time = time.time() - start
    logger.info(f"Single image: {single_time:.4f}s for {num_images} images ({num_images/single_time:.2f} img/s)")
    
    # Benchmark batch processing
    start = time.time()
    for i in range(0, num_images, batch_size):
        batch = dummy_images[i:i+batch_size]
        batch_tensor = torch.cat(batch, dim=0)
        router.route_batch(batch_tensor)
    batch_time = time.time() - start
    logger.info(f"Batch (size={batch_size}): {batch_time:.4f}s for {num_images} images ({num_images/batch_time:.2f} img/s)")
    
    speedup = single_time / batch_time
    logger.info(f"Batch speedup: {speedup:.2f}x faster")
    
    return {
        'single_time': single_time,
        'batch_time': batch_time,
        'speedup': speedup
    }

def benchmark_caching(num_images: int = 1000, cache_size: int = 500):
    """Benchmark caching effectiveness."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Caching (IndependentMultiCropPipeline)")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy config
    config = {
        'crops': ['tomato', 'pepper', 'corn'],
        'cache_enabled': True,
        'cache_size': cache_size
    }
    
    pipeline = IndependentMultiCropPipeline(config, device=device)
    
    # Create dummy images (some duplicates for cache hits)
    dummy_images = []
    for i in range(num_images):
        # Every 10th image is a duplicate to simulate cache hits
        if i % 10 == 0 and i > 0:
            # Use same image as 10 steps back
            img = dummy_images[i-10]
        else:
            img = torch.randn(1, 3, 224, 224)
        dummy_images.append(img)
    
    # Clear any initial cache
    pipeline.clear_cache()
    
    # Process with caching
    start = time.time()
    for img in dummy_images:
        # Mock processing (skip actual adapter prediction)
        cache_key = pipeline._generate_cache_key(img)
        if cache_key in pipeline.adapter_cache:
            pipeline.cache_hits += 1
        else:
            pipeline.cache_misses += 1
            # Simulate some work
            time.sleep(0.001)
            # Add to cache
            pipeline.adapter_cache[cache_key] = {'mock': 'result'}
            if len(pipeline.adapter_cache) > cache_size:
                oldest_key = next(iter(pipeline.adapter_cache))
                del pipeline.adapter_cache[oldest_key]
    cached_time = time.time() - start
    
    stats = pipeline.get_cache_stats()
    logger.info(f"Cached processing: {cached_time:.4f}s")
    logger.info(f"Cache hits: {stats['cache_hits']}")
    logger.info(f"Cache misses: {stats['cache_misses']}")
    logger.info(f"Hit rate: {stats['hit_rate']:.2%}")
    
    # Clear cache and reprocess without caching
    pipeline.clear_cache()
    pipeline.cache_enabled = False
    
    start = time.time()
    for img in dummy_images:
        # Simulate work without cache
        time.sleep(0.001)
    uncached_time = time.time() - start
    
    logger.info(f"Uncached processing: {uncached_time:.4f}s")
    speedup = uncached_time / cached_time if cached_time > 0 else 0
    logger.info(f"Caching speedup: {speedup:.2f}x")
    
    return {
        'cached_time': cached_time,
        'uncached_time': uncached_time,
        'speedup': speedup,
        'hit_rate': stats['hit_rate']
    }

def benchmark_prototype_computation(num_samples: int = 1000, feature_dim: int = 1024, num_classes: int = 5):
    """Benchmark prototype computation optimization."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: Prototype Computation (Vectorization)")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy data
    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    
    # Create prototype computer
    pc = PrototypeComputer(feature_dim=feature_dim, device=device)
    
    # Benchmark vectorized version
    start = time.time()
    prototypes, class_stds = pc.compute_prototypes_from_features(features, labels)
    vectorized_time = time.time() - start
    logger.info(f"Vectorized computation: {vectorized_time:.4f}s")
    
    # Simulate old non-vectorized approach (loop-based)
    start = time.time()
    features_per_class = {}
    for feat, label in zip(features, labels):
        class_idx = label.item()
        if class_idx not in features_per_class:
            features_per_class[class_idx] = []
        features_per_class[class_idx].append(feat.cpu())
    
    old_prototypes = torch.zeros(num_classes, feature_dim, device=device)
    for class_idx, feat_list in features_per_class.items():
        if len(feat_list) >= 2:
            feats = torch.stack(feat_list)
            mean = feats.mean(dim=0)
            old_prototypes[class_idx] = mean
    
    old_time = time.time() - start
    logger.info(f"Old loop-based: {old_time:.4f}s")
    
    speedup = old_time / vectorized_time if vectorized_time > 0 else 0
    logger.info(f"Vectorization speedup: {speedup:.2f}x")
    
    return {
        'vectorized_time': vectorized_time,
        'old_time': old_time,
        'speedup': speedup
    }

def benchmark_lru_cache(num_operations: int = 10000, cache_size: int = 100):
    """Benchmark LRU cache implementation."""
    logger.info("=" * 60)
    logger.info("BENCHMARK: LRU Caching")
    logger.info("=" * 60)
    
    # Create cache
    cache = LRUCache(cache_size)
    
    # Create dummy data
    keys = [f"key_{i}" for i in range(num_operations)]
    values = [torch.randn(3, 224, 224) for _ in range(num_operations)]
    
    # Benchmark with cache hits
    start = time.time()
    hits = 0
    misses = 0
    for i in range(num_operations):
        key = keys[i]
        val = cache.get(key)
        if val is not None:
            hits += 1
        else:
            misses += 1
            cache.put(key, values[i])
    cache_time = time.time() - start
    
    hit_rate = hits / num_operations
    logger.info(f"LRU cache operations: {cache_time:.4f}s for {num_operations} ops")
    logger.info(f"Hits: {hits}, Misses: {misses}, Hit rate: {hit_rate:.2%}")
    logger.info(f"Cache size: {len(cache)}")
    
    return {
        'cache_time': cache_time,
        'hits': hits,
        'misses': misses,
        'hit_rate': hit_rate
    }

def run_all_benchmarks():
    """Run all benchmarks and report results."""
    logger.info("Starting Stage 2 Performance Benchmarks (v5.5.2-router)")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. OpenCV vs PIL
    try:
        results['opencv_vs_pil'] = benchmark_open_cv_vs_pil(num_images=100)
    except Exception as e:
        logger.error(f"OpenCV vs PIL benchmark failed: {e}")
        results['opencv_vs_pil'] = None
    
    # 2. Batch processing
    try:
        results['batch_processing'] = benchmark_batch_processing(num_images=50, batch_size=16)
    except Exception as e:
        logger.error(f"Batch processing benchmark failed: {e}")
        results['batch_processing'] = None
    
    # 3. Caching
    try:
        results['caching'] = benchmark_caching(num_images=200, cache_size=100)
    except Exception as e:
        logger.error(f"Caching benchmark failed: {e}")
        results['caching'] = None
    
    # 4. Prototype computation
    try:
        results['prototype_computation'] = benchmark_prototype_computation(
            num_samples=500, feature_dim=1024, num_classes=5
        )
    except Exception as e:
        logger.error(f"Prototype computation benchmark failed: {e}")
        results['prototype_computation'] = None
    
    # 5. LRU cache
    try:
        results['lru_cache'] = benchmark_lru_cache(num_operations=1000, cache_size=100)
    except Exception as e:
        logger.error(f"LRU cache benchmark failed: {e}")
        results['lru_cache'] = None
    
    # Print summary
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    for benchmark, data in results.items():
        if data:
            logger.info(f"{benchmark}:")
            for key, value in data.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
    
    # Save results
    output_file = Path("benchmarks/stage2_results.json")
    output_file.parent.mkdir(exist_ok=True)
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Stage 2 optimizations')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images for benchmarks')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch processing')
    args = parser.parse_args()
    
    run_all_benchmarks()
