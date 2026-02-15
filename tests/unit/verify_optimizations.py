#!/usr/bin/env python3
"""
Simple verification script for Stage 2 optimizations.
Tests that batch processing produces same results as single image.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import time

def verify_batch_processing():
    """Verify batch processing gives same results as single image."""
    print("=" * 60)
    print("VERIFICATION: Batch Processing Consistency")
    print("=" * 60)
    
    try:
        from src.router.simple_crop_router import SimpleCropRouter
        
        # Initialize router
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, model_name='facebook/dinov3-base', device='cpu')
        
        # Create test images
        num_test = 5
        test_images = [torch.randn(1, 3, 224, 224) for _ in range(num_test)]
        
        # Get single image results
        single_results = []
        for img in test_images:
            crop, conf = router.route(img)
            single_results.append((crop, conf))
        
        # Get batch results
        batch_tensor = torch.cat(test_images, dim=0)
        batch_crops, batch_confs = router.route_batch(batch_tensor)
        batch_results = list(zip(batch_crops, batch_confs))
        
        # Compare
        all_match = True
        for i, (single, batch) in enumerate(zip(single_results, batch_results)):
            match = single[0] == batch[0] and abs(single[1] - batch[1]) < 1e-6
            print(f"  Image {i}: Single={single}, Batch={batch}, Match={match}")
            if not match:
                all_match = False
        
        print(f"\nResult: {'PASS' if all_match else 'FAIL'}")
        return all_match
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False

def verify_caching():
    """Verify caching mechanism works."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Caching Mechanism")
    print("=" * 60)
    
    try:
        from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
        import torch
        
        config = {
            'crops': ['tomato', 'pepper', 'corn'],
            'cache_enabled': True,
            'cache_size': 10
        }
        
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        # Create test image
        test_img = torch.randn(1, 3, 224, 224)
        
        # First access should be a miss
        pipeline.clear_cache()
        result1 = pipeline.process_image(test_img)
        misses1 = pipeline.cache_misses
        
        # Second access should be a hit
        result2 = pipeline.process_image(test_img)
        hits1 = pipeline.cache_hits
        
        # Check results match
        results_match = result1 == result2  # Simple equality check
        cache_works = misses1 == 1 and hits1 == 1 and results_match
        
        print(f"  First access: misses={pipeline.cache_misses}")
        print(f"  Second access: hits={pipeline.cache_hits}")
        print(f"  Results match: {results_match}")
        print(f"\nResult: {'PASS' if cache_works else 'FAIL'}")
        
        return cache_works
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_opencv_loading():
    """Verify OpenCV loading works."""
    print("\n" + "=" * 60)
    print("VERIFICATION: OpenCV Image Loading")
    print("=" * 60)
    
    try:
        from src.utils.data_loader import CropDataset
        import cv2
        
        # Create a dummy dataset (will fail if no data, but we just want to check imports)
        print("  Checking OpenCV import and LRUCache...")
        
        from src.utils.data_loader import LRUCache
        
        # Test LRU cache
        cache = LRUCache(capacity=5)
        cache.put("key1", torch.randn(3, 224, 224))
        val = cache.get("key1")
        
        if val is not None:
            print("  LRU cache working")
        else:
            print("  LRU cache failed")
            return False
        
        print("\nResult: PASS")
        return True
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_prototype_optimization():
    """Verify prototype computation works with vectorization."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Prototype Computation Vectorization")
    print("=" * 60)
    
    try:
        from src.ood.prototypes import PrototypeComputer
        import torch
        
        # Create prototype computer
        pc = PrototypeComputer(feature_dim=128, device='cpu')
        
        # Create dummy features and labels
        num_samples = 100
        num_classes = 5
        features = torch.randn(num_samples, 128)
        labels = torch.randint(0, num_classes, (num_samples,))
        
        # Compute prototypes
        prototypes, class_stds = pc.compute_prototypes_from_features(features, labels)
        
        # Check shapes
        expected_shape = (num_classes, 128)
        shape_match = prototypes.shape == expected_shape
        stds_match = len(class_stds) == num_classes
        
        print(f"  Prototypes shape: {prototypes.shape} (expected {expected_shape})")
        print(f"  Class stds count: {len(class_stds)} (expected {num_classes})")
        print(f"\nResult: {'PASS' if shape_match and stds_match else 'FAIL'}")
        
        return shape_match and stds_match
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Stage 2 Optimization Verification (v5.5.2-router)")
    print("=" * 60)
    
    results = {
        'Batch Processing': verify_batch_processing(),
        'Caching': verify_caching(),
        'OpenCV Loading': verify_opencv_loading(),
        'Prototype Vectorization': verify_prototype_optimization()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
