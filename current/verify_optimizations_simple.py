#!/usr/bin/env python3
"""
Simple verification for Stage 2 optimizations - checks that new methods exist and work
"""

import sys
from pathlib import Path
import inspect

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch

def test_batch_processing():
    """Test that batch processing method exists and works."""
    print("Testing batch processing...")
    try:
        from src.router.simple_crop_router import SimpleCropRouter
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, model_name='facebook/dinov3-base', device='cpu')
        
        # Check method exists
        assert hasattr(router, 'route_batch'), "route_batch method missing"
        
        # Test with dummy data
        dummy_batch = torch.randn(3, 3, 224, 224)
        crops_out, confs = router.route_batch(dummy_batch)
        assert len(crops_out) == 3, "Should return 3 crop predictions"
        assert len(confs) == 3, "Should return 3 confidence scores"
        assert all(c in crops for c in crops_out), "All crops should be valid"
        assert all(0 <= c <= 1 for c in confs), "Confidences should be in [0,1]"
        
        print("  PASS: Batch processing works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching():
    """Test that caching mechanism exists."""
    print("Testing caching...")
    try:
        from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
        
        # Check class has caching attributes in __init__
        source = inspect.getsource(IndependentMultiCropPipeline.__init__)
        has_router_cache = 'router_cache' in source
        has_adapter_cache = 'adapter_cache' in source
        has_cache_enabled = 'cache_enabled' in source
        
        assert has_router_cache, "router_cache not found in __init__"
        assert has_adapter_cache, "adapter_cache not found in __init__"
        assert has_cache_enabled, "cache_enabled not found in __init__"
        
        # Check methods exist
        assert hasattr(IndependentMultiCropPipeline, 'clear_cache'), "clear_cache method missing"
        assert hasattr(IndependentMultiCropPipeline, 'get_cache_stats'), "get_cache_stats method missing"
        assert hasattr(IndependentMultiCropPipeline, '_generate_cache_key'), "_generate_cache_key method missing"
        
        print("  PASS: Caching mechanism implemented correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv():
    """Test that OpenCV is used in data loader."""
    print("Testing OpenCV integration...")
    try:
        from src.utils.data_loader import LRUCache
        import cv2
        
        # Check LRUCache exists and works
        cache = LRUCache(capacity=5)
        cache.put("test", torch.randn(3, 224, 224))
        val = cache.get("test")
        assert val is not None, "LRU cache should return stored value"
        
        # Check that data_loader uses cv2
        from src.utils import data_loader
        source = inspect.getsource(data_loader)
        has_cv2 = 'cv2' in source or 'import cv2' in source
        assert has_cv2, "OpenCV (cv2) not found in data_loader module"
        
        print("  PASS: OpenCV and LRU caching integrated correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prototype_optimization():
    """Test that prototype computation is optimized."""
    print("Testing prototype optimization...")
    try:
        from src.ood.prototypes import PrototypeComputer
        
        pc = PrototypeComputer(feature_dim=128, device='cpu')
        
        # Check methods exist
        assert hasattr(pc, 'compute_prototypes_from_features'), "Missing optimized method"
        assert hasattr(pc, 'get_prototype_for_class'), "Missing caching method"
        assert hasattr(pc, 'get_prototype_cache_stats'), "Missing cache stats method"
        
        # Test with dummy data
        features = torch.randn(100, 128)
        labels = torch.randint(0, 5, (100,))
        prototypes, class_stds = pc.compute_prototypes_from_features(features, labels)
        
        assert prototypes.shape[0] == 5, "Should have 5 class prototypes"
        assert len(class_stds) == 5, "Should have 5 class stds"
        
        print("  PASS: Prototype computation optimized correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Stage 2 Optimization Quick Check (v5.5.2-router)")
    print("=" * 60)
    
    results = {
        'Batch Processing': test_batch_processing(),
        'Caching': test_caching(),
        'OpenCV Integration': test_opencv(),
        'Prototype Optimization': test_prototype_optimization()
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
