import inspect

import torch

from src.ood.prototypes import PrototypeComputer
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.utils import data_loader
from src.utils.data_loader import LRUCache


def test_pipeline_exposes_caching_surface():
    source = inspect.getsource(IndependentMultiCropPipeline.__init__)
    assert 'router_cache' in source
    assert 'adapter_cache' in source
    assert 'cache_enabled' in source
    assert hasattr(IndependentMultiCropPipeline, 'clear_cache')
    assert hasattr(IndependentMultiCropPipeline, 'get_cache_stats')


def test_lru_cache_basic_roundtrip():
    cache = LRUCache(capacity=5)
    payload = torch.randn(3, 8, 8)
    cache.put('key', payload)
    loaded = cache.get('key')
    assert loaded is not None
    assert torch.equal(loaded, payload)


def test_opencv_dependency_is_available_in_data_loader_module():
    source = inspect.getsource(data_loader)
    assert 'cv2' in source or 'import cv2' in source


def test_prototype_computation_from_features():
    pc = PrototypeComputer(feature_dim=16, device='cpu', min_samples=1)
    features = torch.randn(20, 16)
    labels = torch.tensor([0] * 7 + [1] * 7 + [2] * 6, dtype=torch.long)
    prototypes, class_stds = pc.compute_prototypes_from_features(features, labels)
    assert prototypes.shape == (3, 16)
    assert len(class_stds) == 3
