#!/usr/bin/env python3
"""Unit tests for SimpleCropRouter."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.router import simple_crop_router as router_module
from src.router.simple_crop_router import SimpleCropRouter


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]
        hidden_size = int(self.config.hidden_size)
        cls = torch.zeros((batch_size, 1, hidden_size), device=images.device)
        return SimpleNamespace(last_hidden_state=cls)


@pytest.fixture(autouse=True)
def patch_backbone_load(monkeypatch):
    monkeypatch.setattr(
        router_module.AutoModel,
        'from_pretrained',
        lambda *_args, **_kwargs: _DummyBackbone(hidden_size=32),
    )


class TestSimpleCropRouter:
    """Test cases for SimpleCropRouter."""

    def test_initialization(self):
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, model_name='facebook/dinov2-base', device='cpu')

        assert router.crops == crops
        assert router.model_name == 'facebook/dinov2-base'
        assert router.classifier.out_features == len(crops)
        assert router.classifier.in_features == 32
        assert router.device.type == 'cpu'

    def test_route_returns_crop_name(self):
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, device='cpu')

        dummy_input = torch.randn(1, 3, 224, 224)
        crop = router.route(dummy_input)

        assert isinstance(crop, str)
        assert crop in crops

    def test_save_load_aliases(self, tmp_path):
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, device='cpu')

        save_path = tmp_path / "router_test.pt"
        router.save_model(str(save_path))

        new_router = SimpleCropRouter(crops, device='cpu')
        new_router.load_model(str(save_path))

        for p1, p2 in zip(router.classifier.parameters(), new_router.classifier.parameters()):
            assert torch.allclose(p1, p2)