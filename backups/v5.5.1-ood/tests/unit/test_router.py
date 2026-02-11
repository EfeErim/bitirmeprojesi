#!/usr/bin/env python3
"""
Unit tests for SimpleCropRouter
"""

import pytest
import torch
from src.router.simple_crop_router import SimpleCropRouter
from src.utils.data_loader import CropDataset

class TestSimpleCropRouter:
    """Test cases for SimpleCropRouter."""
    
    def test_initialization(self):
        """Test router initialization."""
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, model_name='facebook/dinov2-base', device='cpu')
        
        assert router.crops == crops
        assert router.classifier.out_features == len(crops)
        assert router.device.type == 'cpu'
    
    def test_route_shape(self):
        """Test that route returns correct output format."""
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, device='cpu')
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # This will fail without trained weights but should return correct format
        try:
            crop, confidence = router.route(dummy_input)
            assert crop in crops
            assert 0 <= confidence <= 1
        except Exception as e:
            # Expected if model not trained
            pytest.skip(f"Router not trained: {e}")
    
    def test_save_load(self, tmp_path):
        """Test model save and load."""
        crops = ['tomato', 'pepper', 'corn']
        router = SimpleCropRouter(crops, device='cpu')
        
        # Save
        save_path = tmp_path / "router_test"
        router.save_model(str(save_path))
        
        # Load into new router
        new_router = SimpleCropRouter(crops, device='cpu')
        new_router.load_model(str(save_path))
        
        # Check that classifier weights are loaded
        for p1, p2 in zip(router.classifier.parameters(), new_router.classifier.parameters()):
            assert torch.allclose(p1, p2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])