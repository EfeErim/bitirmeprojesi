#!/usr/bin/env python3
"""
Integration tests for the full AADS-ULoRA pipeline
"""

import pytest
import torch
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.utils.data_loader import CropDataset, preprocess_image
from src.router.simple_crop_router import SimpleCropRouter
from src.adapter.independent_crop_adapter import IndependentCropAdapter

class TestFullPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline can be initialized with config."""
        import json
        
        # Create minimal config
        config = {
            "adapter_id": "test_v55",
            "architecture": "independent_multicrop_dynamic_ood",
            "crops": ["tomato", "pepper"],
            "data": {
                "crops": {
                    "tomato": {"classes": ["healthy", "disease"]},
                    "pepper": {"classes": ["healthy"]}
                }
            },
            "targets": {}
        }
        
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Create pipeline
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        assert pipeline.crops == ["tomato", "pepper"]
        assert pipeline.router is None
        assert len(pipeline.adapters) == 0
    
    def test_router_initialization_without_data(self, tmp_path):
        """Test router initialization fails gracefully without data."""
        config = {
            "crops": ["tomato", "pepper", "corn"],
            "data": {}
        }
        
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        with pytest.raises(ValueError, match="Cannot train router without datasets"):
            pipeline.initialize_router()
    
    def test_adapter_registration(self, tmp_path):
        """Test registering a crop adapter."""
        config = {"crops": ["tomato"]}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        # Create a dummy adapter directory structure
        adapter_dir = tmp_path / "tomato_adapter"
        adapter_dir.mkdir()
        
        # This will fail because adapter is not actually trained
        # but tests the registration logic
        with pytest.raises(Exception):
            pipeline.register_crop("tomato", str(adapter_dir))
    
    def test_process_image_without_router(self, tmp_path):
        """Test that processing fails if router not initialized."""
        config = {"crops": ["tomato"]}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        # Create dummy image
        dummy_image = torch.randn(1, 3, 224, 224)
        
        with pytest.raises(RuntimeError, match="Router not initialized"):
            pipeline.process_image(dummy_image)
    
    def test_batch_processing(self, tmp_path):
        """Test batch processing of multiple images."""
        config = {"crops": ["tomato"]}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        # Create batch of dummy images
        images = [torch.randn(1, 3, 224, 224) for _ in range(3)]
        
        # Will fail due to no router, but tests batch interface
        with pytest.raises(RuntimeError):
            pipeline.batch_process(images)
    
    def test_get_crop_status(self, tmp_path):
        """Test getting crop adapter status."""
        config = {"crops": ["tomato", "pepper"]}
        pipeline = IndependentMultiCropPipeline(config, device='cpu')
        
        status = pipeline.get_crop_status()
        
        assert isinstance(status, dict)
        assert "tomato" in status
        assert "pepper" in status
        assert status["tomato"]["is_trained"] == False

class TestDataFlow:
    """Test data flow through the system."""
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        from PIL import Image
        
        # Create dummy image
        dummy_img = Image.new('RGB', (300, 400), color='red')
        processed = preprocess_image(dummy_img, target_size=224)
        
        assert processed.shape == (3, 224, 224)
        assert processed.dtype == torch.float32
    
    def test_dataset_loading(self, tmp_path):
        """Test dataset loading with minimal data."""
        # Create minimal dataset structure
        data_dir = tmp_path / "data" / "tomato" / "phase1" / "healthy"
        data_dir.mkdir(parents=True)
        
        # Create dummy image
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='blue')
        img.save(data_dir / "test.jpg")
        
        # Load dataset
        dataset = CropDataset(
            data_dir=str(tmp_path / "data"),
            crop="tomato",
            split="train",
            transform=False
        )
        
        assert len(dataset) >= 1
        assert "healthy" in dataset.classes

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])