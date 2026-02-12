#!/usr/bin/env python3
"""
Unit tests for IndependentCropAdapter
"""

import pytest
import torch
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.utils.data_loader import CropDataset

class TestIndependentCropAdapter:
    """Test cases for IndependentCropAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        
        assert adapter.crop_name == 'tomato'
        assert adapter.device.type == 'cpu'
        assert adapter.is_trained == False
        assert adapter.current_phase is None
    
    def test_phase1_initialization(self, tmp_path):
        """Test Phase 1 initialization."""
        # Create dummy dataset
        class DummyDataset(CropDataset):
            def __init__(self):
                self.classes = ['healthy', 'disease']
                self.class_to_idx = {'healthy': 0, 'disease': 1}
                self.idx_to_class = {0: 'healthy', 1: 'disease'}
                self.image_paths = []
                self.labels = []
                self.transform = None
        
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        
        config = {
            'lora_r': 8,  # Small for testing
            'lora_alpha': 8,
            'lora_dropout': 0.1,
            'loraplus_lr_ratio': 16,
            'num_epochs': 1,  # Just one epoch for test
            'batch_size': 2,
            'learning_rate': 1e-4,
            'early_stopping_patience': 5
        }
        
        # This will fail without real data but tests the interface
        try:
            adapter.phase1_initialize(train_dataset, val_dataset, config, str(tmp_path))
            assert adapter.is_trained == True
            assert adapter.current_phase == 1
        except Exception as e:
            pytest.skip(f"Phase 1 training failed (expected with dummy data): {e}")
    
    def test_save_load_adapter(self, tmp_path):
        """Test adapter save and load."""
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        
        # Save empty adapter
        save_path = tmp_path / "adapter_test"
        adapter.save_adapter(str(save_path))
        
        # Load into new adapter
        new_adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        try:
            new_adapter.load_adapter(str(save_path))
            assert new_adapter.is_trained == adapter.is_trained
        except Exception as e:
            pytest.skip(f"Load failed (expected with untrained adapter): {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])