"""
Test fixtures for AADS-ULoRA unit tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock data for router tests
@pytest.fixture
def mock_router_data():
    """Mock data for SimpleCropRouter tests."""
    crops = ['tomato', 'pepper', 'corn']
    
    # Mock image tensor (batch size 1)
    mock_image = torch.randn(1, 3, 224, 224)
    
    # Mock model outputs
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = torch.randn(1, 1, 768)
    
    # Mock probabilities
    mock_probabilities = torch.tensor([[0.95, 0.03, 0.02]])
    
    return {
        'crops': crops,
        'mock_image': mock_image,
        'mock_outputs': mock_outputs,
        'mock_probabilities': mock_probabilities,
        'confidence_threshold': 0.92,
        'top_k_alternatives': 3
    }

# Mock data for pipeline tests
@pytest.fixture
def mock_pipeline_data():
    """Mock data for IndependentMultiCropPipeline tests."""
    # Mock image tensors
    mock_image = torch.randn(1, 3, 224, 224)
    mock_image_batch = torch.stack([mock_image, mock_image])
    
    # Mock metadata
    mock_metadata = {
        'capture_timestamp': '2026-02-12T23:00:00Z',
        'device_model': 'Pixel 7',
        'os_version': 'Android 14'
    }
    
    # Mock adapter
    mock_adapter = MagicMock()
    mock_adapter.predict_with_ood.return_value = {
        'status': 'success',
        'disease': {
            'class_index': 0,
            'name': 'healthy',
            'confidence': 0.98
        },
        'ood_analysis': {
            'is_ood': False,
            'ood_score': 0.1,
            'threshold': 25.0,
            'dynamic_threshold_applied': True
        }
    }
    
    return {
        'mock_image': mock_image,
        'mock_image_batch': mock_image_batch,
        'mock_metadata': mock_metadata,
        'mock_adapter': mock_adapter,
        'mock_adapters': {
            'tomato': mock_adapter,
            'pepper': mock_adapter,
            'corn': mock_adapter
        }
    }

# Mock data for OOD threshold tests
@pytest.fixture
def mock_ood_data():
    """Mock data for DynamicOODThreshold tests."""
    # Mock Mahalanobis distance calculator
    mock_mahalanobis = MagicMock()
    mock_mahalanobis.num_classes = 3
    mock_mahalanobis.compute_distance.return_value = torch.tensor(15.0)
    
    # Mock validation data
    mock_val_data = torch.randn(100, 3, 224, 224)
    mock_val_labels = torch.randint(0, 3, (100,))
    
    # Mock validation loader
    mock_val_loader = MagicMock()
    mock_val_loader.__iter__.return_value = zip(
        [mock_val_data[i:i+16] for i in range(0, 100, 16)],
        [mock_val_labels[i:i+16] for i in range(0, 100, 16)]
    )
    mock_val_loader.__len__.return_value = 7
    
    # Mock model
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.return_value = type('Output', (), {
        'last_hidden_state': torch.randn(16, 1, 768)
    })()
    
    # Mock distances per class
    mock_distances = {
        0: [15.0, 14.5, 16.0, 15.2, 14.8, 15.5, 15.1, 14.9, 15.3, 15.4],
        1: [20.0, 19.5, 21.0, 20.2, 19.8, 20.5, 20.1, 19.9, 20.3, 20.4],
        2: [25.0, 24.5, 26.0, 25.2, 24.8, 25.5, 25.1, 24.9, 25.3, 25.4]
    }
    
    return {
        'mock_mahalanobis': mock_mahalanobis,
        'mock_val_loader': mock_val_loader,
        'mock_model': mock_model,
        'mock_distances': mock_distances,
        'feature_dim': 768,
        'device': 'cpu'
    }

# Mock data for adapter tests
@pytest.fixture
def mock_adapter_data():
    """Mock data for IndependentCropAdapter tests."""
    # Mock datasets
    class MockCropDataset:
        def __init__(self, classes):
            self.classes = classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
            
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), torch.randint(0, len(self.classes), (1,)).item()
    
    mock_train_classes = ['healthy', 'disease1', 'disease2']
    mock_train_dataset = MockCropDataset(mock_train_classes)
    mock_val_dataset = MockCropDataset(mock_train_classes)
    
    # Mock new class dataset
    mock_new_classes = ['disease3', 'disease4']
    mock_new_class_dataset = MockCropDataset(mock_new_classes)
    
    # Mock domain shift dataset
    mock_domain_shift_dataset = MockCropDataset(mock_train_classes)
    
    # Mock config
    mock_config = {
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'loraplus_lr_ratio': 16,
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'early_stopping_patience': 10
    }
    
    # Mock image
    mock_image = torch.randn(1, 3, 224, 224)
    
    return {
        'mock_train_dataset': mock_train_dataset,
        'mock_val_dataset': mock_val_dataset,
        'mock_new_class_dataset': mock_new_class_dataset,
        'mock_domain_shift_dataset': mock_domain_shift_dataset,
        'mock_config': mock_config,
        'mock_image': mock_image,
        'mock_classes': mock_train_classes + mock_new_classes
    }

# Mock data for validation tests
@pytest.fixture
def mock_validation_data():
    """Mock data for validation tests."""
    # Mock base64 image
    mock_image_data = torch.randn(3, 224, 224).numpy().tobytes()
    mock_base64_image = base64.b64encode(mock_image_data).decode('utf-8')
    
    # Mock file bytes
    mock_file_bytes = mock_image_data
    
    # Mock UUID
    mock_uuid = '123e4567-e89b-12d3-a456-426614174000'
    
    # Mock location
    mock_location = {
        'latitude': 37.7749,
        'longitude': -122.4194,
        'accuracy_meters': 10.5
    }
    
    # Mock metadata
    mock_metadata = {
        'capture_timestamp': '2026-02-12T23:00:00Z',
        'device_model': 'Pixel 7',
        'os_version': 'Android 14'
    }
    
    # Mock batch images
    mock_batch_images = [mock_base64_image] * 5
    
    return {
        'mock_base64_image': mock_base64_image,
        'mock_file_bytes': mock_file_bytes,
        'mock_uuid': mock_uuid,
        'mock_location': mock_location,
        'mock_metadata': mock_metadata,
        'mock_batch_images': mock_batch_images
    }

# Helper functions for creating mock tensors
@pytest.fixture
def mock_tensor_factory():
    """Factory for creating mock tensors."""
    def create_mock_tensor(batch_size=1, channels=3, height=224, width=224):
        return torch.randn(batch_size, channels, height, width)
    
    return create_mock_tensor

# Helper functions for creating mock datasets
@pytest.fixture
def mock_dataset_factory():
    """Factory for creating mock datasets."""
    class MockDataset:
        def __init__(self, classes, size=100):
            self.classes = classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return torch.randn(3, 224, 224), torch.randint(0, len(self.classes), (1,)).item()
    
    def create_mock_dataset(classes, size=100):
        return MockDataset(classes, size)
    
    return create_mock_dataset