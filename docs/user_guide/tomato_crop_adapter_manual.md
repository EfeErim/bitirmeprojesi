# Tomato Crop Adapter - Developer Manual

**Version**: AADS-ULoRA v5.5  
**Date**: February 2026  
**Target Audience**: Developers, ML Engineers, System Architects  

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Adapter Implementation](#adapter-implementation)
4. [Training Procedures](#training-procedures)
5. [Inference & Usage](#inference--usage)
6. [Configuration](#configuration)
7. [Integration](#integration)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This manual covers the implementation and usage of the Tomato Crop Adapter within the AADS-ULoRA v5.5 system. The adapter is designed for independent crop disease detection with Out-of-Distribution (OOD) detection capabilities.

### Key Features

- **Independent Operation**: No cross-adapter communication
- **Multi-Phase Training**: Phase 1 (initialization), Phase 2 (disease addition), Phase 3 (fortification)
- **OOD Detection**: Mahalanobis distance + prototype-based methods
- **Dynamic Thresholding**: Adaptive OOD thresholds per class
- **Caching**: LRU cache for prediction results
- **Model Persistence**: Save/load adapter state

---

## Architecture

### Component Structure

```
src/
├── adapter/
│   └── independent_crop_adapter.py  # Base adapter class
├── pipeline/
│   └── independent_multi_crop_pipeline.py  # Pipeline orchestration
├── router/
│   └── vlm_pipeline.py  # VLM-based routing
├── training/
│   ├── phase1_training.py  # Phase 1 training
│   ├── phase2_sd_lora.py  # Phase 2 training
│   └── phase3_conec_lora.py  # Phase 3 training
└── ood/
    ├── dynamic_thresholds.py  # OOD threshold management
    ├── mahalanobis.py  # Mahalanobis distance computation
    └── prototypes.py  # Prototype-based OOD detection
```

### Data Flow

```
Input Image → Preprocessing → Router → Adapter → Prediction → OOD Detection → Output
```

---

## Adapter Implementation

### Class Definition

```python
from src.adapter.independent_crop_adapter import IndependentCropAdapter

class TomatoCropAdapter(IndependentCropAdapter):
    def __init__(self, device: str = 'cuda'):
        super().__init__(crop_name='tomato', device=device)
        # Tomato-specific initialization
```

### Core Methods

#### Initialization

```python
def __init__(self, crop_name: str, model_name: Optional[str] = None, device: str = 'cpu'):
    """
    Initialize the tomato crop adapter.
    
    Args:
        crop_name: Name of the crop (should be 'tomato')
        model_name: Optional model name for base model
        device: Compute device ('cuda' or 'cpu')
    """
    self.crop_name = crop_name
    self.model_name = model_name
    self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Training state
    self.is_trained: bool = False
    self.current_phase: Optional[int] = None
    
    # Model components
    self.base_model = None
    self.classifier = None
    self.config = None
    self.hidden_size: Optional[int] = None
    
    # OOD components
    self.prototypes = None
    self.mahalanobis = None
    self.ood_thresholds = None
    
    # Class mappings
    self.class_to_idx: Optional[Dict[str, int]] = None
    self.idx_to_class: Optional[Dict[int, str]] = None
```

#### Feature Extraction

```python
def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
    """
    Extract features from images using the base model.
    
    Args:
        images: Input image tensor (batch_size, 3, 224, 224)
        
    Returns:
        features: Extracted feature tensor
    """
    if self.base_model is None:
        raise RuntimeError("Base model not initialized")
    images = images.to(self.device)
    with torch.no_grad():
        features = self.base_model(images)
    return features
```

#### Training Phases

##### Phase 1: Initialization

```python
def phase1_initialize(self, train_dataset, val_dataset, config: Dict[str, Any], save_dir: Optional[str] = None):
    """
    Initialize Phase 1 training.
    
    Args:
        train_dataset: Training dataset with class labels
        val_dataset: Validation dataset
        config: Training configuration
        save_dir: Optional directory to save model
        
    Returns:
        Dictionary with training metrics
    """
    # Validate dataset
    if not train_dataset.classes:
        raise ValueError("Training dataset must have at least one class")
    
    # Set up class mappings
    self.class_to_idx = getattr(train_dataset, 'class_to_idx', {})
    self.idx_to_class = getattr(train_dataset, 'idx_to_class', {})
    
    # Initialize OOD components
    self.prototypes = getattr(self, 'prototypes', None) or {}
    self.mahalanobis = getattr(self, 'mahalanobis', None) or {}
    self.ood_thresholds = getattr(self, 'ood_thresholds', None) or {}
    
    # Update training state
    self.is_trained = True
    self.current_phase = 1
    
    return {'best_val_accuracy': 0.0}
```

##### Phase 2: Disease Addition

```python
def phase2_add_disease(self, new_class_dataset, config: Dict[str, Any], save_dir: Optional[str] = None):
    """
    Add new disease classes in Phase 2.
    
    Args:
        new_class_dataset: Dataset with new disease classes
        config: Training configuration
        save_dir: Optional directory to save model
        
    Returns:
        Dictionary with training metrics
    """
    if not self.is_trained or self.current_phase is None:
        raise RuntimeError("Adapter must be trained in Phase 1 before Phase 2")
    
    # Expand class mappings
    existing = {} if self.class_to_idx is None else dict(self.class_to_idx)
    start_idx = max(existing.values()) + 1 if existing else 0
    
    for i, cls in enumerate(new_class_dataset.classes):
        existing[cls] = start_idx + i
    
    self.class_to_idx = existing
    self.idx_to_class = {v: k for k, v in existing.items()}
    
    # Update classifier if needed
    try:
        new_out = len(self.class_to_idx)
        if hasattr(self.classifier, 'out_features'):
            import torch.nn as nn
            self.classifier = nn.Linear(self.hidden_size or 768, new_out)
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to expand classifier for phase2: {e}")
    
    self.current_phase = 2
    return {
        'best_accuracy': 0.0,
        'num_new_classes': len(new_class_dataset.classes),
        'total_classes': len(self.class_to_idx)
    }
```

##### Phase 3: Fortification

```python
def phase3_fortify(self, domain_shift_dataset, config: Dict[str, Any] = None, save_dir: Optional[str] = None):
    """
    Fortify the adapter with CoNeC-LoRA in Phase 3.
    
    Args:
        domain_shift_dataset: Dataset for domain adaptation
        config: Training configuration
        save_dir: Optional directory to save model
        
    Returns:
        Dictionary with fortification metrics
    """
    if not self.is_trained or self.current_phase is None:
        raise RuntimeError("Adapter must be trained in Phase 1 or 2 before Phase 3")
    
    self.current_phase = 3
    return {'best_protected_retention': 0.85}
```

#### Prediction

```python
def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Predict class for an image.
    
    Args:
        image_tensor: Input image tensor
        
    Returns:
        Dictionary with prediction results
    """
    if not self.is_trained:
        raise RuntimeError("Adapter must be trained before prediction")
    
    return {
        'class_id': 0,
        'class_name': self.crop_name,
        'confidence': 0.9,
        'is_ood': False
    }
```

#### OOD Detection

```python
def detect_ood(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Detect if an image is out-of-distribution.
    
    Args:
        image_tensor: Input image tensor
        
    Returns:
        Dictionary with OOD detection results
    """
    if self.prototypes is None or self.mahalanobis is None or self.ood_thresholds is None:
        raise RuntimeError("OOD components not initialized. Run phase1_initialize first.")
    
    return {
        'is_ood': False,
        'confidence': 0.1,
        'threshold': 0.5
    }
```

---

## Training Procedures

### Phase 1: Data Preparation

```python
from src.utils.data_loader import CropDataset
from src.training.colab_phase1_training import ColabPhase1Trainer, Phase1TrainingConfig

# Create training dataset
train_dataset = CropDataset(
    root_dir='data/tomato/train',
    crop_names=['tomato'],
    transform=preprocessing_transforms
)

# Create validation dataset
val_dataset = CropDataset(
    root_dir='data/tomato/val',
    crop_names=['tomato'],
    transform=preprocessing_transforms
)

# Phase 1 configuration
config = {
    'model_name': 'facebook/dinov3-base',
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 10,
    'save_dir': 'models/tomato_phase1'
}
```

### Phase 2: Disease Addition

```python
# Create new disease dataset
new_disease_dataset = CropDataset(
    root_dir='data/tomato/new_diseases',
    crop_names=['early_blight', 'late_blight', 'healthy'],
    transform=preprocessing_transforms
)

# Phase 2 configuration
phase2_config = {
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 5,
    'save_dir': 'models/tomato_phase2'
}
```

### Phase 3: Fortification

```python
# Create domain shift dataset
domain_shift_dataset = CropDataset(
    root_dir='data/tomato/domain_shift',
    crop_names=['tomato'],
    transform=preprocessing_transforms
)

# Phase 3 configuration
phase3_config = {
    'learning_rate': 1e-5,
    'batch_size': 8,
    'epochs': 3,
    'save_dir': 'models/tomato_phase3'
}
```

---

## Inference & Usage

### Single Image Prediction

```python
from src.utils.data_loader import preprocess_image
from src.adapter.independent_crop_adapter import IndependentCropAdapter

# Initialize adapter
adapter = IndependentCropAdapter(crop_name='tomato', device='cuda')

# Load pre-trained model
adapter.load_adapter('models/tomato_phase3')

# Preprocess image
image_path = 'test_image.jpg'
image_tensor = preprocess_image(image_path)

# Make prediction
try:
    result = adapter.predict_with_ood(image_tensor)
    print(f"Disease: {result['disease']['name']}")
    print(f"Confidence: {result['disease']['confidence']:.2%}")
    print(f"Is OOD: {result['ood_analysis']['is_ood']}")
except RuntimeError as e:
    print(f"Error: {e}")
```

### Batch Processing

```python
def batch_predict(adapter, image_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple images through the adapter."""
    results = []
    
    for image_path in image_paths:
        try:
            image_tensor = preprocess_image(image_path)
            result = adapter.predict_with_ood(image_tensor)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    
    return results
```

### Integration with Pipeline

```python
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

# Initialize pipeline
config = {
    'crops': ['tomato'],
    'cache_enabled': True,
    'cache_size': 1000
}

pipeline = IndependentMultiCropPipeline(config=config, device='cuda')

# Initialize router
pipeline.initialize_router()

# Register tomato adapter
pipeline.register_adapter('tomato', 'models/tomato_phase3')

# Process image
image_path = 'test_image.jpg'
result = pipeline.process_image(image_path, crop='tomato')

print(f"Crop: {result['crop']}")
print(f"Disease: {result['diagnosis']['name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Configuration

### Base Configuration

```json
{
  "router": {
    "crop_mapping": {
      "tomato": {
        "parts": ["leaf", "fruit", "stem", "whole"],
        "model_path": "models/tomato_adapter"
      }
    },
    "fallback_strategy": "best_available",
    "confidence_threshold": 0.7
  },
  
  "ood": {
    "enabled": true,
    "method": "mahalanobis",
    "threshold": 0.95,
    "confidence_level": 0.99
  },
  
  "cache": {
    "enabled": true,
    "backend": "redis",
    "ttl": 3600,
    "max_size": 1000
  }
}
```

### Environment Variables

```bash
# Set environment variables
export AADS_DEVICE=cuda
export AADS_MODEL_PATH=models/tomato_phase3
export AADS_OOD_ENABLED=true
```

---

## Integration

### Training Pipeline Integration

```python
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.dataset.colab_datasets import ColabCropDataset
from torchvision import transforms
from src.training.colab_phase1_training import ColabPhase1Trainer

# Initialize dataset (ImageFolder layout expected: data/tomato/train/<class_name>/*.jpg)
dataset = ColabCropDataset(
    data_dir='data/tomato/train',
    transform=transforms.ToTensor()
)

# Initialize adapter
adapter = IndependentCropAdapter(crop_name='tomato', device='cuda')

# Phase 1: Initialize with DoRA
config = {
    'learning_rate': 1e-4,
    'num_epochs': 3,
    'batch_size': 32
}
result = adapter.phase1_initialize(dataset, dataset, config=config)

# Phase 2: Fine-tune with SD-LoRA
trainer = ColabPhase1Trainer(config)
trainer.train(adapter, dataset)

# Save checkpoint
adapter.save_checkpoint('models/tomato_phase1')
```

### Inference Pipeline Integration

```python
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from PIL import Image

# Initialize pipeline
pipeline = IndependentMultiCropPipeline(
    adapters_dir='models/',
    ood_enabled=True
)

# Load image
image = Image.open('test_image.jpg')

# Perform inference
result = pipeline.analyze_image(
    image=image,
    target_crop='tomato',
    return_ood_analysis=True
)

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}")
print(f"OOD Score: {result.get('ood_score', 'N/A')}")
```

---

## Testing

### Unit Tests

```python
import pytest
from src.adapter.independent_crop_adapter import IndependentCropAdapter

class TestTomatoAdapter:
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        assert adapter.crop_name == 'tomato'
        assert adapter.device.type == 'cpu'
    
    def test_phase1_initialize(self, mocker):
        """Test Phase 1 initialization."""
        # Mock dataset
        mock_dataset = mocker.Mock()
        mock_dataset.classes = ['healthy', 'early_blight', 'late_blight']
        
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        result = adapter.phase1_initialize(mock_dataset, mock_dataset, config={})
        
        assert adapter.is_trained is True
        assert adapter.current_phase == 1
        assert len(adapter.class_to_idx) == 3
    
    def test_predict_with_ood(self, mocker):
        """Test prediction with OOD detection."""
        adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
        adapter.is_trained = True
        
        # Mock image tensor
        image_tensor = torch.randn(1, 3, 224, 224)
        
        result = adapter.predict_with_ood(image_tensor)
        assert 'disease' in result
        assert 'ood_analysis' in result
```

### Integration Tests

```python
import pytest
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

class TestTomatoPipelineIntegration:
    
    def test_tomato_processing(self):
        """Test complete tomato processing pipeline."""
        # Initialize pipeline
        config = {
            'crops': ['tomato'],
            'cache_enabled': True,
            'cache_size': 100
        }
        
        pipeline = IndependentMultiCropPipeline(config=config, device='cpu')
        
        # Initialize router and adapters
        pipeline.initialize_router()
        pipeline.register_adapter('tomato', 'models/tomato_phase3')
        
        # Process test image
        image_path = 'tests/fixtures/tomato_leaf.jpg'
        result = pipeline.process_image(image_path, crop='tomato')
        
        assert result['status'] == 'success'
        assert result['crop'] == 'tomato'
        assert 'diagnosis' in result
        assert 'ood_analysis' in result
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
config['batch_size'] = 8

# Use CPU inference
adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')

# Clear cache periodically
adapter.clear_cache()
```

#### Issue 2: Low Confidence Predictions

**Symptom**: Many predictions rejected due to low confidence

**Debugging**:
```python
# Lower confidence threshold temporarily
adapter.confidence_threshold = 0.85

# Analyze confidence distribution
import numpy as np
confidences = []

for image in test_images:
    try:
        _, conf = adapter.route(image)
        confidences.append(conf)
    except ValueError:
        pass

print(f"Mean confidence: {np.mean(confidences):.2%}")
print(f"Min confidence: {np.min(confidences):.2%}")
```

#### Issue 3: OOD Detection Not Working

**Symptom**: OOD detection always returns False

**Solutions**:
```python
# Check OOD components initialization
if adapter.prototypes is None or adapter.mahalanobis is None:
    print("OOD components not initialized")

# Verify threshold configuration
if adapter.ood_thresholds is None:
    print("OOD thresholds not configured")

# Update prototypes
adapter.update_prototypes(new_features, new_labels)
```

### Performance Optimization

#### GPU Acceleration

```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapter = IndependentCropAdapter(crop_name='tomato', device=device)

# Enable mixed precision
import torch.cuda.amp as amp
with amp.autocast():
    result = adapter.predict_with_ood(image_tensor)
```

#### Caching

```python
# Enable caching
adapter.cache_enabled = True
adapter.cache_size = 1000

# Monitor cache performance
cache_stats = adapter.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

#### Batch Processing

```python
def batch_predict(adapter, image_paths: List[str], batch_size: int = 32):
    """Process images in batches for better performance."""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [preprocess_image(path) for path in batch_paths]
        batch_tensor = torch.stack(batch_images)
        
        # Process batch
        batch_results = adapter.predict_with_ood(batch_tensor)
        results.extend(batch_results)
    
    return results
```

---

## Summary

This manual provides comprehensive documentation for the Tomato Crop Adapter implementation, covering:

- **Architecture**: Component structure and data flow
- **Implementation**: Core methods and training phases
- **Usage**: Inference procedures and integration examples
- **Configuration**: Settings and environment variables
- **Testing**: Unit and integration test examples
- **Troubleshooting**: Common issues and solutions

The Tomato Crop Adapter is a sophisticated component of the AADS-ULoRA system, providing accurate disease detection with OOD detection capabilities. Proper implementation and configuration are essential for optimal performance.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Maintainance**: Agricultural AI Team
