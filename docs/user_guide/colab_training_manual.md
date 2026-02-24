
# Colab Training Manual

## Introduction

This manual provides detailed instructions for training the AADS-ULoRA system on Google Colab. The training pipeline consists of three phases:

1. **Phase 1 (DoRA)**: Domain-specific adapter initialization
2. **Phase 2 (SD-LoRA)**: Stable Diffusion-based data augmentation
3. **Phase 3 (CoNeC-LoRA)**: Contrastive learning with OOD detection

## Prerequisites

- Google account with Google Colab access
- Google Drive with at least 50GB free space
- Basic knowledge of Python and deep learning

## Quick Start Guide

### Step 1: Choose Training Path

**Recommended (one-click):**

1. Open `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
2. Run all cells and follow the interactive configuration UI
3. Use this path for end-to-end training with minimal manual setup

**Manual/diagnostic path:**

1. Open `colab_bootstrap.ipynb` in Google Colab
2. Run all cells sequentially
3. Wait for installation to complete
4. Restart the runtime (Runtime → Restart runtime)

### Step 2: Data Preparation

1. Open `1_data_preparation.ipynb`
2. Mount Google Drive
3. Download and preprocess datasets
4. Verify data integrity

### Step 3: Training

Run notebooks in order:

#### Phase 1: DoRA Training
```bash
# Open 2_phase1_training.ipynb
# Review configuration
# Start training
# Expected: 2-4 hours
```

**Key parameters:**
- Model: `facebook/dinov3-giant` (automatically falls back to local stub if unavailable)
- LoRA rank: 32
- Batch size: 8 (adjust based on GPU)
- Learning rate: 1e-4
- Epochs: 10

**Note:** The trainer automatically handles model loading failures by falling back to a lightweight local stub model for testing. For production training, ensure models are accessible via HuggingFace authentication.

#### Phase 2: SD-LoRA Training
```bash
# Open 3_phase2_training.ipynb
# Ensure Phase 1 adapter is loaded
# Start training
# Expected: 1-3 hours
```

**Key parameters:**
- Model: `stabilityai/stable-diffusion-2-1`
- LoRA rank: 16
- Batch size: 4
- Learning rate: 1e-4
- Epochs: 5

#### Phase 3: CoNeC-LoRA Training
```bash
# Open 4_phase3_training.ipynb
# Ensure Phase 2 adapter is loaded
# Start training
# Expected: 2-4 hours
```

**Key parameters:**
- Model: `facebook/dinov3-giant` with Phase 1 adapters
- CoNeC temperature: 0.07
- Prototype dimension: 128
- Number of prototypes: 10
- Contrastive weight: 0.1
- Orthogonal weight: 0.01
- Batch size: 16
- Epochs: 10

**Resilience Features:**
- Automatic classifier dimension matching for test environments
- Dynamic feature extraction compatible with various model architectures
- Graceful degradation when full OOD dependencies unavailable

### Step 4: Testing and Validation

1. Open `5_testing_validation.ipynb`
2. Load all trained models
3. Evaluate on test set
4. Review OOD detection performance
5. Generate predictions

### Step 5: Performance Monitoring

1. Open `6_performance_monitoring.ipynb`
2. Load training logs
3. Analyze metrics
4. Generate reports
5. Review optimization insights

## Detailed Configuration

### Colab-Specific Settings

The `config/colab.json` file contains all Colab-specific configurations:

```json
{
  "colab": {
    "enabled": true,
    "gpu_type": "auto",
    "memory_optimization": {
      "gradient_checkpointing": true,
      "mixed_precision": true,
      "max_batch_size_gb4": 8,
      "max_batch_size_gb8": 16,
      "max_batch_size_gb16": 32
    },
    "training": {
      "gradient_accumulation_steps": 2,
      "use_amp": true,
      "num_workers": 2,
      "checkpoint_interval": 5,
      "early_stopping_patience": 10
    }
  }
}
```

### GPU-Specific Batch Sizes

The system automatically adjusts batch sizes based on available GPU memory:

| GPU Memory | Phase 1 | Phase 2 | Phase 3 |
|------------|---------|---------|---------|
| < 4GB      | 2       | 1       | 4       |
| 4-8GB      | 8       | 4       | 16      |
| 8-16GB     | 16      | 8       | 32      |
| 16-24GB    | 32      | 16      | 64      |
| 24GB+      | 64      | 32      | 128     |

## Dataset API

### ColabCropDataset

Wrapper around `torchvision.ImageFolder` with Colab-specific optimizations:

```python
from src.dataset.colab_datasets import ColabCropDataset
from torchvision import transforms

# Create dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ColabCropDataset(
    data_dir='./data/plantvillage/train',
    transform=transform
)
```

**Features:**
- Automatic error handling with fallback to dummy tensors
- Compatible with standard PyTorch DataLoader
- Returns tuple format `(image, label)`

### ColabDomainShiftDataset

Extension for Phase 3 training with domain labels:

```python
from src.dataset.colab_datasets import ColabDomainShiftDataset

dataset = ColabDomainShiftDataset(
    data_dir='./data/domain_a',
    transform=transform,
    domain_label=0
)
```

**Features:**
- Returns dict format `{'images': tensor, 'labels': int, 'domain': int}`
- Compatible with CoNeC-LoRA contrastive learning
- Automatic domain label injection

### ColabDataLoader

Optimized DataLoader with retry logic and adaptive settings:

```python
from src.dataset.colab_dataloader import ColabDataLoader

# Option 1: Simple kwargs interface (recommended for notebooks)
loader = ColabDataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2  # Automatically adjusted if multiprocessing fails
)

# Option 2: Config object interface
from src.dataset.colab_dataloader import DataLoaderConfig

config = DataLoaderConfig(
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2
)
loader = ColabDataLoader(dataset, config=config)
```

**Resilience Features:**
- Automatic retry with `num_workers=0` if multiprocessing fails
- Fallback to empty loader if all attempts fail (prevents crashes)
- Automatic CUDA pin_memory detection
- Adaptive worker count based on system resources

## Memory Optimization Techniques

### 1. Mixed Precision Training

Automatic Mixed Precision (AMP) is enabled by default:
- FP16 for forward/backward passes
- FP32 for master weights
- Gradient scaling to prevent underflow

Benefits:
- 2-3x speedup
- 50% memory reduction

### 2. Gradient Accumulation

Accumulate gradients over multiple batches:
```python
gradient_accumulation_steps = 2
effective_batch_size = batch_size * gradient_accumulation_steps
```

### 3. Gradient Checkpointing

Trade compute for memory:
- Checkpoint intermediate activations
- Recompute during backward pass
- Saves 50-70% memory

### 4. Memory Monitoring

Real-time memory tracking:
```python
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    cached = torch.cuda.memory_reserved() / (1024**3)
    print(f"GPU Memory: {allocated:.2f}GB (cached: {cached:.2f}GB)")
```

## Resilience and Error Handling

### Automatic Model Fallbacks

The training system includes automatic fallbacks for model loading failures:

```python
# If HuggingFace model unavailable, automatically uses local stub
trainer = ColabPhase1Trainer(
    model_name='facebook/dinov2-base',  # May fall back to local stub
    num_classes=10,
    device='cpu'
)
```

**Fallback behavior:**
1. Attempts to load from HuggingFace
2. On authentication error, falls back to lightweight local CNN stub
3. Logs warning with details
4. Continues training with stub (useful for testing)

### DataLoader Retry Logic

DataLoaders automatically retry with safer settings:

```python
# Primary attempt with requested settings
loader = ColabDataLoader(dataset, batch_size=16, num_workers=4)
# If fails → retries with num_workers=0
# If fails → returns empty loader to prevent crash
```

### PEFT Injection Resilience

LoRA adapter application includes graceful degradation:

```python
# If PEFT injection fails, continues with base model
# Logs warning: "Failed to apply PEFT model: ..."
# Training proceeds with full model parameters
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```json
{
  "training": {
    "phase1": { "batch_size": 4 },
    "phase2": { "batch_size": 2 },
    "phase3": { "batch_size": 8 }
  }
}
```

**Solution 2: Increase gradient accumulation**
```json
{
  "colab": {
    "training": {
      "gradient_accumulation_steps": 4
    }
  }
}
```

**Solution 3: Disable mixed precision**
```json
{
  "colab": {
    "memory_optimization": {
      "mixed_precision": false
    }
  }
}
```

### Slow Training

**Check GPU type:**
```python
!nvidia-smi
```

**Reduce image size:**
```json
{
  "data": {
    "image_size": 128  # Instead of 224
  }
}
```

**Reduce number of workers:**
```json
{
  "colab": {
    "training": {
      "num_workers": 1
    }
  }
}
```

### Dataset Issues

**Verify data paths:**
```python
import os
print(os.listdir('/content/drive/MyDrive/aads_ulora/data'))
```

**Check class balance:**
```python
from collections import Counter
class_counts = Counter([label for _, label in dataset])
print(class_counts)
```

## Advanced Usage

### Custom Hyperparameters

Edit `config/colab.json` to customize training:

```json
{
  "training": {
    "phase1": {
      "learning_rate": 5e-5,
      "num_epochs": 15,
      "lora_r": 64,
      "lora_alpha": 64
    }
  }
}
```

### Resume Training

To resume from checkpoint:
```python
# Load checkpoint (automatically restores current_epoch)
trainer.load_checkpoint('./checkpoints/phase1/checkpoint_epoch_5.pth', resume=True)

# Continue training from checkpoint epoch
history = trainer.train(train_loader, val_loader, num_epochs=10)

# Note: current_epoch is always restored from checkpoint,
# even if resume=False (for inspection purposes)
print(f"Checkpoint was from epoch: {trainer.current_epoch}")
```

**Phase 3 Checkpoint Handling:**
```python
# Phase 3 supports both file and directory paths
trainer.save_checkpoint('./checkpoints', epoch=5, loss=0.42)
# Saves to: ./checkpoints/checkpoint_epoch_5.pth

trainer.save_checkpoint('./checkpoints/my_checkpoint.pth', epoch=5, loss=0.42)
# Saves to: ./checkpoints/my_checkpoint.pth
```

### Export Models

Export for deployment:
```python
# Save to Drive
trainer.save_adapter('./models/final_adapter')

# Export to ONNX
torch.onnx.export(trainer.model, 'model.onnx')
```

## Performance Tips

### Maximizing GPU Utilization

1. **Use appropriate batch size**: Monitor GPU memory and adjust
2. **Enable prefetching**: `prefetch_factor=2` in DataLoader (automatically applied)
3. **Pin memory**: `pin_memory=True` for faster data transfer (auto-detected when CUDA available)
4. **Use mixed precision**: Already enabled by default
5. **Optimize worker count**: System automatically selects optimal `num_workers` based on CPU/memory
6. **Use notebook-friendly API**: Pass kwargs directly instead of config objects for cleaner code

### CPU-Only Training

For testing without GPU:
```python
trainer = ColabPhase1Trainer(
    model_name='facebook/dinov2-base',
    num_classes=10,
    device='cpu'  # Automatically disables pin_memory and mixed precision
)

loader = ColabDataLoader(
    dataset,
    batch_size=4,  # Smaller for CPU
    num_workers=0  # Required for CPU to avoid multiprocessing overhead
)
```

**Note:** CPU training is significantly slower but useful for:
- Debugging code logic
- Testing configurations
- Validating data pipelines

### Monitoring Training

1. **TensorBoard**:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/tensorboard')
```

2. **CSV logs**: Automatically saved to `./logs/`

3. **Colab output**: Real-time progress bars and metrics

## Inference

After training, use the pipeline directly in your code:

```python
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

# Load trained adapters and run inference
pipeline = IndependentMultiCropPipeline(...)
predictions = pipeline.predict(image)
```

See [Colab quick-start](../README.md) for example inference workflows.

## Cleaning Up

### Free Disk Space

```python
# Remove old checkpoints
!rm -rf ./checkpoints/old/

# Clear logs
!rm -rf ./logs/*.log

# Clear cache
!rm -rf ./cache/*
```

### Unmount Drive

```python
from google.colab import drive
drive.flush_and_unmount()
```

## Support

For issues:
1. Check troubleshooting section
2. Review logs in `./logs/`
3. Open an issue on GitHub

## Appendix

### File Structure

```
/content/drive/MyDrive/aads_ulora/
├── config/
│   └── colab.json
├── data/
│   └── plantvillage/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── phase1_dora_adapter/
│   ├── phase2_sd_lora_adapter/
│   └── phase3_conec_lora_adapter/
├── checkpoints/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── logs/
│   ├── phase1_history.json
│   ├── phase2_history.json
│   ├── phase3_history.json
│   └── colab_training.log
├── outputs/
│   ├── predictions.csv
│   ├── test_summary.json
│   └── model_comparison.csv
└── cache/
```

### Common Commands

```bash
# Check GPU
!nvidia-smi

# Check disk usage
!df -h

# Monitor memory
!watch -n 1 nvidia-smi

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Environment Variables

```python
import os
os.environ['AADS_WORKSPACE'] = '/content/drive/MyDrive/aads_ulora'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging
```

## Conclusion

This manual covers the complete training pipeline for AADS-ULoRA on Google Colab. For additional information, refer to:

- `colab_migration_guide.md` - Migration overview
- `cheatsheet_colab.md` - Quick reference
- API documentation in `docs/api/`