# AADS-ULoRA Colab Migration Guide

## Overview

This guide explains how to migrate the AADS-ULoRA training pipeline to Google Colab for cloud-based training with GPU acceleration.

## Prerequisites

- Google account with access to Google Colab
- Google Drive with at least 50GB free space
- Basic understanding of Python and deep learning

## Quick Start

### 1. Open One-Click Notebook (Recommended)

1. Open `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
2. Run all cells in order
3. Use interactive configuration in the first section

### 2. Manual Path (Optional)

1. Upload the `colab_bootstrap.ipynb` notebook to your Google Drive
2. Open it in Google Colab: Right-click → Open with → Google Colaboratory
3. Follow the instructions in the notebook

### 3. Run Installation

The bootstrap notebook will:
- Detect your GPU type and CUDA version
- Install all dependencies
- Set up workspace directory structure
- Create Colab-specific configuration

### 4. Restart Runtime

After installation completes, restart the runtime:
- Runtime → Restart runtime

### 5. Start Training

Run the training notebooks in order:
1. `1_data_preparation.ipynb` - Prepare and load datasets
2. `2_phase1_training.ipynb` - Train DoRA adapters
3. `3_phase2_training.ipynb` - Train SD-LoRA
4. `4_phase3_training.ipynb` - Train CoNeC-LoRA with OOD detection
5. `5_testing_validation.ipynb` - Test and validate models
6. `6_performance_monitoring.ipynb` - Monitor training metrics

## Configuration

### Colab-Specific Settings

The `config/colab.json` file contains Colab-optimized settings:

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

## Memory Optimization

### Mixed Precision Training

Colab notebooks use automatic mixed precision (AMP) to reduce memory usage:
- FP16 for forward/backward passes
- FP32 for master weights
- Gradient scaling to prevent underflow

### Gradient Accumulation

Gradient accumulation allows training with larger effective batch sizes:
```python
gradient_accumulation_steps = 2  # Accumulate gradients over 2 batches
```

### Memory Monitoring

The system includes comprehensive memory monitoring:
- GPU memory tracking
- Automatic cache clearing
- Memory usage logging
- Early warning for OOM conditions

## Data Management

### Google Drive Integration

All data is stored in Google Drive for persistence:
```
/content/drive/MyDrive/aads_ulora/
├── data/          # Datasets
├── models/        # Saved models
├── checkpoints/   # Training checkpoints
├── logs/          # Training logs
└── outputs/       # Results and metrics
```

### Downloading Datasets

Use the `download_data_colab.py` script or the data preparation notebook to download datasets:
```python
from scripts.download_data_colab import DriveDownloader, DownloadConfig

downloader = DriveDownloader(DownloadConfig(download_dir='./data'))
downloader.download_file(
    file_id='YOUR_FILE_ID',
    destination='dataset.zip',
    description='PlantVillage dataset'
)
```

## Training

### Phase 1: DoRA Initialization

Trains domain-specific adapters using DoRA (Difference of Rectified Activations):
- Model: `facebook/dinov3-giant` (with automatic fallback to local stub)
- LoRA rank: 32
- Batch size: 8 (adjustable based on GPU)
- Epochs: 10

**Resilience Features:**
- Automatic model fallback if HuggingFace unavailable
- Graceful PEFT injection degradation
- Checkpoint epoch tracking for resume

### Phase 2: SD-LoRA

Stable Diffusion-based data augmentation:
- Model: `stabilityai/stable-diffusion-2-1`
- LoRA rank: 16
- Batch size: 4
- Epochs: 5

### Phase 3: CoNeC-LoRA

Contrastive learning with prototype-based OOD detection:
- Model: `facebook/dinov3-giant` with Phase 1 adapters
- CoNeC configuration:
  - Temperature: 0.07
  - Prototype dimension: 128
  - Number of prototypes: 10
  - Contrastive weight: 0.1
  - Orthogonal weight: 0.01
- Batch size: 16
- Epochs: 10

## Monitoring and Logging

### TensorBoard

TensorBoard logs are saved to `./logs/tensorboard`:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/tensorboard')
```

### CSV Logs

Training metrics are saved as CSV files:
- `training_metrics.csv` - Loss and accuracy per epoch
- `memory_metrics.csv` - GPU memory usage
- `ood_metrics.csv` - OOD detection performance

### Colab Output

For Colab notebooks, metrics are also displayed in the notebook output with progress bars.

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce batch size**: Edit `config/colab.json`:
   ```json
   "training": {
     "phase1": { "batch_size": 4 },
     "phase2": { "batch_size": 2 },
     "phase3": { "batch_size": 8 }
   }
   ```

2. **Increase gradient accumulation**:
   ```json
   "colab": {
     "training": {
       "gradient_accumulation_steps": 4
     }
   }
   ```

3. **Disable mixed precision** (if causing issues):
   ```json
   "colab": {
     "memory_optimization": {
       "mixed_precision": false
     }
   }
   ```

### CUDA Out of Memory

Clear GPU cache between epochs:
```python
import torch
torch.cuda.empty_cache()
```

### Slow Training

1. **Check GPU type**: T4 is slower than A100
2. **Reduce image size**: Change `image_size` in config from 224 to 128
3. **Use fewer workers**: Set `num_workers` to 1 or 0

### Dataset Issues

If data loading is slow:
- Use `pin_memory=True` in DataLoader
- Increase `prefetch_factor`
- Cache preprocessed data

## Performance Tips

### Maximizing GPU Utilization

1. **Use appropriate batch size**: Monitor GPU memory usage and adjust
2. **Enable prefetching**: Set `prefetch_factor=2` in DataLoader
3. **Use mixed precision**: Already enabled by default
4. **Gradient checkpointing**: Enabled for memory efficiency

### Saving Checkpoints

Checkpoints are saved automatically:
- Every `checkpoint_interval` epochs
- Best model based on validation loss
- Full training state for resuming

To manually save:
```python
trainer.save_checkpoint('./checkpoints/phase3_best')
```

### Resuming Training

To resume from checkpoint:
```python
trainer.load_checkpoint('./checkpoints/phase3_best/checkpoint_epoch_5.pth')
```

## API Deployment

After training, you can deploy the API in Colab:

```python
# Start API server
# For inference, use the training pipeline's inference functions
```

Use ngrok to expose the API publicly:
```python
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")
```

## Cleaning Up

### Freeing Disk Space

```python
# Remove old checkpoints
!rm -rf ./checkpoints/old_experiments/

# Clear logs
!rm -rf ./logs/*.log

# Clear cache
!rm -rf ./cache/*
```

### Unmount Google Drive

```python
from google.colab import drive
drive.flush_and_unmount()
```

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `./logs/`
- Open an issue on GitHub

## Next Steps

1. Run the full training pipeline
2. Experiment with different hyperparameters
3. Add custom datasets
4. Deploy the trained model