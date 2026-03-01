# Complete Output Files Specification

This document details every file and artifact generated during AADS-ULoRA training, with special emphasis on OOD detection metadata required for deployment.

## Directory Structure

```
Google Drive/aads_ulora/
├── data/                              # Dataset organization
├── models/                            # Trained adapters + OOD components
├── model_checkpoints/                 # Training checkpoints for resumption
├── .checkpoints/                      # Pipeline progress tracking
├── logs/                              # Training metrics and history
└── outputs/                           # Evaluation results and reports
```

---

## 1. Data Directory: `data/`

### `dataset_metadata.json` (REQUIRED FOR INFERENCE)

Essential metadata about the dataset used for training.

```json
{
    "dataset_name": "PlantVillage",
    "crops": ["tomato", "potato", "wheat"],
    "total_classes": 38,
    "classes": {
        "tomato": ["early_blight", "late_blight", "leaf_mold", "..."],
        "potato": ["early_blight", "late_blight", "..."],
        "wheat": ["leaf_rust", "stem_rust", "stripe_rust", "..."]
    },
    "class_counts": {
        "tomato_early_blight": 1000,
        "tomato_late_blight": 925,
        "...": 0
    },
    "splits": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15
    },
    "augmentation": {
        "rotation": [0, 360],
        "brightness": [0.8, 1.2],
        "contrast": [0.8, 1.2],
        "blur": [0, 2]
    },
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "note": "ImageNet normalization"
    },
    "image_size": [224, 224],
    "total_samples": 70295,
    "note": "Required for preprocessing inference images identically"
}
```

**Used by**: Data pipeline, inference preprocessing, validation scripts
**Why important**: Ensures inference data is normalized and augmented consistently with training

---

## 2. Models Directory: `models/`

Contains trained adapters for phases 1-3, each with critical OOD detection components.

### Phase 1 Adapter: `phase1_dora_adapter/`

#### `adapter/` (Directory)
PEFT Low-Rank Adaptation weights:
- `adapter_config.json` - LoRA configuration (rank, alpha, layers)
- `pytorch_model.bin` - Single adapter weight file
- `adapter.state_dict` - PyTorch state dict

#### `classifier.pth` (REQUIRED FOR INFERENCE)
Head classifier network weights mapping features to class logits.
```python
# Shape: [feature_dim=768, num_classes=38]
torch.nn.Linear(768, 38)
```

#### `adapter_meta.json` (REQUIRED FOR INFERENCE)
Adapter metadata necessary for correct loading:
```json
{
    "is_trained": true,
    "current_phase": 1,
    "class_to_idx": {
        "background": 0,
        "tomato_early_blight": 1,
        "tomato_late_blight": 2,
        "...": 38
    },
    "classifier_input_size": 768,
    "classifier_output_size": 38,
    "note": "Maps between class names and indices"
}
```

**Used by**: 
- Inference pipeline for class name lookup
- Validation for consistency checks
- OOD detection for correct indexing

#### `ood_components.pt` ⭐ (CRITICAL FOR OOD DETECTION)

The **most critical file** for production deployment. Contains all OOD detection models:

```python
checkpoint = torch.load('ood_components.pt', map_location='cpu')

checkpoint = {
    # 1. CLASS PROTOTYPES (centroids in feature space)
    'prototypes': torch.Tensor,
    # Shape: [num_classes, feature_dim]
    # Example: [38, 768]
    # What it is: Mean of all training features per class
    # Used for: Prototype-based OOD detection (L2 distance to nearest prototype)
    
    # 2. MAHALANOBIS DISTANCE MODEL
    'mahalanobis': {
        'mean': torch.Tensor,
        # Shape: [feature_dim] - Overall mean of training feature distribution
        
        'covariance': torch.Tensor,
        # Shape: [feature_dim, feature_dim] - Feature covariance matrix
        # Captures correlation structure of features
        
        'inv_covariance': torch.Tensor,
        # Shape: [feature_dim, feature_dim]
        # Pre-computed inverse for efficiency: faster OOD detection
    },
    
    # 3. DYNAMIC OOD THRESHOLDS (per-class)
    'thresholds': {
        0: 25.3452,  # threshold for class 0
        1: 24.8901,  # threshold for class 1
        # ...
        37: 26.1234   # threshold for class 37
    },
    # Computed using: confidence intervals on validation set
    # Method: Statistical mean + k*std with confidence level
    
    # 4. PER-CLASS STATISTICS
    'class_std': {
        0: torch.Tensor([...]),  # Per-feature std for class 0
        1: torch.Tensor([...]),
        # ...
    }
}
```

**Usage in Inference**:
```python
# 1. Extract features from image
features = model.vision_outputs.pooler_output  # Shape: [1, 768]

# 2. Prototype-based OOD detection
distances_to_prototypes = torch.cdist(features, prototypes)  # [1, 38]
prototype_distance = distances_to_prototypes.min()
is_ood_prototype = prototype_distance > THRESHOLD

# 3. Mahalanobis-based OOD detection
centered = features - mean  # [1, 768]
mahalanobis_dist = torch.sqrt(
    torch.sum(centered @ inv_cov * centered, dim=1)
)
is_ood_mahalanobis = mahalanobis_dist > thresholds[predicted_class]

# 4. Ensemble decision
is_ood = is_ood_prototype or is_ood_mahalanobis
confidence = 1 - (prototype_distance / MAX_DISTANCE)
```

**Why critical**:
- Without prototypes: no prototype-based OOD detection
- Without mahalanobis: no statistical OOD detection
- Without thresholds: no way to determine OOD confidence level
- These three together enable robust OOD detection in production

#### `manifest.json` (METADATA)
Phase-level artifact metadata:
```json
{
    "phase": "phase1",
    "created_at": "2025-02-20T10:45:23.456789+00:00",
    "artifacts": {
        "adapter_dir": {
            "path": "/models/phase1_dora_adapter/adapter",
            "exists": true,
            "is_dir": true
        },
        "classifier": {
            "path": "/models/phase1_dora_adapter/classifier.pth",
            "exists": true,
            "is_dir": false
        }
    },
    "metadata": {
        "num_classes": 38,
        "strict_model_loading": false
    }
}
```

---

### Phase 2 & 3 Adapters
Same structure as Phase 1: `adapter/`, `classifier.pth`, `adapter_meta.json`, `ood_components.pt`, `manifest.json`

**Phase 3 (`phase3_conec_lora_adapter/`)** is the **final production-ready model** combining all refinements.

---

## 3. Model Checkpoints: `model_checkpoints/`

Training checkpoints for resuming interrupted training:

```
model_checkpoints/
├── phase1/
│   ├── checkpoint-epoch-0.pt
│   ├── checkpoint-epoch-1.pt
│   ├── checkpoint-epoch-2.pt
│   └── checkpoint-latest.pt      # Always points to most recent
├── phase2/
└── phase3/
```

### Checkpoint File Format
```python
checkpoint = torch.load('checkpoint-latest.pt')

checkpoint = {
    'epoch': 2,                     # Which epoch this is from
    'model_state_dict': {...},      # Model weights
    'classifier_state_dict': {...}, # Classifier weights
    'optimizer_state_dict': {...},  # Optimizer momentum buffers
    'scaler_state_dict': {...},     # Mixed precision scaler
    'scheduler_state_dict': {...},  # LR scheduler state
    'val_accuracy': 0.8234,         # Best validation accuracy so far
    'history': {                    # Full training history up to this point
        'train_loss': [...],
        'val_loss': [...],
        'train_accuracy': [...],
        'val_accuracy': [...]
    },
    'config': {
        'num_classes': 38,
        'hidden_size': 768,
        'lora_r': 8,
        'lora_alpha': 16
    }
}
```

**Used by**: Training resumption, best model selection
**Note**: Not needed for inference - only for training continuation

---

## 4. Pipeline Progress: `.checkpoints/`

Training pipeline progress tracking:

### `checkpoint_log.json` (REQUIRED FOR RESUME)

```json
{
    "setup": {
        "timestamp": "2025-02-20T10:15:23.123456",
        "completed": true,
        "details": {"gpu": "T4", "colab": true}
    },
    "data_prep": {
        "timestamp": "2025-02-20T10:16:45.654321",
        "completed": true,
        "details": {"dataset_path": "/content/drive/MyDrive/datasets/plantvillage"}
    },
    "phase1": {
        "timestamp": "2025-02-20T10:45:23.456789",
        "completed": true,
        "details": {
            "status": "Completed",
            "epochs": 3,
            "crops": ["tomato", "potato"],
            "final_accuracy": 0.8765
        }
    },
    "phase2": {
        "timestamp": "2025-02-20T11:30:15.111111",
        "completed": true,
        "details": {...}
    },
    "phase3": {
        "timestamp": "2025-02-20T12:20:10.222222",
        "completed": true,
        "details": {...}
    },
    "validation": {
        "timestamp": "2025-02-20T12:35:45.333333",
        "completed": true,
        "details": {...}
    },
    "monitoring": {
        "timestamp": "2025-02-20T12:50:30.444444",
        "completed": true,
        "details": {...}
    }
}
```

**Used by**: Resume mechanism, training pipeline progress
**Critical for**: Skipping completed phases and saving GPU compute time

---

## 5. Training Logs: `logs/`

### Phase-specific History Files

#### `phase1_history.json` / `phase2_history.json` / `phase3_history.json`

Complete training history per phase:

```json
{
    "train_loss": [5.234, 3.421, 2.187, 1.654],
    "val_loss": [5.123, 3.512, 2.301, 1.723],
    "train_accuracy": [0.0234, 0.1523, 0.3456, 0.5234],
    "val_accuracy": [0.0123, 0.1234, 0.3234, 0.5123],
    "ood_metrics": {
        "prototype_distance_mean": [4.234, 3.123, 2.456],
        "mahalanobis_score_mean": [25.3, 23.1, 21.5],
        "ood_rate": [0.15, 0.12, 0.08]
    },
    "learning_rates": [0.0001, 0.00008, 0.00006, 0.00004]
}
```

**Used by**: Training analysis, performance tracking, visualization

### `ood_metrics.json` ⭐ (IMPORTANT FOR OOD VALIDATION)

Complete OOD detection evaluation results:

```json
{
    "prototype_embeddings": [
        [0.123, 0.456, ...],  # Prototype embeddings
        [0.234, 0.567, ...],
        "... 36 more total"
    ],
    "ood_metrics_history": [
        {
            "prototype_distances": [2.3, 1.5, 3.2],
            "mahalanobis_scores": [23.4, 21.2, 25.1],
            "prototype_anomaly": [false, false, true],
            "mahalanobis_anomaly": [false, false, true],
            "batch_id": 0
        },
        "... more batches"
    ],
    "avg_prototype_distance": 2.156,
    "avg_mahalanobis_score": 23.45,
    "ood_detection_rate": 0.087,
    "thresholds_by_class": {
        "0": 25.3,
        "1": 24.8,
        "... 36 more": 0
    },
    "notes": "OOD detection evaluation on validation set"
}
```

**Used by**: 
- OOD detection validation
- Threshold tuning and analysis
- False positive rate analysis
- Production confidence calibration

### `training.log` (Optional)

Full text logs from training:
```
2025-02-20 10:15:23,456 - INFO - Starting Phase 1 training
2025-02-20 10:15:25,123 - INFO - GPU: NVIDIA T4 with 16GB memory
2025-02-20 10:45:23,456 - INFO - Phase 1 completed with accuracy: 0.8765
...
```

**Used by**: Debugging, performance analysis, execution timeline

---

## 6. Evaluation Outputs: `outputs/`

### `training_config.json` (REQUIRED FOR REPRODUCIBILITY)

Exact configuration used for this training run:

```json
{
    "crops": ["tomato", "potato"],
    "dataset_path": "/content/drive/MyDrive/datasets/plantvillage",
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 3,
    "phases": {
        "phase1": true,
        "phase2": true,
        "phase3": true
    },
    "device": "cuda",
    "mixed_precision": true,
    "validate": true,
    "output_directory": "/content/drive/MyDrive/aads_ulora"
}
```

**Used by**: 
- Reproducibility
- Experiment tracking
- Configuration comparison between runs
- Automated result annotation

### `validation_results.json` (REQUIRED FOR EVALUATION)

Comprehensive per-class validation metrics:

```json
{
    "overall": {
        "accuracy": 0.8765,
        "weighted_precision": 0.8723,
        "weighted_recall": 0.8765,
        "weighted_f1": 0.8744,
        "macro_f1": 0.8601,
        "macro_precision": 0.8523,
        "macro_recall": 0.8601
    },
    "per_class": {
        "background": {
            "accuracy": 0.9234,
            "precision": 0.9012,
            "recall": 0.9234,
            "f1": 0.9122,
            "support": 234
        },
        "tomato_early_blight": {
            "accuracy": 0.8523,
            "precision": 0.8345,
            "recall": 0.8512,
            "f1": 0.8428,
            "support": 189
        },
        "... 36 more classes": {}
    }
}
```

**Used by**: Model evaluation, class-wise analysis, problem identification

### `confusion_matrices.json`

Confusion matrices per phase for error analysis:

```json
{
    "phase1_confusion_matrix": [
        [234, 5, 0, ...],   # Row: true label, Col: predicted
        [3, 189, 2, ...],
        "... 36x36 matrix"
    ],
    "phase2_confusion_matrix": [...],
    "phase3_confusion_matrix": [...]
}
```

**Used by**: Error analysis, identifying confused classes, model debugging

### `performance_metrics.csv`

Summary metrics in tabular format for spreadsheet analysis:

```csv
phase,accuracy,precision,recall,f1,auc_roc,ood_rate,best_epoch
phase1,0.8234,0.8123,0.8234,0.8178,0.9123,0.087,2
phase2,0.8567,0.8456,0.8567,0.8511,0.9345,0.065,2
phase3,0.8765,0.8723,0.8765,0.8744,0.9512,0.043,3
```

**Used by**: Quick summaries, reporting, stakeholder presentations

### `performance_summary.json`

Structured performance summary across phases:

```json
{
    "training_date": "2025-02-20",
    "total_runtime_hours": 8.5,
    "phases_executed": ["phase1", "phase2", "phase3"],
    "crops_trained": ["tomato", "potato"],
    "devices_used": ["NVIDIA T4"],
    "final_metrics": {
        "phase1": {
            "best_accuracy": 0.8234,
            "final_accuracy": 0.8156
        },
        "phase2": {
            "best_accuracy": 0.8567,
            "final_accuracy": 0.8523
        },
        "phase3": {
            "best_accuracy": 0.8765,
            "final_accuracy": 0.8756
        }
    },
    "ood_detection": {
        "method": ["prototype", "mahalanobis"],
        "overall_ood_rate": 0.065,
        "false_positive_rate": 0.012,
        "false_negative_rate": 0.034
    },
    "test_results": {
        "test_accuracy": 0.8612,
        "test_f1": 0.8567
    }
}
```

**Used by**: Final reporting, model card generation, production readiness assessment

### `training_summary.html`

Interactive HTML report of all training results (browser viewable):
- Training curves (loss, accuracy)
- OOD detection performance
- Per-class metrics
- Confusion matrices visualization
- Phase comparison

**Used by**: Visual analysis, stakeholder review, documentation

---

## File Flow Diagram

```
Training Pipeline
├── Phase 1 generates:
│   ├── phase1_dora_adapter/ (model)
│   ├── phase1_history.json (metrics)
│   └── checkpoint-*.pt (resumption)
├── Phase 2 generates:
│   ├── phase2_sd_lora_adapter/ (model)
│   ├── phase2_history.json (metrics)
│   └── checkpoint-*.pt (resumption)
├── Phase 3 generates:
│   ├── phase3_conec_lora_adapter/ (FINAL model)
│   ├── phase3_history.json (metrics)
│   ├── ood_metrics.json (OOD evaluation)
│   └── checkpoint-*.pt (resumption)
├── Validation generates:
│   ├── validation_results.json (per-class metrics)
│   ├── confusion_matrices.json (error analysis)
│   └── checkpoint_log.json (progress update)
└── Monitoring generates:
    ├── performance_metrics.csv (summary)
    ├── performance_summary.json (structured summary)
    └── training_summary.html (visual report)

For Inference, use:
├── models/phase3_conec_lora_adapter/adapter/ (model weights)
├── models/phase3_conec_lora_adapter/classifier.pth (classification head)
├── models/phase3_conec_lora_adapter/ood_components.pt (OOD detection)
├── models/phase3_conec_lora_adapter/adapter_meta.json (class indices)
└── data/dataset_metadata.json (normalization)
```

---

## Files Essential for Inference

Minimum files required for production deployment:

1. ✅ `phase3_conec_lora_adapter/adapter/` - Main model weights
2. ✅ `phase3_conec_lora_adapter/classifier.pth` - Classification head
3. ✅ `phase3_conec_lora_adapter/ood_components.pt` - OOD detection ⭐⭐⭐
4. ✅ `phase3_conec_lora_adapter/adapter_meta.json` - Class mapping
5. ✅ `dataset_metadata.json` - Preprocessing normalization

Without `ood_components.pt`, the system **cannot detect out-of-distribution samples** and may make unreliable predictions on unseen data.

---

## Files for Reproducibility & Auditing

To fully reproduce results:

1. `training_config.json` - Exact hyperparameters
2. `phase*_history.json` - Training metrics per episode
3. `checkpoint_log.json` - Training progress timeline
4. `validation_results.json` - Evaluation metrics
5. `ood_metrics.json` - OOD detection validation

---

## Storage Estimate

```
Typical output sizes (per crop):
├── models/phase*_adapter/ ≈ 2 GB each (3 phases = 6 GB)
├── ood_components.pt ≈ 100 MB per phase (3 phases = 300 MB)
├── model_checkpoints/ ≈ 500 MB (contains ~3-4 checkpoints)
├── metadata files ≈ 50 MB (JSON, logs)
└── outputs/ ≈ 20 MB

TOTAL: ~7-8 GB per crop training run
```

---

## Version History

This specification documents AADS-ULoRA v5.5 outputs (February 2026).
OOD components are critical and should not be excluded in future versions.
