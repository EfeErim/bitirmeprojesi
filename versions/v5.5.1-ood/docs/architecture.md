# AADS-ULoRA v5.5 Architecture Documentation

## System Overview

AADS-ULoRA v5.5 implements a production-ready multi-crop disease detection system using independent crop adapters with dynamic OOD detection. The system is designed for deployment in agricultural applications, specifically for integration with Uyumsoft ZiraiTakip.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L1: Crop Router                          │
│  (Simple classifier: DINOv2-base)                          │
│  Function: Route image → correct crop adapter              │
│  Target Accuracy: ≥98%                                     │
└─────────────────────────────────────────────────────────────┘
                         ↓ (crop type)
         ┌─────────────────┴─────────────────┐
         ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│  L2: Tomato      │              │  L2: Pepper      │
│  Adapter         │              │  Adapter         │
│  (Independent)   │              │  (Independent)   │
│                  │              │                  │
│  Phases:         │              │  Phases:         │
│  • DoRA (Base)   │              │  • DoRA (Base)   │
│  • SD-LoRA (CIL) │              │  • SD-LoRA (CIL) │
│  • CONEC (DIL)   │              │  • CONEC (DIL)   │
│                  │              │                  │
│  OOD: Dynamic    │              │  OOD: Dynamic    │
│  Mahalanobis     │              │  Mahalanobis     │
└──────────────────┘              └──────────────────┘
```

**Key Design Principle:** Zero cross-adapter communication. Each crop maintains an independent lifecycle, allowing asynchronous updates and zero interference between crops.

## Component Details

### 1. Crop Router (`src/router/simple_crop_router.py`)

**Purpose:** Classify input images by crop type (tomato, pepper, corn, etc.)

**Architecture:**
- Backbone: DINOv2-base (pretrained, frozen)
- Head: Single linear layer (768 → num_crops)
- Training: Linear probe only (backbone frozen)

**Training:**
- Dataset: Images from all supported crops
- Loss: Cross-entropy
- Optimizer: AdamW (lr=1e-3)
- Target: ≥98% accuracy

**Inference:**
- Input: 224×224 RGB image
- Output: (crop_type, confidence)
- Latency: <50ms

### 2. Independent Crop Adapter (`src/adapter/independent_crop_adapter.py`)

**Purpose:** Disease classification for a specific crop with OOD detection

**Three-Phase Training:**

#### Phase 1: DoRA Base Initialization
- **Method:** DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Goal:** Train base adapter on initial disease classes
- **Configuration:**
  - LoRA rank: r=32, alpha=32
  - Target modules: query, value
  - Optimizer: LoRA+ (B matrix LR = 16× A matrix LR)
  - Epochs: 50
- **Output:**
  - Trained adapter with DoRA weights
  - Classifier head
  - Class prototypes (mean features)
  - Class standard deviations
- **Target:** ≥95% clean accuracy

#### Phase 2: SD-LoRA Class-Incremental Learning
- **Method:** SD-LoRA (Selective Directional LoRA)
- **Goal:** Add new disease classes without catastrophic forgetting
- **Mechanism:**
  - Freeze LoRA A/B matrices (directions)
  - Train only LoRA magnitudes and classifier
  - Expand classifier to accommodate new classes
- **Configuration:**
  - LoRA rank: r=32
  - Learning rate: 5e-5 (lower than Phase 1)
  - Epochs: 20
- **Output:**
  - Updated adapter with new classes
  - Updated OOD statistics
- **Target:** ≥90% retention on old classes

#### Phase 3: CONEC-LoRA Domain-Incremental Learning
- **Method:** CONEC-LoRA (Continual Learning with Early-block Consolidation)
- **Goal:** Fortify against domain shifts (different lighting, camera angles, etc.)
- **Mechanism:**
  - Freeze early transformer blocks (0-5) - shared features
  - Add new LoRA to late blocks (6-11) - task-specific adaptation
  - Train on domain-shifted data
- **Configuration:**
  - LoRA rank: r=16 (smaller for task-specific)
  - Frozen blocks: 6
  - Epochs: 15
- **Output:**
  - Fortified adapter
  - Updated OOD thresholds
- **Target:** ≥85% protected retention

### 3. OOD Detection (`src/ood/`)

**Three-Component System:**

#### Prototypes (`prototypes.py`)
- Compute class mean feature vectors from training data
- Store in tensor: (num_classes, feature_dim)
- Updated when new classes are added

#### Mahalanobis Distance (`mahalanobis.py`)
- Compute distance: (x - μ)^T * Σ⁻¹ * (x - μ)
- Σ approximated as diagonal covariance (std²)
- Regularization: Σ += 1e-4 * I for invertibility
- Efficient per-class distance computation

#### Dynamic Thresholds (`dynamic_thresholds.py`)
- Per-class threshold: threshold_c = mean_dist_c + k * std_dist_c
- k=2.0 for 95% confidence (2-sigma)
- Minimum 10 validation samples required per class
- Fallback: 25.0 if insufficient data
- Adaptive threshold manager for online updates

**OOD Decision:**
- Compute Mahalanobis distance to predicted class prototype
- If distance > threshold_c → OOD detected
- Trigger: Queue for expert review, potential Phase 2

### 4. Pipeline Orchestration (`src/pipeline/independent_multi_crop_pipeline.py`)

**Responsibilities:**
- Initialize and manage crop router
- Load and manage multiple independent adapters
- Route images to correct adapter
- Aggregate results
- Handle OOD events
- Save/load pipeline state

**Key Methods:**
- `initialize_router()`: Load or train crop router
- `register_crop()`: Load adapter for a specific crop
- `process_image()`: End-to-end inference
- `batch_process()`: Batch inference
- `update_adapter()`: Hot-swap adapter updates

### 5. Data Pipeline (`src/utils/data_loader.py`)

**Dataset Classes:**

#### `CropDataset`
- Expected structure:
  ```
  data/{crop}/{split}/{class}/*.jpg
  ```
- Splits: train, val, test
- Transforms:
  - Train: Random flip, rotation, color jitter
  - Val/Test: Resize + normalize only
- Automatic class mapping

#### `DomainShiftDataset`
- For Phase 3 fortification
- Contains images with domain variations
- Same interface as CropDataset

**Preprocessing:**
- Resize to 224×224
- Normalize with ImageNet stats
- ToTensor

## Data Flow

### Training Phase 1
```
Raw Images → CropDataset → DoRA Adapter → Classifier
                                    ↓
                            Compute Prototypes
                                    ↓
                            Compute OOD Thresholds
                                    ↓
                            Save Adapter + OOD Stats
```

### Inference
```
Input Image
    ↓
Crop Router (predict crop)
    ↓
Load Crop Adapter
    ↓
Extract Features (DINOv2)
    ↓
Classifier (predict disease)
    ↓
Mahalanobis Distance (to predicted class prototype)
    ↓
Compare to Dynamic Threshold
    ↓
┌───────────────┬──────────────┐
│ In-Distribution │ OOD         │
│ → Return result │ → Flag OOD  │
└───────────────┴──────────────┘
```

## Configuration

See `config/adapter_spec_v55.json` for all configurable parameters:

- Model architectures
- LoRA hyperparameters
- Training schedules
- OOD detection settings
- Performance targets

## Performance Targets

| Metric | Target |
|--------|--------|
| Crop routing accuracy | ≥98% |
| Phase 1 clean accuracy | ≥95% |
| Phase 2 old class retention | ≥90% |
| Phase 3 protected retention | ≥85% |
| OOD detection AUROC | ≥0.92 |
| OOD false positive rate | ≤5% |
| Inference latency | <200ms |
| Memory per adapter | <25MB |

## Error Handling

All components include:
- Input validation
- Graceful degradation (fallback thresholds)
- Comprehensive logging
- Exception handling with informative messages

## Logging

Logging configured throughout:
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Levels: INFO for normal operation, WARNING for issues, ERROR for failures
- Output: Console + file (configurable)

## Extensibility

### Adding a New Crop
1. Add crop name to `config/adapter_spec_v55.json`
2. Prepare data in `data/{crop}/phase1/`
3. Train Phase 1 adapter
4. Register with pipeline: `pipeline.register_crop(crop_name, adapter_path)`

### Adding a New Disease (Phase 2)
1. Prepare data for new disease class
2. Run Phase 2 training with `--new_classes`
3. System automatically:
   - Expands classifier
   - Computes new class prototype
   - Updates OOD thresholds

### Updating OOD Thresholds
- Automatic during Phase 1, 2, 3
- Can manually recalibrate using validation data
- Adaptive threshold manager supports online updates

## Deployment Considerations

### GPU Memory
- DINOv2-giant: ~1.1B parameters
- With LoRA (r=32): ~2-3MB trainable
- Total memory: ~4-5GB per adapter
- Use gradient checkpointing if needed

### Batch Inference
- Pipeline supports batch processing
- Optimal batch size: 8-16 for A100
- Consider batch size 1 for real-time mobile

### Model Versioning
- Each adapter saved with metadata
- OOD stats versioned with timestamps
- Hot-swapping supported via `update_adapter()`

## Monitoring

Key metrics to monitor:
- Crop routing accuracy
- Per-crop disease accuracy
- OOD detection rate (should be 1-5%)
- False positive rate
- Inference latency
- GPU utilization

## Future Enhancements

- iOS mobile SDK
- Model quantization for edge deployment
- Federated learning across farms
- Active learning for OOD samples
- Multi-modal inputs (soil, weather)

---

**Last Updated:** February 2026
**Version:** 5.5.0