# AADS-ULoRA v5.5 Architecture Documentation

## System Overview

AADS-ULoRA v5.5 implements a production-ready multi-crop disease detection system using independent crop adapters with dynamic OOD detection. The system is designed for deployment in agricultural applications, specifically for integration with Uyumsoft ZiraiTakip.

## Enhanced Architecture: Crop+Part Routing with VLM Pipeline

Based on research findings from "Researching Plant Detection Methods.pdf" (2026), the system implements **Scenario B (Diagnostic Scouting)** exclusively, using the multi-stage VLM pipeline for maximum accuracy and explainability.

```
┌─────────────────────────────────────────────────────────────┐
│              L1: Enhanced Crop Router                        │
│  • Dual classifier: Crop + Plant Part                       │
│  • DINOv3-base backbone                                     │
│  • Two heads: (crop, part) → adapter_key                   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (crop, part)
          ┌─────────────────┴─────────────────┐
          ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│  Adapter:        │              │  Adapter:        │
│  tomato_leaf     │              │  tomato_fruit    │
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
          ↓                                   ↓
┌─────────────────────────────────────────────────────────────┐
│              Scenario B: VLM Diagnostic Pipeline            │
│  (Only scenario supported - for diagnostic scouting)       │
│                                                             │
│  Grounding DINO (open-set detection)                       │
│      ↓                                                      │
│  SAM-2 (zero-shot segmentation)                            │
│      ↓                                                      │
│  BioCLIP 2 (taxonomic identification + explanation)       │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principle:** Zero cross-adapter communication. Each (crop, part) combination maintains an independent lifecycle, allowing asynchronous updates and zero interference.

## Component Details

### 1. Enhanced Crop Router (`src/router/enhanced_crop_router.py`)

**Purpose:** Classify both crop type and plant part, then route to specific (crop, part) adapter.

**Architecture:**
- Backbone: DINOv3-base (pretrained, frozen)
- Two classification heads:
  - Crop classifier: (768 → num_crops)
  - Part classifier: (768 → num_parts)
- Adapter registry: {(crop, part): adapter_instance}

**Routing Logic:**
1. Classify crop type (e.g., 'tomato')
2. Classify plant part (e.g., 'leaf')
3. Form adapter key: `(crop, part)`
4. Lookup registered adapter
5. Fallback: If specific (crop, part) not found, use any adapter for that crop

**Supported Plant Parts:**
- leaf
- fruit
- stem
- root
- flower
- (extensible via configuration)

**Inference:**
- Input: 224×224 RGB image
- Output: (crop, part, crop_confidence, part_confidence, routing_info)
- Caching: Configurable cache for repeated images

**Scenario Handling:**
- Only **Scenario B (Diagnostic Scouting)** is supported
- Uses VLM pipeline for all inferences
- No automatic switching to other scenarios

### 2. VLM Pipeline (`src/router/vlm_pipeline.py`)

**Purpose:** High-accuracy diagnostic analysis using multi-stage foundation models.

**Three-Stage Architecture:**

#### Stage 1: Open-Set Detection (Grounding DINO)
- Accepts natural language prompts (e.g., "Find all tomato leaves")
- Returns bounding boxes for plant parts
- Zero-shot capability: can detect novel objects without retraining

#### Stage 2: Zero-Shot Segmentation (SAM-2)
- Takes bounding boxes as prompts
- Generates pixel-perfect masks
- Isolates plant tissue from background (critical for diseased images)

#### Stage 3: Taxonomic Identification (BioCLIP 2)
- Classifies segmented regions
- Uses hierarchical taxonomic embeddings
- Robust to morphological variations (withered, deformed organs)
- Provides natural language explanations

**Performance:**
- Accuracy: 97.27% on tomato disease identification
- Latency: <5 FPS (requires GPU acceleration)
- VRAM: >24GB for full pipeline

**Output:**
- Detections with bounding boxes and confidence
- Segmentation masks
- Classifications with taxonomic hierarchy
- Natural language explanation

### 3. Independent Crop Adapter (`src/adapter/independent_crop_adapter.py`)

**Purpose:** Disease classification for a specific (crop, part) combination with OOD detection.

**Three-Phase Training:**

#### Phase 1: DoRA Base Initialization
- **Method:** DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Goal:** Train base adapter on initial disease classes for this crop+part
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

### 4. OOD Detection (`src/ood/`)

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

### 5. Pipeline Orchestration (`src/pipeline/independent_multi_crop_pipeline.py`)

**Responsibilities:**
- Initialize and manage enhanced crop router
- Load and manage multiple independent adapters keyed by (crop, part)
- Route images to correct adapter via dual classification
- Aggregate results
- Handle OOD events
- Save/load pipeline state

**Key Methods:**
- `initialize_router()`: Load or train enhanced crop router
- `register_crop()`: Load adapter for specific (crop, part)
- `process_image()`: End-to-end inference with VLM pipeline
- `batch_process()`: Batch inference
- `update_adapter()`: Hot-swap adapter updates

**Integration with VLM:**
- All inferences use Scenario B (VLM pipeline)
- Router determines (crop, part) → selects appropriate adapter
- Adapter performs disease classification with OOD detection
- VLM provides segmentation and explanation

### 6. Data Pipeline (`src/utils/data_loader.py`)

**Dataset Classes:**

#### `CropDataset`
- Expected structure:
  ```
  data/{crop}/{part}/{split}/{class}/*.jpg
  ```
  OR (legacy):
  ```
  data/{crop}/{split}/{class}/*.jpg
  ```
- Splits: train, val, test
- Transforms:
  - Train: Random flip, rotation, color jitter
  - Val/Test: Resize + normalize only
- Automatic class mapping

**Preprocessing:**
- Resize to 224×224
- Normalize with ImageNet stats
- ToTensor

## Data Flow

### Training Phase 1 (Per Crop+Part)
```
Raw Images (crop+part) → CropDataset → DoRA Adapter → Classifier
                                          ↓
                                  Compute Prototypes
                                          ↓
                                  Compute OOD Thresholds
                                          ↓
                                  Save Adapter + OOD Stats
```

### Inference (Scenario B - Diagnostic Scouting)
```
Input Image
    ↓
Enhanced Crop Router
    ├─ Classify crop (DINOv3 + linear head)
    ├─ Classify part (DINOv3 + linear head)
    └─ Determine adapter_key = (crop, part)
    ↓
Load Adapter for (crop, part)
    ↓
VLM Pipeline:
    ├─ Grounding DINO (detect plant parts)
    ├─ SAM-2 (segment detected parts)
    ├─ Adapter Classifier (disease ID)
    └─ BioCLIP 2 (taxonomic reasoning + explanation)
    ↓
Return results with OOD analysis and natural language report
```

## Configuration

See `config/adapter_spec_v55.json` for all configurable parameters:

### Routing Configuration

```json
{
  "crop_router": {
    "model_name": "facebook/dinov3-base",
    "training_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3
  },
  "routing": {
    "cache_size": 1000,
    "fallback_enabled": true
  }
}
```

### Supported Crops and Parts

The router supports:
- **Crops**: tomato, pepper, corn (configurable via `data.crops`)
- **Parts**: leaf, fruit, stem, root, flower (configurable)

Each (crop, part) combination requires a separately trained adapter.

## Performance Targets

| Metric | Target |
|--------|--------|
| Crop classification accuracy | ≥98% |
| Part classification accuracy | ≥95% |
| Combined (crop, part) routing accuracy | ≥93% |
| Phase 1 clean accuracy (per adapter) | ≥95% |
| Phase 2 old class retention | ≥90% |
| Phase 3 protected retention | ≥85% |
| OOD detection AUROC | ≥0.92 |
| OOD false positive rate | ≤5% |
| VLM pipeline latency | <500ms |
| Memory per adapter | <25MB |

## Scenario B: Diagnostic Scouting (Only Supported Scenario)

According to research, Scenario B is optimal for disease diagnosis and phenotyping:

**Advantages:**
- Highest accuracy (97.27% on tomato diseases)
- Zero-shot generalization to novel diseases
- Pixel-perfect segmentation for accurate symptom analysis
- Natural language explanations for agronomists
- Robust to morphological variations

**Requirements:**
- GPU with ≥24GB VRAM
- Sufficient power (not battery-constrained)
- Latency tolerance >100ms

**Use Cases:**
- Disease diagnosis and reporting
- Yield estimation
- New pathogen discovery
- Research and phenotyping
- Expert consultation support

## Adding New Crops and Parts

### Step 1: Add Crop and Part Definitions

Update `config/adapter_spec_v55.json`:
```json
{
  "data": {
    "crops": ["tomato", "pepper", "corn", "cucumber"],
    "parts": ["leaf", "fruit", "stem", "root", "flower", "tuber"]
  }
}
```

### Step 2: Prepare Dataset

Organize data:
```
data/
├── cucumber/
│   ├── leaf/
│   │   ├── healthy/
│   │   ├── powdery_mildew/
│   │   └── ...
│   ├── fruit/
│   │   ├── healthy/
│   │   └── ...
│   └── ...
```

### Step 3: Train Router (if new crops/parts added)

The router needs to be trained to classify the new crops and parts. Prepare a dataset with all crop types and parts, then train the dual classifiers.

### Step 4: Train Adapter for Each (Crop, Part) Combination

For each combination (e.g., cucumber_leaf, cucumber_fruit):
```bash
# Phase 1
python -m src.training.phase1_training \
  --data_dir ./data/cucumber/leaf \
  --crop cucumber_leaf \
  --output_dir ./adapters/cucumber_leaf \
  --epochs 50

# Phase 2 (if adding new diseases later)
python -m src.training.phase2_sd_lora \
  --adapter_path ./adapters/cucumber_leaf \
  --new_classes new_disease \
  --output_dir ./adapters/cucumber_leaf_phase2

# Phase 3 (for domain fortification)
python -m src.training.phase3_conec_lora \
  --adapter_path ./adapters/cucumber_leaf_phase2 \
  --domain_shift_dir ./data/cucumber/leaf/domain_shift \
  --output_dir ./adapters/cucumber_leaf_phase3
```

### Step 5: Register Adapters

```python
router = EnhancedCropRouter(crops, parts, config)

# Register each (crop, part) adapter
router.register_adapter('cucumber', 'leaf', './adapters/cucumber_leaf')
router.register_adapter('cucumber', 'fruit', './adapters/cucumber_fruit')
```

## Error Handling

All components include:
- Input validation
- Graceful degradation (fallback to any crop adapter)
- Comprehensive logging
- Exception handling with informative messages
- Cache management for performance

## Logging

Logging configured throughout:
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Levels: INFO for normal operation, WARNING for issues, ERROR for failures
- Output: Console + file (configurable)

## Extensibility

### Adding a New Scenario

The current implementation only supports Scenario B (Diagnostic Scouting). To add other scenarios:
1. Extend the router to include model selection logic
2. Implement YOLO26 adapter for Scenario A
3. Implement ALNet adapter for Scenario C
4. Update routing logic to select scenario based on metadata/hardware

### Adding a New Plant Part

1. Add part name to configuration
2. Prepare training data for that part across all crops
3. Train part classifier (or retrain if adding to existing)
4. Train adapters for each (crop, new_part) combination

### Updating OOD Thresholds

- Automatic during Phase 1, 2, 3
- Can manually recalibrate using validation data
- Adaptive threshold manager supports online updates

## Deployment Considerations

### GPU Memory
- DINOv3-giant: ~1.1B parameters
- With LoRA (r=32): ~2-3MB trainable per adapter
- Total memory: ~4-5GB per adapter
- VLM pipeline: >24GB required
- Use gradient checkpointing if needed

### Batch Inference
- Router supports batch processing
- Optimal batch size: 8-16 for A100
- VLM pipeline typically processes one image at a time

### Model Versioning
- Each adapter saved with metadata
- OOD stats versioned with timestamps
- Hot-swapping supported via `register_adapter()`
- Router state includes classification heads

## Monitoring

Key metrics to monitor:
- Crop classification accuracy
- Part classification accuracy
- Adapter hit/miss rates
- Fallback frequency
- OOD detection rate (should be 1-5%)
- False positive rate
- Inference latency
- GPU utilization
- Cache hit rates

## Future Enhancements

- Implement Scenario A (YOLO26) for real-time actuation
- Implement Scenario C (ALNet) for resource-constrained devices
- Automatic part detection (instead of classification)
- Multi-part handling in single image
- Federated learning across farms
- Active learning for OOD samples
- Multi-modal inputs (soil, weather, satellite)

---

**Last Updated:** February 2026
**Version:** 5.5.0
**Based on Research:** "Researching Plant Detection Methods.pdf" (2026)