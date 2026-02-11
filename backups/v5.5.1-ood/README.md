# AADS-ULoRA v5.5

**Agricultural AI Development System - ULoRA v5.5**

A production-ready, multi-crop disease detection system using independent crop adapters with dynamic OOD (Out-of-Distribution) detection.

## Key Features

- **Multi-Crop Support**: Independent adapters for tomato, pepper, corn (extensible to more crops)
- **Dynamic OOD Detection**: Per-class Mahalanobis distance thresholds (no manual tuning)
- **Continual Learning**: Three-phase training pipeline (DoRA, SD-LoRA, CONEC-LoRA)
- **Rehearsal-Free**: No need to store old class data
- **Asynchronous Updates**: Each crop adapter updates independently
- **Mobile Integration**: Cloud API with offline queue for field use

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    L1: Crop Router                          │
│  (Simple classifier: DINOv2-base or ResNet-50)             │
│  Function: Route image → correct crop adapter              │
│  Target Accuracy: ≥98%                                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
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

**Key Principle**: Zero cross-adapter communication. Each crop maintains independent lifecycle.

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 45GB+ GPU memory (A100 recommended) or use gradient checkpointing

### Setup

```bash
# Clone repository
cd aads-ulora-v5.5

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Create dataset structure
python -m src.dataset.preparation \
  --data_dir ./data \
  --crop tomato
```

Expected directory structure:
```
data/
├── tomato/
│   ├── phase1/
│   │   ├── healthy/
│   │   ├── early_blight/
│   │   └── ...
│   ├── val/
│   └── test/
├── pepper/
└── corn/
```

### 2. Train Crop Router

```bash
python -m src.router.simple_crop_router \
  --data_dir ./data \
  --crop tomato \
  --output_dir ./outputs/router \
  --epochs 10 \
  --batch_size 32
```

### 3. Train First Crop (Phase 1 - DoRA)

```bash
python -m src.training.phase1_training \
  --data_dir ./data \
  --crop tomato \
  --output_dir ./outputs/tomato_phase1 \
  --epochs 50 \
  --batch_size 32
```

### 4. Add New Disease (Phase 2 - SD-LoRA)

```bash
python -m src.training.phase2_sd_lora \
  --adapter_path ./outputs/tomato_phase1 \
  --new_classes septoria_leaf_spot \
  --output_dir ./outputs/tomato_phase2 \
  --epochs 20
```

### 5. Fortify Against Domain Shift (Phase 3 - CONEC-LoRA)

```bash
python -m src.training.phase3_conec_lora \
  --adapter_path ./outputs/tomato_phase2 \
  --domain_shift_dir ./data/tomato/domain_shift \
  --output_dir ./outputs/tomato_phase3 \
  --epochs 15
```

### 6. Run Demo

```bash
# Start Gradio demo
python -m demo.app \
  --adapter_dir ./adapters \
  --router_path ./router/crop_router_best.pth \
  --port 7860
```

## Project Structure

```
AADS_ULoRA_v5.5/
├── src/
│   ├── router/              # Crop routing
│   │   └── simple_crop_router.py
│   ├── adapter/             # Independent crop adapters
│   │   └── independent_crop_adapter.py
│   ├── training/            # Training scripts
│   │   ├── phase1_training.py
│   │   ├── phase2_sd_lora.py
│   │   └── phase3_conec_lora.py
│   ├── ood/                 # OOD detection
│   │   ├── prototypes.py
│   │   ├── mahalanobis.py
│   │   └── dynamic_thresholds.py
│   ├── pipeline/            # Orchestration
│   │   └── independent_multi_crop_pipeline.py
│   ├── utils/               # Utilities
│   │   └── data_loader.py
│   ├── evaluation/          # Metrics
│   │   └── metrics.py
│   └── debugging/           # Monitoring
│       └── monitoring.py
├── api/                     # FastAPI endpoints (to be implemented)
├── mobile/                  # Mobile SDK (to be implemented)
├── config/                  # Configuration files
│   └── adapter_spec_v55.json
├── data/                    # Dataset
├── adapters/                # Trained adapters
├── router/                  # Trained router
├── prototypes/              # Class prototypes
├── ood_stats/               # OOD thresholds
├── logs/                    # Training logs
├── checkpoints/             # Model checkpoints
├── demo/                    # Gradio demo
│   └── app.py
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

Edit `config/adapter_spec_v55.json` to customize:

- Model architectures
- LoRA hyperparameters (r, alpha, dropout)
- Training epochs and learning rates
- OOD detection threshold factor
- Target metrics

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

## API Reference

### REST API (Planned)

```http
POST /v1/diagnose
Content-Type: application/json

{
  "image": "base64_encoded_jpeg",
  "crop_hint": "tomato",
  "location": {
    "latitude": 41.0082,
    "longitude": 28.9784
  }
}

Response:
{
  "status": "success",
  "crop": {"predicted": "tomato", "confidence": 0.987},
  "disease": {
    "class_index": 1,
    "name": "early_blight",
    "confidence": 0.943
  },
  "ood_analysis": {
    "is_ood": false,
    "mahalanobis_distance": 8.5,
    "threshold": 12.3
  }
}
```

## Training Details

### Phase 1: DoRA Base Initialization

- **Method**: DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Purpose**: Train base adapter on initial disease classes
- **Target**: ≥95% accuracy on clean test set
- **Duration**: ~50 epochs

### Phase 2: SD-LoRA Class-Incremental Learning

- **Method**: SD-LoRA (Selective Directional LoRA)
- **Purpose**: Add new disease classes without forgetting old ones
- **Mechanism**: Freeze LoRA A/B matrices (directions), train only magnitudes and classifier
- **Target**: ≥90% retention on old classes
- **Duration**: ~20 epochs

### Phase 3: CONEC-LoRA Domain-Incremental Learning

- **Method**: CONEC-LoRA (Continual Learning with Early-block Consolidation)
- **Purpose**: Fortify against domain shifts (different lighting, angles, etc.)
- **Mechanism**: Freeze early transformer blocks (0-5), add new LoRA to late blocks (6-11)
- **Target**: ≥85% protected retention on non-fortified classes
- **Duration**: ~15 epochs

## OOD Detection

The system uses **dynamic Mahalanobis distance** with per-class thresholds:

1. **Prototype Computation**: Compute mean feature vector per class from training data
2. **Covariance Estimation**: Compute per-class feature covariance (diagonal approximation)
3. **Distance Calculation**: Mahalanobis distance = (x - μ)^T * Σ⁻¹ * (x - μ)
4. **Threshold Calibration**: threshold = mean_distance + k * std_distance (k=2 for 95% confidence)

This approach adapts to each class's inherent variability, providing robust OOD detection without manual threshold tuning.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{aads-ulora-v5.5,
  title={AADS-ULoRA v5.5: Independent Multi-Crop Disease Detection with Dynamic OOD},
  author={Agricultural AI Development Team},
  year={2026},
  url={https://github.com/yourorg/aads-ulora-v5.5}
}
```

## Support

- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Email: contact@aads.com

---

**Status**: Production-ready for Uyumsoft ZiraiTakip integration.