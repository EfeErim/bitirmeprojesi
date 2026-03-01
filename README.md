# AADS-ULoRA v5.5

**Agricultural AI Development System - ULoRA v5.5**

A production-ready, multi-crop disease detection system using independent crop adapters with dynamic OOD (Out-of-Distribution) detection and crop-specific routing for precise disease identification.

## Overview

AADS-ULoRA is focused on the **core ML training and inference engine** for multi-crop disease detection. This project provides:

- Multi-phase training pipeline (DoRA → SD-LoRA → CoNeC-LoRA)
- Crop-specific adapters for specialized inference
- Out-of-Distribution (OOD) detection with Mahalanobis distance
- Colab-first training workflow with complete notebooks
- Comprehensive test suites for reliability

## Project Focus

This project focuses on the **research and training components** of the agricultural AI system. Mobile application deployment, API servers, and web service integrations have been removed to maintain focus on the core product development pipeline.

## Repository Navigation

- **Full file-purpose + relationship map:** [docs/REPO_FILE_RELATIONS.md](docs/REPO_FILE_RELATIONS.md)
- **Documentation index:** [docs/README.md](docs/README.md)
- **Reports archive:** [docs/reports/README.md](docs/reports/README.md)
- **Notebook index:** [colab_notebooks/README.md](colab_notebooks/README.md)
- **Scripts index:** [scripts/README.md](scripts/README.md)

If you're unsure where to start, use the file-relation map first and then jump to the relevant notebook/script index.

### Removed Components (February 2026)
- Mobile application code (`mobile/android/`)
- FastAPI server and REST endpoints (`api/`)
- Demo application (`demo/`)
- Docker containerization (`docker/`)
- Monitoring infrastructure (`monitoring/`)


## Directory Structure

```
d:/bitirme projesi/
├── colab_notebooks/            # Jupyter notebooks (one-click + phase-by-phase)
├── src/                        # Core training/inference code
│   ├── adapter/                # Crop adapters
│   ├── core/                   # Contracts/config managers
│   ├── dataset/                # Data prep/load/cache
│   ├── debugging/              # Debug/perf helpers
│   ├── evaluation/             # Metrics and evaluation
│   ├── monitoring/             # Runtime monitoring metrics
│   ├── ood/                    # OOD detection components
│   ├── pipeline/               # Multi-crop pipeline
│   ├── router/                 # VLM and crop routing
│   ├── training/               # Phase 1/2/3 trainers
│   ├── utils/                  # Shared utilities
│   └── visualization/          # Visualization tools
├── tests/                      # Unit/integration/colab test suites
├── config/                     # Runtime and taxonomy config
│   ├── base.json
│   ├── colab.json
│   ├── plant_taxonomy.json
│   └── pytest.ini
├── docs/                       # Documentation and reports
│   ├── README.md
│   ├── REPO_FILE_RELATIONS.md
│   ├── reports/
│   │   ├── README.md
│   │   └── v55/
│   ├── architecture/
│   ├── development/
│   ├── guides/
│   ├── user_guide/
│   └── security/
├── scripts/                    # Setup/testing/regression scripts
├── data/                       # Local data/test dataset
├── logs/                       # Run logs
├── plans/                      # Planning artifacts
├── specs/                      # Adapter/spec files
├── README.md
├── requirements.txt
├── requirements_colab.txt
├── setup.py
└── validate_notebook_imports.py
```



## Benefits

- **Focused Development**: Pure ML training and inference, no deployment overhead
- **Comprehensive Training Pipeline**: Full 3-phase adapter training workflow
- **Production-Grade Tests**: Extensive Colab and integration test coverage
- **Clear Code Structure**: Well-organized source modules with clean boundaries
- **Colab-First Design**: Optimized for cloud-based training

## Technical Specifications

- **Framework**: PyTorch with Vision Transformers (DINOv3)
- **Adaptation**: ULoRA (Unified Low-Rank Adaptation)
  - Phase 1: DoRA (Difference of Rectified Activations)
  - Phase 2: SD-LoRA (Stable Diffusion LoRA)
  - Phase 3: CoNeC-LoRA (Congruent Enhancement LoRA)
- **OOD Detection**: Mahalanobis distance + prototype-based methods
- **Routing**: Crop-specific adapter dispatch
- **Training Environment**: Colab-optimized with GPU/TPU support
- **Configuration**: JSON-based with environment overrides

## Status

Training and inference engine for AADS-ULoRA. Version 5.5.0. Last updated: February 2026.

## 🚀 Quick Start: Complete Seamless Training

### Canonical Entrypoints (User-Facing)

- **Primary start path:** `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
- **Manual/diagnostic path:** `colab_notebooks/colab_bootstrap.ipynb` then notebooks `1` → `6`
- **Canonical local sanity command:** `python scripts/validate_notebook_imports.py`
- **Compatibility aliases (root scripts):** retained for legacy workflows, but `scripts/` is preferred in documentation.

### Option 1: One-Click Colab Training (Recommended)

**No manual intervention needed** - Just one notebook, everything runs automatically!

1. **Open in Google Colab:**  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EfeErim/bitirmeprojesi/blob/master/colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb)

2. **Configure Training (Interactive UI)**
   - The first cell will display an interactive configuration interface:
     - Select crops (tomato, potato, wheat, custom)
     - Set class-root dataset path on Google Drive (<root>/<class_name>/<images>)
     - Choose training parameters (batch size, learning rate, epochs)
     - Select which phases to train
     - Enable/disable validation
     - Set output directory

3. **Run All Cells** (Shift+Enter repeatedly or Runtime → Run all)
   - The notebook will automatically handle:
     - ✅ GPU detection and setup
     - ✅ Google Drive mounting
     - ✅ Repository cloning/syncing
     - ✅ Dependency installation
     - ✅ Data preparation
     - ✅ Phase 1 DoRA training (3-4 hours)
     - ✅ Phase 2 SD-LoRA training (2-3 hours)
     - ✅ Phase 3 CoNeC-LoRA training (2-3 hours)
     - ✅ Model validation
     - ✅ Performance reporting

4. **Wait for completion** (~8-12 hours total)
   - All results saved to Google Drive automatically
   - No interruptions or manual steps needed
   - **Checkpoints created at each stage** - can resume from any point if interrupted

### ⚡ Checkpoint System: Never Lose Training Progress

The auto-training notebook includes an **automatic checkpoint system** that:

- **🔄 Recovers from Failures** - If Colab times out during Phase 3, resume automatically from Phase 3 without re-running Phases 1-2
- **⏭️ Skips Completed Phases** - Second runs skip finished phases entirely (saves 8+ hours of compute time)
- **🔍 Clear Progress Visibility** - See exactly which phases completed with timestamps
- **📝 Configuration Logging** - Every checkpoint stores training parameters for reproducibility

**How it works in practice:**
```
First run: Setup → Data Prep → Phase 1 → Phase 2 → Phase 3 → Validation → Done
           ✅     ✅          ✅        ✅        ✅        ✅

Second run (if interrupted): Phase 3 resumes from checkpoint
                             ⊘      ⊘          ⊘        ✅        ✅  

Third run (modified config): Phases 1-2 skipped (cached), Phase 3 re-runs with new learning rate
                             ⊘      ⊘          ⊘        ✅(skip)  ✅(rerun) ✅ Done!
```

**Manage checkpoints programmatically:**
```python
# Check which phases are complete
checkpoint_manager.display_checkpoint_status()

# Clear a phase to re-run it
checkpoint_manager.clear_checkpoints(['phase2'])

# Start completely fresh
checkpoint_manager.clear_checkpoints()
```

See [Checkpoint System Guide](docs/guides/CHECKPOINT_SYSTEM_GUIDE.md) for detailed usage.

### 🎯 Interactive Configuration

Before training starts, the notebook displays an interactive UI where you can:
- **🌾 Select Crops** - Choose from tomato, potato, wheat, or provide custom crop names
- **📂 Dataset Path** - Specify the Google Drive class-root dataset path (<root>/<class_name>/<images>). The notebook auto-creates train/val/test splits (80/10/10).
- **⚙️ Training Parameters** - Set batch size (8-128), learning rate (1e-6 to 1e-2), epochs per phase (1-10)
- **🔀 Phase Selection** - Run only the phases you need (Phase 1, 2, 3 independently or together)
- **🖥️ Advanced Options** - Enable mixed precision, validation, device selection, checkpoint resume, etc.

No more manual script editing - just set parameters once and train!

### Option 2: Local Development Setup

```bash
# Clone repository
git clone https://github.com/EfeErim/bitirmeprojesi.git
cd bitirmeprojesi

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/validate_notebook_imports.py
python tests/import_test.py
```

### Option 3: Local Testing

```bash
# Run quick modular suite group (fast default)
python scripts/run_test_suites.py

# Run only router-focused unit tests
python scripts/run_test_suites.py --suite unit/router

# Run all unit suites
python scripts/run_test_suites.py --suite unit

# Run integration suites
python scripts/run_test_suites.py --suite integration

# Run full matrix explicitly
python scripts/run_test_suites.py --suite all

# Run core Python sanity checks (notebook imports + local pipeline checks)
python scripts/run_python_sanity_bundle.py

# Run core import validation smoke check
python tests/import_test.py

# Check documentation
python scripts/check_markdown_links.py --root .

# Run policy profile + stage-order regression bundle
python scripts/run_policy_regression_bundle.py
```

The GitHub Actions CI workflow also runs this policy regression bundle on every push/PR via the `policy-regression` job, and runs modular suites via `python scripts/run_test_suites.py` in the `test` job.

## 💾 Output Files

After training completes (Colab):

```
Google Drive/aads_ulora/
├── data/
│   ├── dataset_metadata.json           # Dataset normalization, augmentation, class counts
│   └── plantvillage/                   # Training/val/test splits
├── models/
│   ├── phase1_dora_adapter/
│   │   ├── adapter/                    # PEFT LoRA weights
│   │   │   ├── adapter_config.json
│   │   │   └── pytorch_model.bin
│   │   ├── classifier.pth              # Classifier head weights
│   │   ├── adapter_meta.json           # Class indices, input/output dims
│   │   ├── ood_components.pt           # OOD detection model (prototypes, mahalanobis, thresholds)
│   │   └── manifest.json               # Artifact metadata and paths
│   ├── phase2_sd_lora_adapter/
│   │   ├── adapter/
│   │   ├── classifier.pth
│   │   ├── adapter_meta.json
│   │   ├── ood_components.pt
│   │   └── manifest.json
│   └── phase3_conec_lora_adapter/
│       ├── adapter/
│       ├── classifier.pth
│       ├── adapter_meta.json
│       ├── ood_components.pt          # FINAL OOD detection model for inference
│       └── manifest.json
├── model_checkpoints/                  # Model weights and optimizer states for resuming
│   ├── phase1/
│   │   └── checkpoint-{epoch}.pt
│   ├── phase2/
│   └── phase3/
├── .checkpoints/                       # Training pipeline progress tracking
│   └── checkpoint_log.json             # Progress log with timestamps per stage
├── logs/
│   ├── phase1_history.json             # Training losses, accuracies
│   ├── phase2_history.json
│   ├── phase3_history.json
│   ├── ood_metrics.json                # OOD detection evaluation results
│   └── training.log                    # Full training logs
└── outputs/
    ├── training_config.json            # Configuration used for this run
    ├── validation_results.json         # Per-class metrics (accuracy, precision, recall, F1)
    ├── confusion_matrices.json         # Confusion matrices per phase
    ├── performance_metrics.csv         # Aggregate metrics summary
    ├── performance_summary.json        # Best metrics per phase
    └── training_summary.html           # HTML report of all results
```

### OOD Components Metadata (in `ood_components.pt`)
Contains PyTorch tensors and objects:
```python
{
    'prototypes': torch.Tensor,                    # Class prototypes [num_classes, feature_dim]
    'mahalanobis': {
        'mean': torch.Tensor,                      # Feature space mean
        'covariance': torch.Tensor,                # Covariance matrix
        'inv_covariance': torch.Tensor             # Pre-computed inverse for efficiency
    },
    'thresholds': Dict[int, float],                # Per-class OOD thresholds
    'class_std': Dict[int, torch.Tensor]           # Per-class feature standard deviations
}
```

### Adapter Metadata (in `adapter_meta.json`)
```json
{
    "is_trained": true,
    "current_phase": 1,
    "class_to_idx": {"tomato_early_blight": 0, "tomato_late_blight": 1, ...},
    "classifier_input_size": 768,
    "classifier_output_size": 10
}
```

**Checkpoint Log Example:**
```json
{
  "setup": {"timestamp": "2025-02-20T10:15:23", "completed": true, "details": {...}},
  "data_prep": {"timestamp": "2025-02-20T10:16:45", "completed": true, "details": {...}},
  "phase1": {"timestamp": "2025-02-20T10:45:23", "completed": true, "details": {"status": "Completed", "epochs": 3, "crops": ["tomato"]}},
  "phase2": {"timestamp": "2025-02-20T11:30:15", "completed": true, "details": {...}},
  "phase3": {"timestamp": "2025-02-20T12:20:10", "completed": true, "details": {...}},
  "validation": {"timestamp": "2025-02-20T12:35:45", "completed": true, "details": {...}},
  "monitoring": {"timestamp": "2025-02-20T12:50:30", "completed": true, "details": {...}}
}
```

The checkpoint log allows you to resume training exactly where it left off. OOD components are essential for inference - without them, the adapters cannot detect out-of-distribution samples during deployment.

## 📚 Documentation

### Getting Started
- [Seamless Auto-Train Guide](docs/guides/SEAMLESS_AUTOTRAIN_GUIDE.md) - Complete end-to-end Colab training walkthrough
- **[Checkpoint System Guide](docs/guides/CHECKPOINT_SYSTEM_GUIDE.md)** - Recovery, resuming, and managing training checkpoints

### Technical Documentation
- [Documentation Index](docs/README.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Crop Router Technical Guide](docs/architecture/crop-router-technical-guide.md)
- [VLM Pipeline Guide](docs/architecture/vlm-pipeline-guide.md)
- [Development Setup](docs/development/development-setup.md)
- [Colab Training Manual](docs/user_guide/colab_training_manual.md)
- [Colab Cheatsheet](docs/user_guide/cheatsheet_colab.md)
- [Tomato Crop Adapter Manual](docs/user_guide/tomato_crop_adapter_manual.md)
