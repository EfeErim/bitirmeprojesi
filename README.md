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

### Removed Components (February 2026)
- Mobile application code (`mobile/android/`)
- FastAPI server and REST endpoints (`api/`)
- Demo application (`demo/`)
- Docker containerization (`docker/`)
- Monitoring infrastructure (`monitoring/`)


## Directory Structure

```
d:/bitirme projesi/
├── colab_notebooks/       # Jupyter training notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_phase1_training.ipynb
│   ├── 3_phase2_training.ipynb
│   ├── 4_phase3_training.ipynb
│   ├── 5_testing_validation.ipynb
│   └── 6_performance_monitoring.ipynb
├── src/                   # Core training and inference code
│   ├── adapter/          # Crop-specific adapters
│   ├── core/             # Configuration and contracts
│   ├── dataset/          # Data preparation and loading
│   ├── debugging/        # Performance monitoring
│   ├── evaluation/       # Metrics and evaluation
│   ├── ood/              # OOD detection components
│   ├── pipeline/         # Multi-crop inference pipeline
│   ├── router/           # Crop routing logic
│   ├── training/         # Phase trainers (Phase 1-3)
│   ├── utils/            # Shared utilities
│   └── visualization/    # Visualization tools
├── tests/                 # Comprehensive test suites
│   ├── colab/            # Colab environment smoke tests
│   ├── integration/       # End-to-end pipeline tests
│   ├── unit/             # Unit tests for components
│   └── fixtures/         # Test data and fixtures
├── config/                # Configuration files
│   ├── base.json         # Base configuration
│   ├── colab.json        # Colab environment config
│   ├── development.json  # Development overrides
│   ├── production.json   # Production settings
│   ├── adapter_spec_v55.json
│   └── pytest.ini
├── docs/                  # Documentation
│   ├── README.md
│   ├── architecture/
│   ├── development/
│   ├── contributing/
│   ├── deployment/
│   ├── security/
│   ├── user_guide/
│   ├── api/
│   └── colab_migration_guide.md
├── scripts/               # Utility scripts
│   ├── check_markdown_links.py
│   ├── download_data_colab.py
│   └── install_colab.py
├── data/                  # Data storage
│   └── test_dataset/
├── logs/                  # Training logs
├── colab_bootstrap.ipynb  # Colab setup notebook
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
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

### Option 1: One-Click Colab Training (Recommended)

**No manual intervention needed** - Just one notebook, everything runs automatically!

1. **Open in Google Colab:**  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EfeErim/bitirmeprojesi/blob/master/colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb)

2. **Configure Training (Interactive UI)**
   - The first cell will display an interactive configuration interface:
     - Select crops (tomato, potato, wheat, custom)
     - Set dataset path on Google Drive
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

See [Checkpoint System Guide](CHECKPOINT_SYSTEM_GUIDE.md) for detailed usage.

### 🎯 Interactive Configuration

Before training starts, the notebook displays an interactive UI where you can:
- **🌾 Select Crops** - Choose from tomato, potato, wheat, or provide custom crop names
- **📂 Dataset Path** - Specify the Google Drive path to your PlantVillage dataset
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
python validate_notebook_imports.py
pytest -c config/pytest.ini tests/import_test.py
```

### Option 3: Local Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Check documentation
python scripts/check_markdown_links.py --root .
```

## 💾 Output Files

After training completes (Colab):

```
Google Drive/aads_ulora/
├── models/
│   ├── phase1_dora_adapter/       # DoRA trained model
│   ├── phase2_sd_lora_adapter/    # SD-LoRA trained model
│   └── phase3_conec_lora_adapter/ # CoNeC-LoRA trained model
├── model_checkpoints/             # Model weights and optimizer states
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
├── .checkpoints/                  # Training progress tracking
│   └── checkpoint_log.json        # Progress log with timestamps
├── logs/
│   ├── phase1_history.json
│   ├── phase2_history.json
│   └── phase3_history.json
└── outputs/
    ├── validation_results.json
    ├── performance_metrics.csv
    ├── training_summary.html
    └── training_config.json       # Configuration used for this run
```

**Checkpoint Log Example:**
```json
{
  "setup": {"timestamp": "2025-02-20T10:15:23", "completed": true},
  "data_prep": {"timestamp": "2025-02-20T10:16:45", "completed": true},
  "phase1": {"timestamp": "2025-02-20T10:45:23", "completed": true, "details": {...}},
  "phase2": {"timestamp": "2025-02-20T11:30:15", "completed": true, "details": {...}},
  "phase3": null,
  "validation": null,
  "monitoring": null
}
```
The checkpoint log allows you to resume training exactly where it left off.

## 📚 Documentation

### Getting Started
- [Seamless Auto-Train Guide](SEAMLESS_AUTOTRAIN_GUIDE.md) - Complete end-to-end Colab training walkthrough
- **[Checkpoint System Guide](CHECKPOINT_SYSTEM_GUIDE.md)** - Recovery, resuming, and managing training checkpoints

### Technical Documentation
- [Documentation Index](docs/README.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Crop Router Technical Guide](docs/architecture/crop-router-technical-guide.md)
- [VLM Pipeline Guide](docs/architecture/vlm-pipeline-guide.md)
- [Development Setup](docs/development/development-setup.md)
- [Colab Training Manual](docs/user_guide/colab_training_manual.md)
- [Colab Cheatsheet](docs/user_guide/cheatsheet_colab.md)
- [Tomato Crop Adapter Manual](docs/user_guide/tomato_crop_adapter_manual.md)
