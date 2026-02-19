# AADS-ULoRA v5.5 - Seamless End-to-End Auto-Training

## ✅ What's New (February 20, 2026)

You now have a **completely seamless, one-click training pipeline** for Google Colab with zero manual intervention required.

## 🚀 How It Works

### **Option 1: One-Click Colab (Recommended)**

Simply open and run the notebook in your browser - everything is automated!

**Link:** [Open in Google Colab](https://colab.research.google.com/github/EfeErim/bitirmeprojesi/blob/master/colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb)

**Steps:**
1. Click the link above
2. Click "Run all cells" (Runtime → Run all)
3. Wait for completion (~8-12 hours)
4. All results automatically saved to Google Drive

**What Gets Done Automatically:**
```
✅ GPU Detection (T4, A100, V100, etc.)
✅ Google Drive Mount
✅ Repository Clone/Sync
✅ Dependency Installation
✅ Data Preparation (20 min)
✅ Phase 1 DoRA Training (3-4 hours)
✅ Phase 2 SD-LoRA Training (2-3 hours)
✅ Phase 3 CoNeC-LoRA Training (2-3 hours)
✅ Model Validation (30 min)
✅ Performance Reports (30 min)
```

### **Option 2: Quick Setup Cell**

Copy and paste this into a single Colab cell:

```python
import subprocess
from pathlib import Path
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')
project_root = Path('/content/drive/MyDrive/aads_ulora')
project_root.mkdir(parents=True, exist_ok=True)

# Clone repo
repo_path = project_root / 'bitirmeprojesi'
subprocess.run(['git', 'clone', 'https://github.com/EfeErim/bitirmeprojesi.git', str(repo_path)], check=True)

# Open auto-train notebook
print(f"Open: {repo_path}/colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb")
```

## 📁 Output Structure

After training completes, all results are saved to Google Drive:

```
MyDrive/aads_ulora/bitirmeprojesi/
├── models/
│   ├── phase1_dora_adapter/
│   │   ├── adapter_config.json
│   │   ├── pytorch_model.bin
│   │   └── adapter.state_dict
│   ├── phase2_sd_lora_adapter/
│   └── phase3_conec_lora_adapter/
├── checkpoints/
│   ├── phase1/
│   │   └── checkpoint-latest.pt
│   ├── phase2/
│   └── phase3/
├── logs/
│   ├── phase1_history.json
│   ├── phase2_history.json
│   ├── phase3_history.json
│   └── training.log
└── outputs/
    ├── validation_results.json
    ├── performance_metrics.csv
    └── confusion_matrices.json
```

## ⏱️ Timeline Breakdown

| Phase | Duration | GPU | Memory |
|-------|----------|-----|--------|
| Setup | 5-10 min | - | - |
| Data Prep | 20 min | Light | 2GB |
| Phase 1 (DoRA) | 3-4 hours | Heavy | 14GB |
| Phase 2 (SD-LoRA) | 2-3 hours | Heavy | 14GB |
| Phase 3 (CoNeC) | 2-3 hours | Heavy | 14GB |
| Validation | 30 min | Medium | 4GB |
| Reports | 30 min | Light | 2GB |
| **Total** | **8-12 hours** | - | - |

## 🔧 How It All Works Behind the Scenes

### Master Orchestration Notebook
**File:** `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`

This notebook:
- Handles all setup and configuration
- Mounts Google Drive and fetches the repo
- Executes each phase notebook sequentially
- Captures output and tracks progress
- Generates final reports

### Python Orchestrator
**File:** `scripts/colab_auto_orchestrator.py`

Alternative programmatic approach:
- Can be called from command line
- Provides logging and error handling
- Useful for batch/automated runs

### Quick Setup Helper
**File:** `scripts/colab_quick_setup.py`

A minimal setup script to initialize everything in one cell.

## 🛠️ Advanced Usage

### Run from Command Line (Colab Terminal)
```bash
# Inside Colab
!python scripts/colab_auto_orchestrator.py
```

### Run Specific Phases Only
Edit the master notebook and comment out phases you don't need:
```python
# Phase 1: Data Preparation
# data_nb = notebook_dir / '1_data_preparation.ipynb'
# if data_nb.exists():
#     self.run_notebook(data_nb, "PHASE 1: Data Preparation", timeout=1800)

# Phase 2: Phase 1 Training (DoRA)
phase1_nb = notebook_dir / '2_phase1_training.ipynb'
if phase1_nb.exists():
    if not self.run_notebook(phase1_nb, "PHASE 2: DoRA Training", timeout=3600):
        logger.error("Phase 1 training failed")
        return False
```

### Resume from Checkpoint
The notebooks automatically load from existing checkpoints if found:
```python
# In any phase notebook, checkpoints are automatically detected
experiment = TrainingExperiment('phase3_checkpoint_latest.pt')
# If checkpoint exists, training resumes from epoch N instead of 0
```

## ✨ Key Improvements

### Before (February 19, 2026)
- Run notebook 1, wait for it to finish, manually run notebook 2, etc.
- Manual configuration tweaks between phases
- No automatic progress tracking
- Results scattered across different locations

### Now (February 20, 2026)
✅ **One notebook, zero manual steps**
✅ **Automatic error handling and recovery**
✅ **Real-time progress tracking**
✅ **Centralized output management**
✅ **Comprehensive logging**
✅ **Easy resumption from checkpoints**

## 📊 Monitoring Progress

The notebook prints real-time updates:
```
🚀 AADS-ULoRA Complete Auto-Training Pipeline
============================================================
Phase 0: ENVIRONMENT SETUP
  ✅ GPU: Tesla T4 (16.0GB)
  ✅ All dependencies verified

PHASE 1: Data Preparation
  Executing 1_data_preparation.ipynb...
  ✅ Data prep completed in 18.5 minutes

PHASE 2: DoRA Training
  Executing 2_phase1_training.ipynb...
  [====>................] Epoch 15/30 - Loss: 0.234
  ✅ Phase 1 completed in 187.3 minutes
```

## 🐛 Troubleshooting

### "No GPU detected"
- Runtime → Change runtime type → GPU (T4 or higher)

### "Out of memory"
- Phase notebooks have adaptive batch sizes
- Can manually adjust `config/colab.json` `batch_size`

### "Interrupted execution"
- Colab sessions timeout after 12 hours inactivity
- Training checkpoints auto-save every epoch
- Simply run the notebook again - it will resume

### "Google Drive quota"
- Need ~100GB free space on Drive
- Check in Google Drive settings

## 🎯 What's Produced

After complete training, you get:

1. **3 Fully Trained Adapters**
   - Phase 1: DoRA initialization and feature learning
   - Phase 2: SD-LoRA stable diffusion adaptation
   - Phase 3: CoNeC-LoRA congruent enhancement

2. **Validation Results**
   - Accuracy metrics by crop
   - OOD detection calibration
   - Confusion matrices

3. **Performance Reports**
   - HTML dashboard
   - CSV metrics
   - Training curves

4. **Complete Logs**
   - Training history per phase
   - Error logs
   - Resource usage

## 🚀 Next Steps

After training:
1. Download results from Google Drive
2. Export models to production format
3. Integrate with inference pipeline
4. Deploy to mobile/web applications

## 📚 Documentation

- [Complete README](README.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Training Manual](docs/user_guide/colab_training_manual.md)
- [Code Documentation](docs/development/development-setup.md)

---

**Happy Training! 🌾🤖**

For issues or questions, check the documentation or open an issue on GitHub.
