# Google Colab Setup for AADS-ULoRA v5.5

This directory contains the Google Colab notebook for running the AADS-ULoRA v5.5 project in the cloud with GPU support.

## Quick Start

### Option 1: Manual Upload (Recommended)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Select `AADS-ULoRA_v5.5_Colab.ipynb`
4. Change runtime to GPU: **Runtime → Change runtime type → GPU** (select T4 or A100)
5. Run all cells: **Runtime → Run all**

### Option 2: Direct Link (if notebook is on GitHub)

If you've uploaded this notebook to GitHub, you can open it directly:

```
https://colab.research.google.com/github/yourusername/your-repo/blob/main/AADS-ULoRA_v5.5_Colab.ipynb
```

Replace `yourusername/your-repo` with your actual GitHub path.

## What the Notebook Does

1. **Checks GPU**: Verifies CUDA and GPU availability
2. **Mounts Google Drive**: Persists data between Colab sessions
3. **Installs dependencies**: All requirements from `requirements.txt`
4. **Prepares data**: Validates and organizes your dataset
5. **Training pipeline**: Runs all 3 phases (DoRA, SD-LoRA, CONEC-LoRA)
6. **Evaluation**: Tests model performance
7. **Visualization**: Generates plots and metrics
8. **API & Demo**: Launches FastAPI server and Gradio UI

## Data Preparation

Before training, you need to upload your dataset to Colab. The expected structure:

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

### Upload Data Options:

1. **Upload via Colab file browser** (left sidebar)
2. **Copy from Google Drive** if already stored there:
   ```python
   shutil.copytree('/content/drive/MyDrive/path/to/data', './data', dirs_exist_ok=True)
   ```
3. **Download from cloud** using `!wget` or `!gdown`

## Important Notes

- **Session Timeout**: Colab disconnects after ~12 hours or inactivity. Save checkpoints to Drive regularly.
- **GPU Memory**: T4 provides ~15GB, A100 provides ~50GB. Adjust batch size if you get OOM errors.
- **Storage**: Colab has ~80GB temporary storage. Use Drive for persistence.
- **First Run**: Dependency installation may take 10-15 minutes.
- **Usage Limits**: Free tier has limits. Consider Colab Pro for extended sessions.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--batch_size` (try 8 or 4) |
| `No module named 'src'` | Ensure `pip install -e .` completed successfully |
| `Data not found` | Verify `./data/` directory exists with proper structure |
| `Training too slow` | Confirm GPU is active (check `nvidia-smi` output) |
| `Session disconnected` | Save checkpoints to Drive; reload and resume from last checkpoint |

## Customization

You can modify the notebook to:

- Train only specific crops (pepper, corn)
- Skip training phases (start from checkpoint)
- Adjust hyperparameters (learning rate, epochs, batch size)
- Run only evaluation without training
- Use different model architectures

## File Locations in Colab

- **Project root**: `/content/drive/MyDrive/aads-ulora-v5.5/` (if using Drive) or `/content/aads-ulora-v5.5/`
- **Data**: `./data/` (relative to project root)
- **Outputs**: `./outputs/` (contains checkpoints, logs, visualizations)
- **Logs**: `./outputs/*/logs/` for TensorBoard/WandB

## Next Steps

1. Upload this notebook to Colab
2. Mount Google Drive
3. Upload your dataset
4. Run all cells sequentially
5. Monitor training progress
6. Access results in `./outputs/`

## Support

For issues specific to the AADS-ULoRA project, refer to:
- `README.md` - Project documentation
- `docs/` - Detailed architecture and API docs
- `PROJECT_FIX_SUMMARY.md` - Known issues and fixes

For Colab-specific issues, see Google's [Colab FAQ](https://research.google.com/colab/faq.html).
