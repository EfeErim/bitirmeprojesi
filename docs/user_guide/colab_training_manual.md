# Colab Training Manual (v6)

## Recommended Path

- Run `colab_notebooks/1_crop_router_pipeline.ipynb` for image upload + VLM routing.
- Run `colab_notebooks/2_interactive_adapter_training.ipynb` for parameterized continual adapter training.
- Use `colab_notebooks/README.md` for status and archive references.

## Active Two-Notebook Flow

1. Open `1_crop_router_pipeline.ipynb`.
2. Upload image(s), choose runtime profile, and validate router detections.
3. Open `2_interactive_adapter_training.ipynb`.
4. Validate dataset path, initialize training engine, run training with live progress.
5. Calibrate OOD and save adapter outputs.

### Runtime Notes (Current)

- Notebook bootstrap logic is now shared via `scripts/colab_repo_bootstrap.py`.
- Notebook artifact/checkpoint plumbing is centralized via `scripts/colab_notebook_helpers.py`.
- Notebook 2 class alignment now uses `config/plant_taxonomy.json` (`crop_specific_diseases`) instead of hardcoded class maps.
- Notebook 1 pipeline execution explicitly disables duplicate analyzer routing before full pipeline call.
- Notebook 2 OOD calibration status widget is displayed explicitly after calibration completes.

### Telemetry, Artifacts, and Recovery

- Notebook telemetry is written continuously to Drive with local spool fallback.
- Default root: `${AADS_DRIVE_LOG_ROOT:-/content/drive/MyDrive/aads_ulora}` under `telemetry/<run_id>/`.
- Notebook 2 exposes resume controls (`Start fresh` / `Resume latest`) before training.
- Checkpoints are saved at epoch boundaries, every 200 batches, and on exception.
- Reconnect troubleshooting runbook: `docs/user_guide/colab_reconnect_investigation.md`.
- Validation artifacts include:
  - `classification_report.txt/json`
  - `confusion_matrix.npy/csv/png`
  - `confusion_matrix_normalized.png`
  - per-class metric CSV

## Superseded Notebook Set

The previous manual stage notebooks are archived/superseded:

1. `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
2. `1_data_preparation.ipynb`
3. `2_continual_sd_lora_training.ipynb`
4. `5_testing_validation.ipynb`
5. `6_performance_monitoring.ipynb`
6. `colab_bootstrap.ipynb`

Archive location: `colab_notebooks/archive/v6_superseded_2026-03-02/`

## Repository Root Resolution

- Notebooks auto-detect repo root from common Colab and Drive locations.
- If no repo root is found, notebooks try auto-clone to `/content/bitirmeprojesi`.
- You can override behavior with:
  - `AADS_REPO_ROOT` or `REPO_ROOT`
  - `AADS_REPO_CLONE_TARGET`
  - `AADS_REPO_URL`
  - `AADS_DISABLE_AUTO_CLONE=1`
