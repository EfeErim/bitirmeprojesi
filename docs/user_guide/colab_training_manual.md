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
