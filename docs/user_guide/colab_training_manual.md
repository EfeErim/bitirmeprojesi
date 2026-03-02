# Colab Training Manual (v6)

## Recommended Path

- Use `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` for one-click continual flow.

## Manual Stage Path

Run notebooks in order when you need per-stage control:

1. `1_data_preparation.ipynb`
2. `2_continual_sd_lora_training.ipynb`
3. `5_testing_validation.ipynb`
4. `6_performance_monitoring.ipynb`

## Repository Root Resolution

- Notebooks auto-detect repo root from common Colab and Drive locations.
- If no repo root is found, notebooks try auto-clone to `/content/bitirmeprojesi`.
- You can override behavior with:
  - `AADS_REPO_ROOT` or `REPO_ROOT`
  - `AADS_REPO_CLONE_TARGET`
  - `AADS_REPO_URL`
  - `AADS_DISABLE_AUTO_CLONE=1`
