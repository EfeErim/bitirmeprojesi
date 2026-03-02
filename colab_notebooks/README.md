# Colab Notebooks Map

## Current Status

- Active notebook workflow now has two canonical notebooks:
	- `1_crop_router_pipeline.ipynb` (image upload + VLM routing)
	- `2_interactive_adapter_training.ipynb` (widget-driven continual adapter training)

## Runtime Notes

- Both notebooks include embedded setup cells (repo root resolution, optional Drive mount, dependency install).
- Notebook 1 supports full VLM router profile selection (`fast`, `balanced`, `calibrated`, `leaf_fruit_production`).
- Notebook 2 includes dataset validation, live training progress callback visualization, OOD calibration, and adapter save.

## Archived Notebook Set (v6 superseded, 2026-03-02)

- Archive location: `colab_notebooks/archive/v6_superseded_2026-03-02/`

- `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
- `1_data_preparation.ipynb`
- `2_continual_sd_lora_training.ipynb`
- `5_testing_validation.ipynb`
- `6_performance_monitoring.ipynb`
- `colab_bootstrap.ipynb` (moved from repository root)

## Legacy Content

- Legacy v5.5 notebooks are archived under `colab_notebooks/archive/v5_legacy/`.
