# Colab Notebooks Map

Use this index to choose the right notebook quickly.

## Main Entrypoints

- `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`  
  One-click end-to-end training (recommended for full training runs).
- `7_VLM_ROUTER_ONECLICK.ipynb`  
  One-click router-focused VLM workflow.
- `TEST_VLM_ROUTER.ipynb`  
  Interactive/manual router checks and debugging.

## Phase-by-Phase Training

- `1_data_preparation.ipynb` → prepares and validates dataset structure.
- `2_phase1_training.ipynb` → DoRA phase training.
- `3_phase2_training.ipynb` → SD-LoRA phase training.
- `4_phase3_training.ipynb` → CoNeC-LoRA phase training.
- `5_testing_validation.ipynb` → model checks and validation passes.
- `6_performance_monitoring.ipynb` → performance and monitoring review.

## Setup Notebook

- `colab_bootstrap.ipynb`  
  Colab environment bootstrap helper.

## Related Files

- `../scripts/colab_auto_orchestrator.py` orchestrates one-click notebook flow.
- `../scripts/colab_setup_dependencies.py` installs notebook dependencies.
- `../requirements_colab.txt` and `requirements_colab.txt` contain Colab dependency sets.
