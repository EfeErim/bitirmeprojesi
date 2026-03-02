# Colab Notebooks Map (v6)

## Canonical Start Path

- Primary: `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
- Manual path: `1_data_preparation.ipynb` -> `2_continual_sd_lora_training.ipynb` -> `5_testing_validation.ipynb` -> `6_performance_monitoring.ipynb`

## Bootstrap Notes

- Each active notebook includes repository root auto-discovery in its first code cell.
- If discovery fails, notebooks can auto-clone to `/content/bitirmeprojesi` by default.
- Useful environment variables:
  - `AADS_REPO_ROOT` or `REPO_ROOT`: force an existing clone path.
  - `AADS_REPO_CLONE_TARGET`: change auto-clone destination.
  - `AADS_REPO_URL`: change clone source URL.
  - `AADS_DISABLE_AUTO_CLONE=1`: disable auto-clone fallback.

## Notebook Intent Matrix

| Notebook | Use When | Priority |
|---|---|---|
| `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` | You want full continual pipeline orchestration | Primary |
| `1_data_preparation.ipynb` | You need manual dataset prep and checks | Secondary |
| `2_continual_sd_lora_training.ipynb` | You run continual SD-LoRA training | Primary |
| `5_testing_validation.ipynb` | You run validation/evaluation | Secondary |
| `6_performance_monitoring.ipynb` | You review monitoring outputs | Secondary |

## Legacy Content

Legacy v5.5 notebooks are archived under `colab_notebooks/archive/v5_legacy/`.
