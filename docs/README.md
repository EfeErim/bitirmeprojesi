# Documentation Map

This repo keeps a small set of maintained Markdown docs. Read them in this order.

## Primary Docs

- [../README.md](../README.md)
  Repo scope, maintained surfaces, quick start, output locations, and deployment paths.

- [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)
  Notebook 2 and Notebook 3 operations, dataset contracts, Colab runtime controls, and adapter handoff.

- [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
  How readiness is computed, when fallback benchmarking runs, and how to interpret `production_readiness.json`.

- [architecture/overview.md](architecture/overview.md)
  Current code map for config, training, inference, Colab helpers, and artifact writers.

## Supporting Notes

- [architecture/ood_recommendation.md](architecture/ood_recommendation.md)
  Forward-looking engineering recommendation note for improving OOD quality.

- [architecture/experimental_leave_one_class_out_ood.md](architecture/experimental_leave_one_class_out_ood.md)
  Historical design note for the earlier held-out OOD prototype.

## Current Source Of Truth

When docs and older notes disagree, prefer:

1. `src/workflows/`, `src/core/config_manager.py`, and `src/shared/contracts.py`
2. `scripts/` helpers and validators
3. `tests/` coverage
4. historical architecture notes

## Validation Surfaces

These scripts reflect the current maintained surfaces:

- `python scripts/validate_notebook_imports.py`
- `python scripts/evaluate_dataset_layout.py --root <flat_class_root>`
- `python scripts/benchmark_surfaces.py`

## Tracked Vs Local Files

Tracked:

- `src/`, `tests/`, `scripts/`, `config/`
- `docs/` and the root `README.md`
- `colab_notebooks/*.ipynb`
- root dependency files such as `requirements.txt` and `requirements_colab.txt`

Local and generated:

- `runs/<RUN_ID>/`
- `models/adapters/<crop>/continual_sd_lora_adapter/`
- `outputs/`
- `.runtime_tmp/`, `.tmp*/`, caches, and virtualenvs

## Similar-Looking Paths

- `requirements_colab.txt` is the canonical Colab dependency list.
- `colab_notebooks/requirements_colab.txt` is a wrapper file so notebook-local bootstrap can resolve the root requirements file correctly.
- Notebook 2 currently exports Drive adapter assets under `artifacts/adapter_export/continual_sd_lora_adapter/`.
- Some helper surfaces also accept the older `artifacts/adapter/` layout.
