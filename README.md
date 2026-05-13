# AADS v6

AADS v6 is a narrow plant-disease training and inference repository. It focuses on three maintained surfaces:

1. Preparing grouped datasets for training
2. Training crop-specific SD-LoRA adapters
3. Running router-guided inference and readiness checks

If you are new to the project, start with [docs/README.md](docs/README.md), then read [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md) and [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md) as needed.

## Maintained entry points

- Notebook 0: [colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb](colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb)
- Notebook 1: [colab_notebooks/1_identify_crop_part_with_router.ipynb](colab_notebooks/1_identify_crop_part_with_router.ipynb)
- Notebook 2: [colab_notebooks/2_train_continual_sd_lora_adapter.ipynb](colab_notebooks/2_train_continual_sd_lora_adapter.ipynb)
- Notebook 3: [colab_notebooks/3_validate_exported_adapter_directly.ipynb](colab_notebooks/3_validate_exported_adapter_directly.ipynb)
- Notebook 5: [colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb](colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb)
- Training workflow: [src/workflows/training.py](src/workflows/training.py)
- Inference workflow: [src/workflows/inference.py](src/workflows/inference.py)

Notebook 4 is kept as a convenience wrapper for direct adapter smoke testing. It is not a separate canonical surface.

## What the repo covers

- Dataset auditing and materialization for Notebook 0
- Continual adapter training for one crop at a time
- OOD calibration and readiness reporting
- Router-driven inference with adapter lookup
- Direct adapter validation and router calibration notebooks

## Core concepts

- `crop`: plant type such as `tomato` or `grape`
- `class`: one label the model can predict for a crop
- `adapter`: exported crop-specific model bundle
- `router`: crop and part selector used before adapter loading
- `OOD`: out-of-distribution input outside the supported class set
- `readiness`: final deployment verdict written to `production_readiness.json`

## Setup

On Windows, use the repo launcher so the local `.venv` is preferred:

```powershell
.\scripts\python.cmd
```

Create or reuse the virtual environment:

```powershell
.\scripts\python.cmd -m venv .venv
```

Install dependencies:

```powershell
.\scripts\python.cmd -m pip install --upgrade pip
.\scripts\python.cmd -m pip install -r requirements.txt
.\scripts\python.cmd -m pip install -r requirements-dev.txt
```

Run the main validation checks:

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/validate_config_schema.py
pytest tests/unit tests/colab/test_smoke_training.py -q
```

For broader verification, also run:

```powershell
pytest tests/integration -q --runintegration
.\scripts\python.cmd scripts/benchmark_surfaces.py
```

## Data layout

Notebook 0 starts from a flat class-root layout:

```text
<root>/<class>/<images>
```

Notebook 2 and the workflow code train from the prepared runtime layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
```

Prepared runtime datasets are written under `data/prepared_runtime_datasets/`. Reusable OOD pools live under `data/ood_dataset/`.

## Training and inference

The canonical training entry point is `TrainingWorkflow.run(...)` in [src/workflows/training.py](src/workflows/training.py). The canonical inference entry point is `InferenceWorkflow.predict(...)` in [src/workflows/inference.py](src/workflows/inference.py).

Training loads `config/base.json`, optionally merges `config/colab.json`, trains the adapter, calibrates OOD behavior, and writes the exported adapter plus readiness artifacts.

Inference loads the router first. If the router cannot make a reliable decision, the runtime returns an abstaining result instead of forcing a crop or class prediction. When a crop is accepted, the matching adapter is loaded and the response includes the structured router result together with calibrated adapter output.

## Default paths

The default adapter deployment path is:

```text
models/adapters/<crop>/<part>/continual_sd_lora_adapter/
```

Workflow and notebook runs may also export adapters under `runs/` and `outputs/`. Those locations are for generated artifacts, not maintained source.

## Repository layout

- `src/`: workflow, pipeline, and runtime code
- `scripts/`: validation tools and notebook helpers
- `config/`: shipped JSON configuration
- `docs/`: maintained documentation
- `colab_notebooks/`: maintained notebook surfaces
- `tests/`: unit, integration, and smoke coverage
- `data/`: tracked examples and local staging roots

## Read next

- [docs/README.md](docs/README.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md)
- [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md)

## License

See [LICENSE](LICENSE) for terms.
