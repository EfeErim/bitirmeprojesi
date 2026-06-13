# AADS v6

AADS v6 is a narrow plant-disease training and inference repository. The handoff surface is intentionally small:

1. Prepare grouped datasets for training
2. Train crop-specific SD-LoRA adapters
3. Run router-guided inference and readiness checks

If you are new to the project, start with [docs/README.md](docs/README.md), then read [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md) and [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md) as needed.

## Canonical Surfaces

| Surface | Purpose |
|---|---|
| [Notebook 0](colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb) | Grouped dataset preparation |
| [Notebook 1](colab_notebooks/1_identify_crop_part_with_router.ipynb) | Router crop and part identification |
| [Notebook 2](colab_notebooks/2_train_continual_sd_lora_adapter.ipynb) | Continual SD-LoRA training |
| [Notebook 3](colab_notebooks/3_validate_exported_adapter_directly.ipynb) | Direct adapter validation |
| [Notebook 5](colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb) | Router calibration |
| [Notebook 8](colab_notebooks/8_auto_router_adapter_prediction.ipynb) | Router-to-adapter inference |
| [Notebook 16](colab_notebooks/16_ablation_dual_view_inference.ipynb) | ROI/bbox evidence-gate ablation |
| [Training workflow](src/workflows/training.py) | Canonical training entrypoint |
| [Inference workflow](src/workflows/inference.py) | Canonical inference entrypoint |

Notebook 4 is a convenience wrapper for direct adapter smoke testing. It is not a separate canonical surface. Notebook 6 is a batch-training regression surface used to exercise the maintained Notebook 2 cell contract, and Notebook 7 is a prepared-runtime OOD/OE quality audit surface.

Notebook 8 is a thin Colab wrapper over Notebook 1's router cells plus the canonical inference workflow for single-image router-to-adapter prediction.

Notebook 16 is the maintained ROI/bbox ablation surface. It keeps full-image adapter prediction as the final decision and uses router/Grounding DINO bbox evidence only for review flags. Historical ROI ablation reports remain under `docs/ablation_results/<condition>/`.

## What The Repo Covers

- Dataset auditing and materialization for Notebook 0
- Continual adapter training for one crop at a time
- OOD calibration and readiness reporting
- Router-driven inference with adapter lookup
- Direct adapter validation and router calibration notebooks
- Validation-only notebook surfaces used by tests and maintenance checks

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

## Handoff Boundaries

The following paths are generated or local-only surfaces and should not be treated as client-facing source:

- `runs/`
- `models/adapters/`
- `outputs/`
- `data/prepared_runtime_datasets/`
- `.runtime_tmp/`

The repo root may also contain exported plots or analysis files. Treat those as reports or evidence artifacts unless a file is explicitly called out as canonical documentation.

## Repository layout

- `src/`: workflow, pipeline, and runtime code
- `scripts/`: validation tools and notebook helpers
- `config/`: shipped JSON configuration
- `docs/`: maintained documentation
- `colab_notebooks/`: maintained notebook surfaces
- `tests/`: unit, integration, and smoke coverage
- `data/`: tracked examples and local staging roots; prepared runtime datasets are generated locally

## Read next

- [docs/README.md](docs/README.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md)
- [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md)

## License

See [LICENSE](LICENSE) for terms.

## References

- Hendrycks, D., Mazeika, M., & Dietterich, T. (2018). Deep Anomaly Detection with Outlier Exposure. arXiv. https://arxiv.org/abs/1812.04606
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv. https://arxiv.org/abs/2107.07511
