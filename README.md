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
- Notebook 8: [colab_notebooks/8_auto_router_adapter_prediction.ipynb](colab_notebooks/8_auto_router_adapter_prediction.ipynb)
- Notebook 9 presentation demo: [colab_notebooks/9_presentation_recording_demo.ipynb](colab_notebooks/9_presentation_recording_demo.ipynb)
- Notebooks 10-14 ROI ablations: [colab_notebooks/10_ablation_full_image_baseline.ipynb](colab_notebooks/10_ablation_full_image_baseline.ipynb), [colab_notebooks/11_ablation_primary_roi_inference.ipynb](colab_notebooks/11_ablation_primary_roi_inference.ipynb), [colab_notebooks/12_ablation_hybrid_roi_fallback.ipynb](colab_notebooks/12_ablation_hybrid_roi_fallback.ipynb), [colab_notebooks/13_ablation_roi_trained_adapter.ipynb](colab_notebooks/13_ablation_roi_trained_adapter.ipynb), [colab_notebooks/14_ablation_mixed_full_roi_training.ipynb](colab_notebooks/14_ablation_mixed_full_roi_training.ipynb)
- Training workflow: [src/workflows/training.py](src/workflows/training.py)
- Inference workflow: [src/workflows/inference.py](src/workflows/inference.py)

Notebook 4 is kept as a convenience wrapper for direct adapter smoke testing. It is not a separate canonical surface. Notebook 6 is a batch-training regression surface used to exercise the maintained Notebook 2 cell contract, and Notebook 7 is a prepared-runtime OOD/OE quality audit surface. Notebook 8 is a thin Colab wrapper over Notebook 1's router cells plus the canonical inference workflow for single-image router-to-adapter prediction.

Notebook 9 is a recording-oriented presentation wrapper over Notebook 8. Its preview cell runs the same canonical inference path once before recording, then its render-only recording cell immediately displays an audience-facing explanation of SAM3 region proposals, BioCLIP-2.5 routing, the safety gate, specialist adapter loading, the model prediction, and the OOD assessment.

Notebooks 10-12 are inference-only part-aware SAM box ROI ablations. They compare full-image baseline, primary ROI crop, and hybrid ROI fallback behavior through `scripts/colab_roi_ablation.py` without changing production inference. Notebooks 13-14 reserve the second-phase ROI-training and mixed full+ROI training ablation contracts. Reports are written under `docs/ablation_results/<condition>/` so Colab-produced results can be committed and viewed on GitHub.

## What the repo covers

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

## References

- Hendrycks, D., Mazeika, M., & Dietterich, T. (2018). Deep Anomaly Detection with Outlier Exposure. arXiv. https://arxiv.org/abs/1812.04606
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. https://www.deeplearningbook.org/
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv. https://arxiv.org/abs/2107.07511
