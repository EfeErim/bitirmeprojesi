# AADS v6

This repo now keeps only two supported flows:

- `colab_notebooks/2_interactive_adapter_training.ipynb` for Colab training
- `colab_notebooks/1_router_adapter_inference.ipynb` or `scripts/colab_router_adapter_inference.py` for router-driven inference

The canonical app entrypoints are:

- `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`

The workflow layer delegates to `src/pipeline/router_adapter_runtime.py`, `src/training/continual_sd_lora.py`, and `src/adapter/independent_crop_adapter.py`.

Experimental design notes that are not part of the supported workflow live under `docs/architecture/`. The current OOD prototype note is `docs/architecture/experimental_leave_one_class_out_ood.md`.

## Core Model Behavior

The training engine is a continual SD-LoRA adapter pipeline:

- a frozen vision backbone is loaded once
- LoRA adapters are attached to the selected transformer linear layers
- multi-scale backbone features are fused into one feature vector
- a classifier head is trained on top of that fused representation

The same adapter bundle also carries the runtime OOD state:

- OOD calibration is run after normal training on the known classes
- optional Bi-directional Energy Regularization (BER) can be applied during training as an additive loss wrapper
- radial L2 feature normalization can auto-tune a calibration-time $\beta$ and rescale features before OOD stats are computed
- per-class calibration stores feature mean/variance, energy statistics, and SURE+ thresholds
- inference scores each prediction with a weighted ensemble of Mahalanobis z-score and energy z-score, then applies SURE+ semantic/confidence rejection
- optional split conformal prediction adds a set-valued guarantee (`conformal_set`) in addition to the top-1 diagnosis
- class-specific thresholds and SURE+/conformal calibration artifacts are persisted with the adapter

The architecture guide expands these details in `docs/architecture/overview.md`, and the experimental OOD evaluation note lives in `docs/architecture/experimental_leave_one_class_out_ood.md`.

Key controls are under `training.continual.ood` in `config/base.json`, including:

- `ber_enabled`, `ber_lambda_old`, `ber_lambda_new`
- `radial_l2_enabled`, `radial_beta_range`, `radial_beta_steps`
- `sure_enabled`, `sure_semantic_percentile`, `sure_confidence_percentile`
- `conformal_enabled`, `conformal_alpha`

## Quick Start

Install minimal local dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the lightweight validation surface:

```powershell
python scripts/validate_notebook_imports.py
pytest tests/unit tests/colab/test_smoke_training.py -q
pytest tests/integration -q --runintegration
python scripts/benchmark_surfaces.py
```

You can also use the thin CLI facade:

```powershell
python -m src.app.cli inference path\to\image.jpg --config-env colab
python -m src.app.cli training tomato data\runtime_notebook_datasets outputs\training_run --config-env colab
```

Training surfaces and output paths differ by entrypoint:

- Notebook 2 (`colab_notebooks/2_interactive_adapter_training.ipynb`)
	- adapter: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
	- notebook artifacts: `outputs/colab_notebook_training/artifacts/`
	- checkpoint stream: `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/checkpoints/` (or `AADS_DRIVE_LOG_ROOT` override)
- Workflow / CLI (`TrainingWorkflow.run(...)`, `python -m src.app.cli training ...`)
	- adapter: `<output_dir>/continual_sd_lora_adapter/`
	- artifacts: `<output_dir>/training_metrics/`
		- `training/results.png`
		- `training/results.csv`
		- `training/batch_metrics.csv`
		- `validation/confusion_matrix.png`
		- `validation/confusion_matrix_normalized.png`
		- `validation/classification_report.json`
		- `validation/metric_gate.json`
		- `training/summary.json`

## Colab

- Root Colab dependencies live in `requirements_colab.txt`
- Notebook bootstrap helpers live in `scripts/colab_repo_bootstrap.py`
- Telemetry and checkpoints live in `scripts/colab_live_telemetry.py` and `scripts/colab_checkpointing.py`
- Notebook and script wrappers now sit on top of the workflow layer instead of owning the core orchestration

## Adapter Layout

Inference expects adapters under:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

If a training run writes the adapter elsewhere (for example Notebook 2 under `outputs/colab_notebook_training/`), move or copy it into `models/adapters/<crop>/`, or pass `--adapter-root` to inference entrypoints.

## End-to-End Path Map

Notebook-to-inference handoff in 3 steps:

1. Train in `colab_notebooks/2_interactive_adapter_training.ipynb`.
2. Take adapter output from `outputs/colab_notebook_training/continual_sd_lora_adapter/`.
3. Deploy either by:
	 - copying to `models/adapters/<crop>/continual_sd_lora_adapter/` (default inference lookup), or
	 - keeping custom location and running inference with `--adapter-root <parent_of_crop_dirs>`.

Example default deploy layout:

```text
models/adapters/
	tomato/
		continual_sd_lora_adapter/
			adapter_meta.json
			...
```
