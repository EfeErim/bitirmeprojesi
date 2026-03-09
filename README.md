# AADS v6

This repo now keeps three supported notebook/runtime flows:

- `colab_notebooks/2_interactive_adapter_training.ipynb` for Colab training
- `colab_notebooks/1_router_adapter_inference.ipynb` or `scripts/colab_router_adapter_inference.py` for router-driven inference
- `colab_notebooks/3_adapter_smoke_test.ipynb` for direct adapter validation on top of the DINO backbone

The canonical app entrypoints are:

- `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`

The workflow layer delegates to `src/pipeline/router_adapter_runtime.py`, `src/training/continual_sd_lora.py`, and `src/adapter/independent_crop_adapter.py`.

User-facing OOD behavior, dataset layout, fallback benchmarking, and readiness outputs are documented in [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md). Historical design notes live under `docs/architecture/`, including `docs/architecture/experimental_leave_one_class_out_ood.md`.

## Core Model Behavior

The training engine is a continual SD-LoRA adapter pipeline:

- a frozen vision backbone is loaded once
- LoRA adapters are attached to the selected transformer linear layers
- multi-scale backbone features are fused into one feature vector
- a classifier head is trained on top of that fused representation

The same adapter bundle also carries the runtime OOD state:

- OOD calibration is run after normal training on the known classes
- when the extended OOD stack is enabled, calibration materializes one feature/logit snapshot and reuses it across radial, SURE+, and conformal phases instead of rescanning the loader repeatedly
- optional experimental Bi-directional Energy Regularization (BER) can be applied during training as an additive loss wrapper
- radial L2 feature normalization can auto-tune a calibration-time $\beta$ and rescale features before OOD stats are computed
- per-class calibration stores feature mean/variance, energy statistics, and SURE+ thresholds
- inference scores each prediction with a weighted ensemble of Mahalanobis z-score and energy z-score, then applies SURE+ semantic/confidence rejection
- optional split conformal prediction adds a set-valued guarantee (`conformal_set`) in addition to the top-1 diagnosis
- class-specific thresholds and SURE+/conformal calibration artifacts are persisted with the adapter

The architecture guide expands these details in [docs/architecture/overview.md](docs/architecture/overview.md), and the user-facing OOD workflow guide lives in [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md).

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
	- local outputs stay under `outputs/colab_notebook_training/`
		- adapter: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
		- notebook artifacts: `outputs/colab_notebook_training/artifacts/`
	- when the run finishes, non-checkpoint outputs are mirrored into `runs/<RUN_ID>/`
		- notebook export: `runs/<RUN_ID>/notebooks/2_interactive_adapter_training.executed.ipynb`
		- local notebook outputs: `runs/<RUN_ID>/outputs/colab_notebook_training/`
		- telemetry logs and artifacts: `runs/<RUN_ID>/telemetry/`
		- checkpoint metadata only: `runs/<RUN_ID>/checkpoint_state/` (manifests/index, no `checkpoints/` tree)
	- Drive outputs stay under `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/` (or `AADS_DRIVE_LOG_ROOT/telemetry/<RUN_ID>/`)
		- `artifacts/adapter/`
		- `artifacts/validation/`
		- `artifacts/best_checkpoint.json`
		- `artifacts/export_layout.json`
		- `artifacts/crop_info.json`
		- `checkpoints/`
	- simple rule:
		- `artifacts/adapter/` = adapter from the best checkpoint
		- `artifacts/validation/` = metrics for that best checkpoint
		- `artifacts/best_checkpoint.json` = which checkpoint was chosen as best
		- `artifacts/export_layout.json` = saved path summary for the run
		- `artifacts/crop_info.json` = crop name marker for the run
		- `checkpoints/` = rolling checkpoint saves during training
- Workflow / CLI (`TrainingWorkflow.run(...)`, `python -m src.app.cli training ...`)
	- adapter: `<output_dir>/continual_sd_lora_adapter/`
	- artifacts: `<output_dir>/training_metrics/`
		- `training/results.png`
		- `training/results.csv`
		- `training/batch_metrics.csv`
		- `validation/confusion_matrix.png`
		- `validation/confusion_matrix_normalized.png`
		- `validation/classification_report.json`
		- `validation/metric_gate.json` (`accuracy`, `ood_auroc`, `ood_false_positive_rate` as FPR@95TPR, plus optional `sure_ds_f1` / `conformal_empirical_coverage`)
		- `training/summary.json`

## Colab

- Root Colab dependencies live in `requirements_colab.txt`
- Notebook bootstrap helpers live in `scripts/colab_repo_bootstrap.py`
- Notebooks 1 and 2 now resolve `HF_TOKEN` from env or Colab secrets in their second code cell and validate the login before model access
- Notebook 2 now mirrors all non-checkpoint run outputs into `runs/<RUN_ID>/` inside the repo
- Telemetry and checkpoints live in `scripts/colab_live_telemetry.py` and `scripts/colab_checkpointing.py`
- Notebook and script wrappers now sit on top of the workflow layer instead of owning the core orchestration
- Adapter smoke-test helpers live in `scripts/colab_adapter_smoke_test.py`

## Adapter Smoke Test

- Use `colab_notebooks/3_adapter_smoke_test.ipynb` to load one trained crop adapter directly, inspect its metadata, and run smoke predictions without the router.
- By default the notebook searches configured Drive roots for adapter bundles, lists the discovered candidates, and lets you choose which adapter to load.
- `ADAPTER_DIR` is the forgiving manual override. It can point to a direct adapter asset directory, a parent export directory, a telemetry run directory, a telemetry `artifacts/` directory, or directly to `adapter_meta.json`.
- Typical `ADAPTER_DIR` examples:
  - `outputs/colab_notebook_training/continual_sd_lora_adapter/`
  - `outputs/colab_notebook_training/`
  - `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/`
  - `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/artifacts/`
  - `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/artifacts/adapter/adapter_meta.json`
- `ADAPTER_ROOT` is different: it should be the parent of crop directories, usually `models/adapters/`, so the notebook resolves `<ADAPTER_ROOT>/<crop>/continual_sd_lora_adapter/`.
- `IMAGE_PATH` must be one image file. `BATCH_IMAGE_DIR` must be one directory containing image files.
- The notebook includes:
  - one-image direct prediction for quick adapter verification
  - an optional folder pass that summarizes predicted labels, OOD decisions, and failed files

## Adapter Layout

Inference expects adapters under:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

If a training run writes the adapter elsewhere (for example Notebook 2 under `outputs/colab_notebook_training/` or Drive telemetry under `telemetry/<RUN_ID>/artifacts/adapter/`), move or copy it into `models/adapters/<crop>/`, or pass `--adapter-root` to inference entrypoints.

## End-to-End Path Map

Notebook-to-inference handoff in 3 steps:

1. Train in `colab_notebooks/2_interactive_adapter_training.ipynb`.
2. Take adapter output from either `outputs/colab_notebook_training/continual_sd_lora_adapter/` or `telemetry/<RUN_ID>/artifacts/adapter/`.
3. Optionally verify the adapter directly in `colab_notebooks/3_adapter_smoke_test.ipynb`.
4. Deploy either by:
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
