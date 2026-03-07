# Colab Training

Use `colab_notebooks/2_interactive_adapter_training.ipynb`.

## Dataset Contract

```text
<root>/<class>/<images>
```

Notebook 2 materializes the runtime split layout automatically:

```text
data/runtime_notebook_datasets/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

Each generated runtime dataset includes:

- `split_manifest.json` with source root, crop, seed, allowed classes, per-class counts, and split policy
- `_split_metadata.json` as the legacy compatibility mirror

## Training Controls

Normalized training controls live in `training.continual`:

- `adapter.lora_r`, `adapter.lora_alpha`, `adapter.lora_dropout`
- `adapter.target_modules_strategy`
- `fusion.layers`, `fusion.output_dim`, `fusion.dropout`, `fusion.gating`
- `seed`, `deterministic`
- `optimization.grad_accumulation_steps`
- `optimization.max_grad_norm`
- `optimization.mixed_precision`
- `optimization.label_smoothing`
- `optimization.scheduler`
- `ood.threshold_factor`
- `early_stopping`
- `evaluation.best_metric`
- `evaluation.emit_ood_gate`
- `evaluation.require_ood_for_gate`

Colab runtime-only controls live in `colab.training`:

- `num_workers`
- `pin_memory`
- `stdout_progress_batch_interval`
- `stdout_progress_min_interval_sec`
- `checkpoint_every_n_steps`
- `checkpoint_on_exception`

Legacy `checkpoint_interval` is still accepted as an alias for `checkpoint_every_n_steps`.

## What The Notebook Trains

Notebook 2 trains a continual SD-LoRA adapter rather than fine-tuning the full backbone.

The training stack is:

- frozen pretrained backbone
- LoRA adapters on selected transformer linear layers
- multi-scale feature fusion across configured backbone layers
- classifier head trained for the current crop classes

This keeps most backbone weights fixed and saves only the adapter bundle, classifier, fusion state, and metadata.

## OOD Behavior

After normal training, the workflow calibrates OOD statistics on known-class data.

Calibration stores per-class:

- fused-feature mean and variance for Mahalanobis scoring
- logit energy mean and standard deviation
- ensemble threshold derived from the calibrated score distribution

Inference then returns both the predicted class and an OOD payload built from:

- `mahalanobis_z`
- `energy_z`
- `ensemble_score`
- `class_threshold`
- `is_ood`

The notebook user usually only needs to tune `training.continual.ood.threshold_factor` and decide whether validation gates should require OOD metrics through `training.continual.evaluation.require_ood_for_gate`.

## Outputs

Notebook 2 (`colab_notebooks/2_interactive_adapter_training.ipynb`) writes:

- Adapter: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- Notebook artifacts: `outputs/colab_notebook_training/artifacts/`
- Validation metric gate: `outputs/colab_notebook_training/artifacts/validation/metric_gate.json`
- Training checkpoints + manifests: `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/checkpoints/` (default root `/content/drive/MyDrive/aads_ulora`)

Workflow / CLI training (`TrainingWorkflow.run(...)` and `python -m src.app.cli training ...`) writes:

- Adapter: `<output_dir>/continual_sd_lora_adapter/`
- Workflow metrics board: `<output_dir>/training_metrics/training/results.png`
- Workflow epoch metrics: `<output_dir>/training_metrics/training/results.csv`
- Workflow batch telemetry: `<output_dir>/training_metrics/training/batch_metrics.csv`
- Workflow confusion/report artifacts: `<output_dir>/training_metrics/validation/`

Inference default adapter lookup remains:

- `models/adapters/<crop>/continual_sd_lora_adapter/`

If your adapter was produced by Notebook 2, copy or move it under `models/adapters/<crop>/`, or use inference `--adapter-root`.

## Validation

Run before or after notebook updates:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook_imports.py
.\.venv\Scripts\python.exe scripts/evaluate_dataset_layout.py --root data\<your_class_root_dataset>
```
