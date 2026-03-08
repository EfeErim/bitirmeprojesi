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
- `ood.ber_enabled`, `ood.ber_lambda_old`, `ood.ber_lambda_new`
- `ood.radial_l2_enabled`, `ood.radial_beta_range`, `ood.radial_beta_steps`
- `ood.sure_enabled`, `ood.sure_semantic_percentile`, `ood.sure_confidence_percentile`
- `ood.conformal_enabled`, `ood.conformal_alpha`
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
- SURE+ semantic and confidence thresholds (when enabled)

Detector-level calibration also stores:

- calibrated radial normalization $\beta$ (when enabled)
- conformal nonconformity quantile $\hat{q}$ (when enabled)

Inference then returns both the predicted class and an OOD payload built from:

- `mahalanobis_z`
- `energy_z`
- `ensemble_score`
- `class_threshold`
- `is_ood`

When enabled, payloads additionally include:

- `radial_beta`
- `sure_semantic_score`, `sure_confidence_score`
- `sure_semantic_ood`, `sure_confidence_reject`
- `conformal_set`, `conformal_set_size`, `conformal_coverage`

Typical tuning order for notebook users:

1. Set `training.continual.ood.threshold_factor` for baseline sensitivity.
2. Enable/disable SURE+ and adjust `sure_semantic_percentile` / `sure_confidence_percentile`.
3. Enable conformal prediction and set `conformal_alpha` for target coverage level.
4. Enable BER when continual old/new class energy separation needs stabilization.

## Outputs

Notebook 2 (`colab_notebooks/2_interactive_adapter_training.ipynb`) writes:

- Local outputs stay under `outputs/colab_notebook_training/`
  - adapter: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
  - notebook artifacts: `outputs/colab_notebook_training/artifacts/`
- Drive outputs stay under `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/`
  Default root: `/content/drive/MyDrive/aads_ulora/telemetry/<RUN_ID>/`
  - `artifacts/adapter/`
  - `artifacts/validation/`
  - `artifacts/best_checkpoint.json`
  - `artifacts/export_layout.json`
  - `artifacts/crop_info.json`
  - `checkpoints/`
- Meaning of each Drive path:
  - `artifacts/adapter/` holds the adapter exported from the best checkpoint
  - `artifacts/validation/` holds the validation metrics and plots for that best checkpoint
  - `artifacts/best_checkpoint.json` identifies which checkpoint was selected as best
  - `artifacts/export_layout.json` records the final local/Drive paths for the run
  - `artifacts/crop_info.json` records the crop name for the run
  - `checkpoints/` holds the rolling checkpoint history during training

Workflow / CLI training (`TrainingWorkflow.run(...)` and `python -m src.app.cli training ...`) writes:

- Adapter: `<output_dir>/continual_sd_lora_adapter/`
- Workflow metrics board: `<output_dir>/training_metrics/training/results.png`
- Workflow epoch metrics: `<output_dir>/training_metrics/training/results.csv`
- Workflow batch telemetry: `<output_dir>/training_metrics/training/batch_metrics.csv`
- Workflow confusion/report artifacts: `<output_dir>/training_metrics/validation/`

Inference default adapter lookup remains:

- `models/adapters/<crop>/continual_sd_lora_adapter/`

If your adapter was produced by Notebook 2, copy or move it from either the local notebook export or the Drive telemetry adapter export under `models/adapters/<crop>/`, or use inference `--adapter-root`.

## Validation

Run before or after notebook updates:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook_imports.py
.\.venv\Scripts\python.exe scripts/evaluate_dataset_layout.py --root data\<your_class_root_dataset>
```
