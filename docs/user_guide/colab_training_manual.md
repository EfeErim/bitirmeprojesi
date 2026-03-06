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

- `seed`, `deterministic`
- `optimization.grad_accumulation_steps`
- `optimization.max_grad_norm`
- `optimization.mixed_precision`
- `optimization.label_smoothing`
- `optimization.scheduler`
- `early_stopping`
- `evaluation.best_metric`
- `evaluation.emit_ood_gate`
- `evaluation.require_ood_for_gate`

Colab runtime-only controls live in `colab.training`:

- `num_workers`
- `pin_memory`
- `stdout_progress_batch_interval`
- `checkpoint_every_n_steps`
- `checkpoint_on_exception`

Legacy `checkpoint_interval` is still accepted as an alias for `checkpoint_every_n_steps`.

## Outputs

- Adapter: `models/adapters/<crop>/continual_sd_lora_adapter/`
- Training checkpoints: `checkpoints/`
- Notebook artifacts: `outputs/colab_notebook_training/artifacts/`
- Validation metric gate: `outputs/colab_notebook_training/artifacts/validation/metric_gate.json`
- Telemetry: Drive or local spool from `scripts/colab_live_telemetry.py`

## Validation

Run before or after notebook updates:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook_imports.py
.\.venv\Scripts\python.exe scripts/evaluate_dataset_layout.py --root data\<your_class_root_dataset>
```
