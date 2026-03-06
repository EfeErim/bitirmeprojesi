# Colab Training

Use `colab_notebooks/2_interactive_adapter_training.ipynb`.

## Expected Dataset Layout

```text
data/<crop>/continual/<class>/*
data/<crop>/val/<class>/*
data/<crop>/test/<class>/*
```

## Outputs

- Adapter: `models/adapters/<crop>/continual_sd_lora_adapter/`
- Training checkpoints: `checkpoints/`
- Notebook artifacts: `outputs/colab_notebook_training/artifacts/`
- Telemetry: Drive or local spool from `scripts/colab_live_telemetry.py`

## Validation

Run before or after notebook updates:

```powershell
python scripts/validate_notebook_imports.py
python scripts/evaluate_dataset_layout.py --root data/<your_class_root_dataset>
```
