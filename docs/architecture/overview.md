# Architecture

The repo is intentionally narrow.

## Training

- `colab_notebooks/2_interactive_adapter_training.ipynb`
- `src/adapter/independent_crop_adapter.py`
- `src/training/continual_sd_lora.py`
- `src/training/session.py`
- `src/training/validation.py`
- `src/training/types.py`
- `scripts/colab_checkpointing.py`
- `scripts/colab_live_telemetry.py`

Training is split into three layers:

1. `ContinualSDLoRATrainer` owns model initialization, LoRA wrapping, optimizer setup, batch stepping, snapshot/restore, OOD calibration, and adapter save/load.
2. `ContinualTrainingSession` owns epoch/batch orchestration, resume, validation timing, stop checks, observer events, and history accumulation.
3. `evaluate_model(...)` computes validation metrics outside the trainer loop.

Notebook 2 validates dataset layout, creates per-crop loaders, builds a session around one crop adapter trainer, calibrates OOD, and saves assets to `models/adapters/<crop>/continual_sd_lora_adapter/`.

## Inference

- `src/pipeline/router_adapter_runtime.py`
- `src/router/vlm_pipeline.py`
- `scripts/colab_router_adapter_inference.py`
- `colab_notebooks/1_router_adapter_inference.ipynb`

Inference is one path only:

1. Router resolves the crop.
2. Runtime loads that crop adapter lazily.
3. Adapter returns diagnosis and OOD payload.
