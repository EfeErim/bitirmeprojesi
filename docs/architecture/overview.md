# Architecture

The repo is intentionally narrow.

## Training

- `colab_notebooks/2_interactive_adapter_training.ipynb`
- `src/workflows/training.py`
- `src/adapter/independent_crop_adapter.py`
- `src/training/continual_sd_lora.py`
- `src/training/session.py`
- `src/training/validation.py`
- `src/training/types.py`
- `src/data/`
- `src/utils/data_loader.py`
- `scripts/colab_checkpointing.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_live_telemetry.py`
- `scripts/colab_notebook_helpers.py`

Training is split into four layers:

1. `TrainingWorkflow.run(...)` is the canonical orchestration entrypoint for the supported adapter-training flow.
2. `ContinualSDLoRATrainer` owns model initialization, LoRA wrapping, optimizer setup, batch stepping, snapshot/restore, OOD calibration, and adapter save/load, while `src/training/services/` holds shared metric/runtime/persistence helpers.
3. `ContinualTrainingSession` owns epoch/batch orchestration, resume, validation timing, checkpoint requests, best-metric updates, early stopping, observer events, and history accumulation.
4. `evaluate_model(...)` computes validation metrics outside the trainer loop, while notebook helpers persist reports, confusion matrices, and the metric gate artifact.

Notebook 2 now uses a two-stage dataset contract:

1. User input is a flat class-root directory `<root>/<class>/<images>`.
2. `prepare_runtime_dataset_layout(...)` materializes `data/runtime_notebook_datasets/<crop>/{continual,val,test}/<class>` and writes `split_manifest.json`.

The session emits stable observer payloads. `batch_end` exposes `loss`, and epoch/validation payloads carry the validation summary plus the history snapshot used for artifact persistence and best-checkpoint decisions.

Checkpoint payloads persist the normalized trainer contract, optimizer state, scheduler state, scaler state, best-metric state, optimizer-step counters, RNG state, and serialized OOD calibration state. Adapter bundles persist LoRA weights, classifier/fusion weights, and the public metadata contract in one place.

## Inference

- `src/workflows/inference.py`
- `src/pipeline/router_adapter_runtime.py`
- `src/pipeline/inference_payloads.py`
- `src/router/vlm_pipeline.py`
- `scripts/colab_router_adapter_inference.py`
- `colab_notebooks/1_router_adapter_inference.ipynb`

Inference is one path only:

1. `InferenceWorkflow.predict(...)` is the canonical entrypoint.
2. Router resolves the crop.
3. Runtime loads that crop adapter lazily.
4. Adapter returns diagnosis and OOD payload through a typed inference contract.
