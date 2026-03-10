# Architecture

This repo is intentionally narrow. The current project state is one adapter family, one workflow layer, one router runtime, and a small set of Colab helper surfaces.

## Configuration

Current config flow:

1. `src/core/config_manager.py` loads `config/base.json`.
2. Optional environment overrides such as `config/colab.json` are deep-merged on top.
3. `src/training/services/config_surface.py` normalizes the public `training.continual` surface.
4. Legacy top-level OOD aliases are backfilled into `training.continual.ood` and kept in sync.
5. `src/training/quantization.py` rejects prohibited 4-bit flags before the merged config is used.

Important files:

- `config/base.json`
- `config/colab.json`
- `src/core/config_manager.py`
- `src/training/services/config_surface.py`

## Training Stack

Current training path:

1. `src/workflows/training.py` is the canonical orchestration entrypoint.
2. `src/workflows/training_support.py` resolves loaders, class names, adapter initialization, and artifact payload shaping.
3. `src/adapter/independent_crop_adapter.py` owns the public adapter lifecycle and delegates the actual engine to the trainer.
4. `src/training/continual_sd_lora.py` owns backbone initialization, LoRA wrapping, optimization, prediction, and OOD calibration.
5. `src/training/session.py` runs epochs and emits observer events for telemetry and checkpoints.
6. `src/training/services/` persists plots, metric gates, OOD benchmark summaries, and readiness artifacts.

The supported adapter path is continual SD-LoRA only:

- frozen pretrained DINOv3 backbone
- LoRA adapters on selected transformer linear layers
- multi-scale feature fusion
- classifier head over the fused representation
- OOD state persisted with the exported adapter bundle

## Data Contracts

There are two dataset contracts in the current project:

- Notebook input contract:
  `<root>/<class>/<images>`

- Runtime training contract:
  `<data_dir>/<crop>/{continual,val,test[,ood]}/...`

`scripts/colab_dataset_layout.py` converts the flat class-root notebook input into the runtime split layout and writes:

- `split_manifest.json`
- `_split_metadata.json`

`src/data/datasets.py` maps workflow split `train` onto runtime split `continual`.

## Inference Stack

Current inference path:

1. `src/workflows/inference.py` exposes `InferenceWorkflow.predict(...)`.
2. `src/pipeline/router_adapter_runtime.py` loads the router lazily, chooses the crop, loads the per-crop adapter, and returns a typed result.
3. `src/router/vlm_pipeline.py` owns the router-side crop and part analysis.
4. `src/pipeline/inference_payloads.py` maps raw adapter output into the public payload.
5. `src/shared/contracts.py` defines the shared `InferenceResult`, `OODAnalysis`, and adapter metadata contracts.

Default adapter resolution:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

`crop_hint` bypasses router crop resolution. `part_hint` is optional metadata, not a separate adapter selector.

## OOD Stack

Current OOD behavior is part of the adapter runtime, not a separate model family.

The supported path is:

1. train on known classes
2. calibrate OOD after training
3. persist calibration with the adapter
4. score inference output with the calibrated OOD detector

Implemented pieces:

- detector logic: `src/ood/continual_ood.py`
- radial normalization: `src/ood/radial_normalization.py`
- SURE+ scoring: `src/ood/sure_scoring.py`
- conformal prediction: `src/ood/conformal_prediction.py`
- BER loss wrapper: `src/training/ber_loss.py`
- OOD calibration orchestration: `src/training/services/ood_calibration.py`
- readiness metrics and gates: `src/training/services/metrics.py`
- held-out fallback benchmark: `src/training/services/ood_benchmark.py`

Default readiness targets currently live in `DEFAULT_PLAN_TARGETS` inside `src/training/services/metrics.py`.

## Colab Support

Maintained Colab helpers:

- `scripts/colab_repo_bootstrap.py`
- `scripts/colab_live_telemetry.py`
- `scripts/colab_checkpointing.py`
- `scripts/colab_notebook_helpers.py`
- `scripts/colab_adapter_smoke_test.py`

These helpers cover:

- repo discovery or auto-clone
- dependency installation for notebooks
- Drive mounting and Hugging Face login checks
- local spool plus Drive telemetry sync
- rolling checkpoint management
- repo mirror exports for notebook runs
- notebook completion checks and optional runtime auto-disconnect
- direct adapter discovery and smoke prediction

## Artifact Flow

Workflow and CLI training write:

```text
<output_dir>/
  continual_sd_lora_adapter/
  training_metrics/
```

`training_metrics/` currently contains:

- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

Notebook 2 writes:

- local outputs under `outputs/colab_notebook_training/`
- repo mirrors under `runs/<RUN_ID>/`
- Drive telemetry under `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/`

Current adapter export detail:

- local notebook export: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- workflow export: `<output_dir>/continual_sd_lora_adapter/`
- current Notebook 2 Drive telemetry export: `artifacts/adapter_export/continual_sd_lora_adapter/`
- some smoke-test helper paths also accept `artifacts/adapter/`

## Validation Surfaces

The repo currently validates itself through:

- `scripts/validate_notebook_imports.py`
- `scripts/evaluate_dataset_layout.py`
- `scripts/benchmark_surfaces.py`
- `tests/unit/`
- `tests/integration/`
- `tests/colab/test_smoke_training.py`

## Related Docs

- [../user_guide/colab_training_manual.md](../user_guide/colab_training_manual.md)
- [../user_guide/ood_readiness_guide.md](../user_guide/ood_readiness_guide.md)
- [ood_recommendation.md](ood_recommendation.md)
- [experimental_leave_one_class_out_ood.md](experimental_leave_one_class_out_ood.md)
