# Architecture Overview

This document explains the current AADS v6 architecture in plain language.

The project is intentionally narrow:

- one maintained training workflow
- one maintained router-driven inference workflow
- one adapter family
- a small set of notebook helper surfaces around those same workflows

If you are new to the repo, read [../../README.md](../../README.md) before this file.

## The Big Picture

At a high level, AADS v6 has four layers:

1. Configuration: load JSON config, merge environment overrides, normalize the public training surface.
2. Training: build one crop adapter, train it, calibrate OOD, and write artifacts.
3. Inference: route an image to the right crop adapter and return a typed prediction payload.
4. Notebook support: thin Colab wrappers for training, inference, telemetry, and smoke testing.

## Important Terms

- `workflow`: the stable app-facing facade that other surfaces should call
- `adapter`: the saved crop-specific bundle produced by training and loaded at inference time
- `router`: the crop-and-part analysis stage used before adapter inference
- `telemetry`: the logs and mirrored artifacts written during notebook runs
- `artifact`: any generated output file such as a JSON report, CSV, plot, or exported adapter
- `runtime dataset`: the split dataset layout used by the training workflow

## Configuration Flow

The configuration path is intentionally simple.

1. `src/core/config_manager.py` loads `config/base.json`.
2. If an environment is requested, such as `colab`, it deep-merges `config/colab.json` on top.
3. `src/training/services/config_surface.py` normalizes the public training surface under `training.continual`.
4. Legacy top-level OOD aliases are backfilled into `training.continual.ood` and then kept in sync.
5. `src/training/quantization.py` rejects prohibited 4-bit flags before the merged config is used.

Why this matters:

- users edit a small public config surface
- workflow code sees a normalized shape
- older top-level OOD keys do not silently drift away from the canonical training surface

## Training Architecture

The canonical training entrypoint is:

```text
src/workflows/training.py -> TrainingWorkflow.run(...)
```

### End-to-end training flow

1. `TrainingWorkflow.run(...)` receives the crop name, runtime dataset root, output directory, and optional overrides.
2. `src/workflows/training_support.py` prepares the run:
   - config selection
   - loader creation
   - class detection
   - adapter construction
   - loader-size and split-count summaries
3. `src/data/loaders.py` creates the training loaders from the runtime dataset.
4. `src/adapter/independent_crop_adapter.py` exposes the public adapter lifecycle.
5. `src/training/continual_sd_lora.py` owns the actual training engine:
   - backbone loading
   - LoRA wrapping
   - optimization
   - prediction
   - OOD calibration
6. `src/training/session.py` runs epochs and emits observer events for progress, telemetry, and checkpoints.
7. `src/training/services/reporting.py` writes plots, CSV files, JSON metrics, confusion matrices, and readiness artifacts.
8. `src/training/services/ood_benchmark.py` runs the held-out fallback benchmark when real OOD data is missing and the config allows it.
9. The workflow saves the adapter and returns a structured `TrainingWorkflowResult`.

### What the training stack is optimizing

The supported adapter path is continual SD-LoRA only:

- frozen pretrained DINOv3 backbone
- LoRA adapters on selected transformer linear layers
- multi-scale feature fusion
- classifier head over the fused representation
- OOD state saved with the adapter bundle

## Dataset Contracts

There are two dataset contracts and they serve different purposes.

### Contract 1: Notebook input

Notebook 2 accepts a flat class-root layout:

```text
<root>/<class>/<images>
```

This is easier for a beginner to prepare by hand.

### Contract 2: Runtime training layout

Workflow and CLI training use a runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

This layout is what the code actually trains from.

### The conversion step

`scripts/colab_dataset_layout.py` converts the flat notebook input into the runtime split layout.

It writes:

- `split_manifest.json`
- `_split_metadata.json`

The current split policy is effectively 80/10/10 with small-class safeguards.

Important detail:

- the workflow uses the runtime folder name `continual`
- workflow loading maps the logical training split onto that folder

## Inference Architecture

The canonical inference entrypoint is:

```text
src/workflows/inference.py -> InferenceWorkflow.predict(...)
```

### End-to-end inference flow

1. `InferenceWorkflow` builds `src/pipeline/router_adapter_runtime.py`.
2. The runtime loads config and resolves the adapter root.
3. If the caller did not supply `crop_hint`, the runtime loads `src/router/vlm_pipeline.py`.
4. The router analyzes the image and proposes the crop and plant part.
5. The runtime resolves the crop adapter directory.
6. The runtime loads that adapter through `IndependentCropAdapter`.
7. The image is preprocessed to the configured target size.
8. The adapter predicts disease plus OOD information.
9. `src/pipeline/inference_payloads.py` converts the raw output into the public payload.

### Default adapter resolution

The default deployment path is:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

If `crop_hint` is provided, the router step is skipped.

`part_hint` is metadata only. It is not a separate adapter selector.

### Router-driven inference vs direct adapter smoke testing

These are intentionally separate surfaces:

- router-driven inference uses the router first, then the crop adapter
- direct adapter smoke testing loads one adapter directly and bypasses the router

Notebook 3 and `scripts/colab_adapter_smoke_test.py` are for the second case.

## OOD And Readiness Architecture

OOD behavior is part of the adapter runtime. It is not a separate model family.

The supported path is:

1. train on known classes
2. calibrate OOD after training
3. save the OOD state with the adapter
4. evaluate OOD evidence
5. write a deployment verdict

### Main OOD pieces

- detector logic: `src/ood/continual_ood.py`
- radial normalization: `src/ood/radial_normalization.py`
- SURE+ scoring: `src/ood/sure_scoring.py`
- conformal prediction: `src/ood/conformal_prediction.py`
- BER loss wrapper: `src/training/ber_loss.py`
- OOD calibration orchestration: `src/training/services/ood_calibration.py`
- readiness metrics and gates: `src/training/services/metrics.py`
- held-out fallback benchmark: `src/training/services/ood_benchmark.py`

### Why there are multiple JSON artifacts

The workflow writes more than one decision-looking file, but they are not equal.

- `validation/metric_gate.json`: split-local diagnostic gate
- `test/metric_gate.json`: split-local diagnostic gate
- `production_readiness.json`: final deployment verdict

The final deployment decision must come from `production_readiness.json`.

## Artifact Flow

### Workflow and CLI training output

Training writes:

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

### Notebook 2 output

Notebook 2 writes to three places:

- local outputs under `outputs/colab_notebook_training/`
- repo mirrors under `runs/<RUN_ID>/`
- Drive telemetry under `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/`

The repo mirror keeps notebook outputs, telemetry copies, and checkpoint manifests plus only the best checkpoint; rolling checkpoint history remains under the Drive telemetry root.

Current adapter export detail:

- local notebook export: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- workflow export: `<output_dir>/continual_sd_lora_adapter/`
- Drive telemetry export: `artifacts/adapter_export/continual_sd_lora_adapter/`
- some smoke-test helper paths also accept `artifacts/adapter/`

## Colab Support Layer

The notebook support layer is intentionally thin. The notebooks should wrap the maintained workflows, not replace them.

Maintained Colab helper scripts:

- `scripts/colab_repo_bootstrap.py`
- `scripts/colab_live_telemetry.py`
- `scripts/colab_checkpointing.py`
- `scripts/colab_notebook_helpers.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_router_adapter_inference.py`
- `scripts/colab_adapter_smoke_test.py`

These helpers handle:

- repo discovery or auto-clone
- dependency installation
- Drive mount coordination
- Hugging Face token checks
- dataset materialization
- local output mirroring
- telemetry sync
- checkpoint management
- direct adapter smoke testing

## File-to-Responsibility Map

Use this as the quick code map.

- `src/core/config_manager.py`: config loading, merge, normalization, and alias sync
- `src/workflows/training.py`: canonical training workflow
- `src/workflows/training_support.py`: run preparation and artifact payload shaping
- `src/workflows/inference.py`: canonical inference workflow
- `src/adapter/independent_crop_adapter.py`: public adapter lifecycle
- `src/training/continual_sd_lora.py`: actual adapter training engine
- `src/training/services/metrics.py`: metric thresholds and readiness logic
- `src/training/services/reporting.py`: artifact persistence
- `src/training/services/ood_benchmark.py`: held-out fallback OOD benchmark
- `src/pipeline/router_adapter_runtime.py`: router-plus-adapter inference runtime
- `src/router/vlm_pipeline.py`: router implementation
- `src/shared/contracts.py`: shared result and metadata contracts

## Validation Surfaces

The repo validates the maintained surface through:

- `scripts/validate_notebook_imports.py`
- `scripts/evaluate_dataset_layout.py`
- `scripts/benchmark_surfaces.py`
- `tests/unit/`
- `tests/integration/`
- `tests/colab/test_smoke_training.py`

## Related Docs

- [../../README.md](../../README.md)
- [../user_guide/colab_training_manual.md](../user_guide/colab_training_manual.md)
- [../user_guide/ood_readiness_guide.md](../user_guide/ood_readiness_guide.md)
- [ood_recommendation.md](ood_recommendation.md)
- [experimental_leave_one_class_out_ood.md](experimental_leave_one_class_out_ood.md)
