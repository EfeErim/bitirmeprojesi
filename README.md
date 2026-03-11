# AADS v6

AADS v6 is a focused plant-disease training and inference repository.

In plain language, this repo does three things:

1. It trains one crop-specific disease adapter at a time.
2. It checks whether that adapter is accurate enough and safe enough to deploy.
3. It uses a router plus the saved adapter to predict disease from a new image.

You do not need prior machine learning experience to read this repo if you follow the docs in order. This README is the starting point.

## Start Here

If you are completely new, read in this order:

1. This file.
2. [docs/README.md](docs/README.md)
3. [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md) if you want to train.
4. [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md) if you want to decide whether a model is deployable.
5. [docs/architecture/overview.md](docs/architecture/overview.md) if you want the code map.

## What The Main Words Mean

- `crop`: The plant type, such as `tomato`, `potato`, or `wheat`.
- `class`: One label the model can predict for that crop, such as `healthy` or `late_blight`.
- `adapter`: The saved crop-specific model bundle that contains LoRA weights, classifier state, metadata, and OOD state.
- `router`: The front part of inference that looks at an image and guesses the crop and plant part before the crop adapter runs.
- `OOD`: "Out of distribution." This means the input does not look like the disease classes the adapter was trained to support.
- `readiness`: The final deployment verdict written to `production_readiness.json`.
- `artifact`: A file produced by training, evaluation, or notebook telemetry, such as a JSON report, plot, or exported adapter.
- `runtime dataset`: The split dataset layout used by the workflow code during training.

## What This Repo Actually Supports

This repository is intentionally narrow. The maintained user surfaces are:

- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook 1 router inference: `colab_notebooks/1_router_adapter_inference.ipynb`
- Notebook 3 direct adapter smoke test: `colab_notebooks/3_adapter_smoke_test.ipynb`
- CLI training: `.\scripts\python.cmd -m src.app.cli training ...`
- CLI inference: `.\scripts\python.cmd -m src.app.cli inference ...`

The canonical app-facing entrypoints are:

- `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`

This repo does not aim to be:

- a general-purpose ML framework
- a multi-model research playground
- an autonomous agent system

## The System In One Minute

Training works like this:

1. You start with images for one crop.
2. The training flow builds one adapter for that crop.
3. The workflow evaluates the adapter on validation and test data.
4. The workflow calibrates OOD behavior.
5. The workflow writes reports and a final readiness verdict.
6. You can copy the exported adapter into the deployment adapter root.

Inference works like this:

1. The router looks at the image and guesses the crop and part.
2. The runtime loads the adapter for that crop.
3. The adapter predicts the disease class.
4. The runtime also reports whether the image looks OOD.

Notebook 3 is different:

- It skips the router.
- It loads one adapter directly.
- It is used to confirm that an exported adapter still works.

## Key Technical Facts

The maintained training path is:

- frozen DINOv3 backbone
- LoRA adapters on selected transformer linear layers
- multi-scale feature fusion
- classifier head on the fused representation
- OOD calibration saved inside the exported adapter bundle

The default inference deployment path is:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

## Repo Layout

These folders matter most:

- `src/workflows/`: stable training and inference facades
- `src/training/`: training session logic, evaluation, OOD calibration, and artifact writing
- `src/pipeline/` and `src/router/`: router-driven inference runtime
- `scripts/`: notebook helpers, validation tools, and small entrypoints
- `config/`: shipped JSON configuration
- `docs/`: maintained Markdown documentation
- `tests/`: unit, integration, and notebook-surface coverage

## Before You Run Anything

### 1. Use the repo Python launcher on Windows

Windows PowerShell examples in this repo use:

```powershell
.\scripts\python.cmd
```

That launcher prefers the repo `.venv` and avoids the Microsoft Store `python.exe` stub.

### 2. Create or reuse the virtual environment

```powershell
.\scripts\python.cmd -m venv .venv
```

### 3. Install dependencies

```powershell
.\scripts\python.cmd -m pip install --upgrade pip
.\scripts\python.cmd -m pip install -r requirements.txt
.\scripts\python.cmd -m pip install -r requirements-dev.txt
```

### 4. Run the narrow validation commands

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
pytest tests/unit tests/colab/test_smoke_training.py -q
pytest tests/integration -q --runintegration
.\scripts\python.cmd scripts/benchmark_surfaces.py
```

## Dataset Formats Explained

This is the most common beginner mistake, so treat these as two different contracts.

### Notebook 2 input contract

Notebook 2 expects a flat class-root layout:

```text
<root>/<class>/<images>
```

Example:

```text
data/tomato_flat/
  healthy/
    img001.jpg
    img002.jpg
  early_blight/
    img003.jpg
  late_blight/
    img004.jpg
```

Validate that layout with:

```powershell
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\tomato_flat
```

### Workflow and CLI training contract

The workflow code does not train from the flat layout directly. It expects a runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

Notebook 2 creates that layout automatically under:

```text
data/runtime_notebook_datasets/<crop>/
```

The split folder is named `continual` because the project uses continual-training terminology. Internally, workflow loading maps the public training split onto that folder.

`ood/` is one shared pool of unsupported inputs for that crop adapter. It is not another supported class. Nested folders inside `ood/` are allowed for organization and are loaded recursively. For concrete curation guidance, see [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md).

## Training, Step By Step

The canonical training entrypoint is `TrainingWorkflow.run(...)` in `src/workflows/training.py`.

In practice, the flow is:

1. Load `config/base.json`.
2. Optionally merge `config/colab.json` when the environment is `colab`.
3. Normalize the public training surface under `training.continual`.
4. Build data loaders from the runtime dataset.
5. Train the crop adapter.
6. Restore the best in-memory weights.
7. Calibrate OOD using the chosen calibration split.
8. Save the adapter.
9. Write evaluation artifacts for validation and test.
10. Use real `ood/` data if it exists, otherwise run the held-out fallback benchmark automatically.
11. Write the final readiness verdict to `production_readiness.json`.

## Inference, Step By Step

The canonical inference entrypoint is `InferenceWorkflow.predict(...)` in `src/workflows/inference.py`.

In practice, the runtime does this:

1. Load config and locate the adapter root.
2. Run the router unless you pass `crop_hint`.
3. Resolve the crop adapter directory.
4. Load the adapter for that crop.
5. Preprocess the image to the configured target size.
6. Predict the disease class.
7. Return OOD information together with the prediction.

If the router cannot identify a crop, the runtime returns an `unknown` result instead of forcing a disease prediction.
If the router backend itself is unavailable, the runtime returns `router_unavailable` instead of pretending the crop was merely unknown.

## Common Commands

### Train from an already materialized runtime dataset

```powershell
.\scripts\python.cmd -m src.app.cli training tomato data\runtime_notebook_datasets outputs\training_run --config-env colab
```

### Run router-driven inference

```powershell
.\scripts\python.cmd -m src.app.cli inference path\to\image.jpg --config-env colab
```

### Run the script wrapper for inference

```powershell
.\scripts\python.cmd scripts/colab_router_adapter_inference.py path\to\image.jpg --config-env colab
```

### Bypass the router with a known crop

```powershell
.\scripts\python.cmd -m src.app.cli inference path\to\image.jpg --config-env colab --crop tomato
```

## Configuration Overview

The config flow is:

1. `config/base.json` is always loaded.
2. `config/<environment>.json` is merged on top when requested.
3. `ConfigurationManager` normalizes the training surface.
4. Legacy top-level OOD keys are kept in sync with `training.continual.ood`.
5. Prohibited 4-bit flags are rejected before use.

The most important config areas are:

- `training.continual.backbone`
- `training.continual.adapter`
- `training.continual.fusion`
- `training.continual.optimization`
- `training.continual.ood`
- `training.continual.evaluation`
- `colab.training`
- `router`
- `inference`

## What Training Produces

Workflow and CLI training write:

```text
<output_dir>/
  continual_sd_lora_adapter/
  training_metrics/
    training/
      results.png
      results.csv
      history.json
      history.csv
      batch_metrics.csv
      summary.json
    validation/
      classification_report.txt
      classification_report.json
      per_class_metrics.csv
      confusion_matrix.csv
      confusion_matrix.png
      confusion_matrix_normalized.png
      metric_gate.json
    test/
      ...
    ood_benchmark/
      summary.json
      per_fold.csv
    production_readiness.json
```

What these files mean:

- `continual_sd_lora_adapter/`: the exported adapter bundle you can deploy
- `training/`: training curves and summary data
- `validation/` and `test/`: split-specific evaluation reports
- `ood_benchmark/`: fallback OOD evidence when no real `ood/` split exists
- `production_readiness.json`: the final deployment verdict

## What Notebook 2 Produces

Notebook 2 writes to three places.

### Local notebook output

```text
outputs/colab_notebook_training/
  continual_sd_lora_adapter/
  artifacts/
```

### Repo mirror for the run

```text
runs/<RUN_ID>/
  notebooks/2_interactive_adapter_training.executed.ipynb
  outputs/colab_notebook_training/
  telemetry/
  checkpoint_state/
```

`checkpoint_state/` keeps the checkpoint manifests plus only the mirrored best checkpoint. Rolling checkpoint history stays under the Drive telemetry root.

### Drive telemetry root

```text
<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/
  checkpoints/
  artifacts/
    training/
    validation/
    test/
    ood_benchmark/
    adapter_export/
      continual_sd_lora_adapter/
  events.jsonl
  runtime.log
  latest_status.json
  summary.json
  latest_checkpoint.json
  best_checkpoint.json
  checkpoint_index.json
```

Important current detail:

- Notebook 2 exports the Drive adapter bundle under `artifacts/adapter_export/continual_sd_lora_adapter/`.
- Some older helper paths still accept `artifacts/adapter/`.

## How Deployment Handoff Works

Router inference looks for adapters here by default:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

You can deploy a trained adapter by copying one of these outputs there:

- workflow output: `<output_dir>/continual_sd_lora_adapter/`
- Notebook 2 local output: `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- Notebook 2 repo mirror: `runs/<RUN_ID>/outputs/colab_notebook_training/continual_sd_lora_adapter/`
- Notebook 2 Drive export: `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/artifacts/adapter_export/continual_sd_lora_adapter/`

If you keep adapters somewhere else, pass `--adapter-root`.

## OOD And Readiness In Plain Language

The repo does not treat high validation accuracy as enough for deployment.

The workflow also checks whether the adapter can recognize inputs that do not belong to its supported disease set. That decision is written to:

```text
production_readiness.json
```

Use this file as the final go/no-go artifact.

Do not use only these files for the final decision:

- `validation/metric_gate.json`
- `test/metric_gate.json`

Those are useful diagnostics, but they are not the authoritative deployment verdict by themselves.

For the full explanation, read [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md).

## Tracked Vs Local-Only Files

Tracked source of truth:

- `src/`
- `tests/`
- `scripts/`
- `config/`
- `docs/`
- `README.md`
- `colab_notebooks/*.ipynb`
- root dependency files

Local or generated only:

- `runs/<RUN_ID>/`
- `models/adapters/`
- `outputs/`
- `.runtime_tmp/`
- caches and virtual environments

Do not treat generated outputs as tracked implementation files unless you explicitly need to inspect a local run.

## Where To Read Next

- [docs/README.md](docs/README.md): documentation map and reading paths
- [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md): beginner-friendly Notebook 2 and Notebook 3 guide
- [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md): how deployment readiness is decided
- [docs/architecture/overview.md](docs/architecture/overview.md): code and data flow map
