# Colab Training Manual

This guide is written for someone who may be opening the AADS notebooks for the first time.

It covers the maintained Colab surfaces:

- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook 3: `colab_notebooks/3_adapter_smoke_test.ipynb`

Notebook 2 trains an adapter. Notebook 3 checks an already exported adapter directly.

If you are brand new to the repo, read [../../README.md](../../README.md) first.

## What You Will Do In These Notebooks

### Notebook 2

Notebook 2 takes a folder of labeled images for one crop, prepares the runtime dataset, trains a crop adapter, evaluates it, calibrates OOD behavior, and exports the result.

### Notebook 3

Notebook 3 loads one saved adapter directly so you can inspect it and run a quick prediction without the router.

## Important Terms

- `flat class-root dataset`: the simple folder layout you prepare by hand for Notebook 2
- `runtime dataset`: the split layout Notebook 2 creates before training
- `adapter bundle`: the saved training output that contains model weights, metadata, and OOD state
- `telemetry`: the logs and mirrored artifacts saved during a notebook run
- `OOD`: "out of distribution," meaning the image may not belong to the supported disease classes
- `readiness`: the final deployment verdict written to `production_readiness.json`

## Before You Start

Make sure you have these things ready:

1. A dataset for one crop.
2. A Colab runtime that has GPU access if you intend to use the default `cuda` path.
3. A Hugging Face token if the required models need authenticated access.
4. Enough storage for outputs, telemetry, and checkpoints if you enable Drive logging.

Important current behavior:

- requesting `device="cuda"` fails immediately when CUDA is unavailable
- the notebook validates the dataset before training starts

## Notebook 2 In Plain English

This is the current Notebook 2 flow from start to finish:

1. find or initialize the repo workspace
2. install notebook requirements
3. mount Google Drive when available
4. resolve a Hugging Face token from environment variables or Colab secrets
5. validate a flat class-root dataset
6. materialize a runtime split layout under `data/runtime_notebook_datasets/<crop>/`
7. train the continual SD-LoRA adapter
8. restore the best model state
9. calibrate OOD
10. write validation and test artifacts
11. use real `ood/` data when present, otherwise run the held-out fallback benchmark when enabled
12. write `production_readiness.json`
13. mirror outputs into `runs/<RUN_ID>/`
14. optionally auto-push the mirrored run record to GitHub
15. optionally auto-disconnect the Colab runtime after final exports succeed

## The Dataset Format Notebook 2 Accepts

Notebook 2 expects the simplest possible input layout:

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
    img004.jpg
  late_blight/
    img005.jpg
```

That means:

- the root folder contains one folder per class
- each class folder contains image files
- you should not pre-create `train`, `val`, or `test` folders for Notebook 2

### Validate the dataset before training

Use:

```powershell
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\tomato_flat
```

That validator checks:

- whether the root exists
- whether it is a directory
- whether it uses the flat class-root shape
- how many images exist per class
- what the estimated split counts will look like

## What Notebook 2 Builds From That Dataset

Notebook 2 converts the flat dataset into the runtime layout used by the workflow:

```text
data/runtime_notebook_datasets/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

If you also provide unknown examples, the runtime layout can include:

```text
data/runtime_notebook_datasets/<crop>/ood/*
```

The generated runtime dataset includes:

- `split_manifest.json`
- `_split_metadata.json`

Important distinction:

- Notebook 2 accepts the flat class-root layout
- `TrainingWorkflow.run(...)` and CLI training expect the runtime split layout

## How The Split Is Created

The dataset materialization step is handled by `scripts/colab_dataset_layout.py`.

Current behavior:

- the notebook uses an 80/10/10-style split policy with small-class safeguards
- the runtime split names are `continual`, `val`, and `test`
- on Windows, materialization defaults to copying files instead of symlinks

## Training Settings, Explained By Purpose

The config surface can look large when you are new. The easiest way to understand it is by what each section changes.

### Model identity

These decide which base model and LoRA strategy are used:

- `training.continual.backbone.model_name`
- `training.continual.adapter.target_modules_strategy`
- `training.continual.adapter.lora_r`
- `training.continual.adapter.lora_alpha`
- `training.continual.adapter.lora_dropout`

### Feature fusion

These control how features from multiple backbone layers are combined:

- `training.continual.fusion.layers`
- `training.continual.fusion.output_dim`
- `training.continual.fusion.dropout`
- `training.continual.fusion.gating`

### Optimization

These control the actual training process:

- `training.continual.learning_rate`
- `training.continual.weight_decay`
- `training.continual.num_epochs`
- `training.continual.batch_size`
- `training.continual.optimization.grad_accumulation_steps`
- `training.continual.optimization.max_grad_norm`
- `training.continual.optimization.mixed_precision`
- `training.continual.optimization.label_smoothing`
- `training.continual.optimization.scheduler`
- `training.continual.early_stopping`

### Data loading

These affect image loading and error tolerance:

- `training.continual.data.sampler`
- `training.continual.data.loader_error_policy`
- `training.continual.data.target_size`
- `training.continual.data.cache_size`
- `training.continual.data.validate_images_on_init`

### OOD behavior

These control how OOD calibration and scoring behave:

- `training.continual.ood.threshold_factor`
- `training.continual.ood.radial_l2_enabled`
- `training.continual.ood.radial_beta_range`
- `training.continual.ood.radial_beta_steps`
- `training.continual.ood.sure_enabled`
- `training.continual.ood.sure_semantic_percentile`
- `training.continual.ood.sure_confidence_percentile`
- `training.continual.ood.conformal_enabled`
- `training.continual.ood.conformal_alpha`

### Readiness policy

These control how strict the final deployment verdict is:

- `training.continual.evaluation.best_metric`
- `training.continual.evaluation.emit_ood_gate`
- `training.continual.evaluation.require_ood_for_gate`
- `training.continual.evaluation.ood_fallback_strategy`
- `training.continual.evaluation.ood_benchmark_auto_run`
- `training.continual.evaluation.ood_benchmark_min_classes`

Practical interpretation:

- `emit_ood_gate` controls whether split-local `metric_gate.json` files are written to disk
- `require_ood_for_gate` controls whether final production readiness fails when OOD evidence is missing or insufficient

### Colab runtime controls

These do not change the model itself. They change the notebook runtime behavior:

- `colab.training.num_workers`
- `colab.training.pin_memory`
- `colab.training.checkpoint_every_n_steps`
- `colab.training.checkpoint_on_exception`
- `colab.training.stdout_progress_batch_interval`
- `colab.training.stdout_progress_min_interval_sec`

Legacy compatibility note:

- `checkpoint_interval` is still normalized as an alias for `checkpoint_every_n_steps`

Current Colab default tradeoffs:

- the Colab environment disables deterministic training so CuDNN can use faster kernels
- automatic held-out OOD fallback benchmarking is disabled by default for faster iteration
- if you do not provide a real `ood/` split and leave that fallback disabled, `production_readiness.json` will stay failed until you re-enable the benchmark or add real OOD data

## Notebook-Only Top Cell Toggles

Notebook 2 also exposes a small set of notebook-level toggles:

- `BER_ENABLED`
- `BER_LAMBDA_OLD`
- `BER_LAMBDA_NEW`
- `AUTO_DISCONNECT_RUNTIME`
- `AUTO_DISCONNECT_GRACE_SECONDS`

These are convenience controls for the notebook surface. The canonical workflow still lives in `TrainingWorkflow.run(...)`.

## Token Resolution

Notebook 2 resolves the Hugging Face token from these sources:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- matching Colab secrets when running inside Colab

The notebook validates the token before model access.

Notebook 2 also resolves the GitHub push token from:

- `GH_TOKEN`
- `GITHUB_TOKEN`
- matching Colab secrets when running inside Colab

If auto-push is enabled, the notebook uses that token after the repo mirror step.

## What Notebook 2 Produces

Notebook 2 writes to three locations. Each one exists for a different reason.

### 1. Local notebook output

```text
outputs/colab_notebook_training/
  continual_sd_lora_adapter/
  artifacts/
```

Use this when you want the immediate local result of the notebook run.

`artifacts/` contains the same artifact families as the workflow:

- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

### 2. Repo mirror

Notebook 2 mirrors non-checkpoint outputs into:

```text
runs/<RUN_ID>/
  notebooks/2_interactive_adapter_training.executed.ipynb
  outputs/colab_notebook_training/
  telemetry/
  checkpoint_state/
```

Use this when you want one self-contained record of a notebook run inside the repo workspace.

Important detail:

- `checkpoint_state/` keeps checkpoint metadata plus only the mirrored best checkpoint
- the actual rolling checkpoint tree stays under the Drive checkpoint root

Optional current behavior:

- if `AUTO_PUSH_TO_GITHUB` is enabled and `GH_TOKEN` or `GITHUB_TOKEN` is available, Notebook 2 commits and pushes `runs/<RUN_ID>/` after the mirror step
- the auto-push helper skips `.pt` checkpoint blobs, so large resume weights stay out of the normal GitHub history

### 3. Drive telemetry

Current Drive telemetry layout:

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

Use this when you need durable notebook logs and checkpoint recovery across sessions.

Important current-state note:

- Notebook 2 currently exports adapter assets to `artifacts/adapter_export/continual_sd_lora_adapter/`
- some older helpers and fixtures also support `artifacts/adapter/`

## How To Read The Main Training Artifacts

### `training/`

This folder helps you understand how training behaved over time.

- `results.png`: visual overview of loss, accuracy, throughput, and other curves
- `history.json` and `history.csv`: per-epoch history
- `batch_metrics.csv`: batch-level record
- `summary.json`: compact summary of the run

### `validation/` and `test/`

These folders help you understand how the model performed on known-class data.

- `classification_report.txt` and `.json`: precision, recall, and F1-style breakdowns
- `per_class_metrics.csv`: per-class summary
- `confusion_matrix.csv` and `.png`: where predictions are confused
- `metric_gate.json`: split-local threshold checks

### `ood_benchmark/`

This folder appears when the fallback held-out OOD benchmark runs.

- `summary.json`: overall benchmark verdict
- `per_fold.csv`: one row per held-out class fold
- `progress.json`: latest persisted fold and stage while the benchmark is running
- `folds/<held_out_class>/failure.json` and `failure_traceback.txt`: saved Python failure details when a fold raises an exception

### `production_readiness.json`

This is the final deployment verdict. Read this before deployment.

## OOD And Readiness In Notebook 2

Notebook 2 does more than train a classifier. It also:

- calibrates OOD state after training
- evaluates real `ood/` data when present
- otherwise runs the held-out fallback benchmark when enabled
- writes the final readiness artifact to `production_readiness.json`

Important guardrails:

- the best in-memory weights are restored before final calibration, evaluation, and adapter export
- if `val` was used for OOD calibration, an isolated `test/` split is needed for a final readiness verdict
- if no real `ood/` split exists, the fallback benchmark may retrain once per known class

Read [ood_readiness_guide.md](ood_readiness_guide.md) for the full readiness logic.

## BER Rollout

BER is currently an optional training-only experiment.

Recommended comparison workflow:

1. keep the crop, seed, class set, and evidence source fixed
2. run a BER-off baseline
3. run BER candidates with different lambdas
4. compare the same artifact families across runs

Most useful files for that comparison:

- `validation/metric_gate.json`
- `test/metric_gate.json`
- `ood_benchmark/summary.json`
- `production_readiness.json`

Optional helper command:

```powershell
.\scripts\python.cmd scripts/evaluate_ber_rollout.py <baseline_artifact_root> <candidate_artifact_root> [...]
```

## Notebook 3 Adapter Smoke Test

Notebook 3 is for direct adapter validation after training or export.

It does not use the router.

Use it to:

- inspect adapter metadata
- confirm the adapter bundle still loads
- run one-image prediction
- run folder-level sanity prediction with per-file error reporting

### What paths Notebook 3 accepts

Current accepted `ADAPTER_DIR` patterns include:

- direct asset dir: `.../continual_sd_lora_adapter/`
- parent export dir: `outputs/colab_notebook_training/`
- current Drive export dir: `.../telemetry/<RUN_ID>/artifacts/adapter_export/`
- older telemetry adapter dir when present: `.../telemetry/<RUN_ID>/artifacts/adapter/`
- direct metadata file: `.../adapter_meta.json`

Current accepted `ADAPTER_ROOT` pattern:

- parent of crop folders, usually `models/adapters/`

Important caveat:

- if a telemetry run only contains the current `adapter_export/continual_sd_lora_adapter/` layout, point `ADAPTER_DIR` at `adapter_export/` or the `continual_sd_lora_adapter/` folder itself
- `ADAPTER_ROOT` is for deployed adapters under `models/adapters/<crop>/...`, not for telemetry run roots

### Image input rules

- `IMAGE_PATH` must be one image file
- `BATCH_IMAGE_DIR` must be one directory of image files
- `CROP_NAME` can stay `None` when the adapter path itself already implies the crop or nearby metadata provides it

## Deployment Handoff

Router inference looks for adapters here by default:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

You can deploy from any of these outputs:

- `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `runs/<RUN_ID>/outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/artifacts/adapter_export/continual_sd_lora_adapter/`

If you want a different storage location, pass `--adapter-root` to the inference surface.

## Common Beginner Mistakes

### Mistake 1: using a pre-split dataset as Notebook 2 input

Notebook 2 expects the flat class-root layout, not pre-made `train/val/test` folders.

### Mistake 2: confusing local output with deployed adapters

Training output folders are not automatically deployed. Router inference only finds adapters under the configured adapter root unless you override it.

### Mistake 3: reading only `validation/metric_gate.json`

Use `production_readiness.json` for the final deployment verdict.

### Mistake 4: pointing Notebook 3 at the wrong folder

If you are testing a telemetry export, point `ADAPTER_DIR` at the export folder or the `continual_sd_lora_adapter/` folder inside it.

## Validation Commands

Run these before or after notebook-related changes:

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\<your_flat_class_root>
pytest tests/colab/test_smoke_training.py -q
```

## Repo Hygiene

Keep these out of git:

- `runs/`
- `models/adapters/`
- `outputs/`

`colab_notebooks/requirements_colab.txt` should stay in the repo. It is a wrapper around the canonical root `requirements_colab.txt`.
