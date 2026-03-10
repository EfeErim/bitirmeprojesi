# Colab Training Manual

This guide covers the current maintained Colab surfaces:

- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook 3: `colab_notebooks/3_adapter_smoke_test.ipynb`

Notebook 2 is the training surface. Notebook 3 is the post-training direct adapter validation surface.

## Notebook 2 In One Pass

The current Notebook 2 flow is:

1. resolve the repo root, install notebook requirements, and mount Drive when available
2. resolve a Hugging Face token from env or Colab secrets
3. validate a flat class-root dataset
4. materialize a runtime dataset split under `data/runtime_notebook_datasets/<crop>/`
5. train the continual SD-LoRA adapter
6. calibrate OOD and write validation/test artifacts
7. run held-out fallback OOD benchmarking when real `ood/` data is missing
8. write `production_readiness.json`
9. mirror local outputs into `runs/<RUN_ID>/`
10. optionally auto-disconnect the Colab runtime after final exports succeed

## Dataset Contract

Notebook 2 expects a flat class-root dataset:

```text
<root>/<class>/<images>
```

Before training, validate it with:

```powershell
python scripts/evaluate_dataset_layout.py --root data\<your_flat_class_root>
```

Notebook 2 materializes the runtime split layout automatically:

```text
data/runtime_notebook_datasets/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

If a shared unknown pool exists, the workflow also reads:

```text
data/runtime_notebook_datasets/<crop>/ood/*
```

The generated runtime dataset includes:

- `split_manifest.json`
- `_split_metadata.json`

Important distinction:

- Notebook 2 accepts flat class-root input.
- `TrainingWorkflow.run(...)` and `python -m src.app.cli training ...` expect the already materialized runtime root.

## Configuration And Controls

Current normalized training controls live under `training.continual`:

- `backbone.model_name`
- `adapter.target_modules_strategy`, `adapter.lora_r`, `adapter.lora_alpha`, `adapter.lora_dropout`
- `fusion.layers`, `fusion.output_dim`, `fusion.dropout`, `fusion.gating`
- `learning_rate`, `weight_decay`, `num_epochs`, `batch_size`
- `seed`, `deterministic`
- `optimization.grad_accumulation_steps`
- `optimization.max_grad_norm`
- `optimization.mixed_precision`
- `optimization.label_smoothing`
- `optimization.scheduler`
- `data.sampler`
- `data.loader_error_policy`
- `data.target_size`
- `data.cache_size`
- `data.validate_images_on_init`
- `ood.threshold_factor`
- `ood.ber_enabled`, `ood.ber_lambda_old`, `ood.ber_lambda_new`
- `ood.radial_l2_enabled`, `ood.radial_beta_range`, `ood.radial_beta_steps`
- `ood.sure_enabled`, `ood.sure_semantic_percentile`, `ood.sure_confidence_percentile`
- `ood.conformal_enabled`, `ood.conformal_alpha`
- `early_stopping`
- `evaluation.best_metric`
- `evaluation.emit_ood_gate`
- `evaluation.require_ood_for_gate`

Practical behavior:

- `evaluation.emit_ood_gate` controls whether split-local `validation/metric_gate.json` and `test/metric_gate.json` are written
- `evaluation.require_ood_for_gate` controls whether final production readiness hard-fails when OOD evidence is missing
- `evaluation.ood_fallback_strategy`
- `evaluation.ood_benchmark_auto_run`
- `evaluation.ood_benchmark_min_classes`

Colab runtime controls live under `colab.training`:

- `num_workers`
- `pin_memory`
- `checkpoint_every_n_steps`
- `checkpoint_on_exception`
- `stdout_progress_batch_interval`
- `stdout_progress_min_interval_sec`

Legacy compatibility:

- `checkpoint_interval` is still normalized as an alias for `checkpoint_every_n_steps`.

Notebook-only top-cell toggles currently exposed in Notebook 2:

- `BER_ENABLED`
- `BER_LAMBDA_OLD`
- `BER_LAMBDA_NEW`
- `AUTO_DISCONNECT_RUNTIME`
- `AUTO_DISCONNECT_GRACE_SECONDS`

## Hugging Face Access

Notebook 2 resolves a token from:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- matching Colab secrets when running in Colab

The notebook validates the token before model access.

## What Notebook 2 Produces

### Local notebook output

```text
outputs/colab_notebook_training/
  continual_sd_lora_adapter/
  artifacts/
```

`artifacts/` contains the same artifact families as the workflow surface:

- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

### Repo mirror

Notebook 2 mirrors non-checkpoint outputs into:

```text
runs/<RUN_ID>/
  notebooks/2_interactive_adapter_training.executed.ipynb
  outputs/colab_notebook_training/
  telemetry/
  checkpoint_state/
```

`checkpoint_state/` is a metadata mirror only. The rolling checkpoint tree itself stays under the Drive checkpoint root.

### Drive telemetry

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

Important current-state note:

- Notebook 2 currently exports adapter assets to `artifacts/adapter_export/continual_sd_lora_adapter/`.
- Some helper code and older fixtures also support `artifacts/adapter/`.
- Do not rely on older docs that mention `export_layout.json` or `crop_info.json` as guaranteed notebook outputs; they are not part of the current maintained notebook flow.

## OOD And Readiness

Notebook 2 does not stop at validation accuracy. The current workflow also:

- calibrates OOD state after training
- evaluates against real `ood/` data when present
- otherwise runs the held-out fallback benchmark
- writes the final gate artifact to `production_readiness.json`

For the full readiness policy and artifact interpretation, see [ood_readiness_guide.md](ood_readiness_guide.md).

## BER Rollout

BER is currently an optional training-only experiment.

Recommended comparison flow:

1. keep crop, seed, split layout, and OOD evidence source fixed
2. run a BER-off baseline
3. run BER candidates with different `BER_LAMBDA_OLD` / `BER_LAMBDA_NEW` values
4. compare:
   - `validation/metric_gate.json`
   - `test/metric_gate.json`
   - `ood_benchmark/summary.json`
   - `production_readiness.json`

Optional helper:

```powershell
python scripts/evaluate_ber_rollout.py <baseline_artifact_root> <candidate_artifact_root> [...]
```

## Notebook 3 Adapter Smoke Test

Notebook 3 loads one adapter directly through `scripts/colab_adapter_smoke_test.py` without the router.

Use it to:

- inspect saved adapter metadata
- verify the bundle still loads
- run one-image or folder-level sanity predictions

Current accepted `ADAPTER_DIR` patterns include:

- direct asset dir: `.../continual_sd_lora_adapter/`
- parent export dir: `outputs/colab_notebook_training/`
- current Notebook 2 Drive export dir: `.../telemetry/<RUN_ID>/artifacts/adapter_export/`
- older telemetry adapter dir when present: `.../telemetry/<RUN_ID>/artifacts/adapter/`
- direct metadata file: `.../adapter_meta.json`

Current accepted `ADAPTER_ROOT` pattern:

- parent of crop folders, usually `models/adapters/`

Important caveat:

- If your telemetry run only contains the current `adapter_export/continual_sd_lora_adapter/` layout, point `ADAPTER_DIR` at `adapter_export/` or the `continual_sd_lora_adapter/` folder itself.
- `ADAPTER_ROOT` is for deployed adapters under `models/adapters/<crop>/...`, not for telemetry run roots.

For image inputs:

- `IMAGE_PATH` must be one image file.
- `BATCH_IMAGE_DIR` must be one directory of image files.
- `CROP_NAME` can stay `None` when the adapter path itself implies the crop or nearby metadata provides it.

## Deployment Handoff

Router inference looks for adapters at:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

You can deploy from any of these current outputs:

- `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `runs/<RUN_ID>/outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/artifacts/adapter_export/continual_sd_lora_adapter/`

Or keep a custom location and pass `--adapter-root`.

## Validation

Run these before or after notebook changes:

```powershell
python scripts/validate_notebook_imports.py
python scripts/evaluate_dataset_layout.py --root data\<your_flat_class_root>
pytest tests/colab/test_smoke_training.py -q
```

## Repo Hygiene

Keep these out of git:

- `runs/`
- `models/adapters/`
- `outputs/`

`colab_notebooks/requirements_colab.txt` should stay in the repo. It is the notebook-local wrapper around the canonical root `requirements_colab.txt`.
