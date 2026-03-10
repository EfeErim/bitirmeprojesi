# AADS v6

AADS v6 is a narrowed plant-disease training and inference repo built around one adapter family and three maintained user surfaces:

- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Router inference: `colab_notebooks/1_router_adapter_inference.ipynb` or `scripts/colab_router_adapter_inference.py`
- Direct adapter validation: `colab_notebooks/3_adapter_smoke_test.ipynb`

The canonical app-facing entrypoints are:

- `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`
- `src/app/cli.py` via `python -m src.app.cli ...` or local Windows PowerShell `.\scripts\python.cmd -m src.app.cli ...`

The repo keeps one training engine and one runtime contract:

- continual SD-LoRA on top of a frozen DINOv3 backbone
- multi-scale feature fusion plus classifier head
- OOD calibration persisted with the adapter bundle
- router -> crop adapter inference with typed output payloads

Use [docs/README.md](docs/README.md) for the documentation map.

## Current Layout

- `src/workflows/`: stable training and inference facades
- `src/adapter/`, `src/training/`, `src/ood/`: core adapter lifecycle, trainer, and OOD stack
- `src/pipeline/`, `src/router/`: router runtime and VLM routing pipeline
- `src/core/config_manager.py`: config loading, environment merge, alias backfill, and training-surface normalization
- `scripts/`: maintained Colab helpers, validation utilities, and benchmark helpers
- `config/base.json` + `config/colab.json`: shipped config surfaces
- `tests/`: unit, integration, and Colab-surface coverage

## Quick Start

Local Windows PowerShell should prefer `.\scripts\python.cmd ...`. It resolves the repo `.venv` first and ignores the Microsoft Store `python.exe` stub.

Create the repo virtual environment once if it does not already exist:

```powershell
.\scripts\python.cmd -m venv .venv
```

Install dependencies:

```powershell
.\scripts\python.cmd -m pip install --upgrade pip
.\scripts\python.cmd -m pip install -r requirements.txt
.\scripts\python.cmd -m pip install -r requirements-dev.txt
```

Validate the maintained surfaces:

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\<your_flat_class_root>
pytest tests/unit tests/colab/test_smoke_training.py -q
pytest tests/integration -q --runintegration
.\scripts\python.cmd scripts/benchmark_surfaces.py
```

## Running The Project

CLI inference:

```powershell
.\scripts\python.cmd -m src.app.cli inference path\to\image.jpg --config-env colab
.\scripts\python.cmd scripts/colab_router_adapter_inference.py path\to\image.jpg --config-env colab
```

CLI training:

```powershell
.\scripts\python.cmd -m src.app.cli training tomato data\runtime_notebook_datasets outputs\training_run --config-env colab
```

Important dataset rule:

- Notebook 2 expects a flat class-root input: `<root>/<class>/<images>`
- Workflow and CLI training expect an already materialized runtime root: `<data_dir>/<crop>/{continual,val,test[,ood]}/...`

Notebook 2 creates that runtime layout automatically under `data/runtime_notebook_datasets/<crop>/`.

## Configuration

Current shipped config flow:

- `config/base.json` is always loaded
- `config/colab.json` is merged when `environment="colab"` or `--config-env colab`
- `ConfigurationManager` normalizes `training.continual`, keeps the legacy top-level `ood.threshold_factor` alias in sync, and rejects prohibited 4-bit flags

High-signal controls live under:

- `training.continual.backbone`
- `training.continual.adapter`
- `training.continual.fusion`
- `training.continual.ood`
- `training.continual.optimization`
- `training.continual.evaluation`
- `colab.training`
- `inference`

## Outputs

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
    test/
    ood_benchmark/
    production_readiness.json
```

Notebook 2 writes to three places:

1. Local notebook output:

```text
outputs/colab_notebook_training/
  continual_sd_lora_adapter/
  artifacts/
```

2. Repo mirror for the run:

```text
runs/<RUN_ID>/
  notebooks/2_interactive_adapter_training.executed.ipynb
  outputs/colab_notebook_training/
  telemetry/
  checkpoint_state/
```

3. Drive telemetry root:

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

Notes:

- Notebook 2 currently exports the Drive adapter bundle under `artifacts/adapter_export/continual_sd_lora_adapter/`.
- Some helper surfaces and older test fixtures also accept `artifacts/adapter/`.
- `production_readiness.json` is the final deployment verdict. Split-local `metric_gate.json` files are diagnostics, not the final decision by themselves.

## Adapter Handoff

Default router inference lookup is:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

You can deploy a trained adapter by copying one of these outputs there:

- `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `runs/<RUN_ID>/outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/artifacts/adapter_export/continual_sd_lora_adapter/`
- workflow output `<output_dir>/continual_sd_lora_adapter/`

Or keep a custom location and pass `--adapter-root`.

## Notebook 3

`colab_notebooks/3_adapter_smoke_test.ipynb` is the maintained direct-adapter validation surface.

It supports:

- automatic adapter discovery under configured search roots
- direct metadata inspection through `adapter_meta.json`
- one-image prediction
- folder-level sanity passes with per-file error reporting

Useful manual path inputs:

- direct adapter dir: `.../continual_sd_lora_adapter/`
- parent export dir: `outputs/colab_notebook_training/`
- telemetry adapter-export dir: `.../telemetry/<RUN_ID>/artifacts/adapter_export/`
- telemetry adapter dir when present: `.../telemetry/<RUN_ID>/artifacts/adapter/`
- direct metadata file: `.../adapter_meta.json`
- deployed adapter root: `models/adapters/`

## Repo Hygiene

Commit source, config, docs, notebooks, and tests.

Keep these local-only:

- `runs/`
- `models/adapters/`
- `outputs/`
- `.runtime_tmp/`, caches, and virtual environments

## Further Reading

- [docs/README.md](docs/README.md)
- [docs/user_guide/colab_training_manual.md](docs/user_guide/colab_training_manual.md)
- [docs/user_guide/ood_readiness_guide.md](docs/user_guide/ood_readiness_guide.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
