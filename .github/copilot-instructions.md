# AADS v6 Copilot Instructions

This repo is intentionally narrow. The maintained surface is continual SD-LoRA adapter training, OOD readiness, router-driven inference, and Colab notebook wrappers around those flows.

For deeper repo-specific routing and task guidance, also read `AGENTS.md` and the relevant file under `skills/`.

## Source of truth

Prefer these before making assumptions:

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `docs/user_guide/colab_training_manual.md`
- `docs/user_guide/ood_readiness_guide.md`

## Canonical entrypoints

- Training: `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- Inference: `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`
- Notebook training: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook inference: `colab_notebooks/1_router_adapter_inference.ipynb`
- Direct adapter validation: `colab_notebooks/3_adapter_smoke_test.ipynb` and `scripts/colab_adapter_smoke_test.py`

Treat workflow code as canonical over notebook-only behavior.

## Repo boundaries

Tracked source:

- `src/`, `tests/`, `scripts/`, `config/`
- `docs/` and `README.md`
- `colab_notebooks/*.ipynb`
- Root dependency files

Local or generated only:

- `runs/<RUN_ID>/`
- `models/adapters/<crop>/continual_sd_lora_adapter/`
- `outputs/`
- `.runtime_tmp/`, caches, temp folders, virtualenvs

Do not edit generated outputs as if they were maintained source unless the user explicitly asks for that.

## Surface routing

- Training, OOD calibration, readiness artifacts, BER, and training-side adapter export: start from `src/workflows/training.py`, `config/base.json`, and `src/training/services/`
- Notebook 1, 2, or 3 issues, dataset materialization, Hugging Face token handling, Drive telemetry, or notebook export paths: inspect the matching notebook plus `scripts/colab_*`
- Router inference, adapter lookup, deployment handoff, smoke testing, or inference payload behavior: start from `src/workflows/inference.py`, `src/pipeline/router_adapter_runtime.py`, `src/pipeline/inference_payloads.py`, and `src/router/vlm_pipeline.py`
- CI, tests, benchmark capture, and docs consistency: inspect `.github/workflows/ci.yml`, `pyproject.toml`, `scripts/validate_notebook_imports.py`, and `scripts/benchmark_surfaces.py`

## Validation defaults

Use the narrowest relevant validation first:

- `python scripts/validate_notebook_imports.py`
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `python scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json`

Run benchmark capture when workflow entrypoints, router runtime orchestration, or benchmarked interfaces change.
