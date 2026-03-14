# AADS v6 Agent Guide

This repo is intentionally narrow. The maintained product surface is continual SD-LoRA adapter training, OOD readiness, router-driven inference, and Colab notebook wrappers for those same flows. This file is for coding agents working inside this repo.

## Source Of Truth

Read these before making repo-wide assumptions:

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `docs/user_guide/colab_training_manual.md`
- `docs/user_guide/ood_readiness_guide.md`

## Maintained Entrypoints

- Training: `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- Inference: `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`
- Notebook training surface: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook inference surface: `colab_notebooks/1_router_adapter_inference.ipynb`
- Direct adapter validation: `colab_notebooks/3_adapter_smoke_test.ipynb` and `scripts/colab_adapter_smoke_test.py`
- Repo validation and automation: `.github/workflows/ci.yml`, `scripts/validate_notebook_imports.py`, `scripts/validate_config_schema.py`, `scripts/benchmark_surfaces.py`

## Tracked Vs Local-Generated

Tracked source of truth:

- `src/`, `tests/`, `scripts/`, `config/`
- `docs/` and `README.md`
- `colab_notebooks/*.ipynb`
- Root dependency files

Local or generated only:

- `runs/<RUN_ID>/`
- `models/adapters/<crop>/continual_sd_lora_adapter/`
- `outputs/`
- `.runtime_tmp/`, caches, temp folders, virtualenvs

Do not treat generated outputs as tracked implementation files unless the user explicitly asks about generated artifacts. Do not vendor or edit Codex-home global skills from this repo.

## Repo-Local Skills

Project-local skills live under `skills/` and should be preferred for repo-specific work:

- `aads-training-ood`: `skills/aads-training-ood/SKILL.md`
- `aads-colab-notebooks`: `skills/aads-colab-notebooks/SKILL.md`
- `aads-inference-runtime`: `skills/aads-inference-runtime/SKILL.md`
- `aads-repo-hygiene`: `skills/aads-repo-hygiene/SKILL.md`

Use the smallest set that covers the task.

## Routing Rules

- Use `aads-training-ood` for `TrainingWorkflow.run(...)`, continual SD-LoRA config, OOD calibration, readiness artifacts, BER comparisons, and training-side adapter export semantics.
- Use `aads-colab-notebooks` for Notebook 1, 2, or 3 changes, dataset materialization, Hugging Face token handling, Drive telemetry, notebook output mirroring, and notebook-specific troubleshooting.
- Use `aads-inference-runtime` for router inference, adapter lookup and deployment handoff, lazy adapter loading, direct adapter smoke testing, and inference payload behavior.
- Use `aads-repo-hygiene` for CI, tests, benchmark capture, docs consistency, and tracked-vs-generated repo boundaries.

## Overlap Rules

- Use `aads-training-ood` plus `aads-colab-notebooks` for Notebook 2 changes and notebook/export mismatches.
- Use `aads-training-ood` plus `aads-repo-hygiene` for training-side code changes that also affect tests, docs, metrics, or CI coverage.
- Use `aads-inference-runtime` plus `aads-repo-hygiene` for runtime bugfixes, adapter lookup regressions, or inference-facing docs and tests.
- Use `aads-colab-notebooks` plus `aads-inference-runtime` for Notebook 1 or Notebook 3 tasks that stay on inference and adapter-validation surfaces.
- If a task spans training and inference through the saved adapter contract, anchor on the canonical workflow and runtime entrypoints rather than notebook-only behavior.

## Default Validation Commands

Start with the narrowest relevant subset:

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `.\scripts\python.cmd scripts/validate_config_schema.py`
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `.\scripts\python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json`

Run benchmark capture when orchestration interfaces, workflow entrypoints, or router runtime behavior change.

## Non-Goals

- This repo does not define autonomous runtime agents.
- GitHub Actions jobs are automation surfaces, not the "agents" managed by this file.
- Keep project-local skills lean and route to existing docs and scripts instead of copying long guides into skill bodies.
