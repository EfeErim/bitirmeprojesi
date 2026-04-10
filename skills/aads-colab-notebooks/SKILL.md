---
name: aads-colab-notebooks
description: Use when changing AADS v6 Colab Notebook 0, 1, 2, 3, or 4 flows, grouped dataset preparation, dataset materialization, Hugging Face token handling, Drive telemetry, or notebook-specific troubleshooting.
---

# AADS Colab Notebooks

Use this skill for notebook wrapper behavior, Colab-only bootstrap logic, and notebook export or telemetry expectations.

## Inspect First

- `README.md`
- `docs/user_guide/colab_training_manual.md`
- `docs/architecture/overview.md`
- `colab_notebooks/0_grouped_dataset_preparation.ipynb`
- `colab_notebooks/1_router_adapter_inference.ipynb`
- `colab_notebooks/2_interactive_adapter_training.ipynb`
- `colab_notebooks/3_adapter_smoke_test.ipynb`
- `colab_notebooks/4_simple_adapter_smoke_test.ipynb`
- `scripts/colab_repo_bootstrap.py`
- `scripts/colab_notebook_helpers.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_live_telemetry.py`
- `scripts/evaluate_dataset_layout.py`

Load `skills/aads-training-ood/SKILL.md` for Notebook 0 or Notebook 2 training, readiness, or dataset-contract behavior. Load `skills/aads-inference-runtime/SKILL.md` for Notebook 1, Notebook 3, or Notebook 4 inference-side issues.

## Workflow

1. Keep notebooks as thin wrappers over maintained workflow and script surfaces instead of growing notebook-only orchestration.
2. Notebook 0 is the audit-first grouped dataset preparation surface. Preserve its role in duplicate-aware review and optional runtime-dataset materialization.
3. Notebook 2 accepts either a flat class-root dataset or an already prepared runtime dataset root. Workflow and CLI training still expect the runtime dataset contract. Do not blur those two contracts.
4. Notebook 2 writes local outputs under `outputs/colab_notebook_training/`, uses repo-local telemetry/checkpoint storage by default, and mirrors non-checkpoint exports into `runs/<RUN_ID>/`.
5. Notebook 3 is the fuller direct adapter validation surface, and Notebook 4 is its smaller widget wrapper. Both are separate from router-driven inference and should stay usable without the router.
6. Keep Hugging Face token resolution aligned with the maintained sources documented in the Colab manual.
7. When notebook prose, defaults, or exposed controls imply methodological claims about training, OOD handling, inference behavior, or data-prep policy, align them with the canonical workflow docs and literature-backed rationale where available. Avoid notebook-only scientific claims.

## Boundaries

- Do not make notebook behavior the primary source of truth when workflow code already defines the contract.
- Do not rewrite local run exports or checkpoint trees as tracked repo fixtures.
- Use `aads-repo-hygiene` too if the task changes CI coverage, maintained validation commands, or contributor-facing docs.
- Do not imply a paper-faithful method or benchmark claim from notebook UX alone unless the underlying workflow and docs substantiate it.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `.\scripts\python.cmd scripts/validate_config_schema.py`
- `.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root <flat_class_root>` for Notebook 0 or Notebook 2 dataset-contract changes
- `pytest tests/colab/test_smoke_training.py -q`
- Run the narrowest relevant notebook-adjacent tests or helper imports for the touched Colab scripts.
