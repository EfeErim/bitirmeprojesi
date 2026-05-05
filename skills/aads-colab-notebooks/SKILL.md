---
name: aads-colab-notebooks
description: Use when changing AADS v6 Colab Notebook 0, 1, 2, 3, 4, or 5 flows, grouped dataset preparation, router calibration, dataset materialization, Hugging Face token handling, Drive telemetry, or notebook-specific troubleshooting.
---

# AADS Colab Notebooks

Use this skill for notebook wrapper behavior, Colab-only bootstrap logic, and notebook export or telemetry expectations.

## Inspect First

- `README.md`
- `docs/user_guide/colab_training_manual.md`
- `docs/architecture/overview.md`
- `colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb`
- `colab_notebooks/1_identify_crop_part_with_router.ipynb`
- `colab_notebooks/2_train_continual_sd_lora_adapter.ipynb`
- `colab_notebooks/3_validate_exported_adapter_directly.ipynb`
- `colab_notebooks/4_simple_direct_adapter_test_ui.ipynb`
- `colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb`
- `scripts/colab_repo_bootstrap.py`
- `scripts/colab_notebook_helpers.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_live_telemetry.py`
- `scripts/evaluate_dataset_layout.py`

Load `skills/aads-training-ood/SKILL.md` for Notebook 0 or Notebook 2 training, readiness, or dataset-contract behavior. Load `skills/aads-inference-runtime/SKILL.md` for Notebook 1, Notebook 3, Notebook 4, or Notebook 5 inference-side issues.

## Workflow

1. Keep notebooks as thin wrappers over maintained workflow and script surfaces instead of growing notebook-only orchestration.
2. Notebook 0 is the audit-first grouped dataset preparation surface. Preserve its role in duplicate-aware review and optional runtime-dataset materialization.
3. Notebook 2 accepts either a flat class-root dataset or an already prepared runtime dataset root. Workflow and CLI training still expect the runtime dataset contract. Do not blur those two contracts.
4. Notebook 2 writes local outputs under `outputs/colab_notebook_training/`, uses repo-local telemetry/checkpoint storage by default, and mirrors non-checkpoint exports into `runs/<RUN_ID>/`.
5. Notebook 3 is the fuller direct adapter validation surface, and Notebook 4 is its smaller widget wrapper. Both are separate from router-driven inference and should stay usable without the router.
6. Notebook 5 is the router calibration wrapper over `scripts/evaluate_router_surface.py` and `scripts/calibrate_router_surface.py`; keep calibration logic in those scripts.
7. Keep Hugging Face token resolution aligned with the maintained sources documented in the Colab manual.
8. Preserve deterministic handoff cells: setup, dataset validation, training or inference execution, artifact mirroring, and publish/recovery steps should each expose enough state for salvage without rerunning expensive work.
9. When notebook prose, defaults, or exposed controls imply methodological claims about training, OOD handling, inference behavior, or data-prep policy, align them with the canonical workflow docs and literature-backed rationale where available. Avoid notebook-only scientific claims.
10. Prefer validation and recovery cells over hidden notebook state. If a run completed and publishing failed, salvage artifacts before suggesting retraining.

## Literature And Practice Anchors

- Treat notebooks as reproducible wrappers over canonical workflows. The SOTA concern here is not novelty; it is preserving provenance, deterministic setup, and recoverable artifacts.
- For data preparation notebooks, follow the repo's leakage-prevention docs before adding augmentation or materialization convenience.
- For Notebook 2 readiness controls, defer to OOD calibration, Outlier Exposure, and held-out evidence guidance in `docs/user_guide/ood_readiness_guide.md`.
- For Notebook 5 router calibration, defer to `docs/architecture/router_performance_literature_review.md`: evaluate risk-coverage and negative false accepts before changing thresholds.

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
