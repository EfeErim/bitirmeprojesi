# Repository File & Relationship Map

This guide is a practical map of what each major file/group does and how pieces connect.

## Start Here

- Project entry: `../README.md`
- Documentation index: `README.md`
- Notebook index: `../colab_notebooks/README.md`
- Script index: `../scripts/README.md`

## Top-Level Files (Root)

### Core project files
- `../README.md` - main project overview, setup, and run paths.
- `../requirements.txt` - local/dev Python dependencies.
- `../requirements_colab.txt` - Colab-focused dependency set.
- `../setup.py` - package metadata and install configuration.
- `../LICENSE` - project license.
- `../SECURITY.md` - vulnerability reporting policy.
- `../.env.example` - example environment variables.
- `../sitecustomize.py` - runtime/site-level Python environment customization.

### Root Python utilities/tests
- `../validate_notebook_imports.py` - compatibility wrapper to canonical script.
- `../colab_test_upload.py` - compatibility wrapper to canonical script.
- `../test_dynamic_taxonomy.py` - compatibility wrapper to canonical script.
- `../test_pipeline_final_check.py` - compatibility wrapper to canonical script.

### Root notebooks and guides
- `../colab_bootstrap.ipynb` - root-level bootstrap notebook (legacy-compatible location).
- `guides/CHECKPOINT_SYSTEM_GUIDE.md` - checkpointing behavior and recovery.
- `guides/COLAB_QUICK_START.md` - quick Colab onboarding.
- `guides/SEAMLESS_AUTOTRAIN_GUIDE.md` - auto-train flow guidance.
- `guides/OUTPUT_FILES_SPECIFICATION.md` - artifact/output contract.

### Reports/summaries (status artifacts)
- `reports/README.md` - consolidated reports index.
- `reports/FINAL_EXAMINATION_REPORT.md`
- `reports/SESSION_6_COMPLETION_SUMMARY.md`
- `reports/v55/README.md` - V55 report set index.
- `reports/v55/V55_CRITICAL_FIXES_SUMMARY.md`
- `reports/v55/V55_FINAL_STATUS_REPORT.md`
- `reports/v55/V55_IMPLEMENTATION_SUMMARY.md`
- `reports/v55/V55_INTEGRATION_GUIDE.md`
- `reports/v55/V55_ROUTER_ARCHITECTURE.md`
- `reports/v55/V55_SPECIFICATION_AUDIT.md`
- `reports/v55/VLM_FIX_SUMMARY.md`
- `guides/COLAB_MIGRATION_IMPLEMENTATION.md`

## Source Code Relationships (`src/`)

### Execution path (high-level)
1. Training logic is implemented in `../src/training/`.
2. Artifacts/config contracts are defined under `../src/core/`.
3. Inference dispatch uses `../src/router/` + `../src/adapter/`.
4. OOD safety checks come from `../src/ood/`.
5. Integrated multi-crop execution is in `../src/pipeline/`.

### `src/core/` (contracts + orchestration)
- `config_manager.py` - loads/manages configuration.
- `configuration_validator.py` / `validation.py` - validates config integrity.
- `schemas.py` - schema definitions used across config and pipeline contracts.
- `pipeline_manager.py` - coordinates pipeline stage execution.
- `model_registry.py` - model/artifact registration and lookup.
- `artifact_manifest.py` - output artifact manifest contract helpers.
- `colab_contract.py` - Colab-specific artifact/runtime contract.

### `src/router/` (policy + dispatch)
- `vlm_pipeline.py` - policy/profile-driven VLM routing pipeline.
- `simple_crop_router.py` - simpler crop routing utility path.

### `src/adapter/`
- `independent_crop_adapter.py` - crop-specific adapter interface/logic.

### `src/training/`
- `phase1_training.py` - Phase 1 (DoRA).
- `phase2_sd_lora.py` - Phase 2 (SD-LoRA).
- `phase3_conec_lora.py` - Phase 3 (CoNeC-LoRA).
- `colab_phase1_training.py` / `colab_phase2_sd_lora.py` / `colab_phase3_conec_lora.py` - Colab wrappers/entry variants.

### `src/ood/`
- `prototypes.py` - prototype-based OOD components.
- `mahalanobis.py` - Mahalanobis distance OOD scoring.
- `dynamic_thresholds.py` - dynamic/learned threshold handling.

### `src/pipeline/`
- `independent_multi_crop_pipeline.py` - end-to-end multi-crop pipeline assembly.

### Supporting modules
- `src/dataset/*` - dataset prep, loading, caching, Colab data utilities.
- `src/evaluation/*` - evaluation metrics and v5.5 metric variants.
- `src/debugging/*` and `src/monitoring/*` - runtime profiling/monitoring metrics.
- `src/utils/*` - shared data/model helper utilities.
- `src/visualization/visualization.py` - visualization utilities.

## Notebook Relationships (`colab_notebooks/`)

- Preferred full workflow: `../colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`.
- Step-by-step flow: `1_data_preparation` → `2_phase1_training` → `3_phase2_training` → `4_phase3_training` → `5_testing_validation` → `6_performance_monitoring`.
- Router-specific one-click: `7_VLM_ROUTER_ONECLICK.ipynb`.
- Manual router probing: `TEST_VLM_ROUTER.ipynb`.
- See `../colab_notebooks/README.md` for quick notebook selection.

## Script Relationships (`scripts/`)

- CI policy regression job runs `../scripts/run_policy_regression_bundle.py`.
- Local Python sanity bundle runs from `../scripts/run_python_sanity_bundle.py`.
- Canonical sanity scripts: `../scripts/validate_notebook_imports.py`, `../scripts/test_dynamic_taxonomy.py`, `../scripts/test_pipeline_final_check.py`.
- Canonical Colab upload helper: `../scripts/colab_test_upload.py`.
- Colab setup scripts support notebook bootstrapping and automation.
- Debug/test scripts provide targeted troubleshooting for VLM/SAM3/BioCLIP.
- See `../scripts/README.md` for categorized script usage.

## Configuration, Tests, and Specs

- `../config/base.json` and `../config/colab.json` provide runtime config baselines.
- `../config/plant_taxonomy.json` supports taxonomy-driven behavior.
- `../config/pytest.ini` defines pytest behavior.
- `../tests/` mirrors source modules with unit/integration/colab-focused checks.
- `../specs/adapter-spec.json` defines adapter-level structural contract.

## Practical Navigation Rules

- For training runs, start from notebooks first.
- For regression verification, use scripts first.
- For code changes, edit `src/` and validate with mirrored `tests/`.
- For behavior changes, check both config (`config/`) and policy router (`src/router/vlm_pipeline.py`).
