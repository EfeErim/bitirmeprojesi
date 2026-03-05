# Scripts Map

This folder contains operational scripts used for setup, testing, and policy checks.

## Canonical Usage Policy

- Prefer calling scripts through `scripts/...` paths in all user docs.

## Script Intent Matrix

| User Goal | Preferred Script | When to Use |
|---|---|---|
| Run modular test subsets with suite-level status | `scripts/run_test_suites.py` | Day-to-day local testing and targeted debugging |
| Verify core local Python sanity | `scripts/run_python_sanity_bundle.py` | Before local changes/PRs |
| Validate notebook-related imports only | `scripts/validate_notebook_imports.py` | Fast import compatibility check |
| Run policy/profile regression | `scripts/run_policy_regression_bundle.py` | Router/policy changes |
| Check markdown links | `scripts/check_markdown_links.py --root .` | Docs updates |
| Validate class-root dataset layout for training notebook | `scripts/evaluate_dataset_layout.py --root <path>` | Before running `2_interactive_adapter_training.ipynb` |
| Benchmark phase5 router | `scripts/benchmark_router_phase5.py` | Performance baseline updates |
| Enforce phase5 performance guardrails | `scripts/check_phase5_perf_regression.py` | Benchmark regression gate |

## Colab Setup & Orchestration

- `scripts/install_colab.py` - installs Colab prerequisites.
- `scripts/colab_setup_dependencies.py` - dependency setup for Colab runtime.
- `scripts/colab_quick_setup.py` - quick environment setup path.
- `scripts/colab_auto_orchestrator.py` - automation helper for multi-stage Colab runs.
- `scripts/colab_repo_bootstrap.py` - shared repo-root resolution and optional auto-clone helpers used by both active notebooks.
- `scripts/colab_live_telemetry.py` - live event/log/artifact writer with local spool + Drive synchronization.
- `scripts/colab_checkpointing.py` - rolling checkpoint manager (latest/best manifests + retention pruning).
- `scripts/colab_notebook_helpers.py` - reusable notebook artifact/checkpoint helper utilities to keep notebook cells thin.
- `scripts/download_data_colab.py` - dataset download utilities for Colab.

Notebook bootstrap behavior in `colab_notebooks/`:
- Auto-discovers repository root from common Colab and Drive paths.
- Falls back to auto-clone when root is missing.
- Supports `AADS_REPO_ROOT`, `REPO_ROOT`, `AADS_REPO_CLONE_TARGET`, `AADS_REPO_URL`, and `AADS_DISABLE_AUTO_CLONE`.

## VLM/Router Testing

These scripts are router diagnostics and are not required for continual SD-LoRA adapter training.

- `scripts/colab_test_gpu_vlm.py` - GPU availability + VLM smoke checks.
- `scripts/colab_vlm_quick_test.py` - short VLM sanity checks.
- `scripts/colab_interactive_vlm_test.py` - interactive VLM test runner.
- `scripts/colab_test_upload.py` - Colab image-upload + BioCLIP preprocessing helper.
- `scripts/test_vlm_pipeline_standalone.py` - standalone VLM pipeline checks.

### VLM Test Decision Matrix

| Scenario | Preferred Surface | Status |
|---|---|---|
| Quick Colab VLM sanity check | `scripts/colab_vlm_quick_test.py` | Primary |
| Interactive repeated image probing in Colab | `scripts/colab_interactive_vlm_test.py` | Primary |
| End-to-end standalone local VLM pipeline check | `scripts/test_vlm_pipeline_standalone.py` | Primary |
| Legacy GPU-first debug script with overlapping scope | `scripts/colab_test_gpu_vlm.py` | Legacy/secondary |
| Upload-focused BioCLIP preprocessing check | `scripts/colab_test_upload.py` | Specialized |

For new docs and user instructions, prefer the **Primary** surfaces above.

## Policy/Profile Regression

- `scripts/run_policy_regression_bundle.py` - main policy/stage-order regression bundle (used in CI).
- `scripts/run_test_suites.py` - modular pytest runner with named suites (`quick`, `unit`, `colab`, `integration`, `all`) and per-suite pass/fail summary.
- `scripts/profile_policy_sanity.py` - profile and policy sanity validation.
- `scripts/run_python_sanity_bundle.py` - consolidated local Python sanity checks (`scripts/validate_notebook_imports.py`, dynamic taxonomy, final pipeline check).

## Diagnostics & Utilities

- `scripts/debug_sam3_bioclip_pipeline.py` - debug tool for SAM3 + BioCLIP integration.
- `scripts/test_sam3_raw.py` - raw SAM3 probing helper.
- `scripts/test_bioclip_github.py` - BioCLIP source integration helper.
- `scripts/check_markdown_links.py` - repository markdown link validation.
- `scripts/benchmark_router_phase5.py` - deterministic Phase 5 router latency benchmark.
- `scripts/check_phase5_perf_regression.py` - validates benchmark output against Phase 5 guardrail thresholds.

## Recommended Usage

- Router/inference notebook: `../colab_notebooks/1_crop_router_pipeline.ipynb`.
- Interactive training notebook: `../colab_notebooks/2_interactive_adapter_training.ipynb`.
- Superseded v6 archive: `../colab_notebooks/archive/v6_superseded_2026-03-02/`.
- Fast modular local tests: run `python scripts/run_test_suites.py` (defaults to `quick` suite group).
- Full modular test matrix: run `python scripts/run_test_suites.py --suite all`.
- Policy regression testing: run `python scripts/run_policy_regression_bundle.py`.
- Core Python sanity checks: run `python scripts/run_python_sanity_bundle.py`.
- Quick docs validation: run `python scripts/check_markdown_links.py --root .`.
- Phase 5 benchmark baseline: run `python scripts/benchmark_router_phase5.py`.
- Phase 5 guardrail check: run `python scripts/check_phase5_perf_regression.py`.
- Optional VLM router sanity (Colab): run `%run scripts/colab_vlm_quick_test.py`.
- Optional VLM interactive checks (Colab): use `from scripts.colab_interactive_vlm_test import run_interactive_vlm_test`.
