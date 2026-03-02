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
| Check markdown links | `check_markdown_links.py --root .` | Docs updates |
| Validate class-root dataset layout for training notebook | `evaluate_dataset_layout.py --root <path>` | Before running `2_interactive_adapter_training.ipynb` |
| Benchmark phase5 router | `benchmark_router_phase5.py` | Performance baseline updates |
| Enforce phase5 performance guardrails | `check_phase5_perf_regression.py` | Benchmark regression gate |

## Colab Setup & Orchestration

- `install_colab.py` - installs Colab prerequisites.
- `colab_setup_dependencies.py` - dependency setup for Colab runtime.
- `colab_quick_setup.py` - quick environment setup path.
- `colab_auto_orchestrator.py` - automation helper for multi-stage Colab runs.
- `download_data_colab.py` - dataset download utilities for Colab.

Notebook bootstrap behavior in `colab_notebooks/`:
- Auto-discovers repository root from common Colab and Drive paths.
- Falls back to auto-clone when root is missing.
- Supports `AADS_REPO_ROOT`, `REPO_ROOT`, `AADS_REPO_CLONE_TARGET`, `AADS_REPO_URL`, and `AADS_DISABLE_AUTO_CLONE`.

## VLM/Router Testing

These scripts are router diagnostics and are not required for continual SD-LoRA adapter training.

- `colab_test_gpu_vlm.py` - GPU availability + VLM smoke checks.
- `colab_vlm_quick_test.py` - short VLM sanity checks.
- `colab_interactive_vlm_test.py` - interactive VLM test runner.
- `colab_test_upload.py` - Colab image-upload + BioCLIP preprocessing helper.
- `test_vlm_pipeline_standalone.py` - standalone VLM pipeline checks.

### VLM Test Decision Matrix

| Scenario | Preferred Surface | Status |
|---|---|---|
| Quick Colab VLM sanity check | `colab_vlm_quick_test.py` | Primary |
| Interactive repeated image probing in Colab | `colab_interactive_vlm_test.py` | Primary |
| End-to-end standalone local VLM pipeline check | `test_vlm_pipeline_standalone.py` | Primary |
| Legacy GPU-first debug script with overlapping scope | `colab_test_gpu_vlm.py` | Legacy/secondary |
| Upload-focused BioCLIP preprocessing check | `colab_test_upload.py` | Specialized |

For new docs and user instructions, prefer the **Primary** surfaces above.

## Policy/Profile Regression

- `run_policy_regression_bundle.py` - main policy/stage-order regression bundle (used in CI).
- `run_test_suites.py` - modular pytest runner with named suites (`quick`, `unit`, `colab`, `integration`, `all`) and per-suite pass/fail summary.
- `profile_policy_sanity.py` - profile and policy sanity validation.
- `run_python_sanity_bundle.py` - consolidated local Python sanity checks (`scripts/validate_notebook_imports.py`, dynamic taxonomy, final pipeline check).

## Diagnostics & Utilities

- `debug_sam3_bioclip_pipeline.py` - debug tool for SAM3 + BioCLIP integration.
- `test_sam3_raw.py` - raw SAM3 probing helper.
- `test_bioclip_github.py` - BioCLIP source integration helper.
- `check_markdown_links.py` - repository markdown link validation.
- `benchmark_router_phase5.py` - deterministic Phase 5 router latency benchmark.
- `check_phase5_perf_regression.py` - validates benchmark output against Phase 5 guardrail thresholds.

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
