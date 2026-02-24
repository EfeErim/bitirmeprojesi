# Scripts Map

This folder contains operational scripts used for setup, testing, and policy checks.

## Colab Setup & Orchestration

- `install_colab.py` - installs Colab prerequisites.
- `colab_setup_dependencies.py` - dependency setup for Colab runtime.
- `colab_quick_setup.py` - quick environment setup path.
- `colab_auto_orchestrator.py` - automation helper for one-click flows.
- `download_data_colab.py` - dataset download utilities for Colab.

## VLM/Router Testing

- `colab_test_gpu_vlm.py` - GPU availability + VLM smoke checks.
- `colab_vlm_quick_test.py` - short VLM sanity checks.
- `colab_interactive_vlm_test.py` - interactive VLM test runner.
- `colab_test_upload.py` - Colab image-upload + BioCLIP preprocessing helper.
- `test_vlm_pipeline_standalone.py` - standalone VLM pipeline checks.

## Policy/Profile Regression

- `run_policy_regression_bundle.py` - main policy/stage-order regression bundle (used in CI).
- `profile_policy_sanity.py` - profile and policy sanity validation.
- `run_python_sanity_bundle.py` - consolidated local Python sanity checks (`validate_notebook_imports.py`, dynamic taxonomy, final pipeline check).

## Diagnostics & Utilities

- `debug_sam3_bioclip_pipeline.py` - debug tool for SAM3 + BioCLIP integration.
- `test_sam3_raw.py` - raw SAM3 probing helper.
- `test_bioclip_github.py` - BioCLIP source integration helper.
- `check_markdown_links.py` - repository markdown link validation.
- `benchmark_router_phase5.py` - deterministic Phase 5 router latency benchmark.
- `check_phase5_perf_regression.py` - validates benchmark output against Phase 5 guardrail thresholds.

## Recommended Usage

- Full training: prefer `../colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`.
- Policy regression testing: run `python scripts/run_policy_regression_bundle.py`.
- Core Python sanity checks: run `python scripts/run_python_sanity_bundle.py`.
- Quick docs validation: run `python scripts/check_markdown_links.py --root .`.
- Phase 5 benchmark baseline: run `python scripts/benchmark_router_phase5.py`.
- Phase 5 guardrail check: run `python scripts/check_phase5_perf_regression.py`.
