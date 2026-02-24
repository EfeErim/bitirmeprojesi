# Colab Notebooks Map

Use this index to choose the right notebook quickly.

## Canonical Start Path

- **Primary (recommended):** `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
- **Manual/diagnostic path:** `colab_bootstrap.ipynb` then `1` → `6`

## Notebook Intent Matrix

| Notebook | Use When | Priority | Typical Runtime |
|---|---|---|---|
| `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` | You want full training with minimum manual intervention | Primary | 8-12h |
| `colab_bootstrap.ipynb` | You need explicit environment bootstrap/debug before phase notebooks | Secondary | 5-15m |
| `1_data_preparation.ipynb` | You need manual dataset prep/validation | Secondary | 10-30m |
| `2_phase1_training.ipynb` | You are iterating only Phase 1 | Secondary | 2-4h |
| `3_phase2_training.ipynb` | You are iterating only Phase 2 | Secondary | 1-3h |
| `4_phase3_training.ipynb` | You are iterating only Phase 3 | Secondary | 2-4h |
| `5_testing_validation.ipynb` | You need manual test/validation review | Secondary | 20-60m |
| `6_performance_monitoring.ipynb` | You need performance diagnostics/reporting | Secondary | 15-45m |
| `7_VLM_ROUTER_ONECLICK.ipynb` | You want one-click router/VLM focused checks | Specialized | 10-30m |
| `TEST_VLM_ROUTER.ipynb` | You want interactive/manual router debugging | Specialized | variable |

## Main Entrypoints

- `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`  
  One-click end-to-end training (recommended for full training runs).
- `7_VLM_ROUTER_ONECLICK.ipynb`  
  One-click router-focused VLM workflow.
- `TEST_VLM_ROUTER.ipynb`  
  Interactive/manual router checks and debugging.

## VLM Testing & Router Checks

For VLM-specific testing in Colab vs. local environments, see the **VLM Test Decision Matrix** in [`scripts/README.md`](../scripts/README.md#vlm-test-decision-matrix).

Quick reference:
- **Colab preflight VLM check:** `scripts/colab_vlm_quick_test.py`
- **Interactive router debugging:** `TEST_VLM_ROUTER.ipynb` (this folder)
- **One-click router + training:** `7_VLM_ROUTER_ONECLICK.ipynb` (this folder)
- **Standalone local VLM test:** `scripts/test_vlm_pipeline_standalone.py`

See [`scripts/README.md`](../scripts/README.md) for full decision criteria and when to use each approach.

## Phase-by-Phase Training

- `1_data_preparation.ipynb` → prepares and validates dataset structure.
- `2_phase1_training.ipynb` → DoRA phase training.
- `3_phase2_training.ipynb` → SD-LoRA phase training.
- `4_phase3_training.ipynb` → CoNeC-LoRA phase training.
- `5_testing_validation.ipynb` → model checks and validation passes.
- `6_performance_monitoring.ipynb` → performance and monitoring review.

## Setup Notebook

- `colab_bootstrap.ipynb`  
  Colab environment bootstrap helper.

## Related Files

- `../scripts/colab_auto_orchestrator.py` orchestrates one-click notebook flow.
- `../scripts/colab_setup_dependencies.py` installs notebook dependencies.
- `../requirements_colab.txt` is the canonical Colab dependency manifest.
- `requirements_colab.txt` mirrors the canonical file for notebook compatibility paths.
