# AADS-ULoRA Architecture Overview

## Purpose

AADS-ULoRA is a multi-crop disease diagnosis system with:

- a routing stage to identify crop/part,
- crop-specific adapter inference,
- out-of-distribution (OOD) checks,
- training/inference-focused output composition.

This document reflects the current implementation in the repository.

## High-Level Flow

1. **Input ingress** via notebooks/scripts and pipeline entry utilities.
2. **Image decoding + preprocessing** in router/pipeline helpers.
3. **Pipeline orchestration** in `src/pipeline/independent_multi_crop_pipeline.py`.
4. **Router analysis** through `src/router/vlm_pipeline.py` (`VLMPipeline` + `DiagnosticScoutingAnalyzer`).
5. **Adapter inference** per crop (if available).
6. **OOD analysis + response assembly** returned to calling workflows.

## Core Components

### Runtime Integration Layer

- Runtime orchestration is currently centered on notebooks and scripts in `colab_notebooks/` and `scripts/`.
- API/deployment documentation is preserved only as legacy archive material.

### Configuration Layer

- `src/core/config_manager.py` loads base and runtime configuration files.
- Primary runtime configs live in `config/` (`base.json`, `colab.json`, `perf_guardrails_phase5.json`).

### Orchestration Layer

- `IndependentMultiCropPipeline` handles:
  - router/adapters initialization,
  - image routing and inference sequencing,
  - cache and OOD result handling,
  - final normalized output payload.

### Routing Layer

- `VLMPipeline` provides routing-oriented image analysis.
- `DiagnosticScoutingAnalyzer` provides a simplified classification/diagnostic wrapper used by the pipeline.

### Training Layer

- Colab-first training implementation under `src/training/`:
  - `colab_phase1_training.py`
  - `colab_phase2_sd_lora.py`
  - `colab_phase3_conec_lora.py`

### Testing Layer

- Pytest-based test suites in `tests/`:
  - `tests/colab/` for environment/data/training smoke coverage,
  - `tests/integration/` for pipeline integration checks,
  - `tests/unit/` for component-level behavior.

## Runtime Modes

### Local Development

- Use `.venv` + `requirements.txt`.
- Run sanity and regression scripts under `scripts/`.

### Colab Training

- Bootstrapped through `colab_bootstrap.ipynb`.
- Sequential notebooks in `colab_notebooks/` for data prep, 3-phase training, validation, and monitoring.

## Current Architectural Notes

- Router and adapter abstractions are present; some model operations are implemented with fallback/stub behavior for resilience and testing.
- Documentation and tests are primarily aligned with the Colab workflow.
- Configuration supports environment overlays and runtime feature toggles.

## Related Docs

- `docs/api/api-reference.md`
- `docs/colab_migration_guide.md`
- `docs/user_guide/colab_training_manual.md`

> Note: `docs/api/api-reference.md` is archived legacy documentation and not part of the primary workflow.
