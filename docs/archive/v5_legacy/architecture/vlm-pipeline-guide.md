# VLM Pipeline Guide

## Overview

The VLM layer is implemented in `src/router/vlm_pipeline.py` and currently exposes:

- `VLMPipeline`
- `DiagnosticScoutingAnalyzer`

It is designed as the crop-analysis backend used by the independent multi-crop pipeline.

## `VLMPipeline`

### Initialization

`VLMPipeline(config, device='cuda')`:

- resolves runtime device (`cuda` when available, otherwise `cpu`),
- reads VLM flags from nested `router.vlm` and flat compatibility keys,
- stores confidence threshold and max detection parameters.

### Model Loading

`load_models()` currently initializes model placeholders for:

- Grounding DINO
- SAM-2
- BioCLIP 2

This supports integration/testing flows while preserving extension points for production model loading.

### Core Methods

- `process_image(image_tensor)` returns a compact status/scenario payload.
- `analyze_image(...)` returns detection records (crop, part, disease, confidence, bbox).
- `route_batch(batch)` applies routing over image batches.

## `DiagnosticScoutingAnalyzer`

Provides a wrapper for quick or detailed analysis:

- `quick_assessment(...)` gives minimal status + explanation payload.
- `analyze_image(...)` provides crop/part predictions with confidence.

The analyzer maintains compatibility behavior when VLM is disabled.

## Integration with Main Pipeline

`IndependentMultiCropPipeline.initialize_router()` creates:

- `self.router = VLMPipeline(...)`
- `self.router_analyzer = DiagnosticScoutingAnalyzer(...)`

Routing output then drives adapter dispatch and output composition.

## Extension Points

For production hardening, the current code supports extending:

- real model loading in `load_models()`,
- calibrated confidence thresholds,
- richer region/mask outputs,
- per-crop routing metrics for monitoring.

## Validation

Use these checks when modifying VLM behavior:

- `python scripts/validate_notebook_imports.py`
- `pytest -c config/pytest.ini tests/unit/router`
- `pytest -c config/pytest.ini tests/integration`
