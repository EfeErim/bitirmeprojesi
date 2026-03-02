# AADS v6 Architecture Overview

## Component Diagram

```mermaid
flowchart LR
    INPUT[Input Image] --> ROUTER[VLMPipeline / SimpleCropRouter]
    ROUTER --> CROP[Crop + Part Decision]
    CROP --> ADAPTER[IndependentCropAdapter]
    ADAPTER --> TRAINER[ContinualSDLoRATrainer]
    TRAINER --> OOD[ContinualOODDetector]
    OOD --> RESULT[Diagnosis + OOD Analysis]
```

## Runtime Data Flow

1. `src/pipeline/independent_multi_crop_pipeline.py` accepts an image and normalizes input shape.
2. `src/router/vlm_pipeline.py` (or `src/router/simple_crop_router.py`) predicts crop/part with confidence.
3. Pipeline resolves the matching `src/adapter/independent_crop_adapter.py` instance.
4. Adapter runs `src/training/continual_sd_lora.py` inference surfaces and returns disease logits/probabilities.
5. Adapter invokes OOD scoring (`src/ood/continual_ood.py`) and attaches:
   - `ensemble_score`
   - `class_threshold`
   - `is_ood`
   - `calibration_version`
6. Pipeline emits a unified response payload with router confidence, diagnosis, and OOD analysis.

## OOD Architecture

```mermaid
flowchart TD
    FEAT[Encoded Features] --> MAH[Mahalanobis Z]
    FEAT --> ENG[Energy Z]
    MAH --> ENS[Weighted Ensemble]
    ENG --> ENS
    ENS --> THR[Per-Class Threshold]
    THR --> FLAG{is_ood?}
    FLAG -->|true| OODR[OOD Result]
    FLAG -->|false| IDR[In-Distribution Result]
```

- Score contract: `0.6 * mahalanobis_z + 0.4 * energy_z`
- Calibration state is versioned and persisted in adapter metadata.
- Threshold handling lives in `src/ood/dynamic_thresholds.py` and `src/ood/continual_ood.py`.

## Policy Graph (Router Stage Ordering)

```mermaid
flowchart TD
    START[analyze_image] --> ORDER[Resolve Stage Order]
    ORDER --> ROI[SAM3 ROI Candidate Stage]
    ROI --> CLS[ROI Classification Stage]
    CLS --> FILTER[ROI Filter + Postprocess]
    FILTER --> BEST[Best Detection Selection]
    BEST --> OUTPUT[Crop/Part/Confidence Output]
```

- Stage order and profile behavior are defined in `src/router/vlm_pipeline.py`.
- Policy/taxonomy normalization helpers are in `src/router/policy_taxonomy_utils.py`.
- Regression guardrails:
  - `tests/unit/router/test_vlm_policy_stage_order.py`
  - `tests/unit/router/test_vlm_strict_loading.py`

## Config and Contract Anchors

- Runtime config sources:
  - `config/training_config.json`
  - `config/router_config.json`
- Canonical contract specs:
  - `specs/adapter-spec.json`
  - `specs/router-spec.json`
  - `specs/pipeline-spec.json`
