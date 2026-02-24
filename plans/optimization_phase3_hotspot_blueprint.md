# Phase 3 Blueprint: Hotspot Decomposition & Risk-Controlled Refactors

Date: 2026-02-24  
Scope: Decompose oversized modules with high coupling while preserving behavior.

## 1) Hotspot Baseline Metrics

Measured hotspots:

| Module | LOC | Functions | Classes | Primary Risk |
|---|---:|---:|---:|---|
| `src/router/vlm_pipeline.py` | 2615 | 68 | 2 | Router logic concentration + policy staging complexity |
| `src/training/colab_phase3_conec_lora.py` | 966 | 39 | 9 | Trainer lifecycle and OOD logic tightly coupled |
| `src/debugging/performance_monitor.py` | 1079 | 50 | 9 | Monitoring stack combines config, runtime, and optimization |

Largest functions observed (by LOC):
- `VLMPipeline._classify_sam3_roi_candidate` (~152 LOC)
- `VLMPipeline._analyze_image_sam3` (~128 LOC)
- `VLMPipeline.__init__` (~122 LOC)
- `ColabPhase3Trainer.train_epoch` (~106 LOC)
- `ColabPhase3Trainer.train` (~81 LOC)
- `PerformanceMonitor.analyze_and_optimize` (~79 LOC)
- `PerformanceMonitor.record_batch` (~64 LOC)

## 2) Coupling Map (Usage Pressure)

- `VLMPipeline` is consumed heavily by:
  - runtime pipeline (`src/pipeline/independent_multi_crop_pipeline.py`)
  - multiple scripts (`scripts/*vlm*`, `scripts/profile_policy_sanity.py`)
  - extensive tests (`tests/unit/router/*`, `tests/unit/pipeline/*`, integration smoke).

- `ColabPhase3Trainer` is consumed by:
  - notebook import validation (`scripts/validate_notebook_imports.py`)
  - colab/integration/unit tests (`tests/colab/test_smoke_training.py`, `tests/integration/test_colab_integration.py`, `tests/unit/training/test_phase3_conec_lora.py`).

- `PerformanceMonitor` has low external coupling (mostly self-contained in module), making it safest for first extraction wave.

## 3) Implemented in This Phase

### A) Hotspot defect fix (done)
- `src/debugging/performance_monitor.py`
  - Fixed invalid import (`ConfigManager` -> `ConfigurationManager`).
  - Fixed broken standalone path insertion (`.../src/src` issue).
  - Updated `create_performance_monitor()` to load configuration via `ConfigurationManager.load_all_configs()` for both directory and file-path input.

Impact:
- Removes runtime failure risk in monitor factory path.
- Aligns monitor config access with canonical config model from Phase 2.

## 4) Decomposition Plan (Execution Order)

### Wave 3.1 (Low-risk extraction first)
Target: `src/debugging/performance_monitor.py`

1. Extract immutable metric dataclasses and serializers to `src/debugging/monitoring_types.py`.
2. Extract hardware collectors to `src/debugging/collectors/`:
   - `gpu_collector.py`
   - `memory_collector.py`
   - `drive_collector.py`
3. Keep `PerformanceMonitor` as orchestration facade with unchanged public API.
4. Add compatibility imports in original module to avoid import breaks.

Exit criteria:
- Existing monitor tests pass unchanged.
- `create_performance_monitor()` behavior preserved.

### Wave 3.2 (Training trainer decomposition)
Target: `src/training/colab_phase3_conec_lora.py`

1. Extract config and prototype/OOD helpers:
   - `phase3_config.py`
   - `phase3_ood.py`
2. Extract training-loop helpers (`epoch`, `validation`, checkpoint IO) to `phase3_runtime.py`.
3. Keep `ColabPhase3Trainer` as stable facade with thin delegated methods.

Exit criteria:
- `tests/colab/test_smoke_training.py` remains green.
- No changes required in notebook scripts importing `ColabPhase3Trainer`.

### Wave 3.3 (Router staged decomposition)
Target: `src/router/vlm_pipeline.py`

1. Extract policy/profile resolution + taxonomy/compatibility loaders into `router/policy_taxonomy_utils.py`.
2. Extract ROI utility helpers into `router/roi_helpers.py`.
3. Extract ROI scoring/classification orchestration into `router/roi_pipeline.py`.
4. Keep `VLMPipeline` API stable; route internals via composition.

Exit criteria:
- Stage-order tests stay green (`tests/unit/router/test_vlm_policy_stage_order.py`).
- Strict-loading behavior unchanged (`tests/unit/router/test_vlm_strict_loading.py`).

## 5) Guardrails

- No public constructor signature changes for `VLMPipeline`, `ColabPhase3Trainer`, `PerformanceMonitor`.
- No policy-threshold default changes without dedicated regression approval.
- Every extraction wave runs targeted tests before broader suites.

## 6) Validation Plan

Per-wave targeted validation:
1. Router: `pytest -c config/pytest.ini tests/unit/router -v`
2. Training: `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -v`
3. Monitor: `pytest -c config/pytest.ini tests/unit/monitoring/test_monitoring_metrics.py -v`
4. Docs integrity: `python scripts/check_markdown_links.py --root .`

## 7) Next Step

Proceed with Wave 3.2 implementation (Phase 3 trainer decomposition).

## 8) Wave 3.1 Completion Update

Status: **COMPLETED**

Delivered:
- Extracted monitoring dataclasses to `src/debugging/monitoring_types.py`.
- Extracted monitoring collectors to `src/debugging/collectors.py`.
- Refactored `src/debugging/performance_monitor.py` to import extracted modules while preserving public API exports.
- Fixed monitor factory config loading to align with `ConfigurationManager`.

Validation:
- `pytest -c config/pytest.ini tests/unit/monitoring/test_monitoring_metrics.py -v` -> **26 passed**.
- Backward-compatible imports from `src.debugging.performance_monitor` verified.
- Markdown link validation passed.

## 9) Wave 3.2 Progress Update

Status: **COMPLETED**

Delivered in Step 1:
- Extracted Phase 3 configuration and helper components to `src/training/phase3_components.py`:
  - `CoNeCConfig`
  - `PrototypeManager`
  - `MahalanobisDetector`
  - `DynamicThresholdManager`
  - `ColabMemoryMonitor`
- Refactored `src/training/colab_phase3_conec_lora.py` to import these components, removing duplicated in-file definitions.

Compatibility validation:
- `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v` -> **8 passed**.
- `python scripts/validate_notebook_imports.py` -> **8/8 checks passed**.

Delivered in Step 2:
- Extracted runtime helpers to `src/training/phase3_runtime.py`:
  - `phase3_training_step`
  - `phase3_train_epoch`
  - `phase3_validate`
  - `phase3_save_checkpoint`
  - `phase3_load_checkpoint`
- Refactored `src/training/colab_phase3_conec_lora.py` to delegate runtime-heavy methods to helper functions while preserving `ColabPhase3Trainer` method signatures.

Final Wave 3.2 validation:
- `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v` -> **8 passed**.
- `python scripts/validate_notebook_imports.py` -> **8/8 checks passed**.

Next Step:
- Start Wave 3.3 (router staged decomposition) on `src/router/vlm_pipeline.py`.

## 10) Wave 3.3 Progress Update

Status: **COMPLETED**

Delivered in Step 1:
- Added `src/router/policy_taxonomy_utils.py` and extracted:
  - policy/profile utilities (`deep_merge_dicts`, policy graph defaults/building, profile resolution/application)
  - policy access helpers (`policy_value`, `policy_enabled`)
  - taxonomy loaders (`load_taxonomy`, `load_crop_part_compatibility`)
- Refactored `src/router/vlm_pipeline.py` methods to delegate to extracted helpers while preserving the existing `VLMPipeline` API and method names.

Validation:
- `pytest -c config/pytest.ini tests/unit/router -v` -> **16 passed**.
- `python scripts/profile_policy_sanity.py` -> stage-order/profile sanity output validated for `base.json` and `colab.json`.

Delivered in Step 2:
- Added `src/router/roi_helpers.py` and extracted:
  - image coercion/conversion helpers (`tensor_to_pil`, `coerce_image_input`, `extract_roi`)
  - bbox math + postprocess helpers (`sanitize_bbox`, `bbox_area_ratio`, `bbox_iou`, `suppress_overlapping_detections`)
  - selection/normalization helpers (`select_best_detection`, `unique_nonempty`)
- Refactored `src/router/vlm_pipeline.py` helper methods to delegate to extracted implementations while preserving existing method signatures.

Validation (post-Step 2):
- `pytest -c config/pytest.ini tests/unit/router -v` -> **16 passed**.
- `python scripts/profile_policy_sanity.py` -> profile/stage-order output validated for `base.json` and `colab.json`.
- `python scripts/check_markdown_links.py` -> **no broken links**.

Delivered in Step 3:
- Added `src/router/roi_pipeline.py` and extracted:
  - SAM3 ROI candidate collection orchestration (`collect_sam3_roi_candidates`)
  - per-ROI crop/part fusion flow (`classify_sam3_roi_candidate`)
  - ROI classification-stage execution + focus/open-set gating flow (`run_sam3_roi_classification_stage`)
- Refactored `src/router/vlm_pipeline.py` to delegate `_collect_sam3_roi_candidates`, `_classify_sam3_roi_candidate`, and `_run_sam3_roi_classification_stage` while preserving existing method signatures.

Validation (post-Step 3):
- `pytest -c config/pytest.ini tests/unit/router -v` -> **18 passed**.
- `pytest -c config/pytest.ini tests/unit/router/test_vlm_policy_stage_order.py tests/unit/router/test_vlm_strict_loading.py -v` -> **16 passed**.
- `python scripts/profile_policy_sanity.py` -> profile/stage-order output validated for `base.json` and `colab.json`.
- `python scripts/check_markdown_links.py --root .` -> **no broken links**.
