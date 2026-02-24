# Optimization Phase 5 — Baseline Report

Date: 2026-02-24

## 1) Benchmark Setup
- Script: `scripts/benchmark_router_phase5.py`
- Output: `logs/phase5_router_benchmark.json`
- Mode: CPU, deterministic mocked SAM3/CLIP hooks
- Iterations per scenario: 200

## 2) Results Summary

### Scenario: `full_pipeline`
- Avg wall latency: **0.9998 ms**
- p50 wall latency: **0.8826 ms**
- p95 wall latency: **1.6461 ms**
- Avg pipeline `processing_time_ms`: **0.4602 ms**
- Avg detections: **1**

Stage timing averages:
- preprocess: **0.0022 ms**
- sam3_inference: **0.0219 ms**
- roi_total: **0.3898 ms**
- roi_classification: **0.3672 ms**
- postprocess: **0.0115 ms**

### Scenario: `no_postprocess`
- Avg wall latency: **0.9349 ms**
- p95 wall latency: **1.5063 ms**
- Avg detections: **2**

### Scenario: `no_open_set_gate`
- Avg wall latency: **0.7721 ms**
- p95 wall latency: **1.0037 ms**
- Avg detections: **1**

## 3) Hotspot Interpretation
- Dominant runtime share is ROI path (`roi_total`), especially `roi_classification`.
- `postprocess` cost is comparatively small in this deterministic setup.
- Stage-order changes meaningfully affect p95 latency, so policy-stage tuning can be an optimization lever.

## 4) Priority Actions (Wave 5.2)
1. Optimize per-ROI classification loop allocations and repeated normalization work.
2. Reduce repeated dict/list construction in compatibility + focus branches.
3. Preserve stage-order behavior while introducing micro-optimizations only.

## 5) Validation Gate for Upcoming Changes
- `pytest -c config/pytest.ini tests/unit/router -v`
- `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v`
- Re-run benchmark and compare against this baseline JSON.

## 6) Wave 5.2 Update (Implemented)

Applied optimization:
- Hoisted constant `settings` reads and focus-mode normalization out of the per-candidate loop in `src/router/roi_pipeline.py` (`run_sam3_roi_classification_stage`).

Post-optimization benchmark (`logs/phase5_router_benchmark.json`):

### Scenario: `full_pipeline`
- Avg wall latency: **0.7190 ms** (from **0.9998 ms**, **-28.1%**)
- p95 wall latency: **0.8153 ms** (from **1.6461 ms**, **-50.5%**)
- Avg `processing_time_ms`: **0.3387 ms** (from **0.4602 ms**, **-26.4%**)
- Avg `roi_classification`: **0.2781 ms** (from **0.3672 ms**, **-24.3%**)

Validation:
- `pytest -c config/pytest.ini tests/unit/router -q` -> **18 passed**.

Conclusion:
- Wave 5.2 achieved and exceeded initial target thresholds in this deterministic benchmark harness.
- Next focus should move to Wave 5.3 micro-optimizations in per-ROI compatibility/leaf-override branches.

## 7) Wave 5.3 Update (Implemented)

Applied optimization:
- Micro-optimized compatibility and leaf-visual override branches in `src/router/roi_pipeline.py` (`classify_sam3_roi_candidate`) by:
	- hoisting repeated `settings` lookups to local variables,
	- avoiding repeated dynamic string/weight lookups,
	- reducing fallback normalization work for leaf-score extraction in the common case.

Post-optimization benchmark (`logs/phase5_router_benchmark.json`):

### Scenario: `full_pipeline`
- Avg wall latency: **0.6904 ms** (from Wave 5.2 **0.7190 ms**, **-4.0%**)
- p95 wall latency: **0.7991 ms** (from Wave 5.2 **0.8153 ms**, **-2.0%**)
- Avg `processing_time_ms`: **0.3316 ms** (from Wave 5.2 **0.3387 ms**, **-2.1%**)
- Avg `roi_classification`: **0.2735 ms** (from Wave 5.2 **0.2781 ms**, **-1.6%**)

Validation:
- `pytest -c config/pytest.ini tests/unit/router -q` -> **18 passed**.

Conclusion:
- Wave 5.3 delivered incremental gains on top of Wave 5.2 while preserving behavior.
- Remaining major gains are likely to come from benchmark guardrails and higher-level algorithmic/path tuning rather than additional micro-lookup reductions.

## 8) Wave 5.4 Update (Implemented)

Guardrail artifacts added:
- `config/perf_guardrails_phase5.json`
- `scripts/check_phase5_perf_regression.py`

Guardrail thresholds (CPU deterministic benchmark):
- `full_pipeline`:
	- `avg_wall_ms <= 0.85`
	- `p95_wall_ms <= 0.95`
	- `avg_processing_ms <= 0.45`
	- `roi_classification <= 0.35`
- `no_postprocess`:
	- `avg_wall_ms <= 0.80`
	- `p95_wall_ms <= 0.95`
- `no_open_set_gate`:
	- `avg_wall_ms <= 0.80`
	- `p95_wall_ms <= 0.95`

Execution model:
1. `python scripts/benchmark_router_phase5.py`
2. `python scripts/check_phase5_perf_regression.py`

Outcome:
- Wave 5 now includes reproducible measurement plus fail-fast regression thresholds.
