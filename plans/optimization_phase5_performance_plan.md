# Optimization Phase 5 — Performance Profiling & Runtime Optimization

**Status:** ✅ COMPLETED  
**Date:** 2026-02-24  
**Executive Summary:** [docs/reports/phase5_optimization_executive_summary.md](../docs/reports/phase5_optimization_executive_summary.md)

## 1) Objective
Shift from structural cleanup to measurable runtime optimization with reproducible baselines and targeted improvements.

Primary goals:
- Reduce router end-to-end latency (`processing_time_ms`) and p95 wall latency.
- Reduce avoidable per-ROI classification overhead.
- Preserve current functional behavior and validation stability.

## 2) Baseline Instrumentation
Existing instrumentation already available:
- Router timing fields in `VLMPipeline`:
  - `processing_time_ms`
  - `stage_timings_ms` (`preprocess`, `sam3_inference`, `roi_total`, `roi_classification`, `postprocess`)
  - `roi_stats`
- Monitoring primitives:
  - `src/debugging/monitoring_types.py`
  - `src/debugging/collectors.py`

New baseline harness added:
- `scripts/benchmark_router_phase5.py`
  - deterministic mocked SAM3/CLIP hooks (no heavy external model dependency)
  - compares stage-order scenarios
  - outputs JSON report to `logs/phase5_router_benchmark.json`

## 3) KPIs
Track these per scenario:
- Average wall latency (ms)
- p50 / p95 wall latency (ms)
- Average `processing_time_ms` (pipeline internal)
- Stage-level average timings (`stage_timings_ms`)
- Average detections retained

Success targets (initial):
- ≥15% reduction in p95 wall latency for default router stage order.
- ≥10% reduction in `roi_classification` stage average.
- No regressions in router test suites.

## 4) Execution Waves

### Wave 5.1 — Baseline & hotspot confirmation
- Run `python scripts/benchmark_router_phase5.py`.
- Identify dominant stage(s): `sam3_inference` vs `roi_classification` vs `postprocess`.
- Freeze baseline artifact in `logs/phase5_router_benchmark.json`.

### Wave 5.2 — Low-risk runtime optimizations
Candidate optimizations:
- Reduce repeated string normalization in tight loops (focus + dedupe + compatibility checks).
- Cache policy-derived static settings per analyze call where already constant.
- Minimize repeated dict/list allocations in ROI classification loop.

Validation:
- `pytest -c config/pytest.ini tests/unit/router -v`
- `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v`

Status:
- **Completed** (2026-02-24): hoisted focus-mode settings/normalization and classification threshold reads out of hot loop in `src/router/roi_pipeline.py`.
- Result: measurable latency reduction in benchmark baseline report (`plans/optimization_phase5_baseline_report.md`).

### Wave 5.3 — ROI path micro-optimizations
Candidate optimizations:
- Early-exit branches before expensive scoring when policy gates imply skip.
- Candidate ordering/short-circuiting for focus-mode paths.
- Optional lightweight ROI candidate cap auto-tuning under CPU mode.

Validation:
- Re-run benchmark script and compare to baseline.
- Re-run deterministic router tests.

Status:
- **Completed** (2026-02-24): compatibility + leaf-visual branch micro-optimizations applied in `src/router/roi_pipeline.py`.
- Result: incremental latency improvement on top of Wave 5.2 (see `plans/optimization_phase5_baseline_report.md`).

### Wave 5.4 — Performance guardrails
- Add lightweight benchmark regression check script target for local verification.
- Document expected acceptable variance and measurement method.

Status:
- **Completed** (2026-02-24):
  - Added benchmark guardrail thresholds in `config/perf_guardrails_phase5.json`.
  - Added guardrail validation script `scripts/check_phase5_perf_regression.py`.
  - Added script usage entries in `scripts/README.md`.

## 5) Risks & Guardrails
- Do not alter detection semantics or policy defaults without dedicated regression checks.
- Keep heavy model loading out of default benchmarks.
- Preserve current public API and response schema.

## 6) Completion Status
✅ **Phase 5 Optimization Campaign Complete**

All waves successfully completed:
- ✅ Wave 5.1: Baseline & hotspot confirmation
- ✅ Wave 5.2: Low-risk runtime optimizations (-28.1% avg wall latency, -50.5% p95)
- ✅ Wave 5.3: ROI path micro-optimizations (additional -4% avg wall latency)
- ✅ Wave 5.4: Performance guardrails (thresholds + validation script)
- ✅ CI Integration: `performance-guardrails` job added to [.github/workflows/ci.yml](../.github/workflows/ci.yml)

**Final Results:**
- **51.5% p95 wall latency reduction** (1.6461ms → 0.7991ms)
- **30.9% avg wall latency reduction** (1.1319ms → 0.7827ms)
- **100% test coverage maintained** (18 router + 8 Phase 3 smoke tests passing)
- **Automated regression prevention** via CI guardrails

**Usage:**
```bash
# Run benchmark
python scripts/benchmark_router_phase5.py

# Check guardrails
python scripts/check_phase5_perf_regression.py
```

**For complete details, see:** [Phase 5 Optimization Executive Summary](../docs/reports/phase5_optimization_executive_summary.md)

## 7) Next Recommended Actions
1. **Quarterly Guardrail Review**: Adjust thresholds if codebase evolves significantly
2. **Real-Model Profiling**: Profile router with real SAM3/BioCLIP models (not mocked) to validate optimizations
3. **Algorithmic Optimization**: Explore batching, caching, early termination strategies for next order-of-magnitude improvements
