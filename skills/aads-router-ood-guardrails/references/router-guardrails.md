# Router and OOD Guardrails

## Critical Files

- Router core: `src/router/vlm_pipeline.py`
- ROI internals: `src/router/roi_pipeline.py`, `src/router/roi_helpers.py`
- Policy/taxonomy normalization: `src/router/policy_taxonomy_utils.py`
- Lightweight router: `src/router/simple_crop_router.py`
- OOD: `src/ood/dynamic_thresholds.py`, `src/ood/prototypes.py`, `src/ood/mahalanobis.py`
- Policy regression/profile scripts: `scripts/run_policy_regression_bundle.py`, `scripts/profile_policy_sanity.py`
- Performance checks: `scripts/benchmark_router_phase5.py`, `scripts/check_phase5_perf_regression.py`
- Guardrail thresholds: `config/perf_guardrails_phase5.json`

## Behavioral Invariants

- Stage order must remain compatible with `tests/unit/router/test_vlm_policy_stage_order.py`.
- Strict loading behavior must remain compatible with `tests/unit/router/test_vlm_strict_loading.py`.
- Policy/taxonomy normalization must remain compatible with unit tests and current config contracts.
- OOD threshold logic should preserve expected adapter integration behavior.
- Focus-mode fallback behavior should keep current non-breaking defaults.

## Required Validation

```bash
pytest tests/unit/router -v
pytest tests/unit/ood -v
pytest tests/unit/adapters -v
python scripts/run_policy_regression_bundle.py
python scripts/profile_policy_sanity.py
```

## Performance Validation (when router runtime path changed)

```bash
python scripts/benchmark_router_phase5.py
python scripts/check_phase5_perf_regression.py
```

## Optional Validation

- Heavy model path only when needed:

```bash
pytest tests/unit/router -v --runheavymodel
```
