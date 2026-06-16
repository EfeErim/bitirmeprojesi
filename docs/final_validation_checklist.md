# Final Validation Checklist

Last updated: 2026-06-16

Use this file for M5 final freeze. Prefer narrow, literal validation over broad late refactors.

## Validation Policy

- Run the smallest checks that prove the final handoff surface.
- Record exact pass/fail output, not generic summaries.
- Treat legitimate skips separately from blockers.
- Do not start broad refactors during M5 unless a final blocker makes the demo impossible.

## Required Checks

| Check | Command | Required outcome |
|---|---|---|
| Notebook import contract | `.\scripts\python.cmd scripts\validate_notebook_imports.py` | Pass or exact blocker documented. |
| Config schema | `.\scripts\python.cmd scripts\validate_config_schema.py` | Pass. |
| Code organization | `.\scripts\python.cmd scripts\audit_code_organization.py` | No boundary violations. |
| Demo checklist | Manual Notebook 8 run using `docs/demo_checklist.md` | Pass threshold or explicit limitations. |
| Docs routing | Manual check of `README.md`, `docs/README.md`, handoff guide | New user can find demo/training/validation path. |

## Strongly Recommended Checks

| Check | Command | When to run |
|---|---|---|
| Targeted unit tests | `.\scripts\python.cmd -m pytest tests/unit/pipeline/test_notebook16_failure_analysis.py tests/unit/scripts/test_export_notebook16_target_audit.py tests/unit/scripts/test_apply_notebook16_target_audit_decisions.py tests/unit/scripts/test_restore_router_calibration_artifact.py -q` | Run after doc/support-tool changes or audit helper changes. |
| Ruff targeted | `ruff check src scripts tests` | Run if code changed. |
| Router calibration stability | `.\scripts\python.cmd scripts\restore_router_calibration_artifact.py` then `.\scripts\python.cmd scripts\validate_router_calibration_stability.py` | Run when router calibration artifact or router data changed. |
| Dataset integrity | `.\scripts\python.cmd scripts\monitor_dataset_integrity.py` | Run when dataset/generated runtime data changed; distinguish warn from fail. |

## Optional Checks

| Check | Command | Notes |
|---|---|---|
| Unit + Colab smoke | `pytest tests/unit tests/colab/test_smoke_training.py -q` | Useful if code changed and time allows. |
| Integration | `pytest tests/integration -q --runintegration` | Use when runtime dependencies are available. |
| Typecheck | `mypy --follow-imports skip src/shared src/data src/training/continual_sd_lora.py src/router src/workflows src/pipeline/router_adapter_runtime.py src/pipeline/inference_payloads.py src/adapter src/core/config_manager.py scripts/benchmark_surfaces.py` | Useful for code changes; not required for doc-only changes. |
| Benchmark capture | `.\scripts\python.cmd scripts\benchmark_surfaces.py --output .runtime_tmp\benchmarks.json` | Use when runtime workflow contract changed. |

## Final Freeze Rules

- If a required check fails, write the exact blocker and fix only that blocker.
- If an optional check fails late, do not broaden scope unless it affects the final demo or handoff.
- If a target surface cannot meet demo threshold, mark it `low_confidence` or `experimental`.
- If external access blocks validation, use documented fallback artifacts and record the dependency.

## Final Status Template

| Area | Status | Evidence | Blocker or limitation |
|---|---|---|---|
| Notebook imports |  |  |  |
| Config schema |  |  |  |
| Code organization |  |  |  |
| Demo checklist |  |  |  |
| Supported target labels |  |  |  |
| Handoff guide |  |  |  |
| Presentation fallback artifacts |  |  |  |
