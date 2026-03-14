---
name: aads-inference-runtime
description: Use when changing AADS v6 router inference, adapter-root resolution, deployment handoff, direct adapter smoke testing, or inference payload behavior.
---

# AADS Inference Runtime

Use this skill for router-driven inference, adapter loading, deployment layout expectations, and direct adapter validation on the runtime side.

## Inspect First

- `README.md`
- `docs/architecture/overview.md`
- `src/workflows/inference.py`
- `src/pipeline/router_adapter_runtime.py`
- `src/pipeline/inference_payloads.py`
- `src/router/vlm_pipeline.py`
- `scripts/colab_router_adapter_inference.py`
- `scripts/colab_adapter_smoke_test.py`

Load `skills/aads-colab-notebooks/SKILL.md` for Notebook 1 or Notebook 3 wrapper issues. Load `skills/aads-training-ood/SKILL.md` too if the task changes the saved adapter contract or OOD calibration expectations.

## Workflow

1. Treat `InferenceWorkflow.predict(...)` as the canonical inference entrypoint.
2. Preserve the default adapter layout `models/adapters/<crop>/continual_sd_lora_adapter/` unless the user explicitly wants a deployment-contract change.
3. Keep router-driven inference and direct adapter smoke testing as separate supported surfaces.
4. Preserve the typed inference payload contract and OOD metadata semantics when changing runtime behavior.
5. If a change touches the crop-routing boundary, inspect both `src/router/vlm_pipeline.py` and `src/pipeline/router_adapter_runtime.py` before editing.
6. Back router evidence policy, reject-option behavior, calibration logic, and other non-trivial inference heuristics with literature when credible sources exist. Mark repo-specific engineering inference explicitly instead of overstating the citation.

## Boundaries

- Do not fold adapter smoke testing into router inference unless the user explicitly asks for that product change.
- Do not change training-side readiness policy here without also loading `aads-training-ood`.
- Keep generated adapters under `models/adapters/` treated as local deployment artifacts, not tracked fixtures.
- Do not present runtime heuristics as citation-backed guarantees unless the linked literature actually supports the claimed behavior.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `pytest tests/unit/pipeline/test_router_adapter_runtime.py tests/unit/workflows/test_inference_workflow.py -q`
- `.\scripts\python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json` when runtime orchestration or router performance surfaces change.
