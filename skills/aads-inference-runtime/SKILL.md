---
name: aads-inference-runtime
description: Use when changing AADS v6 router inference, adapter-root resolution, deployment handoff, direct adapter smoke testing, or inference payload behavior.
---

# AADS Inference Runtime

Use this skill for router-driven inference, adapter loading, deployment layout expectations, and direct adapter validation on the runtime side.

## Inspect First

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `src/workflows/inference.py`
- `src/pipeline/router_adapter_runtime.py`
- `src/pipeline/inference_payloads.py`
- `src/router/router_pipeline.py`
- `src/router/vlm_pipeline.py`
- `src/shared/contracts.py`
- `scripts/colab_router_adapter_inference.py`
- `scripts/colab_adapter_smoke_test.py`
- `scripts/colab_simple_adapter_smoke_ui.py`

Load `skills/aads-colab-notebooks/SKILL.md` for Notebook 1, Notebook 3, or Notebook 4 wrapper issues. Load `skills/aads-training-ood/SKILL.md` too if the task changes the saved adapter contract or OOD calibration expectations.

## Workflow

1. Treat `InferenceWorkflow.predict(...)` as the canonical inference entrypoint.
2. Preserve the default adapter layout `models/adapters/<crop>/<part>/continual_sd_lora_adapter/` unless the user explicitly wants a deployment-contract change.
3. Keep router-driven inference and direct adapter smoke testing as separate supported surfaces.
4. Preserve the typed inference payload contract and OOD metadata semantics when changing runtime behavior.
5. Treat router confidence-like scores as uncalibrated unless a repo-local evaluation surface proves the threshold behavior. Prefer calibration sweeps over hand-tuned one-example fixes.
6. If a change touches crop routing, inspect `src/router/router_pipeline.py`, `src/router/vlm_pipeline.py`, and `src/pipeline/router_adapter_runtime.py` before editing. If it touches the public payload, inspect `src/shared/contracts.py` too.
7. Back router evidence policy, reject-option behavior, calibration logic, and other non-trivial inference heuristics with literature when credible sources exist. Mark repo-specific engineering inference explicitly instead of overstating the citation.
8. For runtime routing changes, verify ID accuracy, negative false accepts, abstention/risk-coverage behavior, part precision/recall, payload status transitions, and latency where the changed path affects performance.

## Literature Anchors

- Start with `docs/architecture/router_performance_literature_review.md` before changing router thresholds, prompt evidence, global/local fusion, or calibration scripts.
- Calibration literature supports treating neural confidence as unreliable without validation; threshold changes need held-out evidence, not just plausible scores.
- Selective prediction supports abstention as a first-class output. In this repo, `unknown_crop`, `router_uncertain`, `router_unavailable`, and part `unknown` are product states that must remain explicit.
- OOD benchmark literature warns that poorly constructed evaluation sets can make a detector look better than it is. Preserve realistic off-crop, non-plant, ambiguous, and wrong-part cases when evaluating runtime changes.

## Boundaries

- Do not fold adapter smoke testing into router inference unless the user explicitly asks for that product change.
- Do not change training-side readiness policy here without also loading `aads-training-ood`.
- Keep generated adapters under `models/adapters/` treated as local deployment artifacts, not tracked fixtures.
- Do not present runtime heuristics as citation-backed guarantees unless the linked literature actually supports the claimed behavior.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `pytest tests/unit/pipeline/test_router_adapter_runtime.py tests/unit/workflows/test_inference_workflow.py -q`
- `.\scripts\python.cmd scripts/evaluate_router_part_surface.py --root <router_part_eval_root> --config-env colab --output .runtime_tmp/router_part_eval.json` when crop or part routing thresholds change
- `.\scripts\python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json` when runtime orchestration or router performance surfaces change.
