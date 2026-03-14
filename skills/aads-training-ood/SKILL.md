---
name: aads-training-ood
description: Use when working on AADS v6 training, continual SD-LoRA config, OOD calibration, readiness artifacts, BER experiments, or training-side adapter export behavior.
---

# AADS Training And OOD

Use this skill for changes around `TrainingWorkflow.run(...)`, `config/base.json -> training.continual`, `src/training/`, and training-side artifact semantics.

## Inspect First

- `README.md`
- `docs/architecture/overview.md`
- `docs/architecture/ood_recommendation.md`
- `docs/user_guide/ood_readiness_guide.md`
- `config/base.json`
- `src/workflows/training.py`
- `src/training/services/`

Load `skills/aads-colab-notebooks/SKILL.md` too if the task touches Notebook 2, dataset materialization, Drive telemetry, or notebook export paths.

## Workflow

1. Treat `TrainingWorkflow.run(...)` as the canonical training entrypoint. Notebook wrappers should stay aligned with it.
2. Preserve the shipped config surface under `training.continual`, especially `ood` and `evaluation`, unless the user explicitly wants a config-schema change.
3. Treat `production_readiness.json` as the deployment verdict. `validation/metric_gate.json` and `test/metric_gate.json` are split-local diagnostics.
4. Keep the adapter bundle contract coherent: LoRA weights, classifier or fusion state, config metadata, and serialized OOD state should move together.
5. When BER is involved, compare runs on the same crop, seed, split layout, and OOD evidence source. Use `scripts/evaluate_ber_rollout.py` when a rollout comparison is needed.
6. Back OOD scoring, calibration, readiness-policy, and rejection-method changes with primary literature when possible. If the repo adapts an idea rather than reproducing a paper exactly, say so explicitly.
7. If training outputs or readiness semantics change, update the matching docs instead of leaving behavior implicit.

## Boundaries

- Do not treat notebook-only behavior as canonical if it conflicts with workflow code.
- Do not change inference payload shape here unless the task explicitly spans inference. Load `aads-inference-runtime` too in that case.
- Keep generated outputs under `outputs/`, `runs/`, and `models/adapters/` out of tracked edits.
- Do not describe a training or OOD implementation as literature-faithful unless the cited method, config surface, and behavior actually match.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `pytest tests/unit/workflows/test_training_workflow.py tests/colab/test_smoke_training.py -q`
- Add or run the narrowest relevant training-side unit or integration tests for the touched modules.
