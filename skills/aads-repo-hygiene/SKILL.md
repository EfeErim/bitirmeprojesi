---
name: aads-repo-hygiene
description: Use when updating AADS v6 CI, tests, benchmark surfaces, docs consistency, or tracked-vs-generated repo boundaries.
---

# AADS Repo Hygiene

Use this skill for CI, tests, validation commands, benchmark capture, and documentation consistency around the maintained repo surface.

## Inspect First

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `scripts/benchmark_surfaces.py`
- `scripts/validate_config_schema.py`
- `scripts/validate_notebook_imports.py`

Load the feature-specific skill too if the task touches training, notebooks, or inference behavior.

## Workflow

1. Keep CI aligned with the maintained narrow repo surface. Do not add broad checks for deprecated or local-only workflows.
2. Respect the tracked-vs-generated boundary from `docs/README.md`. `runs/`, `outputs/`, `models/adapters/`, and `.runtime_tmp/` are local or generated surfaces.
3. Prefer the narrowest validation that proves the change. Expand to smoke or integration coverage only when the touched surface requires it.
4. Run `scripts/benchmark_surfaces.py` when workflow entrypoints, router runtime orchestration, or benchmarked interfaces change.
5. When docs drift from code, update the canonical docs in the repo instead of adding duplicate side notes.
6. Preserve or improve literature-backed rationale when the task changes ML methods, benchmark claims, evaluation policy, or contributor guidance about model behavior. Prefer primary sources and label engineering inference explicitly.
7. When changing `AGENTS.md`, `skills/*/SKILL.md`, or `skills/*/agents/*.yaml`, keep the routing text and metadata aligned, and sanity-check representative task phrasings before considering the edit done.
8. Prefer pointing skills at canonical docs and contracts over duplicating mutable repo facts across multiple skill files.

## Boundaries

- Do not rewrite generated artifacts just to satisfy hygiene work unless the user asked for generated-output maintenance.
- Do not treat CI automation as a replacement for repo-local skills or agent routing.
- Keep new process documentation minimal and attach it to maintained files, not ad hoc extras.
- Do not add unsupported performance, readiness, or methodology claims just because a workflow passes locally. Use repo evidence and literature when available.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `.\scripts\python.cmd scripts/validate_config_schema.py`
- `.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root <flat_class_root>` when dataset-contract guidance changes
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `.\scripts\python.cmd scripts/evaluate_router_part_surface.py --root <router_part_eval_root> --config-env colab --output .runtime_tmp/router_part_eval.json` when router part-policy guidance changes
- Re-check `.github/workflows/ci.yml` when changing validation commands or coverage expectations.
