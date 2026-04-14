---
name: aads-bugfix-debugging
description: Use when diagnosing or fixing AADS v6 bugs, regressions, silent failures, invariant violations, unexpected fallback paths, or error-handling gaps on maintained surfaces.
---

# AADS Bugfix And Diagnostics

Use this skill when the task is to reproduce, localize, and fix bugs or robustness problems on the maintained AADS v6 surface.

## Inspect First

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- the owning canonical entrypoint: `src/workflows/training.py` or `src/workflows/inference.py`
- the narrowest affected tests under `tests/`
- `skills/aads-bugfix-debugging/references/bugfix_practices.md`

Load the owning feature skill too:

- `aads-training-ood` for training, readiness, or adapter-export bugs
- `aads-inference-runtime` for router/runtime or payload bugs
- `aads-colab-notebooks` for notebook wrapper bugs
- `aads-repo-hygiene` when the fix also changes tests, docs, CI, or benchmark expectations

## Workflow

1. Reduce the report to one maintained surface and one concrete symptom before editing.
2. Reproduce the problem with the smallest deterministic command, test, fixture, or helper path you can run locally. If the trigger is noisy, minimize the input or diff before patching.
3. Convert the symptom into an executable guard first: a unit test, integration test, schema check, contract check, or notebook helper validation.
4. For silent failures, inspect invariants at boundaries before changing core logic: missing files, empty datasets, stale config aliases, dropped payload fields, `None` or NaN propagation, shape drift, wrong status fallback, or silently skipped branches.
5. Prefer fail-fast validation and explicit errors over silent coercion or auto-repair when bad state would corrupt training artifacts, readiness decisions, or inference payloads.
6. When the exact expected output is hard to specify, add property-based or metamorphic checks around invariants such as idempotence, ordering, monotonicity, round-trips, or status transitions. Use `references/bugfix_practices.md` for the literature-backed rationale.
7. Strengthen typed or schema-backed choke points instead of scattering ad hoc guards. In this repo, prefer `src/shared/contracts.py`, config normalization, workflow result objects, and artifact JSON writers.
8. Add focused diagnostics around decision boundaries and failure transitions. Prefer one structured log or assertion at the right boundary over noisy blanket logging.
9. Keep the fix narrow. Do not hide unresolved faults behind broad `except` blocks, silent defaults, or degraded fallbacks unless the product contract explicitly requires that mode.
10. If test adequacy is still doubtful after the fix, use stronger negative cases, property checks, or targeted mutation-style evaluation on the touched logic instead of broad repo-wide tooling churn.

## Boundaries

- Do not rewrite generated outputs under `runs/`, `outputs/`, or `models/adapters/` unless the user explicitly asks for generated-artifact maintenance.
- Do not weaken readiness, OOD, or payload contracts just to make a failing path disappear.
- Do not describe a guardrail or heuristic as literature-backed unless the cited source actually supports it.
- Do not add repo-wide debugging dependencies unless the fix clearly needs them.

## Validate

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `.\scripts\python.cmd scripts/validate_config_schema.py`
- Run the narrowest relevant `pytest` target for the touched module or workflow surface.
- If the public contract changed, update the matching docs and targeted tests in the same edit.
- Run `.\scripts\python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json` when workflow entrypoints, runtime orchestration, or benchmarked interfaces change.
