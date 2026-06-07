# Project State

## Current Objective

Keep the narrow plant-disease repo stable while supporting grouped dataset preparation, continual SD-LoRA adapter training, router-guided inference, and OOD/readiness checks.

## Current Implementation Status

- Canonical workflows live in `src/workflows/training.py` and `src/workflows/inference.py`.
- Maintained notebook surfaces are 0, 1, 2, 3, 5, and 8; Notebook 4 is a convenience wrapper, and Notebooks 6/7 are validation surfaces.
- Notebook 9 is a recording-oriented presentation wrapper over Notebook 8. It renders real router and adapter payload details without changing canonical inference behavior.
- Notebook 9 uses `requirements_presentation_colab.txt`; keep `torchao==0.17.0` in that lightweight profile because Colab's older preinstalled torchao can break PEFT adapter loading during warm-up.
- The repo is pip-based and uses `./scripts/python.cmd` on Windows so the local `.venv` is preferred.
- CI already covers notebook/import validation, config schema checks, OOD evidence consistency, router calibration stability, adapter smoke tests, metadata completeness, dataset integrity, notebook outputs, and benchmark capture.

## Recent Meaningful Changes

- Latest commit added Codex instructions, a config example, and output-capping helpers, and refreshed `AGENTS.md`.
- Recent notebook calibration work tuned Notebook 5 router calibration defaults to `12/12`, exposed adaptive hyperparameters, and published failure-analysis artifacts.
- The SOTA literature updater filters query-specific candidates more narrowly, restricts BioCLIP candidates to plant-domain context, deduplicates titles, and preserves the previous managed scan when every configured query fails.
- `docs/SOTA_AUTOMATION_GUIDE.md` is now an operating guide for the SOTA refresh loop: every pass should refresh machine evidence, run narrow guardrails, classify skips/failures, and select a concrete next repo action instead of maintaining a static wishlist.

## Important Decisions

- Use `requirements.txt` and `requirements-dev.txt` as the dependency source of truth.
- Treat `runs/`, `models/adapters/`, `outputs/`, and `.runtime_tmp/` as generated or local-only surfaces.
- Prefer narrow validation commands before broad test runs.

## Known Issues / Risks

- No dedicated build step is defined; validation scripts are the main maintenance surface.
- Router calibration and OOD evidence can drift if evaluation data changes.
- Notebook-exported artifacts should not be confused with canonical source code.

## Next Recommended Steps

- Before editing, read this file first.
- Make the smallest change that addresses the request.
- If durable state changes, update this file at the end of the task.
- When calibration, runtime, or benchmark surfaces change, run the narrowest relevant validation first and capture benchmarks if the workflow contract changed.

## Verified Commands

- `./scripts/python.cmd -m pip install -r requirements.txt`
- `./scripts/python.cmd -m pip install -r requirements-dev.txt`
- `./scripts/python.cmd scripts/validate_notebook_imports.py`
- `./scripts/python.cmd scripts/validate_config_schema.py`
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `./scripts/python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json`
- `ruff check src scripts tests`
- `mypy --follow-imports skip src/shared src/data src/training/continual_sd_lora.py src/router src/workflows src/pipeline/router_adapter_runtime.py src/pipeline/inference_payloads.py src/adapter src/core/config_manager.py scripts/benchmark_surfaces.py`
