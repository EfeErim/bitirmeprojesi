# Implementation Plan (Current Baseline)

## Goal

Maintain a reproducible Colab-first training pipeline and stable API inference surface.

## Phase A — Stability and Contracts

1. Keep configuration loading deterministic across environments.
2. Preserve notebook import/runtime compatibility (`validate_notebook_imports.py`).
3. Ensure API startup and route mounting remain test-covered.

## Phase B — Training Reliability

1. Continue improving phase trainer resilience in `src/training/`.
2. Keep checkpoint/load behavior consistent across phases.
3. Maintain manifest-based artifact tracking (see `src/core/artifact_manifest.py`).

## Phase C — Docs and Developer Experience

1. Keep docs index synchronized with actual filesystem.
2. Maintain command-level setup and test docs.
3. Add CI checks for markdown link integrity.

## Definition of Done

- Colab notebooks run sequentially without manual code edits.
- Main smoke/integration tests pass.
- Documentation links resolve to real files.
