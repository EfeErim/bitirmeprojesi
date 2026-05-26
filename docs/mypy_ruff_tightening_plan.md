## mypy / ruff tightening plan

Goal: Incrementally increase static checking on `src/` to improve code quality without blocking CI.

Steps:

1. Baseline run
   - Run `mypy` and `ruff` locally with current config to collect baseline issues.

2. File-scoped tightening
   - For each module under `src/` start adding `# type: ignore[import]` removal and fix type errors.
   - Target small modules first: `src/utils`, `src/shared`, `src/pipeline/adapter_discovery.py`.

3. CI gating
   - Add a new CI job that runs `mypy` only on `src/` with `--strict` in a separate matrix column to measure regressions.

4. Automation
   - Add `pre-commit` hooks to run `ruff` and `isort` locally.

5. Timeline
   - Week 1: Baseline + fix low-hanging type errors in utils.
   - Week 2: Expand to pipeline and router modules.

Notes:
 - Keep `colab_notebooks/` and heavy external dependencies excluded from strict checks.
 - Prefer small, test-covered PRs when removing `ignore_missing_imports`.
