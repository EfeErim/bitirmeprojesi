# Repository Organization Audit (2026-02-26)

## Scope

This audit focused on reducing workspace clutter without changing runtime behavior, imports, or documented user entrypoints.

## Current State Summary

- Git state at start: `master...origin/master` with no pending tracked changes.
- Primary clutter source by size: local virtual environment.
  - `.venv`: `51245` files, `1688.78 MB`
- High-entropy generated artifacts were present in repo paths:
  - `.pytest_cache/`, `config/.pytest_cache/`
  - root and nested `__pycache__/`
  - local egg-info outputs (`aads_ulora_v5.5.egg-info/`, `src/aads_ulora_v5.5.egg-info/`)
  - `.coverage`
- Duplicate-name hotspots (expected + intentional mixed with compatibility surfaces):
  - `validate_notebook_imports.py` (root wrapper + canonical script)
  - `test_dynamic_taxonomy.py` (root wrapper + canonical script)
  - `test_pipeline_final_check.py` (root wrapper + canonical script)
  - `colab_test_upload.py` (root wrapper + canonical script)
  - `colab_bootstrap.ipynb` (root mirror + canonical notebook path)
  - `requirements_colab.txt` (root + notebook folder)

## What Was Organized

### 1) Physical cleanup of generated artifacts

Removed local generated clutter from repository paths:

- `.pytest_cache/`
- `config/.pytest_cache/`
- all repo-side `__pycache__/` directories (excluding `.venv`)
- `aads_ulora_v5.5.egg-info/`
- `src/aads_ulora_v5.5.egg-info/`
- `.coverage`

Resulting file count reduction in main code/docs paths:

- `src`: `152 -> 54` files
- `tests`: `151 -> 42` files
- `scripts`: `31 -> 23` files
- `config`: `12 -> 7` files

### 2) Guardrails to prevent recurrence

Updated root `.gitignore` to explicitly ignore:

- `.venv/`
- `config/.pytest_cache/`
- `aads_ulora_v5.5.egg-info/`
- `src/aads_ulora_v5.5.egg-info/`

## Intentionally Preserved (Compatibility-Sensitive)

These were kept intentionally to avoid breaking docs and legacy workflows:

- Root compatibility wrappers for canonical scripts:
  - `validate_notebook_imports.py`
  - `test_dynamic_taxonomy.py`
  - `test_pipeline_final_check.py`
  - `colab_test_upload.py`
- Root `colab_bootstrap.ipynb` mirror (legacy path support)

Rationale: repository docs and relation maps explicitly describe these as compatibility aliases/mirrors.

## Remaining Organization Risks / Opportunities

1. `.venv` remains inside project root and dominates directory noise/size.
- Recommendation: recreate env outside repo root (for example user-level venv path) and keep `.venv/` ignored for local exceptions.

2. Root `setup_git.sh` is an empty tracked file (`0` bytes).
- Recommendation: remove if confirmed unused, or populate with documented bootstrap steps.

3. Docs contain mixed archival payloads (PDFs/zip) with inconsistent naming.
- Recommendation: standardize under `docs/archive/` with a short index file and naming convention.

4. Compatibility wrappers add discoverability overhead.
- Recommendation: keep for now, but define deprecation criteria/date if the team is ready to enforce `scripts/`-first usage only.

## Validation Commands

Use these to verify organization and regressions quickly:

```bash
git status --short --branch
python scripts/run_python_sanity_bundle.py
python scripts/check_markdown_links.py --root .
```

Optional broader checks:

```bash
python scripts/run_policy_regression_bundle.py
pytest tests/integration -v --runintegration
```

