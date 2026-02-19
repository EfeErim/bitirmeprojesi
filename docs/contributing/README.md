# Contributing Guide

## Scope

This document provides contribution conventions for this repository.

## Before You Start

1. Create/activate virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run baseline checks:

```powershell
python validate_notebook_imports.py
pytest -c config/pytest.ini tests/colab/test_environment.py
```

## Contribution Workflow

1. Create a feature/fix branch.
2. Keep changes scoped and minimal.
3. Update docs when behavior changes.
4. Run relevant tests.
5. Open a pull request with clear summary.

## Code Change Expectations

- Preserve existing public APIs unless change is intentional.
- Avoid unrelated refactors.
- Keep configuration changes explicit in PR notes.

## Documentation Expectations

- No dead links.
- Use project-real commands and paths.
- Update `docs/README.md` when adding/moving docs.
