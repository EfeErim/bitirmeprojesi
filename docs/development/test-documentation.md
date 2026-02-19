# Test Documentation

## Framework

- Pytest configuration: `config/pytest.ini`
- Shared fixtures and options: `tests/conftest.py`

## Test Areas

- `tests/colab/`: Colab environment, data pipeline, smoke training
- `tests/integration/`: end-to-end colab integration

- `tests/unit/`: unit-level component coverage

## Core Validation Commands

```powershell
pytest -c config/pytest.ini tests/colab/test_environment.py
pytest -c config/pytest.ini tests/colab/test_smoke_training.py
pytest -c config/pytest.ini tests/integration/test_colab_integration.py
pytest -c config/pytest.ini tests/api/test_endpoints.py
```

## Fast Preflight

```powershell
python validate_notebook_imports.py
```

This script validates imports and key trainer compatibility methods used by notebooks.

## Useful Pytest Options

- Run slow tests: `--runslow`
- Run integration tests: `--runintegration`

Examples:

```powershell
pytest -c config/pytest.ini tests --runintegration
pytest -c config/pytest.ini tests --runslow
```

## Suggested CI Gate Order

1. `python validate_notebook_imports.py`
2. `tests/colab/test_environment.py`
3. `tests/colab/test_smoke_training.py`
4. `tests/api/test_endpoints.py`
5. integration suite
