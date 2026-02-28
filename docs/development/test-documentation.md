# Test Documentation

## Framework

- Primary local runner: `scripts/run_test_suites.py`
- Root pytest configuration: `pytest.ini`
- Shared fixtures and options: `tests/conftest.py`

## Test Areas

- `tests/colab/`: Colab environment, data pipeline, smoke training
- `tests/integration/`: end-to-end colab integration
- `tests/unit/`: unit-level component coverage
- `tests/fixtures/`: fixture and sample-data contracts

## Recommended Commands

```powershell
# Fast default modular bundle
python scripts/run_test_suites.py

# Run one focused suite
python scripts/run_test_suites.py --suite unit/router

# Run all unit suites
python scripts/run_test_suites.py --suite unit

# Run integration suites
python scripts/run_test_suites.py --suite integration

# Run full matrix explicitly
python scripts/run_test_suites.py --suite all
```

## Fast Preflight

```powershell
python tests/import_test.py
```

This script validates imports and key trainer compatibility.

## Useful Pytest Options

- Run slow tests: `--runslow`
- Run integration tests: `--runintegration`
- Run heavy model tests: `--runheavymodel`

Integration tests are auto-marked from `tests/integration/*`, so they are skipped unless `--runintegration` is provided.

Examples:

```powershell
pytest tests/integration --runintegration
pytest tests/integration/test_colab_integration.py --runintegration --runheavymodel
pytest tests --runslow
```

## Suggested CI Gate Order

1. `python tests/import_test.py`
2. `python scripts/run_test_suites.py --suite quick`
3. `python scripts/run_test_suites.py --suite colab`
4. `python scripts/run_test_suites.py --suite integration`
