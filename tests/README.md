# Tests (v6)

The active test contract for AADS v6 is defined by `scripts/run_test_suites.py`.

## Canonical Command

```powershell
python scripts/run_test_suites.py --suite unit --suite colab --suite integration/core
```

## Archive Policy

- Legacy v5.x tests are stored under `tests/archive/v5_legacy/`.
- `pytest.ini` excludes archive paths from normal discovery.
- New runtime behavior must be validated with v6 suites under active `tests/unit`, `tests/colab`, and `tests/integration`.
