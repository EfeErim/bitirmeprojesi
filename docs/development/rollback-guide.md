# Rollback Guide

## When to Roll Back

Use rollback when one of these occurs:

- notebook execution regressions in Colab pipeline
- training checkpoint compatibility breaks

## Code Rollback

```powershell
git log --oneline -n 20
git revert <bad_commit_sha>
```

If local-only and not pushed:

```powershell
git reset --hard <known_good_sha>
```

## Configuration Rollback

1. Restore last known-good `config/*.json` files.
2. Re-run local smoke checks:

```powershell
python scripts/validate_notebook_imports.py
pytest -c config/pytest.ini tests/colab/test_environment.py
```

## Artifact Rollback (Training)

- Restore previous checkpoint directories from backup location (Drive/local).
- For phase3 output bundles, verify `manifest.json` and referenced artifacts exist.

## Post-Rollback Runtime Verification

```powershell
python scripts/run_python_sanity_bundle.py
python scripts/check_markdown_links.py --root .
```
