# Colab and Docs Sync Checklist

## Canonical User Surfaces

- Main project guide: `README.md`
- Notebook index: `colab_notebooks/README.md`
- Script index: `scripts/README.md`
- Docs index: `docs/README.md`
- Repo relation map: `docs/REPO_FILE_RELATIONS.md`
- Canonical full-flow notebook: `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`

## Sync Rules

- If notebook names/roles change, update both notebook index and root README references.
- If canonical scripts change, update script matrix and docs that cite those commands.
- If notebook command snippets change, update matching script references in user guides.
- Keep root compatibility wrappers documented as aliases, not primary entrypoints.

## Validation Commands

```bash
python scripts/validate_notebook_imports.py
python scripts/check_markdown_links.py --root .
pytest tests/colab/test_environment.py -v
```

## Optional Extended Validation

```bash
pytest tests/colab -v
python scripts/run_python_sanity_bundle.py
```

## Drift Check

- Verify every renamed notebook or script appears in all index files.
- Verify canonical `scripts/` command style is preserved in docs.
- Call out intentional deprecations explicitly in the summary.
