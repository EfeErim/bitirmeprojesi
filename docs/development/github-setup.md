# GitHub Setup

## Recommended Branch Workflow

- `main`: stable branch
- feature branches: `feature/<short-topic>`
- fix branches: `fix/<short-topic>`

## Basic Developer Flow

```powershell
git checkout -b feature/docs-sync
git add docs README.md
git commit -m "docs: synchronize index with existing files"
git push -u origin feature/docs-sync
```

## Pull Request Checklist

- [ ] `python scripts/validate_notebook_imports.py` passes
- [ ] Relevant pytest suites pass
- [ ] Documentation links validated
- [ ] Config changes documented

## Suggested Repository Checks

- Python lint/type checks (if enabled)
- pytest suites (`tests/colab`, `tests/integration`, `tests/unit`)
- markdown link checker for `README.md` and `docs/`

## Markdown Link Guardrails

Automated checks are now available in two forms:

1. **CI workflow**: `.github/workflows/markdown-links.yml`
2. **Local pre-commit hook**: `.pre-commit-config.yaml`

Manual run:

```powershell
python scripts/check_markdown_links.py --root .
```

Enable pre-commit locally (optional):

```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
