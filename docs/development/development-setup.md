# Development Setup

## Prerequisites

- Windows/macOS/Linux
- Python 3.9+
- Git

## Local Setup (Windows PowerShell)

```powershell
cd "D:\bitirme projesi"
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify Installation

```powershell
python validate_notebook_imports.py
pytest -c config/pytest.ini tests/import_test.py
python scripts/check_markdown_links.py --root .
```

## Run API Locally

```powershell
$env:APP_ENV="development"
python -m api.main
```

Expected endpoint base:

- `http://localhost:8000/health`

## Configuration Model

- Base config: `config/base.json`
- Environment overrides: `config/development.json`, `config/production.json`, `config/colab.json`
- Loader: `src/core/config_manager.py`

## Colab Workflow (Primary Training Path)

1. Run `colab_bootstrap.ipynb`
2. Run notebooks in order under `colab_notebooks/`
3. Use `config/colab.json` as default training configuration

## Common Issues

- Missing GPU in Colab: switch runtime to GPU.
- Placeholder file IDs in data prep notebook: replace before running download cells.
- OOM: reduce phase batch size in `config/colab.json` and/or increase gradient accumulation.
