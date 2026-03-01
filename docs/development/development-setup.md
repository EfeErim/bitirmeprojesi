# Development Setup (v6)

## Setup

```powershell
cd "D:\bitirme projesi"
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Core Sanity

```powershell
python scripts/validate_notebook_imports.py
python scripts/check_markdown_links.py --root .
python scripts/run_test_suites.py --suite unit --suite colab --suite integration/core
```

## Legacy Material

Legacy v5.5 documentation/notebooks are archived under `docs/archive/v5_legacy/` and `colab_notebooks/archive/v5_legacy/`.
