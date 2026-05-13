<!-- filepath: docs/notebook-refactoring-guide.md -->
# Notebook Refactoring Guide

## Overview

All Colab notebooks have been refactored to minimize non-interactive cells. Bootstrap setup code (token resolution, repo discovery, access checks) has been moved to a centralized script: `scripts/colab_notebooks_bootstrap.py`.

**Result:** Notebooks are now cleaner and easier to read, with focus on interactive user logic rather than infrastructure setup.

## Changes Summary

### New Module: `scripts/colab_notebooks_bootstrap.py`

Centralizes all non-interactive bootstrap operations:

- **Token Resolution**: GitHub and HuggingFace tokens from environment or Colab secrets
- **Repo Discovery**: Finds or clones the repository
- **Access Checks**: Validates GitHub and model access
- **Initialization**: Sets up sys.path, installs requirements

### Key Functions

#### `bootstrap_notebook()`

Main entry point for notebook initialization:

```python
from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status

BOOTSTRAP = bootstrap_notebook(
    notebook_name="My Notebook",
    require_colab_requirements=True,
    auto_clone_repo=True,
)
ROOT = BOOTSTRAP["ROOT"]
print_bootstrap_status(BOOTSTRAP)
```

**Returns:**
```python
{
    "ROOT": Path,           # Repo root directory
    "IN_COLAB": bool,       # Running in Colab?
    "GH_TOKEN": str,        # GitHub token (if found)
    "HF_TOKEN": str,        # HuggingFace token (if found)
    "bootstrap_status": str # "ok" or error message
}
```

#### Token Resolution Functions

```python
from scripts.colab_notebooks_bootstrap import (
    resolve_github_token,      # Get GitHub token
    resolve_huggingface_token, # Get HF token
    ensure_repo_root,          # Find or clone repo
)
```

### Notebook Changes

#### Notebook 0: Dataset Preparation

**Before:** 3 large bootstrap cells (~200+ lines combined)
**After:** 1 simple bootstrap cell + configuration

```python
# All setup happens in one call
BOOTSTRAP = bootstrap_notebook(notebook_name="Notebook 0: Dataset Preparation")
ROOT = BOOTSTRAP["ROOT"]
```

#### Notebook 1: Router Inference

**Before:** 2 bootstrap + access check cells
**After:** 1 unified bootstrap cell

#### Notebook 2: Adapter Training

**Before:** 1 large bootstrap cell
**After:** 1 simple bootstrap cell

#### Notebook 3: Adapter Smoke Test

**Before:** 2 bootstrap cells
**After:** 1 unified bootstrap + access check

#### Notebook 4: Simple Adapter Smoke Test

**Before:** 2 bootstrap cells  
**After:** 1 unified bootstrap + access check

## User Instructions

### Using the Refactored Notebooks

Simply **run the first cell** of any notebook. That's it! The bootstrap cell will:

1. Resolve GitHub and HuggingFace tokens
2. Find or clone the repository
3. Install Colab requirements
4. Check model access
5. Print status

Then proceed with the remaining notebook cells as normal.

### Customization

If you need to override bootstrap behavior:

```python
# Use environment variables
import os
os.environ["AADS_REPO_ROOT"] = "/path/to/custom/repo"
os.environ["GH_TOKEN"] = "your_token_here"
os.environ["HF_TOKEN"] = "your_token_here"

# Then run bootstrap
BOOTSTRAP = bootstrap_notebook()
ROOT = BOOTSTRAP["ROOT"]
```

## Developer Guide

### Adding a New Bootstrap-Dependent Notebook

Create your notebook with this structure:

```python
# Cell 1: Bootstrap
from scripts.colab_notebooks_bootstrap import bootstrap_notebook

BOOTSTRAP = bootstrap_notebook(
    notebook_name="My Notebook",
    require_colab_requirements=True,
    auto_clone_repo=True,
)
ROOT = BOOTSTRAP["ROOT"]

# Cell 2+: Your interactive/main logic
```

### Extending Bootstrap Logic

To add new bootstrap functionality:

1. Edit `scripts/colab_notebooks_bootstrap.py`
2. Add your function to the appropriate section (Environment, Repo, or Initialization)
3. Update `bootstrap_notebook()` if needed
4. Update this guide

Example:

```python
def my_custom_setup(root: Path) -> None:
    """Custom setup function."""
    # Your code here
    pass

def bootstrap_notebook(...) -> dict:
    # ... existing code ...
    result["custom_setup"] = "done"
    my_custom_setup(ROOT)
    return result
```

## Migration Reference

### Old Pattern (Deprecated)

```python
# Cell 1: Token setup
GH_TOKEN = os.environ.get('GH_TOKEN', '') or ...
# ... 50+ lines of token code ...

# Cell 2: Repo discovery
CLONE_TARGET = Path('/content/bitirmeprojesi')
# ... 80+ lines of repo discovery code ...

# Cell 3: Imports
from scripts.colab_repo_bootstrap import ...
# ... import setup ...

# Cell 4+: Your logic
```

### New Pattern (Recommended)

```python
# Cell 1: Bootstrap (all setup in one line)
from scripts.colab_notebooks_bootstrap import bootstrap_notebook
BOOTSTRAP = bootstrap_notebook(notebook_name="My Notebook")
ROOT = BOOTSTRAP["ROOT"]

# Cell 2+: Your logic
```

## Benefits

✅ **Cleaner Notebooks**

## References

- Repository-specific bootstrap pattern influenced by general notebook engineering guidance (modular bootstrap scripts, small starter cells) and the project's internal `scripts/colab_notebooks_bootstrap.py` implementation.
- Focus on interactive/main logic
- Less boilerplate
- Easier to understand flow

✅ **Maintainability**
- Single source of truth for bootstrap code
- Easy to update token resolution, repo discovery, etc.
- DRY (Don't Repeat Yourself)

✅ **Consistency**
- All notebooks follow same pattern
- Familiar structure for contributors
- Standard error handling

✅ **Flexibility**
- Environment variable overrides
- Optional Colab-specific features
- Works in local Python environments too

## Troubleshooting

### "Repo not found" Error

**Problem:** Bootstrap can't locate the repository

**Solutions:**
1. Set environment variable: `export AADS_REPO_ROOT=/path/to/repo`
2. Run notebook from within the repo directory
3. In Colab, provide GitHub token via secret for private repos

### "Requirements installation failed"

**Problem:** Pip fails to install packages

**Solutions:**
1. Run notebook with `require_colab_requirements=False` to skip
2. Check internet connection in Colab
3. Manually install: `pip install -r requirements_colab.txt`

### Token Not Detected

**Problem:** GitHub or HuggingFace tokens not found

**Solutions:**
1. Set environment variables before running:
   ```python
   os.environ["GH_TOKEN"] = "your_token"
   os.environ["HF_TOKEN"] = "your_token"
   ```
2. In Colab, add secrets via "Secrets" panel (click key icon)
3. Check bootstrap output for which tokens are found

## Testing

To verify bootstrap functionality:

```bash
cd scripts
python -c "from colab_notebooks_bootstrap import bootstrap_notebook; result = bootstrap_notebook(require_colab_requirements=False); print(result)"
```

Should print:
```
[BOOTSTRAP] Repo root: /path/to/repo
[BOOTSTRAP] ... bootstrap complete.
{'ROOT': PosixPath(...), 'IN_COLAB': False, 'GH_TOKEN': '', 'HF_TOKEN': '', 'bootstrap_status': 'ok'}
```
