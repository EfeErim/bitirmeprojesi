# Auto-extracted from colab_notebooks/3_validate_exported_adapter_directly.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper
from pathlib import Path
import os, subprocess
import shutil

CLONE_TARGET = Path('/content/bitirmeprojesi')  # Colab GitHub bootstrap contract
REPO_URL = os.environ.get('AADS_REPO_URL', 'https://github.com/your-org/aads-continual.git')

# Git clone if needed
if not CLONE_TARGET.exists():
    clone_url = REPO_URL
    subprocess.run(['git', 'clone', '--depth', '1', clone_url, str(CLONE_TARGET)], check=True)

from scripts.colab_repo_bootstrap import probe_repo_update_status
try:
    from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
    repo_root_for_update_check = _ensure_repo_root_for_update_check()
except Exception:
    repo_root_for_update_check = None

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb3_smoke_test_helpers import run_bootstrap_notebook_nb3, run_access_check_nb3
BOOTSTRAP = run_bootstrap_notebook_nb3()
ROOT = BOOTSTRAP["ROOT"]

# Check model access
ACCESS_REPORT = run_access_check_nb3(ROOT, print_fn=print)
