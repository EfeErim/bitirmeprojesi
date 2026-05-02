# Auto-extracted from colab_notebooks/4_simple_direct_adapter_test_ui.ipynb cell 1.
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

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb4_simple_ui_helpers import run_bootstrap_notebook_nb4
BOOTSTRAP = run_bootstrap_notebook_nb4()
ROOT = BOOTSTRAP["ROOT"]

# Setup adapter search and access check
SEARCH_ROOTS = [str(ROOT / "models/adapters")]

from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
from src.core.config_manager import get_config

ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=[])
print_notebook_access_report(ACCESS_REPORT, print_fn=print)
