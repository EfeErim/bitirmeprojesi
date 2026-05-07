# Auto-extracted from colab_notebooks/4_simple_direct_adapter_test_ui.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper
from pathlib import Path
import os, subprocess
import shutil

import sys

# Ensure repo root is in sys.path before any imports
def _ensure_sys_path_for_cell_script():
    """Ensure sys.path includes repo root so notebook cell scripts can import."""
    # Look for repository root by walking parents from cwd
    def _find_repo_root(start: Path = None):
        start = start or Path.cwd()
        for p in [start] + list(start.parents):
            if (p / 'scripts').is_dir() and (p / 'src').is_dir():
                return p.resolve()
        return None

    repo = _find_repo_root(Path.cwd())
    if repo:
        repo_root = str(repo)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return repo_root
    return None

_ensure_sys_path_for_cell_script()

CLONE_TARGET = Path('/content/bitirmeprojesi')  # Colab GitHub bootstrap contract
REPO_URL = os.environ.get('AADS_REPO_URL', 'https://github.com/EfeErim/bitirmeprojesi.git')

# Git clone if needed
if not CLONE_TARGET.exists():
    clone_url = REPO_URL
    subprocess.run(['git', 'clone', '--depth', '1', clone_url, str(CLONE_TARGET)], check=True)

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb4_simple_ui_helpers import run_bootstrap_notebook_nb4
BOOTSTRAP = run_bootstrap_notebook_nb4()
ROOT = BOOTSTRAP["ROOT"]

# Setup adapter search and access check
SEARCH_ROOTS = [str(ROOT / "models/adapters"), str(ROOT / "runs")]

from scripts.colab_repo_bootstrap import (
    collect_notebook_access_report,
    login_and_check_hf_token,
    print_notebook_access_report,
    resolve_hf_token,
)
from src.core.config_manager import get_config

CONFIG_FOR_ACCESS = get_config(environment="colab")
BACKBONE_MODEL_NAME = str(
    dict(dict(CONFIG_FOR_ACCESS.get("training", {})).get("continual", {}))
    .get("backbone", {})
    .get("model_name", "")
).strip()
ACCESS_REPORT = collect_notebook_access_report(
    repo_root=ROOT,
    hf_model_ids=[BACKBONE_MODEL_NAME] if BACKBONE_MODEL_NAME.strip() else [],
)
print_notebook_access_report(ACCESS_REPORT, print_fn=print)
if BACKBONE_MODEL_NAME:
    print(f"[KONTROL] Adapter backbone modeli: {BACKBONE_MODEL_NAME}")
if resolve_hf_token():
    login_and_check_hf_token(print_fn=print)
else:
    print("[HF] Token bulunamadi. Gated backbone icin Colab secret olarak HF_TOKEN ekleyin ve runtime'i yeniden baslatin.")
