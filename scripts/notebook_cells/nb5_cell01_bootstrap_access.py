# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper
from pathlib import Path
import os, subprocess
import shutil

CLONE_TARGET = Path('/content/bitirmeprojesi')  # Colab GitHub bootstrap contract
REPO_URL = os.environ.get('AADS_REPO_URL', 'https://github.com/EfeErim/bitirmeprojesi.git')

print('[BOOTSTRAP] Notebook 5 bootstrap cell started.', flush=True)
print(f'[BOOTSTRAP] cwd={Path.cwd()}', flush=True)
print(f'[BOOTSTRAP] clone_target={CLONE_TARGET}', flush=True)

# Git clone if needed
if not CLONE_TARGET.exists():
    clone_url = REPO_URL
    print(f'[BOOTSTRAP] cloning repo from {REPO_URL}', flush=True)
    subprocess.run(['git', 'clone', '--depth', '1', '--progress', clone_url, str(CLONE_TARGET)], check=True)
else:
    print('[BOOTSTRAP] clone target already exists.', flush=True)

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb5_calibration_helpers import run_bootstrap_notebook_nb5, run_access_check_nb5
BOOTSTRAP = run_bootstrap_notebook_nb5()
ROOT = BOOTSTRAP["ROOT"]
print(f'[BOOTSTRAP] resolved ROOT={ROOT}', flush=True)

# Check model access
ACCESS_REPORT = run_access_check_nb5(ROOT, print_fn=print)
