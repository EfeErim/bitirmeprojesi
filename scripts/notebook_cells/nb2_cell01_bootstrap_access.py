# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 1.
# Keep notebook execute-only cells thin; edit behavior here.

# Bootstrap notebook via helper
from scripts.colab_repo_bootstrap import probe_repo_update_status
try:
    from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
    repo_root_for_update_check = _ensure_repo_root_for_update_check()
except Exception:
    repo_root_for_update_check = None

def _build_repo_access_url(base: str, params: dict) -> str:
    """Build repo access URL for freshness check (validation contract)."""
    from urllib.parse import urlencode
    return f"{base}?{urlencode(params)}"

# [KONTROL] Ilk hucre: Bootstrap kontrati
from scripts.notebook_helpers.nb2_training_helpers import run_bootstrap_notebook_nb2, run_access_check_nb2
BOOTSTRAP = run_bootstrap_notebook_nb2()
ROOT = BOOTSTRAP["ROOT"]

# Check model access
ACCESS_REPORT = run_access_check_nb2(ROOT, print_fn=print)
