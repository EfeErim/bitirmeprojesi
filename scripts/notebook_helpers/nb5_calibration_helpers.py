from __future__ import annotations
from pathlib import Path


def run_bootstrap_notebook_nb5(notebook_name: str = "Notebook 5: Router Calibration", require_colab_requirements: bool = True, auto_clone_repo: bool = True) -> dict:
    """Bootstrap Notebook 5."""
    from scripts.colab_notebooks_bootstrap import bootstrap_notebook, print_bootstrap_status
    try:
        from scripts.colab_repo_bootstrap import _ensure_repo_root_for_update_check
        repo_root_for_update_check = _ensure_repo_root_for_update_check()
    except Exception:
        repo_root_for_update_check = None
    
    # [KONTROL] Ilk hucre: Bootstrap kontrati
    BOOTSTRAP = bootstrap_notebook(
        notebook_name=notebook_name,
        require_colab_requirements=require_colab_requirements,
        auto_clone_repo=auto_clone_repo,
    )
    ROOT = BOOTSTRAP["ROOT"]
    print_bootstrap_status(BOOTSTRAP)
    return BOOTSTRAP


def run_access_check_nb5(ROOT: Path, print_fn=print) -> dict:
    """Check calibration router model access."""
    from src.core.config_manager import get_config
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report
    
    CONFIG = get_config(environment='colab')
    ROUTER_VLM_CFG = dict(dict(CONFIG.get('router', {})).get('vlm', {}))
    ROUTER_MODEL_IDS = [
        str(model_id).strip()
        for model_id in list(dict(ROUTER_VLM_CFG.get('model_ids', {})).values())
        if str(model_id).strip()
    ]
    
    print_fn("[SETUP] Checking model access for router calibration...")
    ACCESS_REPORT = collect_notebook_access_report(repo_root=ROOT, hf_model_ids=ROUTER_MODEL_IDS)
    print_notebook_access_report(ACCESS_REPORT, print_fn=print_fn)
    return ACCESS_REPORT


__all__ = [
    'run_bootstrap_notebook_nb5',
    'run_access_check_nb5',
]
